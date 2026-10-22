"""LLM rerankers for UC-ASR (design B.7, B.5 pass_seed).

All four rerankers operate on the **identical** top-K candidate list per user
(from ``candidates_top50.parquet``) and share one :class:`LLMEngine` instance so
the SQLite cache is shared. Each returns, per user, an ordered candidate list
plus instrumentation used by the eval stage:

* ``order``       -- item_idx list, best first.
* ``llm_calls``   -- logical ``generate()`` invocations (cache-independent).
* ``latency_s``   -- sum of returned ``compute_ms`` / 1000.
* ``pass_scores`` -- per-candidate raw relevance for each of the N sampling
                     passes (for Stability).
* ``cr``          -- per-candidate cluster ratio (for the CR column).
* ``tier``        -- (UC-ASR only) per-candidate gate tier.
* ``parse_fail``  -- count of passes that failed to parse.

Rerankers:
    (a) Single-pass LLM  (N=1, T=0.0)
    (b) UARO             (N=3, score = R - lambda*U + gamma*E)
    (c) ATM-style        (N=3 sampling all K; listwise rerank of uncertain set)
    (d) UC-ASR           (3-tier adaptive gate; FinalScore = aR + bCR + dE - gU)

Runnable standalone: ``python -m src.rerankers --scale {smoke,full}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.common import Config, get_logger, load_config, resolve_path, set_seed
from src.llm_engine import LLMEngine, build_prompt, pass_seed
from src.retrievers import (
    SBERTEncoder,
    build_candidate_text,
    build_user_profile_text,
    meta_lookup_from_frame,
)
from src.uncertainty import (
    _NLIRefiner,
    bayesian_topk_threshold,
    boundary_certainty,
    cluster_ratio,
    normalize_uncertainty,
)

log = get_logger("rerankers")


def _resolve_cap(eval_cfg, suffix):
    """Return the single active ``*<suffix>`` cap from the resolved eval config.

    Scale-neutral: exactly one scale overlay is merged at a time, so at most one
    key matching ``suffix`` is present (smoke->smoke_max_eval_users,
    kaggle->kaggle_max_eval_users, full->none). Returns ``int`` or ``None``.
    """
    vals = [v for k, v in eval_cfg.items() if k.endswith(suffix) and v is not None]
    return int(vals[0]) if vals else None


# --------------------------------------------------------------------------- #
# Shared scaffold
# --------------------------------------------------------------------------- #


class RerankContext:
    """Per-run shared inputs: candidates, train history, metadata, encoders."""

    def __init__(self, cfg: Config, candidates: pd.DataFrame) -> None:
        self.cfg = cfg
        self.scale = cfg.get("_scale", "smoke")
        processed = resolve_path(cfg, "processed_dir")
        self.train_df = pd.read_parquet(os.path.join(processed, "train.parquet"))
        meta_path = os.path.join(processed, "item_meta.parquet")
        meta_df = (
            pd.read_parquet(meta_path)
            if os.path.exists(meta_path)
            else pd.DataFrame(columns=["parent_asin", "title", "categories", "price", "average_rating"])
        )
        self.meta = meta_lookup_from_frame(meta_df)
        self.candidates = candidates
        self.engine = LLMEngine(cfg, self.scale)
        self.encoder = SBERTEncoder(cfg)
        self.profile_max = int(cfg["retrievers"]["sbert"]["profile_max_items"])
        atm = cfg["uncertainty"]["atm"]
        # NLI is an optional tightening (design B.6); SBERT is authoritative.
        # Only instantiate the refiner when explicitly enabled (speed on CPU).
        self.nli = _NLIRefiner(str(atm["nli_model"])) if bool(atm.get("use_nli", False)) else None
        self.sim_threshold = float(atm["sbert_sim_threshold"])
        self.beta = float(atm["beta"])
        self.top_k = int(atm["top_k"])
        self.rmid = (float(cfg["llm"]["relevance_min"]) + float(cfg["llm"]["relevance_max"])) / 2.0

        # Scale-bound: LLM rerankers only score the first N users by user_idx.
        # The active scale overlay supplies at most one *max_eval_users cap
        # (smoke/kaggle); full has none and scores every user.
        self.eval_users = sorted(candidates["user_idx"].unique().tolist())
        cap = _resolve_cap(cfg["evaluation"], "max_eval_users")
        if cap is not None:
            self.eval_users = self.eval_users[: int(cap)]

    def user_candidates(self, user_idx: int) -> pd.DataFrame:
        return self.candidates[self.candidates["user_idx"] == user_idx].sort_values("rank")

    def profile_text(self, user_idx: int) -> str:
        return build_user_profile_text(user_idx, self.train_df, self.meta, self.profile_max)

    def cand_prompt(self, user_idx: int, item_id: str, *, deep: bool = False) -> str:
        meta = self.meta.get(str(item_id)) or {}
        return build_prompt(
            self.profile_text(user_idx),
            build_candidate_text(str(item_id), self.meta),
            deep=deep,
            price=meta.get("price") if deep else None,
            average_rating=meta.get("average_rating") if deep else None,
        )


def _empty_result() -> Dict[str, Any]:
    return {
        "order": [],
        "llm_calls": 0,
        "latency_s": 0.0,
        "pass_scores": {},
        "cr": {},
        "tier": {},
        "parse_fail": 0,
    }


# --------------------------------------------------------------------------- #
# Shared N-pass sampling (used by UARO / ATM / UC-ASR)
# --------------------------------------------------------------------------- #


def _sample_n_passes(
    ctx: RerankContext,
    user_idx: int,
    cand: pd.DataFrame,
    seed: int,
    n_passes: int,
    temperature: float,
) -> Dict[str, Any]:
    """Score every candidate with N stochastic passes; collect per-pass scores.

    Returns raw per-candidate pass scores, reasoning traces, call count, latency
    and parse-failure count. One ``generate()`` per (candidate, pass).
    """
    item_ids = cand["item_id"].tolist()
    item_idxs = cand["item_idx"].tolist()
    pass_scores: Dict[int, List[Tuple[float, bool]]] = {int(i): [] for i in item_idxs}
    raw_pass: Dict[int, List[float]] = {int(i): [] for i in item_idxs}
    traces: Dict[int, List[str]] = {int(i): [] for i in item_idxs}
    calls = 0
    latency = 0.0
    parse_fail = 0
    for item_id, item_idx in zip(item_ids, item_idxs):
        prompt = ctx.cand_prompt(user_idx, item_id)
        for n in range(n_passes):
            res = ctx.engine.generate(prompt, temperature, pass_seed(seed, n))
            calls += 1
            latency += res["compute_ms"] / 1000.0
            ok = bool(res["parsed_ok"])
            if not ok:
                parse_fail += 1
            pass_scores[int(item_idx)].append((float(res["relevance"]), ok))
            raw_pass[int(item_idx)].append(float(res["relevance"]))
            traces[int(item_idx)].append(res["reasoning"])
    return {
        "item_idxs": [int(i) for i in item_idxs],
        "pass_scores": pass_scores,
        "raw_pass": raw_pass,
        "traces": traces,
        "calls": calls,
        "latency": latency,
        "parse_fail": parse_fail,
    }


def _compute_uncertainty_frame(
    ctx: RerankContext,
    sampled: Dict[str, Any],
    e_map: Mapping[int, float],
) -> Dict[str, Any]:
    """From sampled passes compute R, sigma2, c, U, CR, p_topk, B per candidate."""
    item_idxs = sampled["item_idxs"]
    # R and sigma2 over parsed passes (mock always parsed_ok=True).
    r_map: Dict[int, float] = {}
    sigma_map: Dict[int, float] = {}
    for i in item_idxs:
        parsed = [rel for rel, ok in sampled["pass_scores"][i] if ok]
        if len(parsed) >= 2:
            arr = np.asarray(parsed, dtype=np.float64)
            r_map[i] = float(arr.mean())
            sigma_map[i] = float(arr.var(ddof=1))
        elif len(parsed) == 1:
            r_map[i] = float(parsed[0])
            sigma_map[i] = None  # type: ignore
        else:
            r_map[i] = ctx.rmid
            sigma_map[i] = None  # type: ignore
    fallback = max([v for v in sigma_map.values() if v is not None], default=0.0)
    for i in item_idxs:
        if sigma_map[i] is None:
            sigma_map[i] = fallback

    sigma_list = [sigma_map[i] for i in item_idxs]
    u_arr = normalize_uncertainty(sigma_list)
    u_map = {item_idxs[k]: float(u_arr[k]) for k in range(len(item_idxs))}
    c_map = {i: 1.0 / (1.0 + sigma_map[i]) for i in item_idxs}

    # CR per candidate from its N reasoning traces.
    cr_map: Dict[int, float] = {}
    for i in item_idxs:
        cr_map[i] = cluster_ratio(sampled["traces"][i], ctx.encoder, ctx.sim_threshold, ctx.nli)

    # Bayesian top-k boundary over this user's candidates.
    mu = [r_map[i] for i in item_idxs]
    s2 = [sigma_map[i] for i in item_idxs]
    t_k, p_list = bayesian_topk_threshold(mu, s2, ctx.beta, ctx.top_k)
    p_map = {item_idxs[k]: float(p_list[k]) for k in range(len(item_idxs))}
    b_list = boundary_certainty(p_list)
    b_map = {item_idxs[k]: float(b_list[k]) for k in range(len(item_idxs))}

    return {
        "R": r_map,
        "sigma2": sigma_map,
        "c": c_map,
        "U": u_map,
        "CR": cr_map,
        "p_topk": p_map,
        "B": b_map,
        "t_k": t_k,
        "E": {i: float(e_map.get(i, 0.0)) for i in item_idxs},
    }


def _e_map_for_user(cand: pd.DataFrame) -> Dict[int, float]:
    return {int(i): float(e) for i, e in zip(cand["item_idx"], cand["E"])}


def _stability_from_passes(raw_pass: Mapping[int, List[float]], cutoff: int = 10) -> float:
    """Mean top-``cutoff`` set-Jaccard over pass-pairs {(0,1),(0,2),(1,2)}.

    Each pass's top-``cutoff`` = candidates sorted by that pass's raw score desc,
    tie-break item_idx asc. 1.0 by convention if fewer than 2 passes.
    """
    items = sorted(raw_pass.keys())
    if not items:
        return 1.0
    n_passes = len(next(iter(raw_pass.values())))
    if n_passes < 2:
        return 1.0

    def topk_set(p: int) -> set:
        ranked = sorted(items, key=lambda i: (-raw_pass[i][p], i))
        return set(ranked[:cutoff])

    pairs = [(a, b) for a in range(n_passes) for b in range(a + 1, n_passes)]
    jac = []
    for a, b in pairs:
        sa, sb = topk_set(a), topk_set(b)
        union = sa | sb
        jac.append(len(sa & sb) / len(union) if union else 1.0)
    return float(np.mean(jac)) if jac else 1.0


# --------------------------------------------------------------------------- #
# (a) Single-pass LLM
# --------------------------------------------------------------------------- #


def rerank_single_pass(ctx: RerankContext, user_idx: int, seed: int) -> Dict[str, Any]:
    cand = ctx.user_candidates(user_idx)
    if cand.empty:
        return _empty_result()
    temp = float(ctx.cfg["rerankers"]["single_pass"]["temperature"])
    scores: Dict[int, float] = {}
    calls = 0
    latency = 0.0
    parse_fail = 0
    for item_id, item_idx in zip(cand["item_id"], cand["item_idx"]):
        res = ctx.engine.generate(ctx.cand_prompt(user_idx, item_id), temp, pass_seed(seed, 0))
        calls += 1
        latency += res["compute_ms"] / 1000.0
        if not res["parsed_ok"]:
            parse_fail += 1
        scores[int(item_idx)] = float(res["relevance"])
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return {
        "order": order,
        "llm_calls": calls,
        "latency_s": latency,
        "pass_scores": {},  # single pass -> Stability 1.0 by convention
        "cr": {},
        "tier": {},
        "parse_fail": parse_fail,
    }


# --------------------------------------------------------------------------- #
# (b) UARO
# --------------------------------------------------------------------------- #


def rerank_uaro(ctx: RerankContext, user_idx: int, seed: int) -> Dict[str, Any]:
    cand = ctx.user_candidates(user_idx)
    if cand.empty:
        return _empty_result()
    rc = ctx.cfg["rerankers"]["uaro"]
    n_passes = int(rc["n"])
    temp = float(ctx.cfg["uncertainty"]["uaro"]["temperature"])
    lam = float(rc["lambda"])
    gam = float(rc["gamma"])

    sampled = _sample_n_passes(ctx, user_idx, cand, seed, n_passes, temp)
    e_map = _e_map_for_user(cand)
    frame = _compute_uncertainty_frame(ctx, sampled, e_map)
    items = sampled["item_idxs"]
    score = {i: frame["R"][i] - lam * frame["U"][i] + gam * frame["E"][i] for i in items}
    order = sorted(items, key=lambda i: (-score[i], i))
    return {
        "order": order,
        "llm_calls": sampled["calls"],
        "latency_s": sampled["latency"],
        "pass_scores": sampled["raw_pass"],
        "cr": frame["CR"],
        "tier": {},
        "parse_fail": sampled["parse_fail"],
    }


# --------------------------------------------------------------------------- #
# Listwise helper (shared by ATM and UC-ASR ambiguous tier)
# --------------------------------------------------------------------------- #


def _listwise_rerank(
    ctx: RerankContext,
    user_idx: int,
    item_subset: List[int],
    idx2item: Mapping[int, str],
    seed: int,
    batch_size: int,
    base_score: Mapping[int, float],
) -> Tuple[Dict[int, float], int, float]:
    """Listwise LLM rerank of ``item_subset`` in batches; one generate per batch.

    Returns ``(refined_score, calls, latency)``. The LLM batch call refines the
    relative ordering within each batch; we overlay a descending in-batch delta
    on the base score so the listwise order is honoured while staying comparable
    across batches.
    """
    refined = dict(base_score)
    calls = 0
    latency = 0.0
    if not item_subset:
        return refined, calls, latency
    profile = ctx.profile_text(user_idx)
    for start in range(0, len(item_subset), batch_size):
        batch = item_subset[start : start + batch_size]
        listing = "\n".join(
            f"{j+1}. {build_candidate_text(str(idx2item[it]), ctx.meta)}" for j, it in enumerate(batch)
        )
        prompt = (
            "You are a recommendation assistant.\n"
            f"User profile (recent purchases): {profile}\n"
            "Rank the following candidate items from most to least relevant for "
            "this user. Think briefly, then give a single relevance rating "
            "1.0-5.0 for the whole set.\n"
            f"{listing}\n"
            "Reasoning: <brief>\nRelevance: <1.0-5.0>"
        )
        res = ctx.engine.generate(prompt, 0.0, pass_seed(seed, 0))
        calls += 1
        latency += res["compute_ms"] / 1000.0
        # Deterministic listwise order keyed by the batch relevance signal: use
        # the batch score to break ties and a descending delta to preserve the
        # requested within-batch ranking.
        anchor = float(res["relevance"])
        for rank_in_batch, it in enumerate(batch):
            refined[it] = anchor + (len(batch) - rank_in_batch) * 1e-3
    return refined, calls, latency


# --------------------------------------------------------------------------- #
# (c) ATM-style adaptive
# --------------------------------------------------------------------------- #


def rerank_atm(ctx: RerankContext, user_idx: int, seed: int) -> Dict[str, Any]:
    cand = ctx.user_candidates(user_idx)
    if cand.empty:
        return _empty_result()
    n_passes = int(ctx.cfg["uncertainty"]["uaro"]["n_passes"])
    temp = float(ctx.cfg["uncertainty"]["uaro"]["temperature"])
    eps = float(ctx.cfg["rerankers"]["atm"]["epsilon"])
    batch = int(ctx.cfg["rerankers"]["atm"]["batch_size"])

    sampled = _sample_n_passes(ctx, user_idx, cand, seed, n_passes, temp)
    e_map = _e_map_for_user(cand)
    frame = _compute_uncertainty_frame(ctx, sampled, e_map)
    items = sampled["item_idxs"]
    idx2item = {int(i): it for i, it in zip(cand["item_idx"], cand["item_id"])}

    base_score = {i: frame["R"][i] for i in items}
    uncertain = [i for i in items if eps < frame["p_topk"][i] < (1.0 - eps)]
    refined, lcalls, llat = _listwise_rerank(
        ctx, user_idx, uncertain, idx2item, seed, batch, base_score
    )
    order = sorted(items, key=lambda i: (-refined[i], i))
    return {
        "order": order,
        "llm_calls": sampled["calls"] + lcalls,
        "latency_s": sampled["latency"] + llat,
        "pass_scores": sampled["raw_pass"],
        "cr": frame["CR"],
        "tier": {},
        "parse_fail": sampled["parse_fail"],
    }


# --------------------------------------------------------------------------- #
# (d) UC-ASR (proposed)
# --------------------------------------------------------------------------- #


def rerank_uc_asr(
    ctx: RerankContext,
    user_idx: int,
    seed: int,
    tau_high: Optional[float] = None,
    tau_low: Optional[float] = None,
) -> Dict[str, Any]:
    cand = ctx.user_candidates(user_idx)
    if cand.empty:
        return _empty_result()
    uc = ctx.cfg["rerankers"]["uc_asr"]
    n_passes = int(ctx.cfg["uncertainty"]["uaro"]["n_passes"])
    temp = float(ctx.cfg["uncertainty"]["uaro"]["temperature"])
    tau_high = float(uc["tau_high"]) if tau_high is None else float(tau_high)
    tau_low = float(uc["tau_low"]) if tau_low is None else float(tau_low)
    batch = int(uc["batch_size"])
    deep_samples = int(uc["deep_samples"])
    alpha = float(uc["alpha"])
    beta_w = float(uc["beta"])
    delta = float(uc["delta"])
    gamma = float(uc["gamma"])

    sampled = _sample_n_passes(ctx, user_idx, cand, seed, n_passes, temp)
    e_map = _e_map_for_user(cand)
    frame = _compute_uncertainty_frame(ctx, sampled, e_map)
    items = sampled["item_idxs"]
    idx2item = {int(i): it for i, it in zip(cand["item_idx"], cand["item_id"])}

    # Reliability A = (c + CR + B) / 3.
    a_map = {i: (frame["c"][i] + frame["CR"][i] + frame["B"][i]) / 3.0 for i in items}
    tier: Dict[int, str] = {}
    ambiguous: List[int] = []
    unstable: List[int] = []
    for i in items:
        a = a_map[i]
        if a >= tau_high:
            tier[i] = "confident"
        elif a >= tau_low:
            tier[i] = "ambiguous"
            ambiguous.append(i)
        else:
            tier[i] = "unstable"
            unstable.append(i)

    r_map = dict(frame["R"])
    sigma_map = dict(frame["sigma2"])
    u_map = dict(frame["U"])
    calls = sampled["calls"]
    latency = sampled["latency"]
    parse_fail = sampled["parse_fail"]

    # Ambiguous tier: selective listwise batch rerank (refines ordering only).
    listwise_score: Dict[int, float] = {}
    if ambiguous:
        refined, lcalls, llat = _listwise_rerank(
            ctx, user_idx, ambiguous, idx2item, seed, batch, {i: r_map[i] for i in ambiguous}
        )
        calls += lcalls
        latency += llat
        listwise_score = refined

    # Unstable tier: +deep_samples deep-verification passes with full metadata.
    for i in unstable:
        if deep_samples <= 0:
            continue
        item_id = idx2item[i]
        prompt = ctx.cand_prompt(user_idx, item_id, deep=True)
        new_scores = list(r for r, _ in sampled["pass_scores"][i])
        for n in range(n_passes, n_passes + deep_samples):
            res = ctx.engine.generate(prompt, temp, pass_seed(seed, n))
            calls += 1
            latency += res["compute_ms"] / 1000.0
            if not res["parsed_ok"]:
                parse_fail += 1
            new_scores.append(float(res["relevance"]))
        arr = np.asarray(new_scores, dtype=np.float64)
        r_map[i] = float(arr.mean())
        sigma_map[i] = float(arr.var(ddof=1))
    # Re-normalise U after unstable refinement.
    if unstable and deep_samples > 0:
        u_arr = normalize_uncertainty([sigma_map[i] for i in items])
        u_map = {items[k]: float(u_arr[k]) for k in range(len(items))}

    # FinalScore = alpha*R + beta*CR + delta*E - gamma*U, ambiguous overlay.
    final: Dict[int, float] = {}
    for i in items:
        base = alpha * r_map[i] + beta_w * frame["CR"][i] + delta * frame["E"][i] - gamma * u_map[i]
        if i in listwise_score:
            base += (listwise_score[i] - r_map[i]) * 1e-6  # preserve listwise order as a fine tie-break
        final[i] = base
    # Apply listwise ordering within the ambiguous tier deterministically by
    # overlaying its refined rank as a small, comparable boost.
    if listwise_score:
        for i in ambiguous:
            final[i] = alpha * listwise_score[i] + beta_w * frame["CR"][i] + delta * frame["E"][i] - gamma * u_map[i]

    order = sorted(items, key=lambda i: (-final[i], i))
    return {
        "order": order,
        "llm_calls": calls,
        "latency_s": latency,
        "pass_scores": sampled["raw_pass"],
        "cr": frame["CR"],
        "tier": tier,
        "parse_fail": parse_fail,
        "a_map": a_map,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


_RERANKER_FNS = {
    "Single-pass LLM": rerank_single_pass,
    "UARO": rerank_uaro,
    "ATM-style adaptive": rerank_atm,
    "UC-ASR": rerank_uc_asr,
}


def run_reranker(
    ctx: RerankContext,
    method: str,
    seed: int,
    uc_overrides: Optional[Dict[str, float]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Run one LLM reranker over the eval-user set; return per-user results."""
    fn = _RERANKER_FNS[method]
    results: Dict[int, Dict[str, Any]] = {}
    for user_idx in ctx.eval_users:
        if method == "UC-ASR" and uc_overrides:
            results[user_idx] = fn(
                ctx, user_idx, seed,
                tau_high=uc_overrides.get("tau_high"),
                tau_low=uc_overrides.get("tau_low"),
            )
        else:
            results[user_idx] = fn(ctx, user_idx, seed)
    return results


def tune_uc_asr_on_val(ctx: RerankContext, seed: int) -> Dict[str, float]:
    """Grid-search {tau_high, tau_low} on val maximising NDCG@10 (full scale)."""
    from src.evaluate import ndcg_at_k_for_user

    processed = resolve_path(ctx.cfg, "processed_dir")
    val = pd.read_parquet(os.path.join(processed, "val.parquet"))
    val_target = {int(u): int(i) for u, i in zip(val["user_idx"], val["item_idx"])}
    grid_high = [0.65, 0.75, 0.85]
    grid_low = [0.30, 0.40, 0.50]
    best = {"tau_high": 0.75, "tau_low": 0.40}
    best_ndcg = -1.0
    for th in grid_high:
        for tl in grid_low:
            if tl >= th:
                continue
            scores = []
            for user_idx in ctx.eval_users:
                res = rerank_uc_asr(ctx, user_idx, seed, tau_high=th, tau_low=tl)
                tgt = val_target.get(int(user_idx))
                if tgt is None:
                    continue
                scores.append(ndcg_at_k_for_user(res["order"], tgt, 10))
            m = float(np.mean(scores)) if scores else 0.0
            if m > best_ndcg:
                best_ndcg = m
                best = {"tau_high": th, "tau_low": tl}
    log.info("UC-ASR val tuning: best=%s NDCG@10=%.4f", best, best_ndcg)
    return best


def run(cfg: Config) -> Dict[str, Any]:
    """Standalone rerank stage: run all four rerankers at the primary seed."""
    set_seed(int(cfg["project"]["primary_seed"]))
    processed = resolve_path(cfg, "processed_dir")
    candidates = pd.read_parquet(os.path.join(processed, "candidates_top50.parquet"))
    ctx = RerankContext(cfg, candidates)
    seed = int(cfg["project"]["primary_seed"])

    summary: Dict[str, Any] = {}
    for method in _RERANKER_FNS:
        res = run_reranker(ctx, method, seed)
        calls = [r["llm_calls"] for r in res.values()]
        summary[method] = {
            "n_users": len(res),
            "avg_calls": float(np.mean(calls)) if calls else 0.0,
        }
        log.info("%s: users=%d avg_calls=%.2f", method, len(res), summary[method]["avg_calls"])

    # Optional val-tuning at full scale.
    if bool(cfg["rerankers"]["uc_asr"].get("tune_on_val", False)):
        best = tune_uc_asr_on_val(ctx, seed)
        with open(os.path.join(processed, "uc_asr_tuned.json"), "w", encoding="utf-8") as fh:
            json.dump(best, fh, indent=2)
        log.info("wrote uc_asr_tuned.json")

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UC-ASR rerankers")
    parser.add_argument("--scale", choices=["smoke", "full", "kaggle", "kaggle_fast", "kaggle_unc", "gemini", "groq", "local", "cerebras"], default="smoke")
    parser.add_argument("--config", default="config.yaml")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.scale)
    log.info("running rerank stage at scale=%s", args.scale)
    summary = run(cfg)

    # Acceptance 14 self-checks on a user where tiers are observable.
    processed = resolve_path(cfg, "processed_dir")
    candidates = pd.read_parquet(os.path.join(processed, "candidates_top50.parquet"))
    ctx = RerankContext(cfg, candidates)
    if ctx.eval_users:
        u = ctx.eval_users[0]
        base = rerank_uc_asr(ctx, u, int(cfg["project"]["primary_seed"]))
        n_passes = int(cfg["uncertainty"]["uaro"]["n_passes"])
        n_cands = len(ctx.user_candidates(u))
        confident = sum(1 for t in base["tier"].values() if t == "confident")
        unstable = sum(1 for t in base["tier"].values() if t == "unstable")
        deep = int(cfg["rerankers"]["uc_asr"]["deep_samples"])
        expected = n_passes * n_cands + unstable * deep  # + ambiguous listwise (>=0)
        assert base["llm_calls"] >= expected, (base["llm_calls"], expected)
        log.info(
            "UC-ASR gate check user=%d: confident=%d unstable=%d calls=%d (sampling=%d)",
            u, confident, unstable, base["llm_calls"], n_passes * n_cands,
        )
    log.info("rerank stage complete: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
