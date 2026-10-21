"""Evaluation + orchestration for UC-ASR (design B.8, schema D.4).

Computes Recall@10, NDCG@10, MRR, Avg. LLM calls, Latency, Stability, CR for all
8 methods across 3 seeds; writes the thesis main table (md + csv), 10 ablations,
paired Wilcoxon significance, and 4 publication-ready figures.

Runnable standalone: ``python -m src.evaluate --scale {smoke,full}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend at import (Part C)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.common import Config, ensure_dirs, get_logger, load_config, resolve_path, set_seed  # noqa: E402
from src.rerankers import RerankContext, _resolve_cap, run_reranker, tune_uc_asr_on_val  # noqa: E402
from src.retrievers import (  # noqa: E402
    SBERTEncoder,
    build_candidate_text,
    build_user_profile_text,
    meta_lookup_from_frame,
    popularity_scores,
    retrieve_candidates,
)

log = get_logger("evaluate")

_NON_LLM = ["Popularity", "MF", "LightGCN", "SBERT reranker"]
_LLM = ["Single-pass LLM", "UARO", "ATM-style adaptive", "UC-ASR"]
_MULTI_PASS = {"UARO", "ATM-style adaptive", "UC-ASR"}


# --------------------------------------------------------------------------- #
# Metric primitives (unit-testable)
# --------------------------------------------------------------------------- #


def recall_at_k_for_user(order: List[int], target: int, k: int) -> float:
    return 1.0 if target in order[:k] else 0.0


def ndcg_at_k_for_user(order: List[int], target: int, k: int) -> float:
    for rank, it in enumerate(order[:k], start=1):
        if it == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mrr_for_user(order: List[int], target: int) -> float:
    for rank, it in enumerate(order, start=1):
        if it == target:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------- #
# Non-LLM rerankers (retrieval-order variants over the same candidates)
# --------------------------------------------------------------------------- #


def _order_by_score(cand: pd.DataFrame, score_map: Mapping[int, float]) -> List[int]:
    items = cand["item_idx"].tolist()
    return sorted((int(i) for i in items), key=lambda i: (-score_map.get(i, 0.0), i))


def non_llm_orders(
    cfg: Config, candidates: pd.DataFrame, ctx_meta, train_df, mf_scores=None
) -> Dict[str, Dict[int, List[int]]]:
    """Per-user orderings for Popularity, MF (own embeddings), LightGCN, SBERT."""
    processed = resolve_path(cfg, "processed_dir")
    id_maps = json.load(open(os.path.join(processed, "id_maps.json"), encoding="utf-8"))
    n_items = len(id_maps["item2idx"])
    pop = popularity_scores(train_df, n_items)

    encoder = SBERTEncoder(cfg)
    profile_max = int(cfg["retrievers"]["sbert"]["profile_max_items"])

    out: Dict[str, Dict[int, List[int]]] = {m: {} for m in _NON_LLM}
    for user_idx, grp in candidates.groupby("user_idx"):
        u = int(user_idx)
        grp = grp.sort_values("rank")
        items = grp["item_idx"].tolist()

        # Popularity: by global popularity.
        out["Popularity"][u] = sorted((int(i) for i in items), key=lambda i: (-pop[i], i))
        # LightGCN: the retrieval IP score (LightGCN embeddings drive FAISS).
        retr = {int(i): float(s) for i, s in zip(grp["item_idx"], grp["retrieval_score"])}
        out["LightGCN"][u] = _order_by_score(grp, retr)
        # MF: rank the shared candidates by BPR-MF's OWN user-item dot product
        # (distinct from LightGCN); falls back to retrieval order if MF is off.
        if mf_scores:
            mf = {int(i): float(mf_scores.get((u, int(i)), 0.0)) for i in items}
            out["MF"][u] = _order_by_score(grp, mf)
        else:
            out["MF"][u] = _order_by_score(grp, retr)
        # SBERT: cosine(profile, candidate).
        profile = build_user_profile_text(u, train_df, ctx_meta, profile_max)
        pvec = encoder.encode_one(profile)
        sbert_score = {}
        for item_id, item_idx in zip(grp["item_id"], grp["item_idx"]):
            cvec = encoder.encode_one(build_candidate_text(str(item_id), ctx_meta))
            sbert_score[int(item_idx)] = float(np.dot(pvec, cvec))
        out["SBERT reranker"][u] = _order_by_score(grp, sbert_score)
    return out


# --------------------------------------------------------------------------- #
# Per-seed evaluation
# --------------------------------------------------------------------------- #


def _per_user_metrics(
    orders: Mapping[int, List[int]], test_target: Mapping[int, int], k: int
) -> Dict[str, Dict[int, float]]:
    recall, ndcg, mrr = {}, {}, {}
    for u, order in orders.items():
        tgt = test_target.get(int(u))
        if tgt is None:
            continue
        recall[u] = recall_at_k_for_user(order, tgt, k)
        ndcg[u] = ndcg_at_k_for_user(order, tgt, k)
        mrr[u] = mrr_for_user(order, tgt)
    return {"recall": recall, "ndcg": ndcg, "mrr": mrr}


def evaluate_seed(
    cfg: Config,
    seed: int,
    collect_paired: bool = False,
    uc_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run retrieval + all rerankers at ``seed``; return per-method metric dicts."""
    set_seed(seed)
    k = int(cfg["evaluation"]["cutoffs"][0])
    processed = resolve_path(cfg, "processed_dir")
    train_df = pd.read_parquet(os.path.join(processed, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(processed, "test.parquet"))
    test_target = {int(u): int(i) for u, i in zip(test_df["user_idx"], test_df["item_idx"])}

    candidates, mf_scores = retrieve_candidates(cfg, seed, return_mf=True)
    meta_path = os.path.join(processed, "item_meta.parquet")
    meta_df = pd.read_parquet(meta_path) if os.path.exists(meta_path) else pd.DataFrame(
        columns=["parent_asin", "title", "categories", "price", "average_rating"]
    )
    meta_lookup = meta_lookup_from_frame(meta_df)

    results: Dict[str, Any] = {}
    paired: Dict[str, Dict[str, Dict[int, float]]] = {}

    # --- Non-LLM methods (Latency = 0.0, NIT-1) ---
    nl_orders = non_llm_orders(cfg, candidates, meta_lookup, train_df, mf_scores)
    for method in _NON_LLM:
        pm = _per_user_metrics(nl_orders[method], test_target, k)
        results[method] = _aggregate_method(pm, calls=0.0, latency=0.0, stability=1.0, cr=float("nan"))
        if collect_paired:
            paired[method] = pm

    # --- LLM methods ---
    ctx = RerankContext(cfg, candidates)
    for method in _LLM:
        res = run_reranker(ctx, method, seed, uc_overrides if method == "UC-ASR" else None)
        orders = {u: r["order"] for u, r in res.items()}
        pm = _per_user_metrics(orders, test_target, k)
        avg_calls = float(np.mean([r["llm_calls"] for r in res.values()])) if res else 0.0
        if seed == int(cfg["project"]["primary_seed"]):
            latency = float(np.mean([r["latency_s"] for r in res.values()])) if res else 0.0
        else:
            latency = None  # latency only measured on primary seed (design B.8)
        stability = _mean_stability(res) if method in _MULTI_PASS else 1.0
        cr = _mean_cr_top10(res, k)
        results[method] = _aggregate_method(pm, calls=avg_calls, latency=latency, stability=stability, cr=cr)
        if collect_paired:
            paired[method] = pm

    out = {"metrics": results}
    if collect_paired:
        out["paired"] = paired
    return out


def _aggregate_method(pm, calls, latency, stability, cr) -> Dict[str, float]:
    def avg(d):
        return float(np.mean(list(d.values()))) if d else 0.0

    return {
        "Recall@10": avg(pm["recall"]),
        "NDCG@10": avg(pm["ndcg"]),
        "MRR": avg(pm["mrr"]),
        "Avg. LLM calls": calls,
        "Latency": latency,
        "Stability": stability,
        "CR": cr,
    }


def _mean_stability(res: Mapping[int, Dict[str, Any]]) -> float:
    from src.rerankers import _stability_from_passes

    vals = []
    for r in res.values():
        if r.get("pass_scores"):
            vals.append(_stability_from_passes(r["pass_scores"], cutoff=10))
    return float(np.mean(vals)) if vals else 1.0


def _mean_cr_top10(res: Mapping[int, Dict[str, Any]], k: int) -> float:
    vals = []
    for r in res.values():
        cr_map = r.get("cr") or {}
        order = r.get("order") or []
        top = [cr_map[i] for i in order[:k] if i in cr_map]
        if top:
            vals.append(float(np.mean(top)))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #

_COLUMNS = ["Recall@10", "NDCG@10", "MRR", "Avg. LLM calls", "Latency", "Stability", "CR"]


def aggregate_seeds(per_seed: List[Dict[str, Any]], methods: List[str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Mean and std across seeds per method per column."""
    agg: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for method in methods:
        agg[method] = {}
        for col in _COLUMNS:
            vals = []
            for seed_res in per_seed:
                v = seed_res["metrics"][method][col]
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                vals.append(float(v))
            if vals:
                agg[method][col] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                agg[method][col] = (float("nan"), float("nan"))
    return agg


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def _fmt_cell(method: str, col: str, mean: float, std: float, md: bool) -> str:
    if math.isnan(mean):
        return "" if not md else "—"
    if col in ("Avg. LLM calls", "Latency"):
        return f"{mean:.2f}" if not md else f"{mean:.2f}±{std:.2f}"
    return f"{mean:.4f}" if not md else f"{mean:.4f}±{std:.4f}"


def write_main_table(cfg: Config, agg: Dict[str, Dict[str, Tuple[float, float]]], methods: List[str]) -> Tuple[str, str]:
    results_dir = resolve_path(cfg, "results_dir")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "thesis_main_table.csv")
    md_path = os.path.join(results_dir, "thesis_main_table.md")

    header = ["Method"] + _COLUMNS
    # CSV: mean only.
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        import csv

        w = csv.writer(fh)
        w.writerow(header)
        for m in methods:
            row = [m]
            for col in _COLUMNS:
                mean, std = agg[m][col]
                row.append("" if math.isnan(mean) else _fmt_cell(m, col, mean, std, md=False))
            w.writerow(row)

    # MD: mean±std.
    lines = [
        "# UC-ASR Thesis Main Table",
        "",
        "Latency is modelled backend-compute time (sum of per-call `compute_ms`), "
        "not cold-process wall-clock; non-LLM methods report Latency 0.0 (NIT-1). "
        "Cells are mean±std across seeds; the CSV mirrors the means only.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for m in methods:
        cells = [m]
        for col in _COLUMNS:
            mean, std = agg[m][col]
            cells.append(_fmt_cell(m, col, mean, std, md=True))
        lines.append("| " + " | ".join(cells) + " |")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log.info("wrote main table -> %s and %s", csv_path, md_path)
    return csv_path, md_path


def write_ablations(cfg: Config, base_seed_res: Dict[str, Any], candidates, methods) -> str:
    """Run the 10 config-driven ablations vs full UC-ASR at the primary seed."""
    results_dir = resolve_path(cfg, "results_dir")
    seed = int(cfg["project"]["primary_seed"])
    k = int(cfg["evaluation"]["cutoffs"][0])
    processed = resolve_path(cfg, "processed_dir")
    test_df = pd.read_parquet(os.path.join(processed, "test.parquet"))
    test_target = {int(u): int(i) for u, i in zip(test_df["user_idx"], test_df["item_idx"])}

    full_uc = base_seed_res["metrics"]["UC-ASR"]
    ablations = cfg.get("ablations", [])

    lines = [
        "# UC-ASR Ablation Studies",
        "",
        "Each row toggles one UC-ASR component (config-driven `ablations:`), "
        "reporting Recall@10 / NDCG@10 / Avg. LLM calls and the delta vs full "
        "UC-ASR (primary seed).",
        "",
        "> NIT-5: at smoke scale sparse candidate lists frequently hit the "
        "`t(k)=-inf` degenerate branch, so UC-ASR tiers may be near-empty and "
        "its LLM-call count approaches UARO's. This is expected, not a gate "
        "failure.",
        "",
        "| Ablation | Description | Recall@10 | NDCG@10 | Avg. calls | dRecall | dNDCG | dCalls |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for ab in ablations:
        name = ab["name"]
        desc = ab.get("desc", "")
        variants = _ablation_variants(cfg, ab)
        abl_cap = _resolve_cap(cfg["evaluation"], "max_ablation_users")
        # Average the sweep variants (or single override) for the row.
        rec_l, ndcg_l, call_l = [], [], []
        for vcfg, uc_over in variants:
            ctx = RerankContext(vcfg, candidates)
            if abl_cap is not None:
                ctx.eval_users = ctx.eval_users[: int(abl_cap)]
            res = run_reranker(ctx, "UC-ASR", seed, uc_over)
            orders = {u: r["order"] for u, r in res.items()}
            pm = _per_user_metrics(orders, test_target, k)
            rec_l.append(float(np.mean(list(pm["recall"].values()))) if pm["recall"] else 0.0)
            ndcg_l.append(float(np.mean(list(pm["ndcg"].values()))) if pm["ndcg"] else 0.0)
            call_l.append(float(np.mean([r["llm_calls"] for r in res.values()])) if res else 0.0)
        rec = float(np.mean(rec_l))
        ndcg = float(np.mean(ndcg_l))
        calls = float(np.mean(call_l))
        lines.append(
            f"| {name} | {desc} | {rec:.4f} | {ndcg:.4f} | {calls:.2f} | "
            f"{rec - full_uc['Recall@10']:+.4f} | {ndcg - full_uc['NDCG@10']:+.4f} | "
            f"{calls - full_uc['Avg. LLM calls']:+.2f} |"
        )

    path = os.path.join(results_dir, "ablation_results.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("wrote %d ablations -> %s", len(ablations), path)
    return path


def _ablation_variants(cfg: Config, ab: Mapping[str, Any]) -> List[Tuple[Config, Optional[Dict[str, float]]]]:
    """Materialise the per-ablation config variants (override or sweep)."""
    import copy as _copy

    out: List[Tuple[Config, Optional[Dict[str, float]]]] = []
    if "sweep" in ab:
        (dotted, values), = ab["sweep"].items()
        for val in values:
            vcfg = _copy.deepcopy(cfg)
            _set_dotted(vcfg, dotted, val)
            out.append((vcfg, _uc_overrides_from_cfg(vcfg)))
        return out
    override = ab.get("override", {}) or {}
    vcfg = _copy.deepcopy(cfg)
    _merge_override(vcfg, override)
    out.append((vcfg, _uc_overrides_from_cfg(vcfg)))
    return out


def _uc_overrides_from_cfg(cfg: Config) -> Dict[str, float]:
    uc = cfg["rerankers"]["uc_asr"]
    return {"tau_high": float(uc["tau_high"]), "tau_low": float(uc["tau_low"])}


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


def _merge_override(cfg: Config, override: Mapping[str, Any]) -> None:
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(cfg.get(key), Mapping):
            _merge_override(cfg[key], val)
        else:
            cfg[key] = val


def write_significance(cfg: Config, paired: Dict[str, Dict[str, Dict[int, float]]]) -> str:
    """Paired Wilcoxon UC-ASR vs the configured baseline on the primary seed."""
    results_dir = resolve_path(cfg, "results_dir")
    baseline = cfg["evaluation"]["wilcoxon_baseline"]
    seed = int(cfg["project"]["primary_seed"])
    lines = [
        "# Statistical Significance (paired Wilcoxon signed-rank)",
        "",
        f"UC-ASR vs **{baseline}**, per-user paired vectors on seed {seed} "
        "(primary seed). Significance at alpha=0.05.",
        "",
        "| Metric | W | p-value | Significant@0.05 |",
        "|---|---|---|---|",
    ]
    for metric_key, metric_name in [("recall", "Recall@10"), ("ndcg", "NDCG@10"), ("mrr", "MRR")]:
        uc = paired["UC-ASR"][metric_key]
        bl = paired[baseline][metric_key]
        users = sorted(set(uc) & set(bl))
        a = np.array([uc[u] for u in users], dtype=float)
        b = np.array([bl[u] for u in users], dtype=float)
        if len(users) == 0 or np.allclose(a - b, 0.0):
            w, p, note = float("nan"), 1.0, " (all-zero differences)"
        else:
            try:
                stat = wilcoxon(a, b)
                w, p, note = float(stat.statistic), float(stat.pvalue), ""
            except ValueError:
                w, p, note = float("nan"), 1.0, " (degenerate)"
        wtxt = "nan" if math.isnan(w) else f"{w:.4f}"
        sig = "yes" if p < 0.05 else "no"
        lines.append(f"| {metric_name} | {wtxt} | {p:.4g}{note} | {sig} |")

    path = os.path.join(results_dir, "statistical_significance.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("wrote significance -> %s", path)
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def write_figures(
    cfg: Config,
    agg: Dict[str, Dict[str, Tuple[float, float]]],
    methods: List[str],
    tier_counts: Dict[str, int],
) -> List[str]:
    figures_dir = resolve_path(cfg, "figures_dir")
    os.makedirs(figures_dir, exist_ok=True)
    dpi = int(cfg["evaluation"]["figure_dpi"])
    saved: List[str] = []

    def save(fig, name):
        p = os.path.join(figures_dir, name)
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # fig1: NDCG@10 vs Avg. LLM calls scatter.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in methods:
        x = agg[m]["Avg. LLM calls"][0]
        y = agg[m]["NDCG@10"][0]
        x = 0.0 if math.isnan(x) else x
        y = 0.0 if math.isnan(y) else y
        hl = m == "UC-ASR"
        ax.scatter(x, y, s=120 if hl else 60, marker="*" if hl else "o", zorder=3 if hl else 2)
        ax.annotate(m, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Avg. LLM calls per user")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs LLM compute")
    ax.grid(True, alpha=0.3)
    save(fig, "fig1_accuracy_vs_calls.png")

    # fig2: grouped bar of Recall/NDCG/MRR with seed error bars.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ["Recall@10", "NDCG@10", "MRR"]
    x = np.arange(len(methods))
    width = 0.25
    for j, met in enumerate(metrics):
        means = [0.0 if math.isnan(agg[m][met][0]) else agg[m][met][0] for m in methods]
        errs = [0.0 if math.isnan(agg[m][met][1]) else agg[m][met][1] for m in methods]
        ax.bar(x + (j - 1) * width, means, width, yerr=errs, capsize=3, label=met)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Ranking metrics across methods")
    ax.legend()
    save(fig, "fig2_main_metrics_bar.png")

    # fig3: UC-ASR tier distribution (all three labels always shown).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    tiers = ["confident", "ambiguous", "unstable"]
    counts = [tier_counts.get(t, 0) for t in tiers]
    ax.bar(tiers, counts, color=["#4c72b0", "#dd8452", "#c44e52"])
    ax.set_ylabel("Candidate count")
    ax.set_title("UC-ASR gate tier distribution")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    save(fig, "fig3_uncertainty_gate_distribution.png")

    # fig4: Stability + CR across LLM methods.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    llm_methods = [m for m in methods if m in _LLM]
    x = np.arange(len(llm_methods))
    stab = [0.0 if math.isnan(agg[m]["Stability"][0]) else agg[m]["Stability"][0] for m in llm_methods]
    cr = [0.0 if math.isnan(agg[m]["CR"][0]) else agg[m]["CR"][0] for m in llm_methods]
    ax.bar(x - 0.2, stab, 0.4, label="Stability")
    ax.bar(x + 0.2, cr, 0.4, label="CR")
    ax.set_xticks(x)
    ax.set_xticklabels(llm_methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Stability and Cluster Ratio (LLM methods)")
    ax.legend()
    save(fig, "fig4_stability_cr.png")

    log.info("wrote %d figures -> %s (dpi=%d)", len(saved), figures_dir, dpi)
    return saved


# --------------------------------------------------------------------------- #
# Tier counts for fig3
# --------------------------------------------------------------------------- #


def _collect_tier_counts(cfg: Config, candidates) -> Dict[str, int]:
    from src.rerankers import rerank_uc_asr

    ctx = RerankContext(cfg, candidates)
    seed = int(cfg["project"]["primary_seed"])
    counts = {"confident": 0, "ambiguous": 0, "unstable": 0}
    for u in ctx.eval_users:
        res = rerank_uc_asr(ctx, u, seed)
        for t in res.get("tier", {}).values():
            counts[t] = counts.get(t, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run(cfg: Config) -> Dict[str, str]:
    """Full eval stage: multi-seed metrics, tables, ablations, stats, figures."""
    ensure_dirs(cfg, ["results_dir", "figures_dir"])
    methods = list(cfg["evaluation"]["methods"])
    seeds = list(cfg["project"]["seeds"])
    primary = int(cfg["project"]["primary_seed"])

    # Optional val-tuning (full scale) -> uc_overrides used by all eval seeds.
    uc_overrides: Optional[Dict[str, float]] = None
    processed = resolve_path(cfg, "processed_dir")
    tuned_path = os.path.join(processed, "uc_asr_tuned.json")
    if os.path.exists(tuned_path):
        uc_overrides = json.load(open(tuned_path, encoding="utf-8"))
        log.info("using tuned UC-ASR thresholds: %s", uc_overrides)

    per_seed: List[Dict[str, Any]] = []
    primary_res: Optional[Dict[str, Any]] = None
    for seed in seeds:
        log.info("evaluating seed %d", seed)
        res = evaluate_seed(cfg, seed, collect_paired=(seed == primary), uc_overrides=uc_overrides)
        per_seed.append(res)
        if seed == primary:
            primary_res = res

    agg = aggregate_seeds(per_seed, methods)
    csv_path, md_path = write_main_table(cfg, agg, methods)

    candidates = retrieve_candidates(cfg, primary)
    abl_path = write_ablations(cfg, primary_res, candidates, methods)
    sig_path = write_significance(cfg, primary_res["paired"])
    tier_counts = _collect_tier_counts(cfg, candidates)
    figs = write_figures(cfg, agg, methods, tier_counts)

    return {
        "main_csv": csv_path,
        "main_md": md_path,
        "ablations": abl_path,
        "significance": sig_path,
        "figures": ";".join(figs),
    }


# --------------------------------------------------------------------------- #
# Self-test + CLI
# --------------------------------------------------------------------------- #


def _self_test() -> None:
    order = [5, 3, 9, 1, 7]
    assert recall_at_k_for_user(order, 9, 10) == 1.0
    assert recall_at_k_for_user(order, 42, 10) == 0.0
    assert abs(ndcg_at_k_for_user(order, 5, 10) - 1.0) < 1e-9  # rank 1
    assert abs(ndcg_at_k_for_user(order, 3, 10) - 1.0 / math.log2(3)) < 1e-9  # rank 2
    assert abs(mrr_for_user(order, 9) - 1.0 / 3.0) < 1e-9
    log.info("evaluate metric self-test passed")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UC-ASR evaluation")
    parser.add_argument("--scale", choices=["smoke", "full", "kaggle", "kaggle_fast", "kaggle_unc"], default="smoke")
    parser.add_argument("--config", default="config.yaml")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _self_test()
    cfg = load_config(args.config, args.scale)
    log.info("running eval stage at scale=%s", args.scale)
    artifacts = run(cfg)
    log.info("eval stage complete: %s", {k: os.path.basename(v) if ";" not in v else "4 figs" for k, v in artifacts.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
