"""Retrieval stage for UC-ASR (design B.4, schema D.3).

Implements, all in PyTorch / NumPy from scratch (no recsys library):

* Popularity   -- item score = train interaction count.
* BPR-MF        -- matrix factorisation trained with the BPR pairwise loss.
* LightGCN      -- light graph convolution (layer-average embeddings) + BPR.
* FAISS ``IndexFlatIP`` retrieval over LightGCN item embeddings -> top-K
  candidates per user with a per-user min-max normalised confidence ``E``.
* SBERT reranker (``all-MiniLM-L6-v2``) over the top-K candidates, whose two
  text builders (user-profile text, candidate text) are shared with the ATM
  cluster-ratio encoder in :mod:`src.uncertainty`.

Runnable standalone: ``python -m src.retrievers --scale {smoke,full}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.common import (
    Config,
    ensure_dirs,
    get_logger,
    load_config,
    resolve_path,
    set_seed,
)

log = get_logger("retrievers")


# --------------------------------------------------------------------------- #
# Loading shared inputs
# --------------------------------------------------------------------------- #


def _load_id_maps(processed: str) -> Dict[str, Dict[str, int]]:
    with open(os.path.join(processed, "id_maps.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_train(processed: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(processed, "train.parquet"))


def _load_meta(processed: str) -> pd.DataFrame:
    path = os.path.join(processed, "item_meta.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["parent_asin", "title", "categories", "price", "average_rating"])


# --------------------------------------------------------------------------- #
# Popularity
# --------------------------------------------------------------------------- #


def popularity_scores(train_df: pd.DataFrame, n_items: int) -> np.ndarray:
    """Item popularity = train interaction count, indexed by ``item_idx``."""
    counts = np.zeros(n_items, dtype=np.float64)
    vc = train_df["item_idx"].value_counts()
    for item_idx, c in vc.items():
        if 0 <= int(item_idx) < n_items:
            counts[int(item_idx)] = float(c)
    return counts


# --------------------------------------------------------------------------- #
# BPR-MF and LightGCN (PyTorch, from scratch)
# --------------------------------------------------------------------------- #


def _resolve_device() -> str:
    """Resolve the compute device for retriever training (cuda if available).

    Mirrors the ``"auto"`` resolution in :mod:`src.llm_engine`: use CUDA when a
    GPU is present, otherwise fall back to CPU so the dev laptop runs unchanged.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a locked dependency
        return "cpu"


def _build_user_pos_items(train_df: pd.DataFrame, n_users: int) -> List[np.ndarray]:
    """Per-user array of positive item indices (train history)."""
    pos: List[List[int]] = [[] for _ in range(n_users)]
    for u, i in zip(train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy()):
        if 0 <= int(u) < n_users and int(i) >= 0:
            pos[int(u)].append(int(i))
    return [np.asarray(sorted(set(p)), dtype=np.int64) for p in pos]


def _build_bpr_sampling_tables(
    user_pos: List[np.ndarray],
) -> Tuple["object", "object", "object", "object"]:
    """Build padded tensors for vectorised BPR triple sampling.

    Returns ``(users_with_pos, pos_matrix, pos_lengths, pos_sorted)`` where:

    * ``users_with_pos`` -- 1-D LongTensor of user indices that have >=1 positive.
    * ``pos_matrix`` -- ``(n_u, max_len)`` LongTensor; row ``r`` holds user
      ``users_with_pos[r]``'s positive item indices, right-padded with the
      item in column 0 (padding is harmless because a vectorised gather only
      reads ``offset < length`` columns).
    * ``pos_lengths`` -- ``(n_u,)`` LongTensor of per-user positive counts.
    * ``pos_sorted`` -- same shape/content as ``pos_matrix`` but each row sorted
      ascending (and padded with a sentinel ``-1`` beyond ``length``), used for
      a vectorised ``searchsorted`` membership test during negative rejection.

    All tensors live on CPU; sampling draws small index tensors that are cheap
    to keep on CPU regardless of where the embedding math runs.
    """
    import torch

    users_with_pos = [u for u, p in enumerate(user_pos) if len(p) > 0]
    if not users_with_pos:
        raise RuntimeError("no users with positive interactions for BPR sampling")

    lengths = [len(user_pos[u]) for u in users_with_pos]
    max_len = max(lengths)
    n_u = len(users_with_pos)

    pos_matrix = torch.zeros((n_u, max_len), dtype=torch.long)
    pos_sorted = torch.full((n_u, max_len), -1, dtype=torch.long)
    for r, u in enumerate(users_with_pos):
        p = torch.as_tensor(user_pos[u], dtype=torch.long)
        L = p.shape[0]
        pos_matrix[r, :L] = p
        # user_pos rows are already sorted/unique (see _build_user_pos_items).
        pos_sorted[r, :L] = p

    uw = torch.tensor(users_with_pos, dtype=torch.long)
    pos_lengths = torch.tensor(lengths, dtype=torch.long)
    return uw, pos_matrix, pos_lengths, pos_sorted


def _sample_bpr_triples(
    user_pos: List[np.ndarray],
    n_items: int,
    batch_size: int,
    generator: "object",
    tables: Optional[Tuple["object", "object", "object", "object"]] = None,
) -> Tuple["object", "object", "object"]:
    """Sample a batch of ``(u, i_pos, i_neg)`` triples with a seeded generator.

    Fully vectorised: one RNG draw per quantity (batch of users, batch of
    positive offsets, batch of negatives) plus a bounded resample loop that
    only touches the entries that collided with a true positive. Deterministic
    given ``generator``. Returns three 1-D CPU LongTensors of length
    ``batch_size`` -- identical signature/contract to the previous per-sample
    implementation.

    ``tables`` is the precomputed output of :func:`_build_bpr_sampling_tables`;
    it is built lazily (and cached by callers across steps) to avoid rebuilding
    the padded matrices every batch.
    """
    import torch

    if tables is None:
        tables = _build_bpr_sampling_tables(user_pos)
    uw, pos_matrix, pos_lengths, pos_sorted = tables

    n_u = uw.shape[0]
    # 1) Sample the batch of users (indices into users_with_pos).
    row = torch.randint(0, n_u, (batch_size,), generator=generator)
    u_batch = uw[row]

    # 2) Sample one positive per user: a uniform offset in [0, length) gathered
    #    from that user's padded positive row -- no Python loop.
    lens = pos_lengths[row]
    rand01 = torch.rand((batch_size,), generator=generator)
    offsets = (rand01 * lens.to(torch.float64)).to(torch.long)
    offsets = torch.minimum(offsets, lens - 1)  # guard float rounding at the top
    pos_items = pos_matrix[row, offsets]

    # 3) Sample negatives for the whole batch at once, then vectorised rejection:
    #    resample ONLY the entries whose draw is a true positive for their user.
    #    Membership is a per-row searchsorted against the sorted positive row.
    neg_items = torch.randint(0, n_items, (batch_size,), generator=generator)

    def _is_positive(rows: "object", neg: "object") -> "object":
        sorted_rows = pos_sorted[rows]  # (m, max_len), -1 padded, ascending
        ins = torch.searchsorted(sorted_rows, neg.unsqueeze(1)).clamp_max(
            sorted_rows.shape[1] - 1
        )
        found = sorted_rows.gather(1, ins).squeeze(1) == neg
        return found

    collide = _is_positive(row, neg_items)
    max_retries = 10
    for _ in range(max_retries):
        if not bool(collide.any()):
            break
        idx = collide.nonzero(as_tuple=False).squeeze(1)
        resampled = torch.randint(0, n_items, (idx.shape[0],), generator=generator)
        neg_items[idx] = resampled
        still = _is_positive(row[idx], resampled)
        collide = torch.zeros_like(collide)
        collide[idx] = still
    # Any residual collisions (astronomically rare) are acceptable for BPR.

    return u_batch, pos_items, neg_items


def train_bpr_mf(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    cfg: Mapping[str, Any],
    seed: int,
) -> "object":
    """Train BPR-MF from scratch; return the learned user/item embeddings."""
    import torch

    rc = cfg["retrievers"]["bpr_mf"]
    dim = int(rc["dim"])
    epochs = int(rc["epochs"])
    lr = float(rc["lr"])
    reg = float(rc["reg"])
    batch_size = int(rc["batch_size"])

    device = _resolve_device()
    # Sampling generator stays on CPU (indices are tiny); the embedding math
    # runs on `device` (cuda when present, else cpu -- CPU path unchanged).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    user_emb = torch.nn.Embedding(n_users, dim)
    item_emb = torch.nn.Embedding(n_items, dim)
    torch.nn.init.normal_(user_emb.weight, std=0.1)
    torch.nn.init.normal_(item_emb.weight, std=0.1)
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    log.info("BPR-MF training on device=%s (dim=%d, epochs=%d)", device, dim, epochs)

    opt = torch.optim.Adam(list(user_emb.parameters()) + list(item_emb.parameters()), lr=lr)
    user_pos = _build_user_pos_items(train_df, n_users)
    sampling_tables = _build_bpr_sampling_tables(user_pos)
    n_inter = int(len(train_df))
    steps = max(1, n_inter // batch_size)

    for ep in range(epochs):
        total = 0.0
        for _ in range(steps):
            u, ip, ineg = _sample_bpr_triples(
                user_pos, n_items, batch_size, gen, tables=sampling_tables
            )
            u = u.to(device)
            ip = ip.to(device)
            ineg = ineg.to(device)
            ue = user_emb(u)
            pe = item_emb(ip)
            ne = item_emb(ineg)
            pos_score = (ue * pe).sum(dim=1)
            neg_score = (ue * ne).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
            loss = loss + reg * (ue.pow(2).sum() + pe.pow(2).sum() + ne.pow(2).sum()) / u.shape[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if not np.isfinite(total):
            raise RuntimeError(f"BPR-MF diverged (NaN loss) at epoch {ep}; lower lr")
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            log.info("BPR-MF epoch %d/%d  loss=%.4f", ep + 1, epochs, total / steps)

    with torch.no_grad():
        return item_emb.weight.detach().clone(), user_emb.weight.detach().clone()


def _build_norm_adj(train_df: pd.DataFrame, n_users: int, n_items: int) -> "object":
    """Symmetric-normalised sparse bipartite adjacency for LightGCN propagation."""
    import torch

    u = train_df["user_idx"].to_numpy()
    i = train_df["item_idx"].to_numpy()
    mask = (u >= 0) & (i >= 0)
    u = u[mask].astype(np.int64)
    i = i[mask].astype(np.int64)
    n = n_users + n_items

    # Edges in both directions on the (users | items) node block.
    rows = np.concatenate([u, i + n_users])
    cols = np.concatenate([i + n_users, u])
    deg = np.zeros(n, dtype=np.float64)
    np.add.at(deg, rows, 1.0)
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    vals = d_inv_sqrt[rows] * d_inv_sqrt[cols]

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def train_lightgcn(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    cfg: Mapping[str, Any],
    seed: int,
) -> "object":
    """Train LightGCN from scratch; return layer-averaged item/user embeddings."""
    import torch

    rc = cfg["retrievers"]["lightgcn"]
    dim = int(rc["dim"])
    n_layers = int(rc["n_layers"])
    epochs = int(rc["epochs"])
    lr = float(rc["lr"])
    reg = float(rc["reg"])
    batch_size = int(rc["batch_size"])

    device = _resolve_device()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    emb = torch.nn.Embedding(n_users + n_items, dim)
    torch.nn.init.normal_(emb.weight, std=0.1)
    emb = emb.to(device)
    adj = _build_norm_adj(train_df, n_users, n_items).to(device)
    log.info(
        "LightGCN training on device=%s (dim=%d, layers=%d, epochs=%d)",
        device,
        dim,
        n_layers,
        epochs,
    )
    opt = torch.optim.Adam(emb.parameters(), lr=lr)
    user_pos = _build_user_pos_items(train_df, n_users)
    sampling_tables = _build_bpr_sampling_tables(user_pos)
    n_inter = int(len(train_df))
    steps = max(1, n_inter // batch_size)

    def propagate() -> "object":
        x = emb.weight
        acc = x
        cur = x
        for _ in range(n_layers):
            cur = torch.sparse.mm(adj, cur)
            acc = acc + cur
        return acc / (n_layers + 1)

    for ep in range(epochs):
        total = 0.0
        for _ in range(steps):
            all_emb = propagate()
            u, ip, ineg = _sample_bpr_triples(
                user_pos, n_items, batch_size, gen, tables=sampling_tables
            )
            u = u.to(device)
            ip = ip.to(device)
            ineg = ineg.to(device)
            ue = all_emb[u]
            pe = all_emb[n_users + ip]
            ne = all_emb[n_users + ineg]
            pos_score = (ue * pe).sum(dim=1)
            neg_score = (ue * ne).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
            reg_term = (ue.pow(2).sum() + pe.pow(2).sum() + ne.pow(2).sum()) / u.shape[0]
            loss = loss + reg * reg_term
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if not np.isfinite(total):
            raise RuntimeError(f"LightGCN diverged (NaN loss) at epoch {ep}; lower lr")
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            log.info("LightGCN epoch %d/%d  loss=%.4f", ep + 1, epochs, total / steps)

    with torch.no_grad():
        final = propagate()
        user_e = final[:n_users].detach().clone()
        item_e = final[n_users:].detach().clone()
    return item_e, user_e


# --------------------------------------------------------------------------- #
# FAISS retrieval
# --------------------------------------------------------------------------- #


def faiss_retrieve(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user_pos: List[np.ndarray],
    top_k: int,
    index_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve top-K item indices + raw IP scores per user, excluding train items.

    Uses an exact ``IndexFlatIP`` (deterministic). Over-fetches so that after
    removing each user's train-history items we still have ``top_k`` candidates.
    """
    import faiss

    if index_type != "IndexFlatIP":
        raise RuntimeError(f"unsupported faiss index_type: {index_type!r}")

    n_items = item_emb.shape[0]
    item_emb = np.ascontiguousarray(item_emb.astype(np.float32))
    user_emb = np.ascontiguousarray(user_emb.astype(np.float32))

    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(item_emb)

    max_hist = max((len(p) for p in user_pos), default=0)
    fetch = min(n_items, top_k + max_hist + 1)
    scores, idxs = index.search(user_emb, fetch)

    n_users = user_emb.shape[0]
    out_items = np.full((n_users, top_k), -1, dtype=np.int64)
    out_scores = np.zeros((n_users, top_k), dtype=np.float32)
    for u in range(n_users):
        exclude = set(user_pos[u].tolist()) if u < len(user_pos) else set()
        kept = 0
        for rank in range(idxs.shape[1]):
            it = int(idxs[u, rank])
            if it < 0 or it in exclude:
                continue
            out_items[u, kept] = it
            out_scores[u, kept] = float(scores[u, rank])
            kept += 1
            if kept >= top_k:
                break
    return out_items, out_scores


def normalize_confidence(scores_row: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-user min-max normalise valid IP scores into ``E`` in [0, 1]."""
    e = np.zeros_like(scores_row, dtype=np.float32)
    if valid.sum() == 0:
        return e
    vals = scores_row[valid]
    lo = float(vals.min())
    hi = float(vals.max())
    e[valid] = ((vals - lo) / (hi - lo + 1e-8)).astype(np.float32)
    return e


def build_candidates_frame(
    out_items: np.ndarray,
    out_scores: np.ndarray,
    idx2user: List[str],
    idx2item: List[str],
    top_k: int,
) -> pd.DataFrame:
    """Assemble the ``candidates_top50.parquet`` frame (schema D.3)."""
    rows: List[Dict[str, Any]] = []
    for u in range(out_items.shape[0]):
        valid = out_items[u] >= 0
        e_row = normalize_confidence(out_scores[u], valid)
        rank = 0
        for pos in range(top_k):
            it = int(out_items[u, pos])
            if it < 0:
                continue
            rank += 1
            rows.append(
                {
                    "user_id": idx2user[u],
                    "user_idx": np.int32(u),
                    "item_id": idx2item[it],
                    "item_idx": np.int32(it),
                    "rank": np.int16(rank),
                    "retrieval_score": np.float32(out_scores[u, pos]),
                    "E": np.float32(e_row[pos]),
                }
            )
    df = pd.DataFrame(
        rows,
        columns=["user_id", "user_idx", "item_id", "item_idx", "rank", "retrieval_score", "E"],
    )
    if not df.empty:
        df["user_idx"] = df["user_idx"].astype("int32")
        df["item_idx"] = df["item_idx"].astype("int32")
        df["rank"] = df["rank"].astype("int16")
        df["retrieval_score"] = df["retrieval_score"].astype("float32")
        df["E"] = df["E"].astype("float32")
    return df


# --------------------------------------------------------------------------- #
# SBERT text builders (shared with uncertainty.py) + encoder
# --------------------------------------------------------------------------- #


def build_user_profile_text(
    user_idx: int,
    train_df: pd.DataFrame,
    meta_lookup: Mapping[str, Mapping[str, Any]],
    profile_max_items: int,
) -> str:
    """Most-recent ``profile_max_items`` train item titles joined by ``"; "``.

    Falls back to the literal ``"unknown user"`` when the user has no titles
    (design B.4 / MEDIUM-3).
    """
    rows = train_df[train_df["user_idx"] == user_idx]
    if rows.empty:
        return "unknown user"
    rows = rows.sort_values("timestamp", ascending=False)
    titles: List[str] = []
    for item_id in rows["item_id"].tolist():
        meta = meta_lookup.get(str(item_id))
        title = (meta or {}).get("title") or ""
        title = str(title).strip()
        if title:
            titles.append(title)
        if len(titles) >= profile_max_items:
            break
    if not titles:
        return "unknown user"
    return "; ".join(titles)


def build_candidate_text(item_id: str, meta_lookup: Mapping[str, Mapping[str, Any]]) -> str:
    """``"{title}. Categories: {...}"``; title->item_id if missing, drop empty cats."""
    meta = meta_lookup.get(str(item_id)) or {}
    title = str(meta.get("title") or "").strip()
    if not title:
        title = str(item_id)
    cats = meta.get("categories") or []
    cats = [str(c) for c in cats if str(c).strip()]
    if cats:
        return f"{title}. Categories: {', '.join(cats)}"
    return title


def meta_lookup_from_frame(meta_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Index ``item_meta`` by ``parent_asin`` for O(1) text building."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for rec in meta_df.to_dict("records"):
        cats = rec.get("categories")
        if cats is None:
            cats = []
        elif isinstance(cats, np.ndarray):
            cats = cats.tolist()
        price = rec.get("price")
        avg = rec.get("average_rating")
        lookup[str(rec["parent_asin"])] = {
            "title": rec.get("title") or "",
            "categories": list(cats),
            "price": None if price is None or (isinstance(price, float) and np.isnan(price)) else float(price),
            "average_rating": None
            if avg is None or (isinstance(avg, float) and np.isnan(avg))
            else float(avg),
        }
    return lookup


class SBERTEncoder:
    """Lazy ``all-MiniLM-L6-v2`` encoder with an on-disk per-text embedding cache."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._cfg = cfg
        self._model_name = cfg["retrievers"]["sbert"]["model"]
        self._cache_dir = resolve_path(cfg, "sbert_emb_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._model = None
        self._mem: Dict[str, np.ndarray] = {}

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading SBERT model %s (CPU)", self._model_name)
            self._model = SentenceTransformer(self._model_name, device="cpu")

    def _cache_path(self, text: str) -> str:
        h = hashlib.sha256((self._model_name + "|" + text).encode("utf-8")).hexdigest()
        return os.path.join(self._cache_dir, h + ".npy")

    def encode_one(self, text: str) -> np.ndarray:
        if text in self._mem:
            return self._mem[text]
        path = self._cache_path(text)
        if os.path.exists(path):
            vec = np.load(path)
            self._mem[text] = vec
            return vec
        self._load()
        vec = self._model.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        np.save(path, vec)
        self._mem[text] = vec
        return vec

    def encode_many(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.encode_one(t) for t in texts]) if texts else np.zeros((0, 384), np.float32)


def sbert_rerank_scores(
    candidates: pd.DataFrame,
    train_df: pd.DataFrame,
    meta_lookup: Mapping[str, Mapping[str, Any]],
    encoder: SBERTEncoder,
    profile_max_items: int,
) -> Dict[Tuple[int, int], float]:
    """Cosine(user-profile, candidate-text) per (user_idx, item_idx)."""
    scores: Dict[Tuple[int, int], float] = {}
    for user_idx, grp in candidates.groupby("user_idx"):
        profile = build_user_profile_text(int(user_idx), train_df, meta_lookup, profile_max_items)
        pvec = encoder.encode_one(profile)
        for item_id, item_idx in zip(grp["item_id"], grp["item_idx"]):
            cvec = encoder.encode_one(build_candidate_text(str(item_id), meta_lookup))
            scores[(int(user_idx), int(item_idx))] = float(np.dot(pvec, cvec))
    return scores


# --------------------------------------------------------------------------- #
# Retrieval hit-rate + stats update
# --------------------------------------------------------------------------- #


def _update_hit_rate(processed: str, candidates: pd.DataFrame) -> float:
    """Fraction of eligible test users whose test item is among their candidates."""
    test_path = os.path.join(processed, "test.parquet")
    if not os.path.exists(test_path):
        return 0.0
    test_df = pd.read_parquet(test_path)
    if test_df.empty:
        hr = 0.0
    else:
        cand_pairs = set(zip(candidates["user_idx"].tolist(), candidates["item_idx"].tolist()))
        hits = sum(
            1
            for u, i in zip(test_df["user_idx"].tolist(), test_df["item_idx"].tolist())
            if (int(u), int(i)) in cand_pairs
        )
        hr = hits / len(test_df)
    stats_path = os.path.join(processed, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)
        stats["retrieval_hit_rate_at_50"] = round(hr, 6)
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
    log.info("retrieval hit-rate@%d = %.4f", int(candidates["rank"].max() or 0), hr)
    return hr


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def retrieve_candidates(
    cfg: Config, seed: Optional[int] = None, return_mf: bool = False
):
    """Train retrievers at ``seed`` and return the top-K candidates frame.

    Does not write to disk -- used both by the standalone stage (which writes)
    and by the multi-seed eval loop (which keeps results in memory).

    When ``return_mf`` is True, also returns the per-candidate BPR-MF score map
    ``{(user_idx, item_idx): score}`` so the eval stage can rank the shared
    candidates with MF's OWN embeddings (distinct from the LightGCN retrieval
    order), per design B.4.
    """
    if seed is None:
        seed = int(cfg["project"]["primary_seed"])
    set_seed(seed)

    processed = resolve_path(cfg, "processed_dir")
    train_df = _load_train(processed)
    id_maps = _load_id_maps(processed)
    user2idx = id_maps["user2idx"]
    item2idx = id_maps["item2idx"]
    n_users = len(user2idx)
    n_items = len(item2idx)
    idx2user = [None] * n_users
    for u, i in user2idx.items():
        idx2user[int(i)] = u
    idx2item = [None] * n_items
    for it, i in item2idx.items():
        idx2item[int(i)] = it

    top_k = int(cfg["retrievers"]["top_k_candidates"])
    user_pos = _build_user_pos_items(train_df, n_users)

    # Popularity is computed for completeness/stats; LightGCN embeddings drive
    # FAISS retrieval per design B.4.
    _ = popularity_scores(train_df, n_items)

    mf_item_emb = mf_user_emb = None
    if cfg["retrievers"]["bpr_mf"].get("enabled", True):
        mf_item_emb, mf_user_emb = train_bpr_mf(train_df, n_users, n_items, cfg, seed)

    item_emb, user_emb = train_lightgcn(train_df, n_users, n_items, cfg, seed)
    item_emb_np = item_emb.cpu().numpy()
    user_emb_np = user_emb.cpu().numpy()

    out_items, out_scores = faiss_retrieve(
        user_emb_np, item_emb_np, user_pos, top_k, cfg["retrievers"]["faiss"]["index_type"]
    )
    candidates = build_candidates_frame(out_items, out_scores, idx2user, idx2item, top_k)
    if not return_mf:
        return candidates

    mf_scores: Dict[Tuple[int, int], float] = {}
    if mf_item_emb is not None:
        mf_i = mf_item_emb.cpu().numpy()
        mf_u = mf_user_emb.cpu().numpy()
        for u, i in zip(candidates["user_idx"].tolist(), candidates["item_idx"].tolist()):
            mf_scores[(int(u), int(i))] = float(np.dot(mf_u[int(u)], mf_i[int(i)]))
    return candidates, mf_scores


def run(cfg: Config) -> Dict[str, str]:
    """Execute the retrieve stage at the primary seed and write artifacts."""
    ensure_dirs(cfg, ["processed_dir", "cache_dir"])
    processed = resolve_path(cfg, "processed_dir")

    candidates = retrieve_candidates(cfg, int(cfg["project"]["primary_seed"]))
    cand_path = os.path.join(processed, "candidates_top50.parquet")
    candidates.to_parquet(cand_path, index=False)
    log.info("wrote %d candidate rows -> %s", len(candidates), cand_path)

    _update_hit_rate(processed, candidates)

    # SBERT reranker is a baseline method; verify it runs over the candidates
    # so the model/cache are warm before eval. Scores are recomputed in eval.
    train_df = _load_train(processed)
    meta_lookup = meta_lookup_from_frame(_load_meta(processed))
    encoder = SBERTEncoder(cfg)
    profile_max = int(cfg["retrievers"]["sbert"]["profile_max_items"])
    _ = sbert_rerank_scores(candidates, train_df, meta_lookup, encoder, profile_max)
    log.info("SBERT reranker scored all candidate pairs")

    return {"candidates": cand_path}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UC-ASR retrievers")
    parser.add_argument("--scale", choices=["smoke", "full", "kaggle", "kaggle_fast", "kaggle_unc", "gemini"], default="smoke")
    parser.add_argument("--config", default="config.yaml")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.scale)
    log.info("running retrieve stage at scale=%s", args.scale)
    run(cfg)
    log.info("retrieve stage complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
