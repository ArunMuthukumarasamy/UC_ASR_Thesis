"""Data stage for UC-ASR (design B.3, schemas D.1 / D.2 / D.2b).

Responsibilities
----------------
1. Stream-download the two Amazon Reviews 2023 ``.jsonl.gz`` files to
   ``data/raw/`` via ``requests`` + ``tqdm`` with an automatic HuggingFace
   ``datasets`` fallback (NIT-4 atomic ``.tmp`` rename guard).
2. Stream-parse reviews under an 8 GB RAM budget (never ``read()`` the whole
   file): a counting pass then a collecting pass over a deterministic user
   subset filtered by iterative two-sided k-core.
3. Stream metadata once, keeping only surviving items.
4. Build a chronological Leave-One-Out split and write all processed artifacts
   plus ``dataset_stats.json``.

Runnable standalone: ``python -m src.data_pipeline --scale {smoke,full}``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.common import (
    Config,
    ensure_dirs,
    get_logger,
    load_config,
    resolve_path,
    set_seed,
)

log = get_logger("data_pipeline")

_CHUNK_BYTES = 1 << 20  # 1 MB streaming chunks
_HTTP_TIMEOUT = 60  # seconds for connect/read on the direct download


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _is_present(path: str) -> bool:
    """A file counts as present only if it exists and is non-empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def _cleanup_stale_tmp(final_path: str) -> None:
    """Remove a leftover ``<final>.tmp`` from a previously-failed HF write."""
    tmp_path = final_path + ".tmp"
    if os.path.exists(tmp_path):
        log.warning("removing stale temp file from a prior failed download: %s", tmp_path)
        os.remove(tmp_path)


def _download_direct(url: str, dest: str) -> bool:
    """Stream ``url`` to ``dest`` in 1 MB chunks under a tqdm bar.

    Returns ``True`` on a successful, non-empty download; ``False`` on any
    network/HTTP failure or a 0-byte result (so the caller can fall back).
    """
    tmp = dest + ".part"
    try:
        with requests.get(url, stream=True, timeout=_HTTP_TIMEOUT) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            written = 0
            with open(tmp, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"download {os.path.basename(dest)}",
            ) as bar:
                for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    bar.update(len(chunk))
        if written == 0:
            log.warning("direct download produced 0 bytes: %s", url)
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
        os.replace(tmp, dest)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure triggers fallback
        log.warning("direct download failed for %s (%s)", url, exc)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def _download_hf(hf_repo: str, subset: str, dest: str) -> bool:
    """Fall back to the HuggingFace ``datasets`` stream, re-serializing to ``dest``.

    Writes to ``dest + '.tmp'`` and atomically renames on success (NIT-4). A
    leftover ``.tmp`` is cleaned up before starting.
    """
    _cleanup_stale_tmp(dest)
    tmp = dest + ".tmp"
    try:
        from datasets import load_dataset

        log.info("HF fallback: load_dataset(%s, %s, streaming=True)", hf_repo, subset)
        ds = load_dataset(hf_repo, subset, split="full", streaming=True)
        written = 0
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for record in tqdm(ds, desc=f"hf-stream {subset}", unit=" rec"):
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")
                written += 1
        if written == 0:
            log.warning("HF stream produced 0 records for subset %s", subset)
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
        os.replace(tmp, dest)  # atomic rename on success
        log.info("HF fallback wrote %d records -> %s", written, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("HF fallback failed for subset %s (%s)", subset, exc)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def _ensure_file(
    dest: str,
    url: str,
    hf_repo: str,
    subset: str,
) -> None:
    """Ensure ``dest`` exists non-empty: skip, else direct download, else HF.

    Treats a leftover ``.tmp`` or a size-0 final file as a failed download that
    must be retried (same guard for both the direct and HF paths, NIT-4).
    """
    _cleanup_stale_tmp(dest)
    if os.path.exists(dest) and os.path.getsize(dest) == 0:
        log.warning("found size-0 file, treating as failed download: %s", dest)
        os.remove(dest)
    if _is_present(dest):
        log.info("skip download (already present): %s", dest)
        return

    log.info("downloading %s -> %s", url, dest)
    if _download_direct(url, dest) and _is_present(dest):
        return

    log.warning("direct path failed; attempting HuggingFace fallback for %s", dest)
    if _download_hf(hf_repo, subset, dest) and _is_present(dest):
        return

    raise RuntimeError(
        "Failed to obtain dataset file via BOTH paths.\n"
        f"  destination : {dest}\n"
        f"  direct URL  : {url}\n"
        f"  HF fallback : datasets.load_dataset('{hf_repo}', '{subset}', "
        "split='full', streaming=True)\n"
        "Check your network connection, or download the file manually into "
        "data/raw/ and re-run."
    )


def download_dataset(cfg: Mapping[str, Any]) -> Tuple[str, str]:
    """Ensure both review and meta ``.jsonl.gz`` files are present locally.

    Returns ``(review_path, meta_path)``.
    """
    ds = cfg["dataset"]
    raw_dir = resolve_path(cfg, "raw_dir")
    os.makedirs(raw_dir, exist_ok=True)

    review_path = os.path.join(raw_dir, ds["review_file"])
    meta_path = os.path.join(raw_dir, ds["meta_file"])

    _ensure_file(review_path, ds["review_url"], ds["hf_repo"], ds["hf_review_subset"])
    _ensure_file(meta_path, ds["meta_url"], ds["hf_repo"], ds["hf_meta_subset"])
    return review_path, meta_path


# --------------------------------------------------------------------------- #
# Stream parsing
# --------------------------------------------------------------------------- #


def _iter_jsonl_gz(path: str, desc: str) -> Iterator[Tuple[Optional[dict], bool]]:
    """Yield ``(record, ok)`` per line; ``ok=False`` for a malformed JSON line.

    Streams line-by-line via ``gzip.open(..., 'rt')`` -- the whole file is never
    loaded into memory (NFR2).
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=desc, unit=" lines"):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), True
            except (json.JSONDecodeError, ValueError):
                yield None, False


def _extract_review(rec: dict) -> Optional[Tuple[str, str, float, int]]:
    """Return ``(user_id, item_id, rating, timestamp)`` or ``None`` if unusable.

    Uses ``parent_asin`` as the canonical item id (design B.3). Rows missing a
    user, item, or timestamp are unusable; ``None`` signals the caller to drop
    (and count a missing-timestamp drop separately via ``_review_fields``).
    """
    user_id = rec.get("user_id")
    item_id = rec.get("parent_asin") or rec.get("asin")
    if not user_id or not item_id:
        return None
    ts = rec.get("timestamp")
    if ts is None:
        return None
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return None
    rating = rec.get("rating", 0.0)
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        rating = 0.0
    return str(user_id), str(item_id), rating, ts


def _select_user_subset(user_ids: List[str], n_subset: int, seed: int) -> set:
    """Deterministically pick up to ``n_subset`` users via a seeded permutation.

    Users are sorted lexicographically first so the permutation is stable
    regardless of dict insertion order (design B.3).
    """
    sorted_users = sorted(user_ids)
    if n_subset >= len(sorted_users):
        return set(sorted_users)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sorted_users))
    chosen = {sorted_users[i] for i in perm[:n_subset]}
    return chosen


def _iterative_kcore(
    user_counts: Dict[str, int],
    item_counts: Dict[str, int],
    pair_user_items: Dict[str, Dict[str, int]],
    k_core: int,
) -> Tuple[set, set]:
    """Two-sided iterative k-core to a fixed point on the subset interactions.

    ``pair_user_items[u][i]`` is the interaction count for (u, i) restricted to
    the chosen user subset. Returns the surviving ``(users, items)`` sets.
    """
    users = set(user_counts)
    items = set(item_counts)
    while True:
        drop_users = {u for u in users if user_counts[u] < k_core}
        drop_items = {i for i in items if item_counts[i] < k_core}
        if not drop_users and not drop_items:
            break
        # Remove dropped users: decrement affected item counts.
        for u in drop_users:
            for i, c in pair_user_items[u].items():
                if i in item_counts:
                    item_counts[i] -= c
            users.discard(u)
            user_counts.pop(u, None)
        # Remove dropped items: decrement affected user counts.
        for u in list(users):
            row = pair_user_items.get(u, {})
            removed = 0
            for i in drop_items:
                c = row.pop(i, 0)
                removed += c
            if removed:
                user_counts[u] -= removed
        for i in drop_items:
            items.discard(i)
            item_counts.pop(i, None)
    return users, items


def parse_reviews(
    review_path: str,
    cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, set, int, int]:
    """Stream-parse reviews into a filtered interaction frame.

    Returns ``(interactions_df, surviving_items, bad_lines_skipped,
    missing_ts_dropped)``. ``interactions_df`` has columns
    ``user_id, item_id, rating, timestamp``.
    """
    pre = cfg["preprocessing"]
    k_core = int(pre["k_core"])
    n_subset = int(pre["n_users_subset"])
    max_bad_frac = float(pre["data_parse_max_bad_frac"])
    seed = int(cfg["project"]["primary_seed"])

    # ---- Counting pass: per-user counts over all reviews -------------------
    log.info("counting pass over reviews ...")
    user_total: Dict[str, int] = defaultdict(int)
    total_lines = 0
    bad_lines = 0
    missing_ts = 0
    for rec, ok in _iter_jsonl_gz(review_path, "count reviews"):
        total_lines += 1
        if not ok:
            bad_lines += 1
            continue
        fields = _extract_review(rec)
        if fields is None:
            # Distinguish a missing-timestamp drop from a structural miss where
            # we cannot tell; count it as a missing-ts drop only if user+item
            # were present. We re-check cheaply here.
            if rec.get("user_id") and (rec.get("parent_asin") or rec.get("asin")):
                missing_ts += 1
            continue
        user_id = fields[0]
        user_total[user_id] += 1

    _abort_if_too_many_bad(bad_lines, total_lines, max_bad_frac)

    # ---- Deterministic user subset ----------------------------------------
    chosen_users = _select_user_subset(list(user_total.keys()), n_subset, seed)
    log.info(
        "selected deterministic user subset: %d of %d users (requested %d)",
        len(chosen_users),
        len(user_total),
        n_subset,
    )

    # ---- Build subset interaction counts for iterative k-core -------------
    log.info("collecting subset interactions for k-core ...")
    user_counts: Dict[str, int] = defaultdict(int)
    item_counts: Dict[str, int] = defaultdict(int)
    pair_user_items: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec, ok in _iter_jsonl_gz(review_path, "subset reviews"):
        if not ok:
            continue
        fields = _extract_review(rec)
        if fields is None:
            continue
        user_id, item_id, _rating, _ts = fields
        if user_id not in chosen_users:
            continue
        user_counts[user_id] += 1
        item_counts[item_id] += 1
        pair_user_items[user_id][item_id] += 1

    surviving_users, surviving_items = _iterative_kcore(
        user_counts, item_counts, pair_user_items, k_core
    )
    log.info(
        "k-core (k=%d) fixed point: %d users, %d items surviving",
        k_core,
        len(surviving_users),
        len(surviving_items),
    )
    if not surviving_users or not surviving_items:
        raise RuntimeError(
            f"k-core filtering (k={k_core}) left no surviving users/items on a "
            f"subset of {len(chosen_users)} users. Lower preprocessing.k_core or "
            "raise n_users_subset."
        )

    # ---- Collecting pass: keep surviving (u, i) rows ----------------------
    log.info("collecting pass: materializing surviving interactions ...")
    rows: List[Tuple[str, str, float, int]] = []
    for rec, ok in _iter_jsonl_gz(review_path, "keep reviews"):
        if not ok:
            continue
        fields = _extract_review(rec)
        if fields is None:
            continue
        user_id, item_id, rating, ts = fields
        if user_id in surviving_users and item_id in surviving_items:
            rows.append((user_id, item_id, rating, ts))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    # De-duplicate repeated (user,item) interactions, keeping the latest.
    df = (
        df.sort_values(["user_id", "item_id", "timestamp"])
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )
    log.info("materialized %d unique interactions", len(df))
    return df, set(df["item_id"].unique()), bad_lines, missing_ts


def _abort_if_too_many_bad(bad_lines: int, total_lines: int, max_bad_frac: float) -> None:
    if total_lines > 0 and (bad_lines / total_lines) > max_bad_frac:
        raise RuntimeError(
            f"malformed-line fraction {bad_lines}/{total_lines} = "
            f"{bad_lines / total_lines:.4f} exceeds "
            f"data_parse_max_bad_frac={max_bad_frac}. Aborting parse."
        )


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def parse_metadata(
    meta_path: str,
    surviving_items: set,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Stream metadata once, keeping only surviving items (schema D.2b)."""
    log.info("streaming metadata for %d surviving items ...", len(surviving_items))
    seen: Dict[str, Dict[str, Any]] = {}
    for rec, ok in _iter_jsonl_gz(meta_path, "meta"):
        if not ok:
            continue
        pid = rec.get("parent_asin") or rec.get("asin")
        if not pid:
            continue
        pid = str(pid)
        if pid not in surviving_items or pid in seen:
            continue
        seen[pid] = {
            "parent_asin": pid,
            "title": _as_str(rec.get("title")),
            "categories": _as_str_list(rec.get("categories")),
            "price": _as_float_or_none(rec.get("price")),
            "average_rating": _as_float_or_none(rec.get("average_rating")),
        }

    # Items with no metadata record still need a row so downstream joins work.
    missing = surviving_items - set(seen)
    for pid in missing:
        seen[pid] = {
            "parent_asin": pid,
            "title": "",
            "categories": [],
            "price": None,
            "average_rating": None,
        }
    if missing:
        log.warning("%d surviving items had no metadata record (placeholder rows added)", len(missing))

    meta_df = pd.DataFrame(
        [seen[pid] for pid in sorted(seen)],
        columns=["parent_asin", "title", "categories", "price", "average_rating"],
    )
    meta_df["parent_asin"] = meta_df["parent_asin"].astype(str)
    meta_df["title"] = meta_df["title"].astype(str)
    meta_df["price"] = meta_df["price"].astype("float32")
    meta_df["average_rating"] = meta_df["average_rating"].astype("float32")
    return meta_df


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _as_float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Leave-One-Out split
# --------------------------------------------------------------------------- #


def build_loo_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]], int]:
    """Chronological Leave-One-Out split with contiguous id maps (schema D.1).

    Per user, sort by ``(timestamp asc, item_id)``: latest -> test,
    second-latest -> val, the rest -> train. Users with < 3 interactions stay
    train-only. ``user2idx``/``item2idx`` are built from the TRAIN split.

    Returns ``(train_df, val_df, test_df, id_maps, n_users_train_only)``.
    """
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    train_rows: List[pd.Series] = []
    val_rows: List[pd.Series] = []
    test_rows: List[pd.Series] = []
    n_train_only = 0

    for _user, grp in df.groupby("user_id", sort=True):
        recs = grp.to_dict("records")
        n = len(recs)
        if n < 3:
            n_train_only += 1
            train_rows.extend(recs)
            continue
        train_rows.extend(recs[: n - 2])
        val_rows.append(recs[n - 2])
        test_rows.append(recs[n - 1])

    train_df = pd.DataFrame(train_rows, columns=["user_id", "item_id", "rating", "timestamp"])
    val_df = pd.DataFrame(val_rows, columns=["user_id", "item_id", "rating", "timestamp"])
    test_df = pd.DataFrame(test_rows, columns=["user_id", "item_id", "rating", "timestamp"])

    if train_df.empty:
        raise RuntimeError("Leave-One-Out split produced an empty train set.")

    # Contiguous id maps from TRAIN (design B.4 / D.2b).
    train_users = sorted(train_df["user_id"].unique())
    train_items = sorted(train_df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(train_users)}
    item2idx = {it: i for i, it in enumerate(train_items)}
    id_maps = {"user2idx": user2idx, "item2idx": item2idx}

    train_df = _attach_indices(train_df, user2idx, item2idx)
    # val/test may reference items unseen in train -> item_idx = -1 (allowed;
    # candidate generation and evaluation handle unseen held-out items).
    val_df = _attach_indices(val_df, user2idx, item2idx)
    test_df = _attach_indices(test_df, user2idx, item2idx)

    log.info(
        "LOO split: train=%d val=%d test=%d (train-only users=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        n_train_only,
    )
    return train_df, val_df, test_df, id_maps, n_train_only


def _attach_indices(
    df: pd.DataFrame,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
) -> pd.DataFrame:
    """Add ``user_idx``/``item_idx`` columns and cast to the D.1 schema types."""
    if df.empty:
        out = pd.DataFrame(
            columns=["user_id", "item_id", "user_idx", "item_idx", "rating", "timestamp"]
        )
        return _cast_split_types(out)
    df = df.copy()
    df["user_idx"] = df["user_id"].map(lambda u: user2idx.get(u, -1))
    df["item_idx"] = df["item_id"].map(lambda i: item2idx.get(i, -1))
    df = df[["user_id", "item_id", "user_idx", "item_idx", "rating", "timestamp"]]
    return _cast_split_types(df)


def _cast_split_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["user_idx"] = df["user_idx"].astype("int32")
    df["item_idx"] = df["item_idx"].astype("int32")
    df["rating"] = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def _write_dataset_stats(
    cfg: Mapping[str, Any],
    stats_path: str,
    interactions: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users_train_only: int,
    bad_lines: int,
) -> None:
    """Write ``dataset_stats.json`` per schema D.2."""
    n_users = int(interactions["user_id"].nunique())
    n_items = int(interactions["item_id"].nunique())
    n_inter = int(len(interactions))
    density = (n_inter / (n_users * n_items)) if n_users and n_items else 0.0
    stats = {
        "scale": cfg.get("_scale", "unknown"),
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_inter,
        "density": round(density, 8),
        "k_core": int(cfg["preprocessing"]["k_core"]),
        "n_users_subset_requested": int(cfg["preprocessing"]["n_users_subset"]),
        "avg_interactions_per_user": round(n_inter / n_users, 4) if n_users else 0.0,
        "avg_interactions_per_item": round(n_inter / n_items, 4) if n_items else 0.0,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_users_train_only": int(n_users_train_only),
        "timestamp_min": int(interactions["timestamp"].min()) if n_inter else 0,
        "timestamp_max": int(interactions["timestamp"].max()) if n_inter else 0,
        "bad_lines_skipped": int(bad_lines),
        "retrieval_hit_rate_at_50": None,  # filled during the retrieve stage
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    log.info("wrote dataset stats -> %s", stats_path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run(cfg: Config) -> Dict[str, str]:
    """Execute the full data stage, returning a map of artifact name -> path."""
    set_seed(int(cfg["project"]["primary_seed"]))
    ensure_dirs(cfg, ["raw_dir", "processed_dir", "cache_dir", "results_dir", "figures_dir"])

    processed = resolve_path(cfg, "processed_dir")

    review_path, meta_path = download_dataset(cfg)

    interactions, surviving_items, bad_lines, missing_ts = parse_reviews(review_path, cfg)
    log.info("bad JSON lines skipped=%d, missing-timestamp rows dropped=%d", bad_lines, missing_ts)

    meta_df = parse_metadata(meta_path, surviving_items, cfg)

    train_df, val_df, test_df, id_maps, n_train_only = build_loo_split(interactions)

    # ---- Write artifacts ---------------------------------------------------
    train_path = os.path.join(processed, "train.parquet")
    val_path = os.path.join(processed, "val.parquet")
    test_path = os.path.join(processed, "test.parquet")
    meta_out = os.path.join(processed, "item_meta.parquet")
    idmap_path = os.path.join(processed, "id_maps.json")
    stats_path = os.path.join(processed, "dataset_stats.json")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    meta_df.to_parquet(meta_out, index=False)
    with open(idmap_path, "w", encoding="utf-8") as fh:
        json.dump(id_maps, fh)
    log.info("wrote splits + item_meta + id_maps to %s", processed)

    _write_dataset_stats(
        cfg, stats_path, interactions, train_df, val_df, test_df, n_train_only, bad_lines
    )

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "item_meta": meta_out,
        "id_maps": idmap_path,
        "dataset_stats": stats_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UC-ASR data pipeline")
    parser.add_argument(
        "--scale",
        choices=["smoke", "full", "kaggle"],
        default="smoke",
        help="config scale overlay to apply (default: smoke)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="path to config.yaml (default: config.yaml)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.scale)
    log.info("running data stage at scale=%s", args.scale)
    artifacts = run(cfg)
    log.info("data stage complete: %s", {k: os.path.basename(v) for k, v in artifacts.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
