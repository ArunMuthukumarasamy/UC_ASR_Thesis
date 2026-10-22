"""UC-ASR pipeline orchestrator (design B.8).

Runs the four-stage DAG -- data -> retrieve -> rerank -> eval -- with a scale
overlay and per-stage selection. Every stage is idempotent: it skips work whose
outputs already exist unless ``--force`` is given.

Usage
-----
    python run_pipeline.py --scale smoke --stage all
    python run_pipeline.py --scale full  --stage data
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from src import data_pipeline, evaluate, retrievers, rerankers
from src.common import get_logger, load_config, resolve_path

log = get_logger("run_pipeline")


def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _stage_data(cfg, force: bool) -> None:
    processed = resolve_path(cfg, "processed_dir")
    targets = [os.path.join(processed, f) for f in ("train.parquet", "val.parquet", "test.parquet")]
    if not force and all(_exists(t) for t in targets):
        log.info("[data] outputs present, skipping (use --force to rebuild)")
        return
    data_pipeline.run(cfg)


def _stage_retrieve(cfg, force: bool) -> None:
    processed = resolve_path(cfg, "processed_dir")
    target = os.path.join(processed, "candidates_top50.parquet")
    if not force and _exists(target):
        log.info("[retrieve] candidates present, skipping (use --force to rebuild)")
        return
    retrievers.run(cfg)


def _stage_rerank(cfg, force: bool) -> None:
    # Rerank has no single on-disk artifact (results live in the LLM cache and
    # are recomputed in eval); run it to warm the cache / tune on val.
    rerankers.run(cfg)


def _stage_eval(cfg, force: bool) -> None:
    results = resolve_path(cfg, "results_dir")
    target = os.path.join(results, "thesis_main_table.csv")
    if not force and _exists(target):
        log.info("[eval] main table present, skipping (use --force to rebuild)")
        return
    evaluate.run(cfg)


_STAGES = {
    "data": _stage_data,
    "retrieve": _stage_retrieve,
    "rerank": _stage_rerank,
    "eval": _stage_eval,
}
_ORDER = ["data", "retrieve", "rerank", "eval"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UC-ASR pipeline orchestrator")
    parser.add_argument("--scale", choices=["smoke", "full", "kaggle", "kaggle_fast", "kaggle_unc", "gemini"], default="smoke")
    parser.add_argument("--stage", choices=["data", "retrieve", "rerank", "eval", "all"], default="all")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild stage outputs even if present")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.scale)

    backend = cfg["llm"]["backend"]
    seeds = cfg["project"]["seeds"]
    log.info("=" * 70)
    log.info("UC-ASR pipeline | scale=%s | stage=%s | backend=%s | seeds=%s", args.scale, args.stage, backend, seeds)
    log.info("=" * 70)

    stages = _ORDER if args.stage == "all" else [args.stage]
    for st in stages:
        log.info(">>> stage: %s", st)
        _STAGES[st](cfg, args.force)

    log.info("pipeline complete (scale=%s, stage=%s)", args.scale, args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())

