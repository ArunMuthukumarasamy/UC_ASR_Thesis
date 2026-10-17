"""Shared infrastructure for the UC-ASR pipeline.

Provides the single config loader (:func:`load_config`), deterministic seeding
(:func:`set_seed`), a stdlib logger factory (:func:`get_logger`), and path
helpers that resolve every ``paths.*`` entry against ``project.root``.

Nothing in the codebase should read ``config.yaml`` directly -- everything goes
through :func:`load_config` so the scale-overlay merge and validation happen in
exactly one place (design B.1/B.2).
"""

from __future__ import annotations

import copy
import logging
import os
import random
import sys
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Config object
# --------------------------------------------------------------------------- #


class Config(dict):
    """A plain ``dict`` with optional dotted attribute access.

    Nested dict values are wrapped as :class:`Config` on access so both
    ``cfg["preprocessing"]["k_core"]`` and ``cfg.preprocessing.k_core`` work.
    Dict-style access is the canonical form used throughout the codebase and in
    the acceptance-criteria one-liners; attribute access is a convenience.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _deep_merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto a deep copy of ``base``.

    Dict values are merged key-by-key; any non-dict value (including lists)
    replaces the base value wholesale. The base is never mutated.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _wrap(obj: Any) -> Any:
    """Recursively wrap nested dicts as :class:`Config` for dotted access."""
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate types/ranges, raising ValueError naming the offending key."""
    pre = cfg.get("preprocessing", {})
    _require(
        isinstance(pre.get("k_core"), int) and pre["k_core"] > 0,
        "preprocessing.k_core must be a positive integer",
    )
    _require(
        isinstance(pre.get("n_users_subset"), int) and pre["n_users_subset"] > 0,
        "preprocessing.n_users_subset must be a positive integer",
    )
    bad_frac = pre.get("data_parse_max_bad_frac")
    _require(
        isinstance(bad_frac, (int, float)) and 0.0 <= float(bad_frac) <= 1.0,
        "preprocessing.data_parse_max_bad_frac must be in [0, 1]",
    )

    ret = cfg.get("retrievers", {})
    _require(
        isinstance(ret.get("top_k_candidates"), int) and ret["top_k_candidates"] > 0,
        "retrievers.top_k_candidates must be a positive integer",
    )
    for mf_key in ("bpr_mf", "lightgcn"):
        epochs = ret.get(mf_key, {}).get("epochs")
        _require(
            isinstance(epochs, int) and epochs > 0,
            f"retrievers.{mf_key}.epochs must be a positive integer",
        )

    unc = cfg.get("uncertainty", {})
    n_passes = unc.get("uaro", {}).get("n_passes")
    _require(
        isinstance(n_passes, int) and n_passes >= 1,
        "uncertainty.uaro.n_passes (N) must be an integer >= 1",
    )
    atm = unc.get("atm", {})
    for thr_key in ("sbert_sim_threshold", "epsilon"):
        val = atm.get(thr_key)
        _require(
            isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0,
            f"uncertainty.atm.{thr_key} must be in [0, 1]",
        )

    uc = cfg.get("rerankers", {}).get("uc_asr", {})
    for thr_key in ("tau_high", "tau_low"):
        val = uc.get(thr_key)
        # tau_high can be set to 1.01 by the no_confident_freeze_tier ablation,
        # so allow a small slack above 1.0 for threshold knobs.
        _require(
            isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.01,
            f"rerankers.uc_asr.{thr_key} must be in [0, 1.01]",
        )


def load_config(path: str = "config.yaml", scale: str = "smoke") -> Config:
    """Load ``config.yaml`` and merge the ``scales.<scale>`` overlay onto it.

    Fails loudly if the top-level ``scales:`` block is missing (design B.2) or
    if the requested ``scale`` is not defined. After merging, unconditionally
    sets ``preprocessing.min_user_interactions`` and
    ``preprocessing.min_item_interactions`` equal to ``preprocessing.k_core``
    (NIT-3); when all three keys are present in the resolved config it asserts
    they already agree before deriving.

    Parameters
    ----------
    path:
        Path to the YAML config (default ``"config.yaml"``).
    scale:
        Scale overlay to apply, e.g. ``"smoke"`` or ``"full"``. The default
        matches the CLI default so standalone and orchestrated runs agree
        (design B.1).

    Returns
    -------
    Config
        A dotted/dict-access mapping of the resolved configuration.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path!r}")

    with open(path, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    if not isinstance(base, dict):
        raise ValueError(f"config file {path!r} did not parse to a mapping")

    scales = base.get("scales")
    if not isinstance(scales, dict) or not scales:
        raise RuntimeError(
            "config.yaml is missing the required top-level `scales:` block "
            "(must define at least `full: {}` and a `smoke:` overlay). "
            "Scale resolution cannot proceed without it -- see design B.2. "
            f"Fix {path!r} by adding a `scales:` section."
        )

    if scale not in scales:
        available = ", ".join(sorted(scales)) or "<none>"
        raise ValueError(
            f"unknown scale {scale!r}; config.yaml `scales:` defines: {available}"
        )

    overlay = scales[scale] or {}
    if not isinstance(overlay, Mapping):
        raise ValueError(
            f"scales.{scale} must be a mapping (or empty), got {type(overlay).__name__}"
        )

    # Base config minus the meta-blocks that are not part of the resolved view.
    base_no_meta = {k: v for k, v in base.items() if k != "scales"}
    merged = _deep_merge(base_no_meta, overlay)

    # NIT-3: k_core is the single authoritative threshold; min_user_interactions
    # and min_item_interactions are derived aliases of it, set unconditionally
    # after the overlay merge. The "they match" assertion applies ONLY when the
    # scale overlay did NOT change k_core (e.g. full scale, where all three keys
    # carry their authored base values) -- if an overlay overrides k_core the
    # base min_* are intentionally stale aliases and are simply overwritten.
    pre = merged.setdefault("preprocessing", {})
    k_core = pre.get("k_core")
    if k_core is None:
        raise ValueError("preprocessing.k_core is required but missing from config")
    overlay_changed_k_core = "k_core" in (overlay.get("preprocessing", {}) or {})
    if not overlay_changed_k_core:
        for alias in ("min_user_interactions", "min_item_interactions"):
            if alias in pre and pre[alias] is not None and pre[alias] != k_core:
                raise ValueError(
                    f"preprocessing.{alias}={pre[alias]} disagrees with "
                    f"preprocessing.k_core={k_core}; k_core is the single "
                    "authoritative threshold (NIT-3). Remove the explicit "
                    f"{alias} or set it equal to k_core."
                )
    pre["min_user_interactions"] = k_core
    pre["min_item_interactions"] = k_core

    # --- Kaggle environment auto-detection -------------------------------
    # On Kaggle, /kaggle/input is READ-ONLY and all outputs must go under
    # /kaggle/working. When running there, redirect the project root for
    # writable outputs and, if an attached dataset carries the raw files,
    # point raw_dir at it. Local/other runs are untouched. Opt out with
    # UC_ASR_NO_KAGGLE_PATHS=1.
    if _on_kaggle() and os.environ.get("UC_ASR_NO_KAGGLE_PATHS") != "1":
        _apply_kaggle_paths(merged)

    _validate_config(merged)

    cfg = _wrap(merged)
    cfg["_scale"] = scale
    return cfg


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` deterministically.

    Also sets ``PYTHONHASHSEED``. Called at the top of every stage with the
    active experiment seed (design B.1). Torch is imported lazily so modules
    that do not need it still import quickly.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # warn_only keeps CPU ops that lack a deterministic impl from crashing.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch is a locked dependency
        pass


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger writing to stdout with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def project_root(cfg: Mapping[str, Any]) -> str:
    """Absolute project root resolved from ``project.root`` in the config."""
    root = cfg.get("project", {}).get("root", ".")
    return os.path.abspath(root)


def resolve_path(cfg: Mapping[str, Any], key_or_path: str) -> str:
    """Resolve a ``paths.*`` key (or a raw relative path) against project root.

    If ``key_or_path`` names an entry under ``paths`` it is resolved; otherwise
    the string is treated as a path relative to the project root.
    """
    root = project_root(cfg)
    paths = cfg.get("paths", {})
    raw = paths.get(key_or_path, key_or_path)
    if os.path.isabs(raw):
        return raw
    return os.path.normpath(os.path.join(root, raw))


def ensure_dirs(cfg: Mapping[str, Any], keys: Iterable[str] | None = None) -> None:
    """``os.makedirs(exist_ok=True)`` for the named ``paths.*`` directory keys.

    When ``keys`` is ``None`` every directory-like entry under ``paths`` is
    created (file-valued entries such as ``llm_cache_db`` have their parent
    directory created instead).
    """
    paths = cfg.get("paths", {})
    if keys is None:
        keys = list(paths.keys())
    for key in keys:
        target = resolve_path(cfg, key)
        # Treat entries that look like files (have a suffix) as files and make
        # their parent directory; everything else is a directory.
        if os.path.splitext(target)[1]:
            os.makedirs(os.path.dirname(target), exist_ok=True)
        else:
            os.makedirs(target, exist_ok=True)


# --------------------------------------------------------------------------- #
# Kaggle environment helpers
# --------------------------------------------------------------------------- #


def _on_kaggle() -> bool:
    """True when running inside a Kaggle kernel."""
    return (
        os.path.isdir("/kaggle/working")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
    )


def _apply_kaggle_paths(merged: Dict[str, Any]) -> None:
    """Redirect writable paths to /kaggle/working and raw reads to an input dataset.

    Mutates ``merged`` in place. ``project.root`` becomes /kaggle/working so all
    relative ``paths.*`` resolve to a writable location. If a Kaggle input
    dataset directory contains the raw .jsonl.gz files, ``paths.raw_dir`` is
    pointed at it (read-only reads are fine). The dataset slug can be overridden
    with the UC_ASR_KAGGLE_DATASET env var (expects /kaggle/input/<slug>).
    """
    project = merged.setdefault("project", {})
    project["root"] = "/kaggle/working"

    paths = merged.setdefault("paths", {})
    ds = merged.get("dataset", {})
    review = ds.get("review_file", "Grocery_and_Gourmet_Food.jsonl.gz")

    slug = os.environ.get("UC_ASR_KAGGLE_DATASET")
    candidates = []
    if slug:
        candidates.append(os.path.join("/kaggle/input", slug))
    # Scan attached input datasets for one that holds the raw review file.
    input_root = "/kaggle/input"
    if os.path.isdir(input_root):
        for entry in sorted(os.listdir(input_root)):
            candidates.append(os.path.join(input_root, entry))
            candidates.append(os.path.join(input_root, entry, "raw"))
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, review)):
            paths["raw_dir"] = cand
            break


if __name__ == "__main__":  # pragma: no cover - lightweight self-test
    log = get_logger("common.selftest")
    cfg_smoke = load_config("config.yaml", "smoke")
    assert cfg_smoke["preprocessing"]["n_users_subset"] == 6000
    assert cfg_smoke["preprocessing"]["min_user_interactions"] == 2
    assert cfg_smoke["preprocessing"]["min_item_interactions"] == 2
    assert cfg_smoke["llm"]["backend"] == "mock"

    cfg_full = load_config("config.yaml", "full")
    assert (
        cfg_full["preprocessing"]["min_user_interactions"]
        == 5
        == cfg_full["preprocessing"]["k_core"]
    )

    set_seed(cfg_smoke["project"]["primary_seed"])
    ensure_dirs(cfg_smoke)
    log.info("common self-test passed (smoke + full config resolved)")
