"""UC-ASR research codebase package.

Uncertainty-Conditioned Adaptive Semantic Reranking for LLM-Based Recommendation.

Every module under ``src`` is importable and runnable standalone via
``python -m src.<module>``. Shared infrastructure (config loading, seeding,
logging, path helpers) lives in :mod:`src.common`.

Convenience names are exposed lazily (PEP 562) so that ``python -m
src.<module>`` does not eagerly import sibling modules -- importing a submodule
as ``__main__`` before the package body runs would otherwise emit a spurious
RuntimeWarning.
"""

__all__ = [
    "load_config",
    "set_seed",
    "get_logger",
    "resolve_path",
    "ensure_dirs",
]


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    if name in __all__:
        from src import common

        return getattr(common, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
