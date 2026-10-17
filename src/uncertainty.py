"""Uncertainty quantification for UC-ASR (design B.6).

Pure, unit-testable functions over the N per-candidate relevance scores and
reasoning traces:

* UARO score variance -> mean ``R``, sample variance ``sigma^2``, confidence
  ``c = 1/(1+sigma^2)``, and per-user min-max normalised uncertainty ``U``.
* ATM Semantic Cluster Ratio ``CR = max_m|s_m|/N`` via greedy SBERT-similarity
  clustering (optional bidirectional-NLI tightening).
* ATM Bayesian top-k threshold ``t(k)`` by binary search with the degenerate
  ``t(k) = -inf`` branch (NIT-5), and the boundary probability ``p_topk``.

Runnable standalone: ``python -m src.uncertainty`` (math self-tests).
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.common import get_logger

log = get_logger("uncertainty")


# --------------------------------------------------------------------------- #
# UARO score variance
# --------------------------------------------------------------------------- #


def uaro_stats(scores: Sequence[float]) -> Tuple[float, float, float]:
    """Return ``(R, sigma2, c)`` for one candidate's N relevance scores.

    ``R`` = mean, ``sigma2`` = sample variance (ddof=1), ``c = 1/(1+sigma2)``.
    Requires >= 2 scores (design B.6; callers upstream enforce the >=2 parsed
    rule and the most-uncertain fallback).
    """
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size < 2:
        raise ValueError("uaro_stats requires at least 2 scores")
    r = float(arr.mean())
    sigma2 = float(arr.var(ddof=1))
    c = 1.0 / (1.0 + sigma2)
    return r, sigma2, c


def normalize_uncertainty(sigma2_values: Sequence[float]) -> np.ndarray:
    """Per-user min-max normalise sigma^2 into ``U`` in [0, 1] (design B.6)."""
    arr = np.asarray(list(sigma2_values), dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float64)


# The per-user R/sigma2/c/U/CR/p_topk frame is assembled inline in the rerank
# stage (``rerankers._compute_uncertainty_frame``) from the shared LLM cache,
# using ``uaro_stats`` and ``normalize_uncertainty`` above plus the ATM
# functions below (design B.6: the frame is recomputed in-memory, never a
# required on-disk artifact).


# --------------------------------------------------------------------------- #
# ATM Semantic Cluster Ratio
# --------------------------------------------------------------------------- #


class _NLIRefiner:
    """Optional bidirectional-NLI tightening; degrades to SBERT-only on failure."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._ok = True

    def _load(self) -> None:
        if self._model is not None or not self._ok:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device="cpu")
        except Exception as exc:  # noqa: BLE001
            log.warning("NLI unavailable -> using SBERT-similarity clustering only (%s)", exc)
            self._ok = False

    def entails_both_ways(self, a: str, b: str) -> bool:
        self._load()
        if not self._ok or self._model is None:
            return True  # SBERT is authoritative; NLI only tightens.
        try:
            scores = self._model.predict([(a, b), (b, a)])
            # nli-deberta label order: [contradiction, entailment, neutral].
            labels = np.asarray(scores)
            if labels.ndim == 2 and labels.shape[1] >= 2:
                return bool(labels[0].argmax() == 1 and labels[1].argmax() == 1)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("NLI prediction failed -> SBERT-only for this pair (%s)", exc)
            self._ok = False
            return True


def cluster_ratio(
    traces: Sequence[str],
    encoder: Any,
    sim_threshold: float,
    nli: Optional[_NLIRefiner] = None,
) -> float:
    """Greedy semantic-equivalence clustering -> ``CR = max cluster size / N``.

    Two traces are equivalent iff SBERT cosine >= ``sim_threshold`` and (if an
    NLI refiner is supplied and loadable) bidirectional entailment holds.
    Deterministic given a fixed trace order (design B.6).
    """
    n = len(traces)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    embs = encoder.encode_many(list(traces))
    clusters: List[List[int]] = []
    reps: List[int] = []
    for i in range(n):
        placed = False
        for ci, rep in enumerate(reps):
            cos = float(np.dot(embs[i], embs[rep]))
            if cos >= sim_threshold and (nli is None or nli.entails_both_ways(traces[rep], traces[i])):
                clusters[ci].append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(i)
    largest = max(len(c) for c in clusters)
    return largest / n


# --------------------------------------------------------------------------- #
# ATM Bayesian top-k threshold
# --------------------------------------------------------------------------- #


def _norm_sf(x: float) -> float:
    """Survival function 1 - Phi(x) via the error function (no scipy import cost)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def bayesian_topk_threshold(
    mu: Sequence[float],
    sigma2: Sequence[float],
    beta: float,
    top_k: int,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> Tuple[float, List[float]]:
    """Solve ``sum_i P(x_i > t) = k`` by bisection; degenerate branch -> ``-inf``.

    Returns ``(t_k, p_topk_list)`` where ``p_topk_i = P(x_i > t_k)``. When the
    candidate count is <= ``top_k`` (so ``g(lower) <= k``) the equation has no
    root: set ``t_k = -inf`` and ``p_topk = 1.0`` for all (NIT-5 / design B.6).
    """
    mu_a = np.asarray(list(mu), dtype=np.float64)
    s2_a = np.asarray(list(sigma2), dtype=np.float64)
    n = mu_a.size
    if n == 0:
        return float("-inf"), []
    std = np.sqrt(s2_a + float(beta) ** 2)
    std = np.where(std <= 0, 1e-8, std)

    def g(t: float) -> float:
        return float(sum(_norm_sf((t - mu_a[i]) / std[i]) for i in range(n)))

    max_std = float(std.max())
    lower = float(mu_a.min()) - 5.0 * max_std
    upper = float(mu_a.max()) + 5.0 * max_std

    # Degenerate bracket: g is bounded above by n; if g(lower) <= k no root.
    if g(lower) <= top_k or n <= top_k:
        p = [1.0] * n
        return float("-inf"), p

    lo, hi = lower, upper
    t_mid = lo
    for _ in range(max_iter):
        t_mid = 0.5 * (lo + hi)
        val = g(t_mid)
        if abs(val - top_k) < tol:
            break
        # g decreasing in t: if g > k we must raise t.
        if val > top_k:
            lo = t_mid
        else:
            hi = t_mid
    p = [_norm_sf((t_mid - mu_a[i]) / std[i]) for i in range(n)]
    return float(t_mid), p


def boundary_certainty(p_topk: Sequence[float]) -> List[float]:
    """``B_i = 2*|p_i - 0.5|`` in [0, 1] (design B.7d)."""
    return [2.0 * abs(float(p) - 0.5) for p in p_topk]


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


class _IdentityEncoder:
    """Tiny encoder stub for the cluster-ratio self-test (no model download)."""

    def __init__(self) -> None:
        self._vocab: dict = {}

    def encode_many(self, texts: List[str]) -> np.ndarray:
        import hashlib as _h

        out = []
        for t in texts:
            h = _h.sha256(t.encode()).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float64)[:8]
            v = v / (np.linalg.norm(v) + 1e-8)
            out.append(v)
        return np.vstack(out)


def _self_test() -> int:
    # UARO: scores [1,3,5] -> sample var = 4.0 -> c = 0.2.
    r, s2, c = uaro_stats([1.0, 3.0, 5.0])
    assert abs(r - 3.0) < 1e-9 and abs(s2 - 4.0) < 1e-9 and abs(c - 0.2) < 1e-9, (r, s2, c)

    # CR over 3 identical traces == 1.0.
    enc = _IdentityEncoder()
    assert abs(cluster_ratio(["a", "a", "a"], enc, 0.85) - 1.0) < 1e-9

    # Degenerate top-k: 3 candidates, k=10 -> t = -inf, all p == 1.0.
    t, p = bayesian_topk_threshold([3.0, 3.0, 3.0], [0.0, 0.0, 0.0], beta=0.5, top_k=10)
    assert t == float("-inf") and all(abs(pi - 1.0) < 1e-12 for pi in p), (t, p)

    # Non-degenerate: 20 candidates, k=10 -> finite t, ~half above boundary.
    mu = list(np.linspace(1.0, 5.0, 20))
    s2v = [0.5] * 20
    t2, p2 = bayesian_topk_threshold(mu, s2v, beta=0.5, top_k=10)
    assert math.isfinite(t2), t2
    assert abs(sum(p2) - 10.0) < 1e-2, sum(p2)

    log.info("uncertainty self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
