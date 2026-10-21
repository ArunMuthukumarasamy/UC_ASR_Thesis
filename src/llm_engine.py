"""Think-Before-Recommend LLM engine with permanent SQLite caching (design B.5).

A single :class:`LLMEngine` exposes ``generate(prompt, temperature, pass_seed)``
over three backends:

* ``mock``   -- deterministic, sha256-keyed, microsecond-fast (the smoke default).
* ``hf``     -- local HuggingFace transformers (Qwen2.5-3B-Instruct; Mistral-7B
                fallback). 4-bit only on CUDA + bitsandbytes; CPU always float32
                (NIT-2).
* ``ollama`` -- local Ollama HTTP API.

Every generation is cached permanently in ``cache/llm_responses.sqlite`` keyed by
``sha256(model|prompt|temperature|pass_seed)`` with a permanent ``compute_ms``
column, so Latency is reproducible from the cache and cache hits never load the
model (design B.5 / B.8).

Runnable standalone: ``python -m src.llm_engine`` (mock self-test).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

from src.common import Config, get_logger, load_config, resolve_path

log = get_logger("llm_engine")

# K = 5 fixed stance templates for the mock (pinned module constant per B.5).
_MOCK_STANCES = [
    "This item aligns closely with the user's established tastes.",
    "This item is a reasonable but not perfect match for the user.",
    "This item only loosely relates to what the user usually buys.",
    "This item seems off from the user's typical preferences.",
    "This item is a strong, confident recommendation for this user.",
]
_MOCK_K = len(_MOCK_STANCES)

# pass_seed block width -- structural determinism constant (not config, B.5).
PASS_SEED_BLOCK = 1000


def pass_seed(experiment_seed: int, pass_index: int) -> int:
    """Authoritative derivation ``pass_seed(s, n) = s*1000 + n`` (design B.5)."""
    return int(experiment_seed) * PASS_SEED_BLOCK + int(pass_index)


# --------------------------------------------------------------------------- #
# SQLite cache
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
  cache_key   TEXT PRIMARY KEY,
  model       TEXT NOT NULL,
  prompt      TEXT NOT NULL,
  temperature REAL NOT NULL,
  pass_seed   INTEGER NOT NULL,
  reasoning   TEXT NOT NULL,
  relevance   REAL NOT NULL,
  raw         TEXT NOT NULL,
  compute_ms  REAL NOT NULL,
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_temp ON llm_cache(model, temperature);
"""


class LLMCache:
    """Thread-safe permanent SQLite cache (WAL, INSERT OR IGNORE)."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT reasoning, relevance, raw, compute_ms FROM llm_cache WHERE cache_key=?",
                (cache_key,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"reasoning": row[0], "relevance": float(row[1]), "raw": row[2], "compute_ms": float(row[3])}

    def put(
        self,
        cache_key: str,
        model: str,
        prompt: str,
        temperature: float,
        seed: int,
        reasoning: str,
        relevance: float,
        raw: str,
        compute_ms: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO llm_cache "
                "(cache_key, model, prompt, temperature, pass_seed, reasoning, relevance, raw, compute_ms, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    cache_key,
                    model,
                    prompt,
                    float(temperature),
                    int(seed),
                    reasoning,
                    float(relevance),
                    raw,
                    float(compute_ms),
                    _dt.datetime.now(_dt.timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------------- #
# Prompt builder + tolerant parser
# --------------------------------------------------------------------------- #


def build_prompt(
    profile_text: str,
    candidate_text: str,
    *,
    deep: bool = False,
    price: Optional[float] = None,
    average_rating: Optional[float] = None,
) -> str:
    """Think-Before-Recommend prompt. ``deep`` injects full metadata (NIT-6).

    The deep-verification variant renders a null price as the literal
    ``price: unknown`` and a null average_rating as ``rating: unknown`` so the
    cache key is deterministic regardless of missing values.
    """
    lines = [
        "You are a recommendation assistant.",
        f"User profile (recent purchases): {profile_text}",
        f"Candidate item: {candidate_text}",
    ]
    if deep:
        price_str = "unknown" if price is None else f"{float(price):.2f}"
        rating_str = "unknown" if average_rating is None else f"{float(average_rating):.2f}"
        lines.append(f"Item details -- price: {price_str}; rating: {rating_str}")
    lines += [
        "Rate how relevant this item is for the user on a scale from 1.0 "
        "(irrelevant) to 5.0 (perfect match).",
        "Respond in EXACTLY this format, with the score on the FIRST line:",
        "Relevance: <a single number from 1.0 to 5.0>",
        "Reasoning: <one or two sentences>",
    ]
    return "\n".join(lines)


_REL_RE = re.compile(r"[-+]?\d*\.?\d+")
# "4/5", "3.5 / 5", "85/100" -> captures (numerator, denominator).
_FRACTION_RE = re.compile(r"([-+]?\d*\.?\d+)\s*/\s*(\d+)")
# "4 out of 5", "3 of 5".
_OUT_OF_FIVE_RE = re.compile(r"([-+]?\d*\.?\d+)\s*(?:out of|of)\s*5\b", re.IGNORECASE)


def parse_response(text: str, rmin: float, rmax: float) -> Dict[str, Any]:
    """Tolerant parse: last float in [rmin, rmax] is the score; else midpoint.

    Returns ``{"reasoning", "relevance", "parsed_ok"}``.
    """
    reasoning = text.strip()
    m = re.search(r"Reasoning:\s*(.*?)(?:Relevance:|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    rel_section = text
    mr = re.search(r"Relevance:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if mr:
        rel_section = mr.group(1)

    # Highest priority: fraction/scale phrasings (e.g. "4/5", "3.5/5",
    # "4 out of 5"). A denominator of 5 means the numerator is the score.
    # If a denominator is present but is NOT 5 (e.g. "8/10", "85/100"), skip
    # that token so we do not misread an out-of-scale rating.
    mf = _FRACTION_RE.search(rel_section)
    if mf and int(mf.group(2)) == 5:
        num = float(mf.group(1))
        num = max(rmin, min(rmax, num))
        return {"reasoning": reasoning, "relevance": float(num), "parsed_ok": True}
    ms = _OUT_OF_FIVE_RE.search(rel_section)
    if ms:
        num = float(ms.group(1))
        num = max(rmin, min(rmax, num))
        return {"reasoning": reasoning, "relevance": float(num), "parsed_ok": True}

    candidates = [float(x) for x in _REL_RE.findall(rel_section)]
    valid = [v for v in candidates if rmin <= v <= rmax]
    if valid:
        return {"reasoning": reasoning, "relevance": float(valid[-1]), "parsed_ok": True}

    # Fall back to any in-range float anywhere in the text, else the midpoint.
    anyvalid = [float(x) for x in _REL_RE.findall(text) if rmin <= float(x) <= rmax]
    if anyvalid:
        return {"reasoning": reasoning, "relevance": float(anyvalid[-1]), "parsed_ok": True}
    mid = (rmin + rmax) / 2.0
    log.warning(
        "LLM response parse failure -> defaulting relevance to %.1f; raw(400)=%r",
        mid,
        " ".join(text.split())[:400],
    )
    return {"reasoning": reasoning, "relevance": float(mid), "parsed_ok": False}


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class MockBackend:
    """Deterministic sha256-keyed mock backend (design B.5)."""

    model_id = "mock"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._rmin = float(cfg["llm"]["relevance_min"])
        self._rmax = float(cfg["llm"]["relevance_max"])

    def generate(self, prompt: str, temperature: float, seed: int) -> Dict[str, Any]:
        h = hashlib.sha256(
            f"{self.model_id}|{prompt}|{temperature}|{seed}".encode("utf-8")
        ).hexdigest()
        rel = round(self._rmin + (int(h[:8], 16) / 0xFFFFFFFF) * (self._rmax - self._rmin), 1)
        stance = _MOCK_STANCES[int(h[8:12], 16) % _MOCK_K]
        cand = _extract_candidate_hint(prompt)
        reasoning = f"Considering {cand}: {stance}"
        raw = f"Reasoning: {reasoning}\nRelevance: {rel}"
        return {"reasoning": reasoning, "relevance": float(rel), "raw": raw, "parsed_ok": True}


def _extract_candidate_hint(prompt: str) -> str:
    m = re.search(r"Candidate item:\s*(.*)", prompt)
    if m:
        return m.group(1).strip()[:80]
    return "the candidate"


class HFBackend:
    """Local HuggingFace transformers backend; CPU-safe fallback (NIT-2)."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm = cfg["llm"]
        self._max_new = int(llm["max_new_tokens"])
        self._rmin = float(llm["relevance_min"])
        self._rmax = float(llm["relevance_max"])

        device = llm.get("device", "auto")
        use_cuda = torch.cuda.is_available() if device == "auto" else (device == "cuda")
        self._device = "cuda" if use_cuda else "cpu"

        load_kwargs: Dict[str, Any] = {}
        if use_cuda:
            want_4bit = bool(llm.get("load_in_4bit", False))
            bnb_ok = False
            if want_4bit:
                try:
                    import bitsandbytes  # noqa: F401

                    bnb_ok = True
                except Exception:  # noqa: BLE001
                    bnb_ok = False
            if want_4bit and bnb_ok:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            else:
                dtype = getattr(torch, str(llm.get("dtype", "bfloat16")), torch.bfloat16)
                load_kwargs["torch_dtype"] = dtype
        else:
            # NIT-2: CPU always float32; dtype (bfloat16) is CUDA-only, no probe.
            log.info("4-bit/bitsandbytes unavailable (CPU-only) -> loading in float32 on CPU")
            load_kwargs["torch_dtype"] = torch.float32

        self._model_id = llm["hf_model"]
        try:
            self._tok = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **load_kwargs)
        except Exception as exc:  # noqa: BLE001
            fallback = "mistralai/Mistral-7B-Instruct-v0.2"
            log.warning("HF model %s failed to load (%s); falling back to %s", self._model_id, exc, fallback)
            self._model_id = fallback
            self._tok = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **load_kwargs)
        if not use_cuda:
            self._model = self._model.to("cpu")
        self._model.eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, temperature: float, seed: int) -> Dict[str, Any]:
        import torch
        from transformers import set_seed as hf_set_seed

        hf_set_seed(int(seed))
        messages = [{"role": "user", "content": prompt}]
        # Build the prompt as a STRING via the chat template first (version-robust),
        # then tokenize to a dict/BatchEncoding. Avoids passing a BatchEncoding
        # positionally to model.generate (which crashes on newer transformers).
        try:
            text_in = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = self._tok(text_in, return_tensors="pt")
        except Exception:  # noqa: BLE001 - tokenizer without chat template
            enc = self._tok(prompt, return_tensors="pt")
        # Move the whole encoding to device (BatchEncoding and dict both support .items()).
        enc = {k: v.to(self._device) for k, v in enc.items()}
        input_ids = enc["input_ids"]

        do_sample = float(temperature) > 0.0
        gen_kwargs: Dict[str, Any] = {}
        if do_sample:
            # Only pass temperature when sampling; omit entirely otherwise so no
            # None is passed (which warns/errors on some transformers versions).
            gen_kwargs["temperature"] = float(temperature)
        pad_token_id = self._tok.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tok.eos_token_id
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self._max_new,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                **gen_kwargs,
            )
        prompt_len = input_ids.shape[1]
        text = self._tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        parsed = parse_response(text, self._rmin, self._rmax)
        parsed["raw"] = text
        return parsed


class OllamaBackend:
    """Local Ollama HTTP backend."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        llm = cfg["llm"]
        self._url = str(llm["ollama_url"]).rstrip("/") + "/api/generate"
        self._model_id = llm["ollama_model"]
        self._rmin = float(llm["relevance_min"])
        self._rmax = float(llm["relevance_max"])
        self._max_new = int(llm["max_new_tokens"])

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, temperature: float, seed: int) -> Dict[str, Any]:
        import requests

        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "seed": int(seed),
                "num_predict": self._max_new,
            },
        }
        try:
            resp = requests.post(self._url, json=payload, timeout=300)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Ollama not reachable at {self._url} ({exc}). Start Ollama "
                "(`ollama serve`) or switch llm.backend to 'hf'/'mock'."
            ) from exc
        text = resp.json().get("response", "")
        parsed = parse_response(text, self._rmin, self._rmax)
        parsed["raw"] = text
        return parsed


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class LLMEngine:
    """Caching facade over the selected backend (design B.5)."""

    def __init__(self, cfg: Config, scale: Optional[str] = None) -> None:
        self._cfg = cfg
        self._scale = scale or cfg.get("_scale", "smoke")
        self._cache_enabled = bool(cfg["llm"].get("cache_enabled", True))
        self._rmin = float(cfg["llm"]["relevance_min"])
        self._rmax = float(cfg["llm"]["relevance_max"])
        self._cache = LLMCache(resolve_path(cfg, "llm_cache_db")) if self._cache_enabled else None
        self._backend_name = cfg["llm"]["backend"]
        self._backend = None  # lazy: cache hits never load the model

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        name = self._backend_name
        log.info("initialising LLM backend: %s", name)
        if name == "mock":
            self._backend = MockBackend(self._cfg)
        elif name == "hf":
            self._backend = HFBackend(self._cfg)
        elif name == "ollama":
            self._backend = OllamaBackend(self._cfg)
        else:
            raise ValueError(f"unknown llm.backend: {name!r} (expected mock/hf/ollama)")

    @property
    def model_id(self) -> str:
        if self._backend_name == "mock":
            return "mock"
        if self._backend_name == "hf":
            return self._cfg["llm"]["hf_model"]
        if self._backend_name == "ollama":
            return self._cfg["llm"]["ollama_model"]
        return self._backend_name

    def _cache_key(self, prompt: str, temperature: float, seed: int) -> str:
        return hashlib.sha256(
            f"{self.model_id}|{prompt}|{temperature}|{seed}".encode("utf-8")
        ).hexdigest()

    def generate(self, prompt: str, temperature: float, pass_seed: int) -> Dict[str, Any]:
        """Generate (or fetch cached) a reasoning trace + relevance score."""
        temperature = float(temperature)
        key = self._cache_key(prompt, temperature, pass_seed)

        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return {
                    "raw": hit["raw"],
                    "reasoning": hit["reasoning"],
                    "relevance": float(hit["relevance"]),
                    "compute_ms": float(hit["compute_ms"]),
                    "parsed_ok": True,
                }

        self._ensure_backend()
        t0 = time.perf_counter()
        result = self._backend.generate(prompt, temperature, pass_seed)
        compute_ms = (time.perf_counter() - t0) * 1000.0
        relevance = max(self._rmin, min(self._rmax, float(result["relevance"])))

        if self._cache is not None:
            self._cache.put(
                key,
                self.model_id,
                prompt,
                temperature,
                pass_seed,
                result["reasoning"],
                relevance,
                result["raw"],
                compute_ms,
            )
        return {
            "raw": result["raw"],
            "reasoning": result["reasoning"],
            "relevance": relevance,
            "compute_ms": compute_ms,
            "parsed_ok": bool(result.get("parsed_ok", True)),
        }

    def cache_count(self) -> int:
        return self._cache.count() if self._cache is not None else 0


def _self_test() -> int:
    cfg = load_config("config.yaml", "smoke")
    engine = LLMEngine(cfg, "smoke")
    prompt = build_prompt("coffee; tea", "Dark Roast Coffee. Categories: Beverages")
    s = 42
    rels = [engine.generate(prompt, 0.7, pass_seed(s, n))["relevance"] for n in range(3)]
    log.info("mock 3-pass relevances: %s", rels)
    assert len(set(rels)) > 1, "expected non-constant relevance across 3 mock passes"

    before = engine.cache_count()
    _ = engine.generate(prompt, 0.7, pass_seed(s, 0))  # cache hit, no new row
    after = engine.cache_count()
    assert after == before, "cache hit must not add a row"
    assert before > 0, "cache should contain rows after generation"

    single = engine.generate(prompt, 0.0, pass_seed(s, 0))
    single2 = engine.generate(prompt, 0.0, pass_seed(s, 0))
    assert single["relevance"] == single2["relevance"], "T=0 single pass must be stable"
    log.info("llm_engine self-test passed (rows=%d)", after)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
