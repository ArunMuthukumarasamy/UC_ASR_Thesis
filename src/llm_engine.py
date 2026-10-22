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
import math
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
    # FIX A: instruction/task FIRST, profile + candidate as clearly delimited
    # DATA blocks, answer format unmistakable LAST. The model must treat the
    # profile as data to rate, not text to continue/echo.
    lines = [
        "You are a strict product-recommendation rater. You MUST answer, not repeat the input.",
        "TASK: rate how relevant the CANDIDATE ITEM is for this user's tastes.",
        "Use an integer 1-5 (1=irrelevant, 5=perfect match).",
        "",
        "USER_RECENT_PURCHASES:",
        f"{profile_text}",
        "",
        "CANDIDATE_ITEM:",
        f"{candidate_text}",
    ]
    if deep:
        # NIT-6: render null price/rating as the literal 'unknown' for
        # deterministic cache keys regardless of missing values.
        price_str = "unknown" if price is None else f"{float(price):.2f}"
        rating_str = "unknown" if average_rating is None else f"{float(average_rating):.2f}"
        lines += [
            "",
            "ITEM_DETAILS:",
            f"price: {price_str}; rating: {rating_str}",
        ]
    lines += [
        "",
        "Answer with EXACTLY two lines and nothing else:",
        "Reasoning: <one short sentence>",
        "Score: <integer 1-5>",
    ]
    return "\n".join(lines)


_REL_RE = re.compile(r"[-+]?\d*\.?\d+")
# Explicit "Score: 4" / "Relevance: 3.5" markers. ``\d*\.?\d+`` requires at
# least one digit so a garbled "Relevance: .a" never matches as a number.
_SCORE_RE = re.compile(r"Score:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
_RELEVANCE_RE = re.compile(r"Relevance:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
# "4/5", "3.5 / 5", "85/100" -> captures (numerator, denominator).
_FRACTION_RE = re.compile(r"([-+]?\d*\.?\d+)\s*/\s*(\d+)")
# "4 out of 5", "3 of 5".
_OUT_OF_FIVE_RE = re.compile(r"([-+]?\d*\.?\d+)\s*(?:out of|of)\s*5\b", re.IGNORECASE)


def parse_response(text: str, rmin: float, rmax: float) -> Dict[str, Any]:
    """Tolerant parse of a reason-first / score-last LLM response.

    Handles both the new ``Score: <int>`` marker and the legacy
    ``Relevance: <num>`` marker, in either order, with integer or float
    scores, and tolerates garbled lines (e.g. ``Relevance: .a``). The score
    is searched for in priority order; only a true midpoint default yields
    ``parsed_ok=False``.

    Returns ``{"reasoning", "relevance", "parsed_ok"}``.
    """
    # (a) Reasoning: text after "Reasoning:" up to a trailing Score:/Relevance:
    # line if present, else the whole trimmed text.
    reasoning = text.strip()
    m = re.search(
        r"Reasoning:\s*(.*?)(?:\n?\s*(?:Score|Relevance):|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m and m.group(1).strip():
        reasoning = m.group(1).strip()

    # (b) Score search, first branch to produce an in-range value wins.

    # 1. Explicit "Score:" marker -> take the LAST match (the final line). An
    #    explicit score the model deliberately wrote is a real judgement even
    #    when out of range, so CLAMP it to [rmin,rmax] rather than discard it
    #    (e.g. 7 -> rmax, 30 -> rmax, 0 -> rmin, -2 -> rmin). Only the absence
    #    of any Score: match falls through to the next branch.
    score_matches = _SCORE_RE.findall(text)
    for raw in reversed(score_matches):
        val = float(raw)
        if math.isfinite(val):
            num = max(rmin, min(rmax, val))
            return {"reasoning": reasoning, "relevance": float(num), "parsed_ok": True}

    # 2. Legacy "Relevance:" marker -> LAST match, clamped (same rationale as
    #    the Score: branch). The regex requires a digit, so "Relevance: .a"
    #    never matches here.
    rel_matches = _RELEVANCE_RE.findall(text)
    for raw in reversed(rel_matches):
        val = float(raw)
        if math.isfinite(val):
            num = max(rmin, min(rmax, val))
            return {"reasoning": reasoning, "relevance": float(num), "parsed_ok": True}

    # 3. Fraction / "out of 5" phrasings anywhere in the text. A denominator
    #    of 5 means the numerator is the score; other denominators are skipped
    #    so we do not misread an out-of-scale rating (e.g. "8/10").
    mf = _FRACTION_RE.search(text)
    if mf and int(mf.group(2)) == 5:
        num = max(rmin, min(rmax, float(mf.group(1))))
        return {"reasoning": reasoning, "relevance": float(num), "parsed_ok": True}
    ms = _OUT_OF_FIVE_RE.search(text)
    if ms:
        num = max(rmin, min(rmax, float(ms.group(1))))
        return {"reasoning": reasoning, "relevance": float(num), "parsed_ok": True}

    # 4. Any in-range float anywhere in the text.
    anyvalid = [float(x) for x in _REL_RE.findall(text) if rmin <= float(x) <= rmax]
    if anyvalid:
        return {"reasoning": reasoning, "relevance": float(anyvalid[-1]), "parsed_ok": True}

    # 5. True midpoint default (the only parsed_ok=False path).
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
    # Match the CANDIDATE_ITEM: data block (FIX A format); fall back to the
    # legacy "Candidate item:" phrasing for any older cached/alt prompts.
    m = re.search(r"CANDIDATE_ITEM:\s*\n?(.*)", prompt, re.IGNORECASE)
    if m is None:
        m = re.search(r"Candidate item:\s*(.*)", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:80]
    return "the candidate"


class HFBackend:
    """Local HuggingFace transformers backend; CPU-safe fallback (NIT-2)."""

    # FIX D: module-level (class) counter so the FIRST 3 generate() calls log
    # the exact prompt-token length and the FULL decoded text at INFO level.
    _dbg = 0

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
        # FIX B: proper system+user message list so the chat template frames
        # the whole thing as an instruction (profile = data), not a single user
        # turn the model continues/echoes.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict product-recommendation rater. Always "
                    "answer in the requested 'Reasoning:'/'Score:' format. Never "
                    "repeat or echo the input."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        # Tokenize the chat template DIRECTLY in one step so special tokens
        # (notably BOS) are added exactly once. Applying the template with
        # tokenize=False and then re-tokenizing the string double-adds BOS and
        # breaks the instruction framing, which makes the model echo/continue
        # the prompt instead of responding as an assistant.
        try:
            enc = self._tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,  # get a dict with input_ids AND attention_mask
            )
        except Exception:  # noqa: BLE001 - no chat template OR older API w/o return_dict
            try:
                input_ids_only = self._tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
                # Older API returns a bare tensor -> wrap into a dict with a mask.
                import torch as _torch

                enc = {
                    "input_ids": input_ids_only,
                    "attention_mask": _torch.ones_like(input_ids_only),
                }
            except Exception:  # noqa: BLE001 - tokenizer without chat template at all
                enc = self._tok(prompt, return_tensors="pt")
        # Move the whole encoding to device (BatchEncoding and dict both support .items()).
        enc = {k: v.to(self._device) for k, v in enc.items()}
        input_ids = enc["input_ids"]

        # FIX E: generation sanity. When temperature>0 (N=3 passes @ T=0.7) use
        # do_sample=True with temperature set; when temperature==0 (single-pass)
        # do_sample=False and NO temperature/top_p passed. repetition_penalty is
        # always applied to discourage verbatim echo/repetition (safe, standard).
        do_sample = float(temperature) > 0.0
        gen_kwargs: Dict[str, Any] = {"repetition_penalty": 1.3}
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
        # FIX C: decode only the generated continuation (slice off the prompt).
        prompt_len = input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        text = self._tok.decode(gen_ids, skip_special_tokens=True).strip()

        # FIX D: log the FIRST 3 calls at INFO with exact prompt-token length
        # and the FULL (untruncated) decoded text so the next short Kaggle run
        # definitively shows echo vs real answer vs slice problem.
        if HFBackend._dbg < 3:
            log.info(
                "HFDBG call#%d prompt_tok_len=%d raw_full=%r",
                HFBackend._dbg,
                prompt_len,
                text,
            )
            HFBackend._dbg += 1

        # FIX C: belt-and-suspenders defensive strip. If an echo prefix slipped
        # through, drop everything before the answer so it cannot swamp
        # parse_response. If a 'Score:' line exists, keep from the first
        # 'Reasoning:' (or the 'Score:' line) onward. parse_response's contract
        # is unchanged -- this only trims a leading echo.
        if re.search(r"score:", text, re.IGNORECASE):
            mr = re.search(r"reasoning:", text, re.IGNORECASE)
            if mr is not None:
                text = text[mr.start():]
            else:
                ms = re.search(r"score:", text, re.IGNORECASE)
                # Keep from the start of the line containing 'Score:'.
                line_start = text.rfind("\n", 0, ms.start()) + 1
                text = text[line_start:]
            text = text.strip()

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


class GeminiBackend:
    """Google Gemini hosted HTTP API backend (CPU-only friendly).

    Calls the Gemini ``generateContent`` v1beta endpoint so the pipeline runs
    on a CPU-only laptop with no local GPU/model. The API key is read ONLY from
    the ``GEMINI_API_KEY`` environment variable -- never from config, never
    hardcoded, never logged.

    Gemini v1beta has no ``seed`` parameter: determinism/caching is handled by
    the SQLite cache key (which includes ``seed``), so N stochastic passes at
    T>0 still cache as distinct entries.
    """

    # Module-wide request throttle (shared across all instances/threads) so we
    # never exceed the configured requests-per-minute.
    _throttle_lock = threading.Lock()
    _last_request_ts = [0.0]  # mutable single-slot timestamp of last dispatch

    # System framing identical in spirit to the HF backend.
    _SYSTEM = (
        "You are a strict product-recommendation rater. Always answer in the "
        "requested 'Reasoning:'/'Score:' format. Never repeat or echo the input."
    )

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        llm = cfg["llm"]
        self._model_id = llm.get("gemini_model", "gemini-1.5-flash")
        self._rmin = float(llm["relevance_min"])
        self._rmax = float(llm["relevance_max"])
        self._max_new = int(llm["max_new_tokens"])
        self._rpm = int(llm.get("gemini_rpm", 15))
        self._max_retries = int(llm.get("gemini_max_retries", 5))
        self._timeout = int(llm.get("gemini_timeout", 60))
        self._api_key = os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. In PowerShell:  "
                "$env:GEMINI_API_KEY='your-key'   or cmd:  set GEMINI_API_KEY=your-key"
                "   then re-run."
            )

    @property
    def model_id(self) -> str:
        return self._model_id

    def _throttle(self) -> None:
        """Enforce <= self._rpm requests/minute across all threads."""
        if self._rpm <= 0:
            return
        min_interval = 60.0 / float(self._rpm)
        with GeminiBackend._throttle_lock:
            now = time.monotonic()
            wait = GeminiBackend._last_request_ts[0] + min_interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            GeminiBackend._last_request_ts[0] = now

    @staticmethod
    def _strip_key(text: str, key: str) -> str:
        """Defensive: never let the API key leak into an exception message."""
        if key:
            text = text.replace(key, "<redacted>")
        return text

    def generate(self, prompt: str, temperature: float, seed: int) -> Dict[str, Any]:
        import requests

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model_id}:generateContent?key={self._api_key}"
        )
        headers = {"Content-Type": "application/json"}
        full_prompt = self._SYSTEM + "\n\n" + prompt
        body = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": self._max_new,
            },
        }

        last_err = ""
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=self._timeout)
            except requests.exceptions.RequestException as exc:  # network/timeout
                last_err = self._strip_key(str(exc), self._api_key)
                if attempt < self._max_retries - 1:
                    time.sleep(min(2.0 * (2 ** attempt), 60.0))
                    continue
                raise RuntimeError(
                    f"Gemini request failed after {self._max_retries} attempts: {last_err[:200]}"
                ) from exc

            if resp.status_code == 200:
                data = resp.json()
                text = ""
                try:
                    # Defensive: candidates may be missing/empty or blocked
                    # (finishReason SAFETY etc.) -> leave text="" so
                    # parse_response falls back to midpoint (parsed_ok False).
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    text = ""
                parsed = parse_response(text, self._rmin, self._rmax)
                parsed["raw"] = text
                return parsed

            # Retry on transient server/rate-limit statuses.
            if resp.status_code in (429, 500, 503) and attempt < self._max_retries - 1:
                last_err = self._strip_key(resp.text, self._api_key)
                time.sleep(min(2.0 * (2 ** attempt), 60.0))
                continue

            # Non-retryable, or retries exhausted -> raise (key never included).
            msg = self._strip_key(resp.text, self._api_key)[:200]
            raise RuntimeError(
                f"Gemini API error (HTTP {resp.status_code}): {msg}"
            )

        raise RuntimeError(
            f"Gemini request failed after {self._max_retries} attempts: {last_err[:200]}"
        )


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
        elif name == "gemini":
            self._backend = GeminiBackend(self._cfg)
        else:
            raise ValueError(f"unknown llm.backend: {name!r} (expected mock/hf/ollama/gemini)")

    @property
    def model_id(self) -> str:
        if self._backend_name == "mock":
            return "mock"
        if self._backend_name == "hf":
            return self._cfg["llm"]["hf_model"]
        if self._backend_name == "ollama":
            return self._cfg["llm"]["ollama_model"]
        if self._backend_name == "gemini":
            return self._cfg["llm"].get("gemini_model", "gemini-1.5-flash")
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
