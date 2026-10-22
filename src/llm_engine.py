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
from collections import deque
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

    # Log the FIRST 429 body (key-stripped, truncated) exactly once per process
    # so the user can tell a per-MINUTE limit (clears in ~60s) from a per-DAY
    # quota (does not clear until Pacific midnight) from the raw metric string.
    _logged_429 = False

    # 429-specific backoff: free-tier per-minute limits clear within ~60s, so
    # allow more attempts and a higher cap than the 500/503 transient path.
    _MAX_429_ATTEMPTS = 8
    _BACKOFF_429_BASE = 5.0
    _BACKOFF_429_CAP = 120.0

    # System framing identical in spirit to the HF backend.
    _SYSTEM = (
        "You are a strict product-recommendation rater. Always answer in the "
        "requested 'Reasoning:'/'Score:' format. Never repeat or echo the input."
    )

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        llm = cfg["llm"]
        # Normalize the model id: strip a leading "models/" prefix if present so
        # both "gemini-2.0-flash" and "models/gemini-2.0-flash" work (ListModels
        # returns names with the prefix, the endpoint path adds it back).
        mid = str(llm.get("gemini_model", "gemini-2.0-flash"))
        self._model_id = mid[len("models/"):] if mid.startswith("models/") else mid
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

    @staticmethod
    def _parse_retry_delay(resp) -> Optional[float]:
        """Extract a server-suggested retry delay (seconds) from a 429 response.

        Google's 429 payload typically carries a ``RetryInfo`` entry inside
        ``error.details[]`` with a ``retryDelay`` like ``"27s"``. Some responses
        instead (or also) set a ``Retry-After`` header (seconds or HTTP-date).
        Returns the delay in seconds, or ``None`` when neither is present/parseable.
        """
        # 1) Prefer the structured RetryInfo in the JSON body.
        try:
            data = resp.json()
            details = (data.get("error", {}) or {}).get("details", []) or []
            for det in details:
                if not isinstance(det, Mapping):
                    continue
                if "RetryInfo" in str(det.get("@type", "")):
                    raw = str(det.get("retryDelay", "")).strip()
                    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)s?$", raw)
                    if m:
                        return float(m.group(1))
        except (ValueError, AttributeError, TypeError):
            pass  # body not JSON / unexpected shape -> fall through to header

        # 2) Fall back to the Retry-After header (integer seconds only).
        try:
            hdr = resp.headers.get("Retry-After")
            if hdr is not None:
                m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$", str(hdr))
                if m:
                    return float(m.group(1))
        except (AttributeError, TypeError):
            pass
        return None

    @classmethod
    def _log_first_429(cls, stripped_body: str) -> None:
        """One-time INFO log of the first 429 body so the quota metric is visible."""
        if cls._logged_429:
            return
        cls._logged_429 = True
        log.info(
            "First Gemini HTTP 429 body (key-stripped, first 300 chars) -- look for "
            "'PerMinute'/'per minute' (clears within ~60s) vs 'PerDay'/'per day' "
            "(daily free-tier quota, clears at Pacific midnight): %s",
            stripped_body[:300],
        )

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

        # 429 (rate-limit / quota) gets its own, more generous attempt budget and
        # retry-delay-aware sleeps because a per-MINUTE free-tier limit clears in
        # ~60s; transient 500/503 keep the shorter exponential backoff. The total
        # budget for the loop is the larger of the two so neither path is starved.
        max_429 = max(self._max_retries, GeminiBackend._MAX_429_ATTEMPTS)
        max_attempts = max(self._max_retries, max_429)

        last_err = ""
        attempts_429 = 0
        for attempt in range(max_attempts):
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

            # --- Rate-limit / quota (429): honour server retry delay. --------- #
            if resp.status_code == 429:
                stripped = self._strip_key(resp.text, self._api_key)
                last_err = stripped
                self._log_first_429(stripped)
                attempts_429 += 1
                if attempts_429 < max_429:
                    delay = self._parse_retry_delay(resp)
                    if delay is not None:
                        # Server told us when to retry -- honour it (capped).
                        sleep_s = min(max(delay, 0.0), GeminiBackend._BACKOFF_429_CAP)
                    else:
                        # No hint -> exponential backoff from a higher base/cap.
                        sleep_s = min(
                            GeminiBackend._BACKOFF_429_BASE * (2 ** (attempts_429 - 1)),
                            GeminiBackend._BACKOFF_429_CAP,
                        )
                    log.warning(
                        "Gemini HTTP 429 (attempt %d/%d); sleeping %.1fs before retry.",
                        attempts_429, max_429, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                # 429 retries exhausted -> explicit, cause-revealing error.
                raise RuntimeError(
                    f"Gemini quota exceeded (HTTP 429) after {attempts_429} retries. "
                    "If this is a per-DAY free-tier limit it will not clear until "
                    "tomorrow (Pacific midnight); switch to a model with higher free "
                    "quota (gemini-2.5-flash-lite), reduce evaluation.gemini_max_eval_users, "
                    f"or enable billing. Raw: {stripped[:300]}"
                )

            # --- Transient server errors (500/503): short exponential backoff. - #
            if resp.status_code in (500, 503) and attempt < self._max_retries - 1:
                last_err = self._strip_key(resp.text, self._api_key)
                time.sleep(min(2.0 * (2 ** attempt), 60.0))
                continue

            # Non-retryable, or retries exhausted -> raise (key never included).
            msg = self._strip_key(resp.text, self._api_key)[:200]
            raise RuntimeError(
                f"Gemini API error (HTTP {resp.status_code}): {msg}"
            )

        raise RuntimeError(
            f"Gemini request failed after {max_attempts} attempts: {last_err[:200]}"
        )


class GroqBackend:
    """Groq hosted OpenAI-compatible chat-completions backend (CPU-only friendly).

    Calls the Groq ``/openai/v1/chat/completions`` endpoint so the pipeline runs
    on a CPU-only laptop with no local GPU/model, using Groq's fast + generous
    free-tier hosted inference. The API key is read ONLY from the
    ``GROQ_API_KEY`` environment variable -- never from config, never hardcoded,
    never logged, never included in an exception message.

    Groq's OpenAI-compatible API accepts an optional ``seed`` field, which is
    included below for reproducibility; the SQLite cache key also includes
    ``seed`` so N stochastic passes at T>0 cache as distinct entries.
    """

    _ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    # Module-wide request throttle (SEPARATE from Gemini's) so Groq's own
    # requests-per-minute limit is enforced across all instances/threads.
    _throttle_lock = threading.Lock()
    _last_request_ts = [0.0]  # mutable single-slot timestamp of last dispatch

    # Module-wide rolling-window TOKEN budget. Groq's free tier caps
    # tokens-per-minute (TPM), not just requests-per-minute, so a pure RPM
    # throttle vastly overshoots an 8000-TPM budget and nearly every call 429s.
    # We track (monotonic_ts, estimated_tokens) of recent dispatches in a
    # trailing-60s window and, before each request, sleep until adding this
    # call's estimated cost keeps the window total <= self._tpm. Guarded by the
    # SAME _throttle_lock so RPM and TPM limiters are consistent across threads.
    _TOKEN_WINDOW_S = 60.0
    _token_events: "deque[tuple[float, int]]" = deque()  # (ts, est_tokens)

    # Log the FIRST 429 body (key-stripped, truncated) exactly once per process
    # so the user can see whether a per-MINUTE vs per-DAY limit was hit.
    _logged_429 = False

    # 429-specific backoff (same shape as GeminiBackend): per-minute limits
    # clear within ~60s, so allow more attempts and a higher cap than the
    # transient 500/502/503 path.
    _MAX_429_ATTEMPTS = 8
    _BACKOFF_429_BASE = 5.0
    _BACKOFF_429_CAP = 120.0

    # System framing identical in spirit to the HF/Gemini backends.
    _SYSTEM = (
        "You are a strict product-recommendation rater. Always answer in the "
        "requested 'Reasoning:'/'Score:' format. Never repeat or echo the input."
    )

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        llm = cfg["llm"]
        self._model_id = str(llm.get("groq_model", "llama-3.3-70b-versatile"))
        self._rmin = float(llm["relevance_min"])
        self._rmax = float(llm["relevance_max"])
        self._max_new = int(llm["max_new_tokens"])
        self._rpm = int(llm.get("groq_rpm", 30))
        self._tpm = int(llm.get("groq_tpm", 8000))
        self._max_retries = int(llm.get("groq_max_retries", 8))
        self._timeout = int(llm.get("groq_timeout", 60))
        self._api_key = os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set. In PowerShell:  "
                "$env:GROQ_API_KEY='gsk_...'   or cmd:  set GROQ_API_KEY=gsk_..."
                "   then re-run."
            )

    @property
    def model_id(self) -> str:
        return self._model_id

    @classmethod
    def _prune_token_window(cls, now: float) -> int:
        """Drop token events older than the trailing 60s; return tokens still in-window.

        MUST be called while holding ``_throttle_lock``. Uses ``cls`` state so a
        subclass (e.g. CerebrasBackend) prunes its OWN token window.
        """
        cutoff = now - cls._TOKEN_WINDOW_S
        events = cls._token_events
        while events and events[0][0] <= cutoff:
            events.popleft()
        return sum(tok for _, tok in events)

    def _estimate_tokens(self, prompt: str) -> int:
        """Cheap per-call token estimate: prompt tokens (~4 chars/token) + output reserve.

        Pre-request estimation (not the real ``resp`` usage) is what prevents
        429s, so we deliberately over-count by reserving the full ``max_new``
        output budget on top of the prompt estimate.
        """
        prompt_tokens = max(1, len(prompt) // 4)  # ~4 chars/token
        return prompt_tokens + int(self._max_new)

    def _throttle_tokens(self, est_tokens: int) -> None:
        """Rolling-window TOKEN limiter: stay <= self._tpm within any trailing 60s.

        Before dispatch, if adding this call's estimated cost would exceed the
        budget, sleep just long enough for the oldest in-window events to age
        out past 60s, then record the estimated tokens at dispatch time. No-op
        when ``self._tpm <= 0`` (disabled).
        """
        if self._tpm <= 0:
            return
        with type(self)._throttle_lock:
            now = time.monotonic()
            in_window = self._prune_token_window(now)
            # Sleep until enough of the window has elapsed to fit est_tokens.
            # A single call bigger than the whole budget still proceeds (we
            # drain the window first) rather than deadlock.
            while type(self)._token_events and in_window + est_tokens > self._tpm:
                oldest_ts = type(self)._token_events[0][0]
                wait = oldest_ts + GroqBackend._TOKEN_WINDOW_S - now
                if wait <= 0:
                    now = time.monotonic()
                    in_window = self._prune_token_window(now)
                    continue
                log.info(
                    "Groq TPM throttle: in-window=%d + est=%d > budget=%d; "
                    "sleeping %.2fs for the 60s window to drain.",
                    in_window, est_tokens, self._tpm, wait,
                )
                time.sleep(wait)
                now = time.monotonic()
                in_window = self._prune_token_window(now)
            type(self)._token_events.append((now, int(est_tokens)))

    def _throttle(self, est_tokens: Optional[int] = None) -> None:
        """Enforce BOTH the per-minute REQUEST and per-minute TOKEN budgets.

        Whichever limiter is stricter wins (each may sleep). ``est_tokens``, when
        provided, drives the rolling-window token budget; pass it on every real
        dispatch so steady-state stays under Groq's TPM free-tier limit.
        """
        # --- Request rate limiter (RPM) --------------------------------------- #
        if self._rpm > 0:
            min_interval = 60.0 / float(self._rpm)
            with type(self)._throttle_lock:
                now = time.monotonic()
                wait = type(self)._last_request_ts[0] + min_interval - now
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
                type(self)._last_request_ts[0] = now
        # --- Token budget limiter (TPM) --------------------------------------- #
        if est_tokens is not None:
            self._throttle_tokens(est_tokens)

    @staticmethod
    def _strip_key(text: str, key: str) -> str:
        """Defensive: never let the API key leak into an exception message."""
        if key:
            text = text.replace(key, "<redacted>")
        return text

    @classmethod
    def _log_first_429(cls, stripped_body: str) -> None:
        """One-time INFO log of the first 429 body so the quota metric is visible."""
        if cls._logged_429:
            return
        cls._logged_429 = True
        log.info(
            "First Groq HTTP 429 body (key-stripped, first 300 chars) -- look for "
            "'per minute'/'RPM' (clears within ~60s) vs 'per day'/'RPD' (daily "
            "free-tier quota): %s",
            stripped_body[:300],
        )

    # "...try again in 937.5ms" / "...try again in 1.2s" appears in the Groq 429
    # error MESSAGE body (not always the Retry-After header). This precise,
    # often sub-second hint beats the generic exponential backoff.
    _RETRY_IN_RE = re.compile(
        r"try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s)\b", re.IGNORECASE
    )

    @classmethod
    def _parse_retry_after(cls, resp) -> Optional[float]:
        """Extract a server-suggested retry delay (seconds) from a 429 response.

        Prefers the precise ``"try again in Xms"/"try again in Xs"`` hint in the
        response BODY (Groq puts the sub-second delay there), then falls back to
        the ``Retry-After`` header (seconds, possibly fractional). Returns the
        delay in seconds, or ``None`` when absent/unparseable.
        """
        # 1) Precise body hint: "try again in 937.5ms" / "try again in 1.2s".
        try:
            body = resp.text
            if body:
                m = cls._RETRY_IN_RE.search(str(body))
                if m:
                    value = float(m.group(1))
                    unit = m.group(2).lower()
                    return value / 1000.0 if unit == "ms" else value
        except (AttributeError, TypeError, ValueError):
            pass
        # 2) Fallback: numeric Retry-After header (seconds).
        try:
            hdr = resp.headers.get("Retry-After")
            if hdr is not None:
                m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$", str(hdr))
                if m:
                    return float(m.group(1))
        except (AttributeError, TypeError):
            pass
        return None

    def generate(self, prompt: str, temperature: float, seed: int) -> Dict[str, Any]:
        import requests

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": prompt},
        ]
        body = {
            "model": self._model_id,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": self._max_new,
            # Groq supports an optional seed for reproducibility; the SQLite
            # cache key already includes seed too.
            "seed": int(seed),
        }

        # 429 gets its own, more generous attempt budget and Retry-After-aware
        # sleeps (per-minute free-tier limits clear in ~60s); transient
        # 500/502/503 keep the shorter exponential backoff. The total budget is
        # the larger of the two so neither path is starved.
        max_429 = max(self._max_retries, GroqBackend._MAX_429_ATTEMPTS)
        max_attempts = max(self._max_retries, max_429)

        # Pre-request token-cost estimate (prompt ~4 chars/token + output
        # reserve). Drives the rolling-window TPM throttle so we stay under
        # Groq's free-tier tokens/min budget and almost never 429.
        est_tokens = self._estimate_tokens(prompt)

        last_err = ""
        attempts_429 = 0
        for attempt in range(max_attempts):
            self._throttle(est_tokens)
            try:
                resp = requests.post(
                    self._ENDPOINT, json=body, headers=headers, timeout=self._timeout
                )
            except requests.exceptions.RequestException as exc:  # network/timeout
                last_err = self._strip_key(str(exc), self._api_key)
                if attempt < self._max_retries - 1:
                    time.sleep(min(2.0 * (2 ** attempt), 60.0))
                    continue
                raise RuntimeError(
                    f"Groq request failed after {self._max_retries} attempts: {last_err[:200]}"
                ) from exc

            if resp.status_code == 200:
                data = resp.json()
                text = ""
                try:
                    # Defensive: shape may be unexpected/empty -> leave text=""
                    # so parse_response falls back to midpoint (parsed_ok False).
                    text = data["choices"][0]["message"]["content"]
                    if text is None:
                        text = ""
                except (KeyError, IndexError, TypeError):
                    text = ""
                parsed = parse_response(text, self._rmin, self._rmax)
                parsed["raw"] = text
                return parsed

            # --- Rate-limit / quota (429): honour Retry-After header. --------- #
            if resp.status_code == 429:
                stripped = self._strip_key(resp.text, self._api_key)
                last_err = stripped
                self._log_first_429(stripped)
                attempts_429 += 1
                if attempts_429 < max_429:
                    delay = self._parse_retry_after(resp)
                    if delay is not None:
                        # Honour the server's precise hint + a small cushion so
                        # we clear the window edge rather than racing it.
                        sleep_s = min(
                            max(delay, 0.0) + 0.25, GroqBackend._BACKOFF_429_CAP
                        )
                    else:
                        sleep_s = min(
                            GroqBackend._BACKOFF_429_BASE * (2 ** (attempts_429 - 1)),
                            GroqBackend._BACKOFF_429_CAP,
                        )
                    log.warning(
                        "Groq HTTP 429 (attempt %d/%d); sleeping %.2fs before retry.",
                        attempts_429, max_429, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                # 429 retries exhausted -> explicit, cause-revealing error.
                raise RuntimeError(
                    f"Groq rate limit / quota exceeded (HTTP 429) after {attempts_429} "
                    "retries. A per-MINUTE limit clears within ~60s (just wait and re-run, "
                    "the SQLite cache resumes); a per-DAY free-tier quota needs a longer "
                    "wait. Consider switching to a smaller/faster model (groq_model: "
                    "llama-3.1-8b-instant) or lowering llm.groq_rpm. Raw: "
                    f"{stripped[:300]}"
                )

            # --- Transient server errors (500/502/503): short backoff. -------- #
            if resp.status_code in (500, 502, 503) and attempt < self._max_retries - 1:
                last_err = self._strip_key(resp.text, self._api_key)
                time.sleep(min(2.0 * (2 ** attempt), 60.0))
                continue

            # Non-retryable, or retries exhausted -> raise (key never included).
            msg = self._strip_key(resp.text, self._api_key)[:200]
            raise RuntimeError(f"Groq API error (HTTP {resp.status_code}): {msg}")

        raise RuntimeError(
            f"Groq request failed after {max_attempts} attempts: {last_err[:200]}"
        )


class CerebrasBackend(GroqBackend):
    """Cerebras hosted OpenAI-compatible chat-completions backend.

    Cerebras' inference API is OpenAI-compatible, so this reuses GroqBackend's
    entire machinery (rolling-window RPM+TPM throttle, 429 Retry-After handling,
    response parsing) and overrides only: the endpoint, the API key env var, the
    model/throttle config keys, and -- importantly -- its OWN module-wide throttle
    state (lock + request timestamp + token window) so Cerebras pacing is tracked
    SEPARATELY from Groq's. Free tier: ~1M tokens/day, ~5 RPM, very fast inference.
    The API key is read ONLY from $CEREBRAS_API_KEY -- never config/chat/git.
    """

    _ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"

    # SEPARATE module-wide throttle state (do NOT share Groq's class attrs).
    _throttle_lock = threading.Lock()
    _last_request_ts = [0.0]
    _token_events: "deque[tuple[float, int]]" = deque()
    _logged_429 = False

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        llm = cfg["llm"]
        self._model_id = str(llm.get("cerebras_model", "llama-3.3-70b"))
        self._rmin = float(llm["relevance_min"])
        self._rmax = float(llm["relevance_max"])
        self._max_new = int(llm["max_new_tokens"])
        # Free tier is ~5 RPM and ~1M tokens/day. We pace on a per-MINUTE token
        # window (same mechanism as Groq); default the per-minute token budget
        # conservatively so steady-state stays well under the daily cap.
        self._rpm = int(llm.get("cerebras_rpm", 5))
        self._tpm = int(llm.get("cerebras_tpm", 60000))
        self._max_retries = int(llm.get("cerebras_max_retries", 8))
        self._timeout = int(llm.get("cerebras_timeout", 60))
        self._api_key = os.environ.get("CEREBRAS_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY environment variable not set. In PowerShell:  "
                "$env:CEREBRAS_API_KEY='csk-...'   or cmd:  set CEREBRAS_API_KEY=csk-..."
                "   then re-run."
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
        elif name == "groq":
            self._backend = GroqBackend(self._cfg)
        elif name == "cerebras":
            self._backend = CerebrasBackend(self._cfg)
        else:
            raise ValueError(
                f"unknown llm.backend: {name!r} "
                "(expected mock/hf/ollama/gemini/groq/cerebras)"
            )

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
        if self._backend_name == "groq":
            return self._cfg["llm"].get("groq_model", "llama-3.3-70b-versatile")
        if self._backend_name == "cerebras":
            return self._cfg["llm"].get("cerebras_model", "llama-3.3-70b")
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
