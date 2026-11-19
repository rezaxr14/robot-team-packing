"""Kaggle Model Proxy chat transport with per-turn journaling (stdlib only).

Follows ``KAGGLE_SKILL.md`` (sections 4, 8, 9):

* OpenAI-compatible ``POST {MODEL_PROXY_URL}/openapi/chat/completions`` (inside a hosted run the
  URL already ends in ``/openapi``); multimodal ``image_url`` parts pass through unchanged.
* Failures are classified: ``auth`` / ``quota`` / ``geo`` -> pause (record nothing for the item in
  flight); ``rate_limit`` -> back off, then pause; anything else is evidence.
* Every answered turn is journaled immediately. A resumed episode replays answered turns for free
  and only re-sends the failed one; if the request would differ from the journal it fails closed
  (``JournalMismatch``). We never pay twice for a turn.
* Per-model parameter quirks are *learned from the provider's HTTP 400* and recorded with every
  request (``request_params``); nothing is assumed greedy or repeatable.

``JournaledChat`` is a plain ``chat(messages) -> str`` so it plugs into ``rtp.toolbrain.ToolBrain``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class ProxyError(RuntimeError):
    kind = "other"


class ProxyAuthError(ProxyError):
    kind = "auth"


class ProxyQuotaError(ProxyError):
    kind = "quota"


class ProxyRateLimit(ProxyError):
    kind = "rate_limit"


class ProxyGeoBlocked(ProxyError):
    kind = "geo"


class JournalMismatch(RuntimeError):
    """The request for turn i differs from the journaled one: refuse to resume silently."""


_BY_KIND = {"auth": ProxyAuthError, "quota": ProxyQuotaError, "rate_limit": ProxyRateLimit,
            "geo": ProxyGeoBlocked, "other": ProxyError}
PAUSE_KINDS = ("auth", "quota", "geo")


def classify(status: int, body: str) -> str:
    b = (body or "").lower()
    if status == 403 and "location" in b:
        return "geo"
    if status == 401 or (status == 403 and any(w in b for w in ("expired", "unauthor", "invalid api key", "token"))):
        return "auth"
    if status == 402 or (status in (403, 429) and "quota" in b):
        return "quota"
    if status == 429:
        return "rate_limit"
    return "other"


def endpoint(url: str | None = None) -> str:
    base = (url or os.environ["MODEL_PROXY_URL"]).rstrip("/")
    if not base.endswith("/openapi"):
        base += "/openapi"
    return base + "/chat/completions"


def usage_cost_usd(usage) -> float | None:
    """True spend from ``usage.cost.*_nanodollars`` (hidden reasoning is included); None if absent."""
    if not isinstance(usage, dict) or not isinstance(usage.get("cost"), dict):
        return None
    c = usage["cost"]
    parts = [c.get("input_tokens_cost_nanodollars"), c.get("output_tokens_cost_nanodollars")]
    if any(not isinstance(p, (int, float)) for p in parts):
        return None
    return sum(parts) / 1e9


def _opener(proxy: str | None):
    """Local runs route through ``RTP_HTTP_PROXY`` (e.g. http://localhost:10808); hosted workers set none."""
    proxy = proxy or os.environ.get("RTP_HTTP_PROXY")
    return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy})) if proxy else urllib.request.build_opener()


def _urllib_post(url: str, headers: dict, payload: dict, timeout: float = 300.0, proxy: str | None = None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={**headers, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with _opener(proxy).open(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), round(time.perf_counter() - t0, 3)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), round(time.perf_counter() - t0, 3)


class ProxyTransport:
    """One model, one decoding setup. ``post`` is injectable for tests."""

    def __init__(self, model: str, *, url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 1200, temperature: float | None = 0.0,
                 reasoning_effort: str | None = "none", post=None, proxy: str | None = None):
        self.model = model
        self.url = url
        self.api_key = api_key
        self.params = {"max_tokens": max_tokens, "temperature": temperature, "reasoning_effort": reasoning_effort}
        if "claude-sonnet-5" in model:                      # measured: rejects `temperature` (skill section 4)
            self.params["temperature"] = None
        self.proxy = proxy
        self._post = post or (lambda u, h, p: _urllib_post(u, h, p, proxy=self.proxy))

    def _payload(self, messages: list) -> dict:
        p = {"model": self.model, "messages": messages}
        for k, v in self.params.items():
            if v is not None:
                p[k] = v
        return p

    def __call__(self, messages: list) -> dict:
        url = self.url or endpoint()
        headers = {"Authorization": "Bearer " + (self.api_key or os.environ.get("MODEL_PROXY_API_KEY", ""))}
        for _ in range(4):                                  # at most three learned parameter changes
            status, text, latency = self._post(url, headers, self._payload(messages))
            if status == 200:
                break
            low = text.lower()
            changed = False
            if status == 400:
                if "temperature" in low and self.params.get("temperature") is not None:
                    self.params["temperature"] = None
                    changed = True
                if "reasoning_effort" in low and self.params.get("reasoning_effort") is not None:
                    self.params["reasoning_effort"] = None
                    changed = True
                if "max_completion_tokens" in low and "max_tokens" in self.params:
                    self.params["max_completion_tokens"] = self.params.pop("max_tokens")
                    changed = True
            if not changed:
                raise _BY_KIND[classify(status, text)](f"HTTP {status}: {text[:400]}")
        else:
            raise ProxyError("too many parameter retries: " + text[:300])
        data = json.loads(text)
        choice = data["choices"][0]
        return {"completion": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
                "response_model": data.get("model"), "latency_s": latency,
                "request_params": {"model": self.model, **{k: v for k, v in self.params.items()}}}


def request_hash(messages: list) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


class JournaledChat:
    """``chat(messages) -> str`` that journals each answered turn and replays it on resume."""

    def __init__(self, transport, journal_path=None):
        self.transport = transport
        self.path = Path(journal_path) if journal_path else None
        self.turns: list = []
        self._replay: list = []
        if self.path and self.path.exists():
            self._replay = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def __call__(self, messages: list, **_kw) -> str:
        i, h = len(self.turns), request_hash(messages)
        if i < len(self._replay):
            rec = self._replay[i]
            if rec["request_hash"] != h:
                raise JournalMismatch(f"turn {i}: request differs from the journal ({rec['request_hash']} vs {h})")
            self.turns.append({**rec, "replayed": True})
            return rec["completion"]
        t = self.transport(messages)                         # may raise: nothing is recorded for a failed turn
        rec = {"i": i, "request_hash": h, "n_messages": len(messages),
               "delta": messages if i == 0 else messages[-1], **t, "replayed": False}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.turns.append(rec)
        return t["completion"]

    def totals(self) -> dict:
        live = [t for t in self.turns if not t.get("replayed")]
        usage = [t.get("usage") for t in live]
        costs = [usage_cost_usd(u) for u in usage]

        def tok(k):
            return sum(u[k] for u in usage if isinstance(u, dict) and isinstance(u.get(k), (int, float)))
        return {"turns": len(self.turns), "replayed": len(self.turns) - len(live),
                "prompt_tokens": tok("prompt_tokens"), "completion_tokens": tok("completion_tokens"),
                "cost_usd": (sum(costs) if live and all(c is not None for c in costs) else None),
                "latency_s": round(sum(t.get("latency_s") or 0 for t in live), 3),
                "finish_reasons": [t.get("finish_reason") for t in self.turns]}
