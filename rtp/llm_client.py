"""Minimal OpenAI-compatible chat client (stdlib only) for vLLM / any /v1/chat/completions server."""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.request


class ChatError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "default",
                 api_key: str = "EMPTY", timeout: float = 300.0, retries: int = 4, chat_template_kwargs: dict | None = None):
        import os
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        # RTP_NO_THINK=1 asks Qwen-style templates not to emit <think> reasoning (ignored by templates without the switch)
        self.chat_template_kwargs = chat_template_kwargs or ({"enable_thinking": False} if os.environ.get("RTP_NO_THINK") == "1" else None)
        self.timeout, self.retries = timeout, retries

    def chat(self, messages: list, temperature: float = 0.0, max_tokens: int = 2000, **extra) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra}
        if self.chat_template_kwargs and "chat_template_kwargs" not in payload:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError) as e:
                last = e
                time.sleep(min(2 ** attempt, 8))
        raise ChatError(f"chat failed after {self.retries + 1} attempts: {last}")

    def generate(self, system: str | None = None, **kw):
        """Return prompt:str -> completion:str for rtp.brain.LLMBrain."""
        def gen(prompt: str) -> str:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            return self.chat(msgs, **kw)
        return gen
