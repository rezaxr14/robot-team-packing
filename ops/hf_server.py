"""A real, small OpenAI-compatible chat server (transformers only) for local pipeline runs.

  python ops/hf_server.py --model Qwen/Qwen2.5-1.5B-Instruct --port 8001 --name brain
  python scripts/run_experiments.py --seeds 20 --only full,probe \
      --llm-endpoint http://localhost:8001/v1 --llm-model brain --out reports/llm_dev

Why this exists: `ops/fake_server.py` returns canned text (plumbing checks only) and vLLM needs a datacentre GPU
and its own torch build. This server lets the *real* LLMBrain / ToolBrain / Track-A paths run against a *real*
small checkpoint on a laptop GPU, so prompt formats, JSON parsing, retries and token accounting are exercised
with a model that actually reads the prompt. It is a development tool: a 1.5B base model is not the paper's
Brain, so `reports/llm_dev` is a wiring/scale check, not evidence.

Endpoints: GET /v1/models, POST /v1/chat/completions (the subset `rtp.llm_client.ChatClient` uses), GET /health.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else (
            "float16" if device == "cuda" else "float32")
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype).to(device).eval()
    return tok, model, device, torch_dtype


class Server:
    def __init__(self, model_id: str, name: str, device: str = "auto", dtype: str = "auto",
                 max_new_tokens_cap: int = 2048, verbose: bool = False):
        self.name, self.cap, self.verbose = name, max_new_tokens_cap, verbose
        self.tok, self.model, self.device, self.dtype = load(model_id, device, dtype)
        self.lock = threading.Lock()          # one generation at a time: a 6 GB laptop has no batch headroom
        self.n_calls, self.n_prompt_tokens, self.n_completion_tokens = 0, 0, 0

    def chat(self, body: dict) -> dict:
        import torch
        messages = body.get("messages") or []
        kwargs = dict(body.get("chat_template_kwargs") or {})
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        max_new = min(int(body.get("max_tokens") or 512), self.cap)
        temp = float(body.get("temperature") or 0.0)
        gen_kwargs = dict(max_new_tokens=max_new, do_sample=temp > 0, pad_token_id=self.tok.pad_token_id)
        if temp > 0:
            gen_kwargs.update(temperature=temp, top_p=float(body.get("top_p") or 1.0))
        t0 = time.time()
        with self.lock, torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)
        new = out[0, enc["input_ids"].shape[1]:]
        content = self.tok.decode(new, skip_special_tokens=True)
        pt, ct = enc["input_ids"].shape[1], int(new.shape[0])
        self.n_calls += 1
        self.n_prompt_tokens += pt
        self.n_completion_tokens += ct
        if self.verbose:
            print(f"[hf_server] call {self.n_calls}: {pt}+{ct} tok in {time.time() - t0:.1f}s", flush=True)
        return {"id": f"chatcmpl-{self.n_calls}", "object": "chat.completion", "created": int(time.time()),
                "model": self.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}


def handler_for(srv: Server):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, obj, code: int = 200):
            out = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path.endswith("/v1/models"):
                self._send({"object": "list", "data": [{"id": srv.name, "object": "model"}]})
            elif path.endswith("/health") or path == "":
                self._send({"ok": True, "model": srv.name, "calls": getattr(srv, "n_calls", 0),
                            "prompt_tokens": getattr(srv, "n_prompt_tokens", 0),
                            "completion_tokens": getattr(srv, "n_completion_tokens", 0)})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send({"error": "not found"}, 404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(srv.chat(body))
            except Exception as e:                      # noqa: BLE001  (surface the model error to the client)
                self._send({"error": {"message": f"{type(e).__name__}: {e}"}}, 500)

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--name", default=None, help="served model name (default: the --model value)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    ap.add_argument("--max-new-tokens-cap", type=int, default=2048)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    srv = Server(a.model, a.name or a.model, a.device, a.dtype, a.max_new_tokens_cap, a.verbose)
    httpd = ThreadingHTTPServer((a.host, a.port), handler_for(srv))
    print(f"[hf_server] {a.model} on http://{a.host}:{a.port}/v1 (device={srv.device}, dtype={srv.dtype})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[hf_server] {srv.n_calls} calls, {srv.n_prompt_tokens}+{srv.n_completion_tokens} tokens", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()
