"""A tiny OpenAI-compatible /v1/chat/completions server for integration tests and dry runs (no GPU, no model).

  python ops/fake_server.py --port 8099 --reply-file reply.txt      # static reply
In tests: `with FakeServer(responder) as url:` where responder(messages, body) -> str.
It is NOT a model: it exists so CLI scripts (Track A eval, run_experiments --llm-endpoint, eye_rft) can be run end to end
and their plumbing, output files and error handling checked before real servers exist.
"""
from __future__ import annotations
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeServer:
    def __init__(self, responder, port: int = 0):
        self.responder, self.requests = responder, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/").endswith("/v1/models"):
                    self._send({"data": [{"id": "fake"}]})
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                try:
                    text = outer.responder(body["messages"], body)
                    self._send({"choices": [{"message": {"content": text}}]})
                except Exception as e:                                      # noqa: BLE001
                    self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

            def _send(self, obj):
                out = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", port), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self.url

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--reply-file", required=True)
    a = ap.parse_args()
    reply = open(a.reply_file).read()
    s = FakeServer(lambda m, b: reply, a.port)
    print("fake server on", s.url, flush=True)
    s.srv.serve_forever()
