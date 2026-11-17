"""The local transformers-backed OpenAI server: endpoint shape only, no model weights.

`ops/hf_server.Server` wraps a real checkpoint; here a stub with the same `chat()` contract is served so the HTTP
surface used by `rtp.llm_client.ChatClient` (and therefore every --llm-endpoint script) is tested without a GPU.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from ops.hf_server import handler_for


class _StubServer:
    name = "stub-model"
    n_calls = 0

    def __init__(self):
        self.seen = []

    def chat(self, body):
        self.seen.append(body)
        self.n_calls += 1
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": '{"tasks": []}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}


def _serve(srv):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(srv))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_models_health_and_chat():
    stub = _StubServer()
    httpd, url = _serve(stub)
    try:
        assert _get(f"{url}/models")[1]["data"][0]["id"] == "stub-model"
        assert _get(url.replace("/v1", "/health"))[1]["ok"] is True

        body = {"model": "stub-model", "messages": [{"role": "user", "content": "pack the bag"}], "max_tokens": 32}
        req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=5) as r:
            out = json.loads(r.read())
        assert out["choices"][0]["message"]["content"] == '{"tasks": []}'
        assert out["usage"]["total_tokens"] == 10 and stub.seen == [body]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_unknown_route_is_404_and_bad_body_is_500():
    stub = _StubServer()
    httpd, url = _serve(stub)
    try:
        try:
            _get(f"{url}/nope")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        else:
            raise AssertionError("expected 404")

        class Boom(_StubServer):
            def chat(self, body):
                raise ValueError("no such model")

        httpd2, url2 = _serve(Boom())
        try:
            req = urllib.request.Request(f"{url2}/chat/completions", data=b"{}", headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=5)
            except urllib.error.HTTPError as e:
                assert e.code == 500 and "no such model" in json.loads(e.read())["error"]["message"]
            else:
                raise AssertionError("expected 500")
        finally:
            httpd2.shutdown()
            httpd2.server_close()
    finally:
        httpd.shutdown()
        httpd.server_close()
