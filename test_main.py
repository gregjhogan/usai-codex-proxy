import contextlib
import json
import os
import struct
import subprocess
import tempfile
import threading
import unittest
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import main
from catalog import build_catalog
from main import build_response_from_output, responses_to_chat

TEST_MODEL = "gpt-6.1-sol-latest-guardrails-defaultv2"


class ServerFixture:
    def __init__(self, handler_cls):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class FakeUsaiState:
    requests = []
    stream = False
    response = None
    stream_chunks = None

    @classmethod
    def reset(cls):
        cls.requests = []
        cls.stream = False
        cls.response = {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        cls.stream_chunks = None


FakeUsaiState.reset()


class FakeUsaiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/models":
            self._json(200, {"object": "list", "data": [{"id": TEST_MODEL}]})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeUsaiState.requests.append({"headers": dict(self.headers), "body": body})
        if body.get("stream") or FakeUsaiState.stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunks = FakeUsaiState.stream_chunks or [{
                "id": "chatcmpl_test",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": None}],
            }]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._json(200, FakeUsaiState.response)

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextlib.contextmanager
def run_proxy(upstream_url, debug=False):
    old_values = (main.UPSTREAM, main.API_KEY, main.DEBUG)
    main.UPSTREAM = upstream_url
    main.API_KEY = "test-key"
    main.DEBUG = debug
    server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        main.UPSTREAM, main.API_KEY, main.DEBUG = old_values


def codex_config_args(base_url):
    return [
        "--ignore-user-config",
        "-c", "model_provider=\"usai-test\"",
        "-c", f"model=\"{TEST_MODEL}\"",
        "-c", f"model_providers.usai-test.name=\"USAi Test\"",
        "-c", f"model_providers.usai-test.base_url=\"{base_url}\"",
        "-c", "model_providers.usai-test.env_key=\"USAI_API_KEY\"",
        "-c", "model_providers.usai-test.wire_api=\"responses\"",
        "-c", "include_apps_instructions=false",
        "-c", "features.apps=false",
        "-c", "features.enable_mcp_apps=false",
    ]


def run_codex_exec(base_url, prompt, extra_args=None, timeout=60):
    env = os.environ.copy()
    env["USAI_API_KEY"] = "test-key"
    args = [
        "codex", "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "-s", "read-only",
        *codex_config_args(base_url),
    ]
    if extra_args:
        args.extend(extra_args)
    args.extend(["--", prompt])
    return subprocess.run(
        args,
        cwd=tempfile.gettempdir(),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def png_1x1_rgb():
    raw_scanline = b"\x00\xff\x00\x00"

    def chunk(kind, data):
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xffffffff)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_scanline))
        + chunk(b"IEND", b"")
    )


class CodexProxyIntegrationTests(unittest.TestCase):
    def setUp(self):
        FakeUsaiState.reset()

    def test_installed_codex_exec_posts_responses_request_through_proxy(self):
        with ServerFixture(FakeUsaiHandler) as upstream, run_proxy(upstream.url) as base_url:
            result = run_codex_exec(base_url, "Reply exactly: done")

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(len(FakeUsaiState.requests), 1)
        sent = FakeUsaiState.requests[0]["body"]
        self.assertEqual(sent["model"], TEST_MODEL)
        self.assertEqual(sent["messages"][-1]["role"], "user")
        self.assertIn("Reply exactly: done", json.dumps(sent["messages"][-1]["content"]))
        self.assertEqual(FakeUsaiState.requests[0]["headers"].get("Authorization"), "Bearer test-key")

    def test_installed_codex_output_schema_reaches_usai_as_response_format(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        FakeUsaiState.response["choices"][0]["message"]["content"] = json.dumps({"answer": "ok"})
        with tempfile.TemporaryDirectory() as td:
            schema_path = Path(td) / "schema.json"
            schema_path.write_text(json.dumps(schema))
            with ServerFixture(FakeUsaiHandler) as upstream, run_proxy(upstream.url) as base_url:
                result = run_codex_exec(
                    base_url,
                    "Return JSON with answer set to ok.",
                    extra_args=["--output-schema", str(schema_path)],
                )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        sent = FakeUsaiState.requests[0]["body"]
        self.assertEqual(sent["response_format"]["type"], "json_schema")
        self.assertEqual(sent["response_format"]["json_schema"]["schema"], schema)

    def test_installed_codex_image_prompt_reaches_usai_as_image_url_content(self):
        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / "pixel.png"
            image_path.write_bytes(png_1x1_rgb())
            with ServerFixture(FakeUsaiHandler) as upstream, run_proxy(upstream.url) as base_url:
                result = run_codex_exec(
                    base_url,
                    "Describe this image in one word.",
                    extra_args=["--image", str(image_path)],
                )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        content = FakeUsaiState.requests[0]["body"]["messages"][-1]["content"]
        if isinstance(content, str) and "image content omitted because it could not be processed" in content:
            self.skipTest("installed Codex accepted --image but could not process the fixture image")
        self.assertIsInstance(content, list)
        self.assertTrue(any(part.get("type") == "image_url" for part in content))

    def test_models_request_with_codex_query_string_returns_model_list(self):
        with ServerFixture(FakeUsaiHandler) as upstream, run_proxy(upstream.url) as base_url:
            with urllib.request.urlopen(base_url + "/models?client_version=test", timeout=10) as resp:
                payload = json.loads(resp.read())

        self.assertEqual(len(payload["models"]), 1)
        self.assertEqual(payload["models"][0]["slug"], TEST_MODEL)
        self.assertIn("display_name", payload["models"][0])
        self.assertIn("context_window", payload["models"][0])


class ProxyTranslationTests(unittest.TestCase):
    # These cover protocol cases that the installed Codex CLI does not expose a
    # deterministic switch for, or that depend on upstream stream chunk timing.
    def test_responses_input_covers_tools_format_controls_and_image_detail(self):
        chat = responses_to_chat({
            "model": "gpt-test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "inspect"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "high"},
                ],
            }],
            "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "read"},
            "parallel_tool_calls": False,
            "max_output_tokens": 17,
            "text": {"format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            }},
        })

        self.assertEqual(chat["tool_choice"], {"type": "function", "function": {"name": "read"}})
        self.assertEqual(chat["parallel_tool_calls"], False)
        self.assertEqual(chat["max_tokens"], 17)
        self.assertEqual(chat["response_format"]["type"], "json_schema")
        self.assertEqual(chat["messages"][0]["content"][1]["image_url"]["detail"], "high")

    def test_build_response_from_output_reuses_streamed_item_ids(self):
        output = [{
            "type": "function_call",
            "id": "fc_streamed",
            "call_id": "call_streamed",
            "name": "read",
            "arguments": "{}",
            "status": "completed",
        }, {
            "type": "message",
            "id": "msg_streamed",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }]

        response = build_response_from_output("resp_1", "gpt-test", output, None)

        self.assertEqual([item["id"] for item in response["output"]], ["fc_streamed", "msg_streamed"])

    def test_streaming_tool_arguments_buffer_until_function_name(self):
        FakeUsaiState.reset()
        FakeUsaiState.stream = True
        FakeUsaiState.stream_chunks = [
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"arguments": "{\"path\""},
            }]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"name": "read", "arguments": ":\"README.md\"}"},
            }]}}]},
        ]
        with ServerFixture(FakeUsaiHandler) as upstream, run_proxy(upstream.url) as base_url:
            req = urllib.request.Request(
                base_url + "/responses",
                data=json.dumps({"model": "gpt-test", "input": "hi", "stream": True}).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                lines = []
                while True:
                    line = resp.readline().decode()
                    if not line:
                        break
                    lines.append(line)
                    if line.strip() == "data: [DONE]":
                        break
                raw = "".join(lines)

        events = []
        for block in raw.split("\n\n"):
            lines = block.splitlines()
            data_lines = [line[5:].strip() for line in lines if line.startswith("data:")]
            if data_lines and data_lines[0] != "[DONE]":
                events.append(json.loads(data_lines[0]))
        deltas = [e["delta"] for e in events if e.get("type") == "response.function_call_arguments.delta"]
        completed = [e for e in events if e.get("type") == "response.completed"][0]

        self.assertEqual(deltas, ['{"path":"README.md"}'])
        self.assertTrue(completed["response"]["output"][0]["id"].startswith("fc_"))
        self.assertEqual(completed["response"]["output"][0]["call_id"], "call_1")
        self.assertEqual(completed["response"]["output"][0]["arguments"], '{"path":"README.md"}')

    def test_streaming_response_declares_connection_close(self):
        # Under HTTP/1.1 the SSE stream has no Content-Length and no chunked
        # framing, so the response MUST send `Connection: close` for the client
        # to detect end-of-body. Without it, clients hang until timeout and
        # Codex reports the turn (e.g. /review) as interrupted.
        FakeUsaiState.reset()
        FakeUsaiState.stream = True
        FakeUsaiState.stream_chunks = [
            {"choices": [{"delta": {"content": "hello"}}]},
        ]
        with ServerFixture(FakeUsaiHandler) as upstream, run_proxy(upstream.url) as base_url:
            req = urllib.request.Request(
                base_url + "/responses",
                data=json.dumps({"model": "gpt-test", "input": "hi", "stream": True}).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                connection_header = resp.headers.get("Connection")
                body = resp.read().decode()

        self.assertEqual((connection_header or "").lower(), "close")
        self.assertIn("data: [DONE]", body)


class CatalogTests(unittest.TestCase):
    def test_build_catalog_clones_matching_bundled_entry(self):
        bundled = [
            {"slug": "gpt-5.5", "context_window": 1},
            {"slug": "gpt-6", "context_window": 2},
            {"slug": "gpt-6.1", "context_window": 3},
        ]
        models = build_catalog(
            ["gpt-6.1-guardrails", "claude-opus-5", "text-embedding-005"], bundled
        )

        by_slug = {m["slug"]: m for m in models}
        self.assertEqual(set(by_slug), {"gpt-6.1-guardrails", "claude-opus-5"})
        self.assertEqual(by_slug["gpt-6.1-guardrails"]["context_window"], 3)
        self.assertEqual(by_slug["claude-opus-5"]["context_window"], 1)


if __name__ == "__main__":
    unittest.main()
