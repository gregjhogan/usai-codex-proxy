#!/usr/bin/env python3
"""
Adapter that lets the Codex CLI talk to the USAi gateway.

Codex (>=0.85) only speaks the OpenAI "responses" wire API, while USAi only
speaks the older "chat/completions" API. This proxy exposes a Responses-API
endpoint for Codex to connect to: it accepts POST /v1/responses (and
/responses), translates each request into a USAi chat/completions call, and
translates the reply back into Responses-API SSE events. It also serves
/v1/models so Codex can list the available models.

Requires the USAI_BASE_URL and USAI_API_KEY environment variables.

Stdlib only. Run:  uv run usai-codex-proxy   (or: python3 main.py)
Then point codex at http://127.0.0.1:8787/v1
"""

import json
import os
import sys
import time
import uuid
import argparse
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("USAI_BASE_URL")
API_KEY = os.environ.get("USAI_API_KEY")
LISTEN_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8787"))
DEBUG = False
MODEL_CATALOG_PATH = Path(__file__).with_name("usai_models.json")


def log(*a):
    print("[proxy]", *a, file=sys.stderr, flush=True)


def debug_log(*a):
    if DEBUG:
        log("debug", *a)


def debug_json(label, obj):
    if DEBUG:
        log("debug", label, json.dumps(obj, default=str))


def _response_request_summary(body):
    inp = body.get("input")
    if isinstance(inp, list):
        input_summary = ",".join(
            item.get("type", "message") if isinstance(item, dict) else type(item).__name__
            for item in inp
        )
    else:
        input_summary = type(inp).__name__
    return {
        "model": body.get("model"),
        "stream": bool(body.get("stream", False)),
        "input": input_summary,
        "tools": len(body.get("tools") or []),
        "tool_choice": body.get("tool_choice"),
    }


def _responses_text_format_to_chat_response_format(text):
    if not isinstance(text, dict):
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None

    ftype = fmt.get("type")
    if ftype == "json_object":
        return {"type": "json_object"}
    if ftype == "json_schema":
        json_schema = {
            "name": fmt.get("name"),
            "schema": fmt.get("schema", {}),
        }
        if fmt.get("description") is not None:
            json_schema["description"] = fmt["description"]
        if fmt.get("strict") is not None:
            json_schema["strict"] = fmt["strict"]
        return {"type": "json_schema", "json_schema": json_schema}
    return None


def _load_codex_model_catalog():
    with MODEL_CATALOG_PATH.open() as f:
        data = json.load(f)
    models = data.get("models", []) if isinstance(data, dict) else []
    return [m for m in models if isinstance(m, dict) and m.get("slug")]


def _upstream_model_ids(upstream):
    data = upstream.get("data", upstream) if isinstance(upstream, dict) else upstream
    ids = set()
    for m in data or []:
        if isinstance(m, dict) and m.get("id"):
            ids.add(m["id"])
        elif isinstance(m, str):
            ids.add(m)
    return ids



# ---------------------------------------------------------------------------
# Request translation: Responses API  ->  Chat Completions
# ---------------------------------------------------------------------------
def _content_to_text(content):
    """Responses content parts -> chat string (or list of parts)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        # Unexpected shape (e.g. a dict); coerce to text rather than forward
        # something USAi would reject.
        return json.dumps(content) if content is not None else ""
    parts = []
    for p in content:
        if not isinstance(p, dict):
            parts.append({"type": "text", "text": str(p)})
            continue
        t = p.get("type")
        if t in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif t in ("input_image", "image_url"):
            url = p.get("image_url") or p.get("url")
            if isinstance(url, dict):
                image_url = dict(url)
            else:
                image_url = {"url": url}
            if p.get("detail") is not None and "detail" not in image_url:
                image_url["detail"] = p["detail"]
            parts.append({"type": "image_url", "image_url": image_url})
        else:
            # fall back to any text field
            if "text" in p:
                parts.append({"type": "text", "text": p["text"]})
    # collapse to plain string if it's all text
    if all(x.get("type") == "text" for x in parts):
        return "".join(x["text"] for x in parts)
    return parts


def responses_to_chat(body):
    """Translate a Responses API request dict into a chat/completions dict."""
    messages = []
    pending_tool_calls = []

    def flush_tool_calls():
        if pending_tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": list(pending_tool_calls),
            })
            pending_tool_calls.clear()

    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue
            itype = item.get("type", "message")
            role = item.get("role", "user")

            if itype == "message":
                flush_tool_calls()
                # USAi only accepts user/system/assistant/tool.
                if role == "developer":
                    role = "system"
                messages.append({
                    "role": role,
                    "content": _content_to_text(item.get("content", "")),
                })
            elif itype == "function_call":
                pending_tool_calls.append({
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments", "") or "",
                    },
                })
            elif itype == "function_call_output":
                flush_tool_calls()
                out = item.get("output", "")
                if not isinstance(out, str):
                    out = json.dumps(out)
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id"),
                    "content": out,
                })
            elif itype in ("reasoning",):
                continue  # drop reasoning items; chat api has no equivalent
            else:
                # unknown item, best-effort
                c = item.get("content")
                if c is not None:
                    flush_tool_calls()
                    messages.append({"role": role, "content": _content_to_text(c)})

        flush_tool_calls()

    chat = {
        "model": body["model"],
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }

    if body.get("temperature") is not None:
        chat["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        chat["top_p"] = body["top_p"]
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]

    for key in (
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "parallel_tool_calls",
        "presence_penalty",
        "seed",
        "stop",
        "top_logprobs",
        "user",
    ):
        if body.get(key) is not None:
            chat[key] = body[key]

    if body.get("response_format") is not None:
        chat["response_format"] = body["response_format"]
    else:
        response_format = _responses_text_format_to_chat_response_format(body.get("text"))
        if response_format is not None:
            chat["response_format"] = response_format

    # tools: responses uses flat {type:function,name,parameters}; chat nests them
    tools = body.get("tools")
    if tools:
        chat_tools = []
        for t in tools:
            if t.get("type") == "function":
                fn = {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                }
                chat_tools.append({"type": "function", "function": fn})
        if chat_tools:
            chat["tools"] = chat_tools

    tc = body.get("tool_choice")
    if tc is not None:
        if isinstance(tc, dict) and tc.get("type") == "function" and tc.get("name"):
            chat["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }
        else:
            chat["tool_choice"] = tc

    return chat


# ---------------------------------------------------------------------------
# Upstream calls
# ---------------------------------------------------------------------------
def upstream_request(chat_body, api_key, stream):
    url = UPSTREAM.rstrip("/") + "/chat/completions"
    debug_json("upstream request", chat_body)
    data = json.dumps(chat_body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "text/event-stream" if stream else "application/json")
    return urllib.request.urlopen(req, timeout=600)


# ---------------------------------------------------------------------------
# Response translation: Chat Completions -> Responses API
# ---------------------------------------------------------------------------
def sse(event, data_obj):
    return f"event: {event}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8")


def build_response_object(resp_id, model, text, tool_calls, usage, status="completed"):
    output = []
    for tc in tool_calls:
        output.append({
            "type": "function_call",
            "id": "fc_" + uuid.uuid4().hex,
            "call_id": tc.get("id") or ("call_" + uuid.uuid4().hex),
            "name": tc["function"]["name"],
            "arguments": tc["function"].get("arguments", "") or "",
            "status": "completed",
        })
    if text:
        output.append({
            "type": "message",
            "id": "msg_" + uuid.uuid4().hex,
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    return build_response_from_output(resp_id, model, output, usage, status)


def build_response_from_output(resp_id, model, output, usage, status="completed"):
    obj = {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
    }
    if usage:
        obj["usage"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    return obj


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _api_key(self):
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return API_KEY

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path.rstrip("/").endswith("/models"):
            self._proxy_models()
            return
        self._send_json(200, {"status": "ok"})

    def _proxy_models(self):
        try:
            url = UPSTREAM.rstrip("/") + "/models"
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self._api_key()}")
            with urllib.request.urlopen(req, timeout=60) as r:
                upstream = json.loads(r.read())
            # USAi returns OpenAI list shape {object,data:[{id,...}]}.
            # Codex expects its full model catalog schema, keyed by slug.
            upstream_ids = _upstream_model_ids(upstream)
            models = [
                m for m in _load_codex_model_catalog()
                if m["slug"] in upstream_ids
            ]
            self._send_json(200, {"models": models})
        except urllib.error.HTTPError as e:
            self._send_raw(e.code, e.read(), "application/json")
        except Exception as e:
            log("models error", repr(e))
            self._send_json(200, {"models": []})

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if not path.rstrip("/").endswith("/responses"):
            self._send_json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": {"message": f"bad json: {e}"}})
            return

        debug_json("responses request", body)
        debug_json("responses request summary", _response_request_summary(body))
        stream = bool(body.get("stream", False))
        chat_body = responses_to_chat(body)
        api_key = self._api_key()
        debug_json("chat request summary", {
            "model": chat_body.get("model"),
            "stream": chat_body.get("stream"),
            "messages": [m.get("role") for m in chat_body.get("messages", [])],
            "tools": len(chat_body.get("tools") or []),
            "tool_choice": chat_body.get("tool_choice"),
        })

        try:
            up = upstream_request(chat_body, api_key, stream)
        except urllib.error.HTTPError as e:
            payload = e.read()
            log("upstream error", e.code, payload[:300])
            self._send_raw(e.code, payload, "application/json")
            return
        except Exception as e:
            log("upstream exception", repr(e))
            self._send_json(502, {"error": {"message": f"upstream: {e}"}})
            return

        if stream:
            self._stream(up, chat_body["model"])
        else:
            self._non_stream(up, chat_body["model"])

    # -- non streaming --------------------------------------------------
    def _non_stream(self, up, model):
        data = json.loads(up.read())
        debug_json("upstream response", data)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        debug_json("non-stream response summary", {
            "finish_reason": choice.get("finish_reason"),
            "text_len": len(text),
            "tool_calls": len(tool_calls),
            "usage": data.get("usage"),
        })
        resp = build_response_object(
            "resp_" + uuid.uuid4().hex, model, text, tool_calls, data.get("usage")
        )
        debug_json("responses response", resp)
        self._send_json(200, resp)

    # -- streaming ------------------------------------------------------
    def _stream(self, up, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        resp_id = "resp_" + uuid.uuid4().hex
        seq = 0

        def emit(event, obj):
            nonlocal seq
            obj.setdefault("sequence_number", seq)
            seq += 1
            debug_json(f"responses event {event}", obj)
            self.wfile.write(sse(event, obj))
            self.wfile.flush()

        base_resp = {
            "id": resp_id, "object": "response", "created_at": int(time.time()),
            "status": "in_progress", "model": model, "output": [],
        }
        emit("response.created", {"type": "response.created", "response": dict(base_resp)})
        emit("response.in_progress", {"type": "response.in_progress", "response": dict(base_resp)})

        text_parts = []
        # accumulate tool calls by index
        tools = {}
        usage = None
        msg_item_id = "msg_" + uuid.uuid4().hex
        text_item_open = False
        msg_out_index = None      # output index assigned to the text message
        next_out_index = 0        # next contiguous output index to hand out

        try:
            for line in up:
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:"):].strip()
                if payload == b"[DONE]":
                    debug_log("upstream stream done")
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                debug_json("upstream stream chunk", chunk)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # text
                dc = delta.get("content")
                if dc:
                    if not text_item_open:
                        msg_out_index = next_out_index
                        next_out_index += 1
                        emit("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": msg_out_index,
                            "item": {"type": "message", "id": msg_item_id,
                                     "role": "assistant", "status": "in_progress",
                                     "content": []},
                        })
                        emit("response.content_part.added", {
                            "type": "response.content_part.added",
                            "item_id": msg_item_id, "output_index": msg_out_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        })
                        text_item_open = True
                    text_parts.append(dc)
                    emit("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": msg_item_id, "output_index": msg_out_index,
                        "content_index": 0, "delta": dc,
                    })

                # tool calls
                for tcd in delta.get("tool_calls") or []:
                    idx = tcd.get("index", 0)
                    slot = tools.setdefault(idx, {
                        "id": None, "call_id": None, "name": "", "arguments": "",
                        "emitted_arguments_len": 0,
                        "item_id": "fc_" + uuid.uuid4().hex, "started": False,
                    })
                    if tcd.get("id"):
                        slot["id"] = tcd["id"]
                        if not slot["started"]:
                            slot["call_id"] = tcd["id"]
                    fn = tcd.get("function", {})
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                    if not slot["started"] and slot["name"]:
                        slot["call_id"] = slot["call_id"] or slot["id"] or slot["item_id"]
                        slot["out_index"] = next_out_index
                        next_out_index += 1
                        emit("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": slot["out_index"],
                            "item": {"type": "function_call", "id": slot["item_id"],
                                     "call_id": slot["call_id"],
                                     "name": slot["name"], "arguments": "",
                                     "status": "in_progress"},
                        })
                        slot["started"] = True
                    if slot["started"]:
                        new_arguments = slot["arguments"][slot["emitted_arguments_len"]:]
                        if new_arguments:
                            emit("response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta",
                                "item_id": slot["item_id"],
                                "output_index": slot["out_index"],
                                "delta": new_arguments,
                            })
                            slot["emitted_arguments_len"] = len(slot["arguments"])

            # close text
            final_text = "".join(text_parts)
            final_output_by_index = {}
            if text_item_open:
                final_output_by_index[msg_out_index] = {
                    "type": "message", "id": msg_item_id, "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": final_text,
                                 "annotations": []}],
                }
                emit("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": msg_item_id, "output_index": msg_out_index,
                    "content_index": 0, "text": final_text,
                })
                emit("response.content_part.done", {
                    "type": "response.content_part.done",
                    "item_id": msg_item_id, "output_index": msg_out_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": final_text, "annotations": []},
                })
                emit("response.output_item.done", {
                    "type": "response.output_item.done", "output_index": msg_out_index,
                    "item": {"type": "message", "id": msg_item_id, "role": "assistant",
                             "status": "completed",
                             "content": [{"type": "output_text", "text": final_text,
                                          "annotations": []}]},
                })

            # close tool calls
            tool_call_list = []
            for idx in sorted(tools):
                slot = tools[idx]
                call_id = slot["call_id"] or slot["id"] or slot["item_id"]
                tool_call_list.append({
                    "id": call_id,
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                })
                if slot["started"]:
                    final_output_by_index[slot["out_index"]] = {
                        "type": "function_call", "id": slot["item_id"],
                        "call_id": call_id, "name": slot["name"],
                        "arguments": slot["arguments"], "status": "completed",
                    }
                    emit("response.function_call_arguments.done", {
                        "type": "response.function_call_arguments.done",
                        "item_id": slot["item_id"],
                        "output_index": slot.get("out_index"),
                        "arguments": slot["arguments"],
                    })
                    emit("response.output_item.done", {
                        "type": "response.output_item.done",
                        "output_index": slot.get("out_index"),
                        "item": {"type": "function_call", "id": slot["item_id"],
                                 "call_id": call_id,
                                 "name": slot["name"], "arguments": slot["arguments"],
                                 "status": "completed"},
                    })

            final_output = [final_output_by_index[i] for i in sorted(final_output_by_index)]
            final = build_response_from_output(resp_id, model, final_output, usage)
            debug_json("stream response summary", {
                "text_len": len(final_text),
                "tool_calls": len(tool_call_list),
                "output_items": len(final_output),
                "usage": usage,
            })
            emit("response.completed", {"type": "response.completed", "response": final})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-stream; nothing more to send.
            log("client disconnected during stream")
        finally:
            up.close()

    # -- helpers --------------------------------------------------------
    def _send_json(self, code, obj):
        self._send_raw(code, json.dumps(obj).encode(), "application/json")

    def _send_raw(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global DEBUG
    parser = argparse.ArgumentParser(description="Responses API to USAi chat/completions proxy")
    parser.add_argument("--debug", action="store_true", help="log request, response, and streaming payloads")
    args = parser.parse_args()
    DEBUG = args.debug

    if not UPSTREAM:
        sys.exit("error: USAI_BASE_URL environment variable is required")
    if not API_KEY:
        sys.exit("error: USAI_API_KEY environment variable is required")
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log(f"listening on http://{LISTEN_HOST}:{LISTEN_PORT}  -> {UPSTREAM}")
    log(f"point codex base_url at http://{LISTEN_HOST}:{LISTEN_PORT}/v1")
    if DEBUG:
        log("debug logging enabled")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")


if __name__ == "__main__":
    main()
