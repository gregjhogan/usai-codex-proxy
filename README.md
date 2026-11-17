# usai-codex-proxy

A small local proxy that lets the [Codex CLI](https://github.com/openai/codex)
talk to a **USAi** LLM gateway.

## Why this exists

- Codex `>= 0.85` only speaks the OpenAI **Responses API** (`/responses`).
  `wire_api = "chat"` is a hard error in current versions.
- USAi only exposes the legacy **Chat Completions API**
  (`/chat/completions`); it has no `/responses` endpoint.
- OpenAI's own API (which Codex would otherwise fall back to) may be blocked by
  a corporate network proxy.

This proxy bridges the gap. It listens on `127.0.0.1:8787`, accepts the
Responses-API requests Codex sends, translates them into USAi
`chat/completions` calls, and streams the result back as Responses-API SSE
events. It also serves a `/models` endpoint in the shape Codex expects.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`)
- `USAI_BASE_URL` and `USAI_API_KEY` set in your environment (both required)

The proxy itself is pure Python stdlib (no third-party dependencies).

## Starting the proxy

```bash
cd ~/code/usai-codex-proxy
export USAI_BASE_URL=https://<your-usai-host>/api/v1
export USAI_API_KEY=<your-key>
uv run usai-codex-proxy
```

It prints:

```
[proxy] listening on http://127.0.0.1:8787  -> https://<your-usai-host>/api/v1
[proxy] point codex base_url at http://127.0.0.1:8787/v1
```

To run it in the background:

```bash
cd ~/code/usai-codex-proxy
nohup uv run usai-codex-proxy >/tmp/usai-proxy.log 2>&1 &
```

For full local request/response tracing, start it with `--debug`:

```bash
uv run usai-codex-proxy --debug
```

Debug mode logs incoming Responses payloads, translated Chat Completions
payloads, upstream responses or stream chunks, and emitted Responses events to
stderr. It includes message content and tool arguments, so use it only when the
log destination is appropriate for that data.

Stop a background instance with:

```bash
pkill -f usai-codex-proxy
```

Quick health check:

```bash
curl -s http://127.0.0.1:8787/         # -> {"status": "ok"}
```

### Configuration (env vars)

| Variable        | Default       | Purpose                            |
| --------------- | ------------- | ---------------------------------- |
| `USAI_BASE_URL` | (required)    | Upstream USAi base URL             |
| `USAI_API_KEY`  | (required)    | Bearer token sent to USAi          |
| `PROXY_HOST`    | `127.0.0.1`   | Address the proxy binds to         |
| `PROXY_PORT`    | `8787`        | Port the proxy binds to            |

The proxy exits with an error if `USAI_BASE_URL` or `USAI_API_KEY` is missing.

The API key resolution prefers a client-supplied `Authorization: Bearer`
header (which is what Codex sends via `env_key = "USAI_API_KEY"`) and only
falls back to the proxy's own `USAI_API_KEY` environment variable when no such
header is present.

## Codex configuration

Codex reads `~/.codex/config.toml`. The setup below points Codex at the proxy,
disables the OpenAI-hosted "apps" connector (which fails behind the proxy), and
loads a local model catalog so `/model` lists the USAi models.

Replace `/path/to/usai-codex-proxy` and the project path with your own.

```toml
model = "gpt-6.1-sol-latest-guardrails-defaultv2"
model_provider = "usai"

# Local list of USAi models so `/model` in the TUI shows them instead of
# trying (and failing) to reach OpenAI's model catalog through the proxy.
model_catalog_json = "/path/to/usai-codex-proxy/usai_models.json"

# Disable the built-in "codex_apps" MCP connector, which tries to reach
# OpenAI-hosted endpoints and fails (HTTP 403) behind a corporate proxy.
include_apps_instructions = false

model_reasoning_effort = "high"

[projects."/path/to/your/project"]
trust_level = "trusted"

[model_providers.usai]
name = "USAi"
base_url = "http://127.0.0.1:8787/v1"
env_key = "USAI_API_KEY"
wire_api = "responses"

[features]
apps = false
enable_mcp_apps = false

[tui.model_availability_nux]
"gpt-5.5" = 2
```

Notes:

- `base_url` points at **the proxy**, not USAi directly.
- `wire_api = "responses"` is required (Codex no longer supports `"chat"`).
- `env_key = "USAI_API_KEY"` means Codex passes your key through; the proxy
  forwards it to USAi as the `Authorization: Bearer` header.
- `apps = false` (with `enable_mcp_apps = false` and
  `include_apps_instructions = false`) suppresses the `codex_apps` MCP
  connector, which otherwise fails with `HTTP 403` behind a corporate proxy.

## The model catalog (`usai_models.json`)

Codex builds its `/model` picker from an OpenAI-hosted catalog it can't reach
here, so `model_catalog_json` points it at this local file instead.

The proxy rebuilds it automatically on every startup (`catalog.py`): it asks
USAi which models it serves and clones each entry from Codex's own bundled
catalog, so the list stays current with both USAi and your installed Codex. If
the update fails (e.g. USAi is unreachable), the proxy logs it and keeps the
existing file. Restart the proxy to pick up new models.

Models are matched to a bundled entry by slug prefix (e.g.
`gpt-6.1-sol-latest-guardrails-defaultv2` uses `gpt-6.1-sol`); anything without
a match, like Claude or Gemini, uses `gpt-5.5`, so the picker shows its context
window (272k). This is display-only.

## Everyday usage

1. Start the proxy (see above). It must be running whenever you use Codex.
2. Run `codex` as normal.
3. Switch models in the TUI with `/model`, or set `model = "<slug>"` in
   `config.toml`.

## How the translation works

`main.py`:

- Converts Responses-API `input` items (messages, `function_call`,
  `function_call_output`, etc.) into chat `messages`. `developer` role is
  remapped to `system` since USAi rejects it.
- Converts Responses-API `tools` (flat `{type, name, parameters}`) into chat
  `tools` (nested `{type: function, function: {...}}`).
- Re-emits streamed chat deltas as Responses-API SSE events
  (`response.created`, `response.output_text.delta`,
  `response.function_call_arguments.delta`, `response.completed`, etc.).
- Serves `/models` by fetching USAi's list and reshaping it to
  `{"models": [{"id": ...}]}`.
