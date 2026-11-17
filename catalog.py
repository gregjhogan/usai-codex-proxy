"""Builds usai_models.json, the model list Codex shows in its /model picker.

Codex can't reach OpenAI's hosted catalog here, so ~/.codex/config.toml points
`model_catalog_json` at this file. The proxy rebuilds it on startup from the
models USAi currently serves. Each entry is cloned from Codex's own bundled
catalog so it always matches the installed Codex's schema.
"""

import json
import subprocess
import urllib.request
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("usai_models.json")
FALLBACK_TEMPLATE = "gpt-5.5"
# USAi lists these but they can't be used for chat.
NON_CHAT = ("embedding", "cohere")


def build_catalog(usai_ids, bundled_models):
    bundled = {m["slug"]: m for m in bundled_models}
    fallback = bundled.get(FALLBACK_TEMPLATE, bundled_models[0])
    models = []
    for priority, slug in enumerate(sorted(usai_ids), start=1):
        if any(word in slug for word in NON_CHAT):
            continue
        # e.g. "gpt-6.1-sol-latest-guardrails-defaultv2" uses the "gpt-6.1-sol" entry.
        template = max((s for s in bundled if slug.startswith(s)), key=len, default=None)
        models.append({
            **(bundled[template] if template else fallback),
            "slug": slug,
            "display_name": slug,
            "description": f"{slug} via USAi",
            "priority": priority,
            # Bundled upgrade/"new model" prompts point at slugs USAi doesn't serve.
            "upgrade": None,
            "availability_nux": None,
        })
    return models


def update_catalog(base_url, api_key):
    req = urllib.request.Request(base_url.rstrip("/") + "/models")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as r:
        usai_ids = [m["id"] for m in json.load(r)["data"]]

    bundled_json = subprocess.run(
        ["codex", "debug", "models", "--bundled", "-c", 'model_catalog_json=""'],
        capture_output=True, text=True, check=True,
    ).stdout
    models = build_catalog(usai_ids, json.loads(bundled_json)["models"])

    CATALOG_PATH.write_text(json.dumps({"models": models}, indent=2) + "\n")
    return len(models)
