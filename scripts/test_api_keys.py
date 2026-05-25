"""Ping each LLM provider with a minimal call to verify API keys.

Loads `.env` from the project root (without requiring python-dotenv to be
installed by parsing it manually), then makes one short call per provider
using the models listed in MODEL_PRICES.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(PROJECT_ROOT / ".env")

from src.config import resolve_api_keys  # noqa: E402
from src.llm_client import FatalLLMError, LLMClient  # noqa: E402


TEST_MODELS = [
    "openai:gpt-4o-mini",
    "anthropic:claude-haiku-4-5-20251001",
    "mistral:mistral-small-latest",
    "deepseek:deepseek-chat",
]


async def _ping(client: LLMClient, model_id: str) -> tuple[str, bool, str]:
    try:
        resp = await client.call(
            model_id,
            system="You are a terse assistant.",
            user="Reply with the single word: pong",
            temperature=0.0,
            max_tokens=10,
            json_mode=False,
            stage="ping",
        )
        return model_id, True, resp.response_text.strip()[:60]
    except FatalLLMError as e:
        return model_id, False, f"FATAL: {e}"
    except Exception as e:
        return model_id, False, f"ERROR: {type(e).__name__}: {e}"


async def main() -> int:
    api_keys = resolve_api_keys()
    print("Key presence:")
    for prov, key in api_keys.items():
        status = f"set (len={len(key)})" if key else "MISSING"
        print(f"  {prov:10s} {status}")
    print()

    client = LLMClient(api_keys=api_keys, concurrency_per_provider=1, max_retries=1)
    results = await asyncio.gather(*(_ping(client, m) for m in TEST_MODELS))

    print("Live calls:")
    any_fail = False
    for model_id, ok, msg in results:
        tag = "OK   " if ok else "FAIL "
        print(f"  [{tag}] {model_id:50s} -> {msg}")
        if not ok:
            any_fail = True
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
