"""Unified async LLM client for OpenAI, Anthropic, Mistral, and DeepSeek.

All calls return a standard ``LLMResponse`` dict regardless of provider.
Retries on rate limits and transient errors use exponential backoff via
tenacity. Per-provider concurrency is enforced with asyncio semaphores.

Model identifier convention: ``"<provider>:<model_name>"``
  - openai:gpt-4o-mini
  - anthropic:claude-haiku-4-5-20251001
  - mistral:mistral-small-latest
  - deepseek:deepseek-chat

Cost is computed from a per-model price table (USD per 1M tokens). Update the
table when new models are added.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Per-million-token prices in USD. Verify before running.
# Format: provider:model -> (input_per_1m, output_per_1m)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openai:gpt-4o-mini": (0.15, 0.60),
    "anthropic:claude-haiku-4-5-20251001": (1.00, 5.00),
    "mistral:mistral-small-latest": (0.20, 0.60),
    "deepseek:deepseek-chat": (0.27, 1.10),
}


class LLMError(Exception):
    """Base class for retriable LLM errors (rate limits, 5xx, network)."""


class FatalLLMError(Exception):
    """Non-retriable errors (auth failures, malformed requests)."""


@dataclass
class LLMResponse:
    response_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    model: str
    timestamp: str
    raw_response: str


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Split 'provider:model_name' into its parts."""
    if ":" not in model_id:
        raise ValueError(f"Model id must be 'provider:model_name', got {model_id!r}")
    provider, model_name = model_id.split(":", 1)
    return provider, model_name


def compute_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    if model_id not in MODEL_PRICES:
        # Unknown model — return 0 rather than crash, but log via assertion-like error
        return 0.0
    in_price, out_price = MODEL_PRICES[model_id]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


class LLMClient:
    """Unified async LLM client.

    Usage:
        client = LLMClient(api_keys, concurrency_per_provider=4, log_path=path)
        resp = await client.call("openai:gpt-4o-mini", system="...", user="...")
    """

    def __init__(
        self,
        api_keys: dict[str, str],
        *,
        concurrency_per_provider: int = 4,
        max_retries: int = 5,
        log_path: Path | None = None,
    ) -> None:
        self.api_keys = api_keys
        self.max_retries = max_retries
        self.log_path = log_path
        # Mistral free-tier enforces ~1 request/second and ~500K tokens/minute.
        # Even 2 concurrent calls trigger 429s in bursts — hard-cap to 1.
        # Other providers can fan out at the configured rate.
        self._semaphores: dict[str, asyncio.Semaphore] = {
            "openai": asyncio.Semaphore(concurrency_per_provider),
            "anthropic": asyncio.Semaphore(concurrency_per_provider),
            "mistral": asyncio.Semaphore(1),
            "deepseek": asyncio.Semaphore(concurrency_per_provider),
        }
        # Lazy-init SDK clients (avoid importing if provider unused)
        self._openai_client: Any = None
        self._anthropic_client: Any = None
        self._mistral_client: Any = None
        self._deepseek_client: Any = None

    # ---------- Public API ----------
    async def call(
        self,
        model_id: str,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
        json_mode: bool = False,
        stage: str = "",
        item_id: str | None = None,
    ) -> LLMResponse:
        provider, model_name = parse_model_id(model_id)
        if provider not in self._semaphores:
            raise ValueError(f"Unsupported provider: {provider}")

        async with self._semaphores[provider]:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                    retry=retry_if_exception_type(LLMError),
                    reraise=True,
                ):
                    with attempt:
                        resp = await self._dispatch(
                            provider=provider,
                            model_id=model_id,
                            model_name=model_name,
                            system=system,
                            user=user,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                        )
                self._log_call(stage, provider, model_id, item_id, resp, success=True)
                return resp
            except (LLMError, FatalLLMError, RetryError) as e:
                self._log_call(
                    stage, provider, model_id, item_id, None, success=False, error=str(e)
                )
                raise

    # ---------- Provider dispatch ----------
    async def _dispatch(
        self,
        *,
        provider: str,
        model_id: str,
        model_name: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        if provider == "openai":
            return await self._call_openai(
                model_id, model_name, system, user, temperature, max_tokens, json_mode
            )
        if provider == "anthropic":
            return await self._call_anthropic(
                model_id, model_name, system, user, temperature, max_tokens
            )
        if provider == "mistral":
            return await self._call_mistral(
                model_id, model_name, system, user, temperature, max_tokens, json_mode
            )
        if provider == "deepseek":
            return await self._call_deepseek(
                model_id, model_name, system, user, temperature, max_tokens, json_mode
            )
        raise ValueError(f"Unknown provider: {provider}")

    # ---------- OpenAI ----------
    async def _call_openai(
        self,
        model_id: str,
        model_name: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        if self._openai_client is None:
            from openai import AsyncOpenAI

            key = self.api_keys.get("openai", "")
            if not key:
                raise FatalLLMError("OPENAI_API_KEY is not set")
            self._openai_client = AsyncOpenAI(api_key=key)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            resp = await self._openai_client.chat.completions.create(**kwargs)
        except Exception as e:
            raise self._classify_openai_error(e) from e
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return LLMResponse(
            response_text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=compute_cost(model_id, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model_id,
            timestamp=_utcnow(),
            raw_response=text,
        )

    @staticmethod
    def _classify_openai_error(e: Exception) -> Exception:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "timeout" in msg or "503" in msg or "502" in msg:
            return LLMError(str(e))
        if "401" in msg or "403" in msg or "invalid" in msg:
            return FatalLLMError(str(e))
        # Default: treat as retriable
        return LLMError(str(e))

    # ---------- Anthropic ----------
    async def _call_anthropic(
        self,
        model_id: str,
        model_name: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic

            key = self.api_keys.get("anthropic", "")
            if not key:
                raise FatalLLMError("ANTHROPIC_API_KEY is not set")
            self._anthropic_client = AsyncAnthropic(api_key=key)

        start = time.perf_counter()
        try:
            resp = await self._anthropic_client.messages.create(
                model=model_name,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise self._classify_openai_error(e) from e
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Anthropic returns content as a list of blocks; concatenate text blocks
        text = "".join(
            getattr(block, "text", "") for block in resp.content if getattr(block, "type", "") == "text"
        )
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        return LLMResponse(
            response_text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=compute_cost(model_id, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model_id,
            timestamp=_utcnow(),
            raw_response=text,
        )

    # ---------- Mistral ----------
    async def _call_mistral(
        self,
        model_id: str,
        model_name: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        if self._mistral_client is None:
            try:
                from mistralai import Mistral  # mistralai 1.x
            except ImportError:
                from mistralai.client.sdk import Mistral  # mistralai 2.x

            key = self.api_keys.get("mistral", "")
            if not key:
                raise FatalLLMError("MISTRAL_API_KEY is not set")
            self._mistral_client = Mistral(api_key=key)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            resp = await self._mistral_client.chat.complete_async(**kwargs)
        except Exception as e:
            raise self._classify_openai_error(e) from e
        latency_ms = int((time.perf_counter() - start) * 1000)
        # Mistral free tier: 1 req/sec hard cap. Sleep ~1s after each call so
        # the next caller (holding the semaphore) doesn't immediately fire.
        await asyncio.sleep(1.1)

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return LLMResponse(
            response_text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=compute_cost(model_id, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model_id,
            timestamp=_utcnow(),
            raw_response=text,
        )

    # ---------- DeepSeek (OpenAI-compatible) ----------
    async def _call_deepseek(
        self,
        model_id: str,
        model_name: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        if self._deepseek_client is None:
            from openai import AsyncOpenAI

            key = self.api_keys.get("deepseek", "")
            if not key:
                raise FatalLLMError("DEEPSEEK_API_KEY is not set")
            self._deepseek_client = AsyncOpenAI(
                api_key=key, base_url="https://api.deepseek.com/v1"
            )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            resp = await self._deepseek_client.chat.completions.create(**kwargs)
        except Exception as e:
            raise self._classify_openai_error(e) from e
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return LLMResponse(
            response_text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=compute_cost(model_id, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model_id,
            timestamp=_utcnow(),
            raw_response=text,
        )

    # ---------- Logging ----------
    def _log_call(
        self,
        stage: str,
        provider: str,
        model_id: str,
        item_id: str | None,
        resp: LLMResponse | None,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        if self.log_path is None:
            return
        record = {
            "stage": stage,
            "provider": provider,
            "model": model_id,
            "item_id": item_id,
            "input_tokens": resp.input_tokens if resp else 0,
            "output_tokens": resp.output_tokens if resp else 0,
            "cost_usd": resp.cost_usd if resp else 0.0,
            "latency_ms": resp.latency_ms if resp else 0,
            "timestamp": resp.timestamp if resp else _utcnow(),
            "success": success,
            "error": error,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
