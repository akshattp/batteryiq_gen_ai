"""LLMClient Protocol and provider implementations (Section 8.1).

Supported providers:
- groq (default): Groq cloud API with rate-limit exponential backoff and on-disk caching
- template: deterministic offline template generator (fallback)
- local_hf: local HuggingFace / LoRA model
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from dotenv import load_dotenv

from guardian.utils.config import get_config
from guardian.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)


class LLMUnavailable(Exception):
    """Raised when LLM API calls fail after all retry attempts."""
    pass


@runtime_checkable
class LLMClient(Protocol):
    """Unified client protocol for language model inference."""

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        """Generate response text given chat messages."""
        ...


class GroqClient:
    """Groq API client with caching, exponential backoff, and fallback model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        cfg = get_config()
        self.model = model or os.getenv("GROQ_MODEL") or cfg.get("llm", {}).get("model", "qwen/qwen3.8-27b")
        self.fallback_model = (
            fallback_model
            or os.getenv("GROQ_FALLBACK_MODEL")
            or cfg.get("llm", {}).get("fallback_model", "openai/gpt-oss-20b")
        )
        self.cache_dir = Path(cache_dir or cfg.get("paths", {}).get("llm_cache", "artifacts/llm_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

        self._client: Optional[Any] = None

    def _get_raw_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise LLMUnavailable("GROQ_API_KEY is missing or empty.")
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as exc:
                raise LLMUnavailable(f"Failed to initialize Groq client: {exc}") from exc
        return self._client

    def _get_cache_path(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        json_schema: Optional[Dict[str, Any]],
    ) -> Path:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "schema": json_schema,
        }
        cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{cache_key}.json"

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        """Call Groq API with caching, retry backoff, and model fallback."""
        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        last_error: Optional[Exception] = None

        for current_model in models_to_try:
            cache_file = self._get_cache_path(current_model, messages, temperature, json_schema)
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    return cached_data.get("response", "")
                except Exception:
                    pass

            client = self._get_raw_client()
            delay = 1.0

            for attempt in range(1, self.max_retries + 1):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": current_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if json_schema is not None:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""

                    # Save to cache
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "model": current_model,
                                    "response": content,
                                    "created_at": time.time(),
                                },
                                f,
                                indent=2,
                            )
                    except Exception as cache_exc:
                        logger.warning(f"Failed to write to LLM cache: {cache_exc}")

                    return content

                except Exception as exc:
                    last_error = exc
                    err_msg = str(exc)
                    is_rate_limit = "429" in err_msg or "RateLimit" in type(exc).__name__
                    is_not_found = "404" in err_msg or "model_not_found" in err_msg

                    if is_not_found:
                        logger.warning(f"Groq model {current_model} not found/accessible: {exc}")
                        break  # Break retry loop to try fallback model

                    if is_rate_limit and attempt < self.max_retries:
                        logger.warning(
                            f"Groq rate limit (429) hit. Backing off for {delay:.1f}s (attempt {attempt}/{self.max_retries})"
                        )
                        time.sleep(delay)
                        delay *= 2.0
                    elif attempt < self.max_retries:
                        time.sleep(delay)
                        delay *= 1.5
                    else:
                        logger.error(f"Groq API call failed after {self.max_retries} attempts: {exc}")

        raise LLMUnavailable(f"All Groq model attempts failed. Last error: {last_error}")


class TemplateClient:
    """Offline deterministic mock LLM client conforming to LLMClient protocol."""

    def __init__(self, default_response: Optional[str] = None) -> None:
        self.default_response = default_response

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        """Produce deterministic JSON response based on prompt contents."""
        if self.default_response:
            return self.default_response

        # Inspect prompt to infer risk level and key telemetry
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        level = "NORMAL"
        for candidate in ("CRITICAL", "HIGH", "WARNING", "NORMAL"):
            if candidate in user_msg.upper():
                level = candidate
                break

        # Build consistent JSON
        data = {
            "summary": f"Battery risk level is assessed as {level} based on active telemetry analysis.",
            "reasons": [
                f"Evaluation confirmed current operational status aligns with {level} classification thresholds.",
                "Telemetry parameters remained within anticipated observation boundaries.",
            ],
            "recommended_actions": (
                [
                    "Reduce battery load immediately to lower thermal dissipation.",
                    "Inspect thermal management cooling system and cell voltages.",
                    "Maintain active monitoring and escalate to engineer if parameters stay elevated.",
                ]
                if level in ("CRITICAL", "HIGH")
                else [
                    "Continue continuous telemetry monitoring.",
                    "Review cell operating trends at routine maintenance check.",
                ]
            ),
            "confidence_note": "Evaluated with offline deterministic template client.",
        }
        return json.dumps(data, indent=2)


class LocalHFClient:
    """Client utilizing local HuggingFace / PEFT LoRA model conforming to LLMClient protocol."""

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        adapter_path: Optional[str | Path] = None,
    ) -> None:
        from guardian.models.lora_explainer import LoRAExplainer

        self.explainer = LoRAExplainer(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        return self.explainer.generate(messages, max_new_tokens=max_tokens, temperature=temperature)


def get_llm_client(provider: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """Factory function returning the configured LLM client instance."""
    cfg = config or get_config()
    chosen_provider = provider or os.getenv("LLM_PROVIDER") or cfg.get("llm", {}).get("provider", "groq")

    if chosen_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY is not set. Falling back to TemplateClient.")
            return TemplateClient()
        return GroqClient()
    elif chosen_provider == "template":
        return TemplateClient()
    elif chosen_provider == "local_hf":
        try:
            adapter_dir = cfg.get("paths", {}).get("lora_adapter", "artifacts/models/lora_explainer")
            return LocalHFClient(adapter_path=adapter_dir)
        except Exception as exc:
            logger.warning(f"Failed to initialize LocalHFClient: {exc}. Falling back to TemplateClient.")
            return TemplateClient()
    else:
        logger.warning(f"Provider '{chosen_provider}' unsupported or not configured. Using TemplateClient.")
        return TemplateClient()
