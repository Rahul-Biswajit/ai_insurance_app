"""
llm_factory.py — Provider-agnostic LLM factory with automatic fallbacks.

Design goals
------------
* One entry point: :func:`get_llm`.
* Provider agnostic via LangChain's ``init_chat_model`` — OpenAI, Anthropic,
  Groq, Together, Ollama, vLLM, LM Studio, or any OpenAI-compatible endpoint
  reached through ``base_url``.
* Resilient: the primary model is wrapped in ``RunnableWithFallbacks`` so
  rate limits / 5xx / timeouts transparently roll over to a stronger model.
* Never fatal: if LangChain is not installed, or no credentials exist, the
  factory returns a deterministic :class:`OfflineChatModel` so the whole
  application still runs (rule-based extraction + templated responses).

Usage
-----
    from llm_factory import get_llm
    llm = get_llm(temperature=0.0)
    reply = llm.invoke([{"role": "user", "content": "Hi"}]).content
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from config import settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional LangChain imports
# --------------------------------------------------------------------------- #
try:
    from langchain.chat_models import init_chat_model  # type: ignore

    _LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    try:
        from langchain_core.language_models import init_chat_model  # type: ignore

        _LANGCHAIN_AVAILABLE = True
    except Exception:
        init_chat_model = None  # type: ignore[assignment]
        _LANGCHAIN_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Offline / degraded model
# --------------------------------------------------------------------------- #
@dataclass
class OfflineMessage:
    """Minimal stand-in for ``langchain_core.messages.AIMessage``."""

    content: str
    response_metadata: dict[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.content


class OfflineChatModel:
    """
    Deterministic no-network chat model.

    It is intentionally *not* a language model: it returns a stable, honest
    marker string. Callers detect it via :attr:`is_offline` and switch to
    their deterministic code path (regex entity extraction, templated Q&A)
    rather than pretending an LLM answered.
    """

    is_offline = True

    def __init__(self, reason: str = "no credentials configured") -> None:
        self.reason = reason

    # -- LangChain-compatible surface ------------------------------------- #
    def invoke(self, messages: Any, **_: Any) -> OfflineMessage:
        return OfflineMessage(
            content="[offline-mode] No LLM configured; deterministic fallback used.",
            response_metadata={"offline": True, "reason": self.reason},
        )

    async def ainvoke(self, messages: Any, **kwargs: Any) -> OfflineMessage:
        return self.invoke(messages, **kwargs)

    def stream(self, messages: Any, **kwargs: Any) -> Iterable[OfflineMessage]:
        yield self.invoke(messages, **kwargs)

    def with_structured_output(self, schema: Any, **_: Any) -> "OfflineStructuredModel":
        return OfflineStructuredModel(schema)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> "OfflineChatModel":
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return f"OfflineChatModel(reason={self.reason!r})"


class OfflineStructuredModel:
    """Structured-output shim that always signals 'nothing extracted'."""

    is_offline = True

    def __init__(self, schema: Any) -> None:
        self.schema = schema

    def invoke(self, messages: Any, **_: Any) -> None:
        return None

    async def ainvoke(self, messages: Any, **kwargs: Any) -> None:
        return None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def _build_kwargs(
    temperature: float,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int | None,
    timeout: int | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temperature": temperature}
    if base_url:
        # ``base_url`` is the modern kwarg; ``openai_api_base`` covers older
        # community integrations. init_chat_model forwards unknown kwargs to
        # the provider class, so we only send the canonical one.
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if timeout:
        kwargs["timeout"] = timeout
    if extra:
        kwargs.update(extra)
    return kwargs


def _instantiate(model_name: str, **kwargs: Any) -> Any:
    """Instantiate a chat model, tolerating providers that reject a kwarg."""
    try:
        return init_chat_model(model_name, **kwargs)  # type: ignore[misc]
    except TypeError as exc:
        # Some providers do not accept `timeout` / `max_tokens`; retry lean.
        logger.warning("init_chat_model rejected kwargs (%s); retrying minimal.", exc)
        lean = {k: v for k, v in kwargs.items() if k in {"temperature", "api_key", "base_url"}}
        return init_chat_model(model_name, **lean)  # type: ignore[misc]


_cache: dict[tuple, Any] = {}
_cache_lock = threading.Lock()


def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    fallback_model_name: str | None = None,
    *,
    max_tokens: int | None = None,
    timeout: int | None = None,
    use_cache: bool = True,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Return a ready-to-use chat model.

    Parameters
    ----------
    model_name:
        Provider-qualified id, e.g. ``"openai:gpt-4o-mini"``,
        ``"anthropic:claude-sonnet-4-5"``, ``"ollama:llama3.1"``.
        Defaults to :data:`config.settings.model_name`.
    temperature:
        Sampling temperature. Defaults to :data:`config.settings.temperature`.
    base_url:
        OpenAI-compatible endpoint (vLLM, LM Studio, LiteLLM, Ollama proxy).
    api_key:
        Overrides the environment credential.
    fallback_model_name:
        Model used when the primary raises. Defaults to
        :data:`config.settings.fallback_model_name`. Pass ``""`` to disable.

    Returns
    -------
    A LangChain ``Runnable`` (``RunnableWithFallbacks`` when a fallback is
    configured), or :class:`OfflineChatModel` when no live model is possible.
    """
    model_name = model_name or settings.model_name
    fallback_model_name = (
        settings.fallback_model_name if fallback_model_name is None else fallback_model_name
    )
    temperature = settings.temperature if temperature is None else temperature
    base_url = base_url or settings.base_url
    api_key = api_key or settings.api_key
    max_tokens = max_tokens or settings.max_tokens
    timeout = timeout or settings.request_timeout

    cache_key = (model_name, fallback_model_name, temperature, base_url, bool(api_key))
    if use_cache:
        with _cache_lock:
            if cache_key in _cache:
                return _cache[cache_key]

    llm = _create(
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        model_kwargs=model_kwargs,
    )

    if use_cache:
        with _cache_lock:
            _cache[cache_key] = llm
    return llm


def _create(
    *,
    model_name: str,
    fallback_model_name: str,
    temperature: float,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int | None,
    timeout: int | None,
    model_kwargs: dict[str, Any] | None,
) -> Any:
    if not _LANGCHAIN_AVAILABLE:
        logger.info("LangChain unavailable — using OfflineChatModel.")
        return OfflineChatModel("langchain not installed")

    # A local endpoint needs no key; a hosted one does.
    if not api_key and not base_url:
        logger.info("No API key or base_url — using OfflineChatModel.")
        return OfflineChatModel("no credentials configured")

    kwargs = _build_kwargs(temperature, base_url, api_key, max_tokens, timeout, model_kwargs)

    try:
        primary = _instantiate(model_name, **kwargs)
    except Exception as exc:
        logger.error("Primary model %s failed to initialise: %s", model_name, exc)
        return OfflineChatModel(f"primary init failed: {exc}")

    if not fallback_model_name or fallback_model_name == model_name:
        return primary

    try:
        secondary = _instantiate(fallback_model_name, **kwargs)
    except Exception as exc:
        logger.warning("Fallback model %s unavailable: %s", fallback_model_name, exc)
        return primary

    try:
        return primary.with_fallbacks([secondary], exceptions_to_handle=(Exception,))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not attach fallbacks: %s", exc)
        return primary


def is_offline(llm: Any) -> bool:
    """True when *llm* is the deterministic offline stand-in."""
    return bool(getattr(llm, "is_offline", False))


def describe_llm(llm: Any) -> str:
    """Human-readable one-liner for the UI status panel."""
    if is_offline(llm):
        return f"Offline deterministic mode ({getattr(llm, 'reason', 'unknown')})"
    inner = getattr(llm, "runnable", llm)
    name = getattr(inner, "model_name", None) or getattr(inner, "model", None)
    has_fallback = hasattr(llm, "fallbacks")
    suffix = f" → fallback: {settings.fallback_model_name}" if has_fallback else ""
    return f"{name or settings.model_name}{suffix}"


def reset_cache() -> None:
    """Drop cached model instances (used by the UI when settings change)."""
    with _cache_lock:
        _cache.clear()


__all__ = [
    "get_llm",
    "is_offline",
    "describe_llm",
    "reset_cache",
    "OfflineChatModel",
    "OfflineMessage",
]
