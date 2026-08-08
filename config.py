"""
config.py — Central system configuration.

Every tunable in the platform is resolved here, in this order:
    1. Explicit environment variable
    2. `.env` file (if python-dotenv is installed)
    3. Hard-coded safe default

The application is designed to run end-to-end with *zero* credentials
(deterministic offline mode). Supplying an API key upgrades the intake /
Q&A agents from rule-based extraction to true LLM reasoning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# .env loading (optional dependency)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
GUARDRAILS_DIR: Final[Path] = BASE_DIR / "security" / "guardrails"
VECTOR_DIR: Final[Path] = BASE_DIR / ".chroma"


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key, default)
    if value is not None:
        value = value.strip()
    return value or None


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable, process-wide configuration object."""

    # ---------------- LLM ---------------- #
    model_name: str = _env("LLM_MODEL", "openai:gpt-4o-mini")
    fallback_model_name: str = _env("LLM_FALLBACK_MODEL", "openai:gpt-4o")
    temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 1024)
    request_timeout: int = _env_int("LLM_TIMEOUT_SECONDS", 45)
    base_url: str | None = _env("LLM_BASE_URL")  # vLLM / LM Studio / Ollama proxy
    api_key: str | None = _env("OPENAI_API_KEY") or _env("LLM_API_KEY")

    # ---------------- Security ---------------- #
    enable_guardrails: bool = _env_bool("ENABLE_GUARDRAILS", True)
    enable_pii_firewall: bool = _env_bool("ENABLE_PII_FIREWALL", True)
    strict_output_rails: bool = _env_bool("STRICT_OUTPUT_RAILS", True)
    guardrails_path: Path = GUARDRAILS_DIR

    # ---------------- RAG / Vector store ---------------- #
    #   "auto"   -> ChromaDB when importable, else deterministic in-memory index
    #   "chroma" -> force ChromaDB (raises if unavailable)
    #   "memory" -> force the dependency-free in-memory index
    vector_backend: str = _env("VECTOR_BACKEND", "auto")
    vector_persist_dir: Path = VECTOR_DIR
    collection_name: str = _env("VECTOR_COLLECTION", "insurance_policies")
    rag_top_k: int = _env_int("RAG_TOP_K", 5)
    recommendation_count: int = _env_int("RECOMMENDATION_COUNT", 3)

    # ---------------- Business rules ---------------- #
    min_age: int = 18
    max_age: int = 100
    min_vehicle_year: int = 1990
    max_vehicle_year: int = field(default_factory=lambda: date.today().year + 1)
    min_annual_mileage: int = 100
    max_annual_mileage: int = 100_000

    # ---------------- KYC ---------------- #
    aadhaar_length: int = 12
    mock_otp: str = _env("MOCK_OTP", "123456")
    otp_ttl_seconds: int = _env_int("OTP_TTL_SECONDS", 300)
    max_otp_attempts: int = _env_int("MAX_OTP_ATTEMPTS", 3)

    # ---------------- Pricing engine ---------------- #
    currency_symbol: str = _env("CURRENCY_SYMBOL", "$")
    deductible_options: tuple[int, ...] = (250, 500, 1000, 2000)
    claim_scenarios: tuple[int, ...] = (3_000, 18_000, 35_000)

    # ---------------- Runtime ---------------- #
    app_title: str = "AegisAI — Insurance Onboarding & Risk Intelligence"
    debug: bool = _env_bool("APP_DEBUG", False)
    mock_api_latency_ms: int = _env_int("MOCK_API_LATENCY_MS", 0)

    # ---------------- Derived helpers ---------------- #
    @property
    def llm_enabled(self) -> bool:
        """True when a live model can plausibly be reached."""
        return bool(self.api_key or self.base_url)

    @property
    def risk_tier_bands(self) -> tuple[tuple[int, str, str], ...]:
        """(minimum score, tier label, class code) — higher score == safer."""
        return (
            (85, "Low Risk", "A+"),
            (70, "Preferred", "A"),
            (55, "Standard", "B"),
            (40, "Elevated Risk", "C"),
            (0, "High Risk", "D"),
        )


settings: Final[Settings] = Settings()


def tier_for_score(score: float) -> tuple[str, str]:
    """Map a 0-100 safety score onto ``(tier_label, class_code)``."""
    for threshold, label, code in settings.risk_tier_bands:
        if score >= threshold:
            return label, code
    return "High Risk", "D"


__all__ = ["Settings", "settings", "tier_for_score", "BASE_DIR", "GUARDRAILS_DIR"]
