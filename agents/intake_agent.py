"""
agents/intake_agent.py — Conversational intake + structured entity extraction.

Two extraction paths, always run in this order:

1. **Deterministic extractor** (regex + vocabulary). Runs *in-process on the
   raw text*, so nothing has to leave the machine for basic entity capture.
   It is the authority for numeric fields.
2. **LLM extractor** (``with_structured_output``). Runs on the *sanitised*
   text only, and fills gaps the regex layer could not resolve — free-form
   phrasing, unusual vehicle names, implied values.

The LLM never overrides a confidently-parsed deterministic value; it only
supplies missing keys. That keeps a prompt-injected model from rewriting an
applicant's age or mileage.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agents.state import OnboardingState
from config import settings
from llm_factory import get_llm, is_offline
from security.validations import REQUIRED_FIELDS, normalise_payload

logger = logging.getLogger(__name__)

__all__ = ["ExtractedEntities", "IntakeAgent", "intake_agent"]


# --------------------------------------------------------------------------- #
# Extraction schema
# --------------------------------------------------------------------------- #
class ExtractedEntities(BaseModel):
    """Entities an intake turn may contain. Every field is optional."""

    model_config = ConfigDict(extra="ignore")

    full_name: str | None = Field(default=None, description="Applicant's full legal name")
    age: int | None = Field(default=None, description="Age in years")
    zip_code: str | None = Field(default=None, description="5-digit US ZIP or 6-digit Indian PIN")
    vehicle_year: int | None = Field(default=None, description="Model year of the vehicle")
    vehicle_make: str | None = Field(default=None, description="Manufacturer, e.g. Toyota")
    vehicle_model: str | None = Field(default=None, description="Model, e.g. Camry")
    vehicle_value: int | None = Field(default=None, description="Approximate vehicle value")
    annual_mileage: int | None = Field(default=None, description="Miles driven per year")
    years_licensed: int | None = Field(default=None, description="Years holding a driving licence")
    prior_claims: int | None = Field(default=None, description="Insurance claims in the last 5 years")
    occupation: str | None = None
    marital_status: str | None = Field(default=None, description="single, married, divorced or widowed")
    coverage_preference: str | None = Field(default=None, description="basic, balanced or comprehensive")


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
_MAKES = (
    "toyota", "honda", "ford", "chevrolet", "chevy", "nissan", "bmw", "mercedes",
    "mercedes-benz", "audi", "volkswagen", "vw", "volvo", "tesla", "hyundai", "kia",
    "mazda", "subaru", "lexus", "jeep", "dodge", "ram", "gmc", "buick", "cadillac",
    "porsche", "jaguar", "land rover", "mini", "fiat", "renault", "peugeot", "skoda",
    "maruti", "maruti suzuki", "suzuki", "tata", "mahindra", "kawasaki", "rivian",
    "lucid", "polestar", "byd", "mg", "citroen", "seat", "acura", "infiniti", "genesis",
)

_MARITAL = {
    "single": "single", "unmarried": "single", "bachelor": "single",
    "married": "married", "wed": "married",
    "divorced": "divorced", "separated": "divorced",
    "widow": "widowed", "widowed": "widowed", "widower": "widowed",
}

_COVERAGE = {
    "basic": "basic", "minimum": "basic", "cheapest": "basic", "budget": "basic",
    "minimal": "basic", "bare": "basic", "liability only": "basic",
    "balanced": "balanced", "standard": "balanced", "middle": "balanced",
    "moderate": "balanced", "normal": "balanced",
    "comprehensive": "comprehensive", "full": "comprehensive", "maximum": "comprehensive",
    "premium": "comprehensive", "best": "comprehensive", "everything": "comprehensive",
}

_CURRENT_YEAR = settings.max_vehicle_year - 1


# --------------------------------------------------------------------------- #
# Deterministic extractor
# --------------------------------------------------------------------------- #
def _num(text: str) -> int | None:
    try:
        return int(re.sub(r"[,_\s]", "", text))
    except (TypeError, ValueError):
        return None


def extract_deterministic(text: str) -> dict[str, Any]:
    """Regex + vocabulary extraction. Conservative: only high-confidence hits."""
    if not text:
        return {}

    lowered = text.lower()
    found: dict[str, Any] = {}

    # ---- age ------------------------------------------------------------- #
    for pattern in (
        r"\b(?:i(?:'m| am)|age(?:d)?(?:\s+is)?|i'?m)\s*:?\s*(\d{1,3})\b(?!\s*(?:%|miles|km|k\b))",
        r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y/?o)\b",
        r"\bage\s*[:=]\s*(\d{1,3})\b",
    ):
        match = re.search(pattern, lowered)
        if match:
            value = _num(match.group(1))
            if value is not None and 0 < value <= 120:
                found["age"] = value
                break

    # ---- postal code ------------------------------------------------------ #
    # `USER_ZIP_90210` is the PII-firewall token; accept it as well as raw codes.
    token = re.search(r"user_zip_(\d{5,6})", lowered)
    if token:
        found["zip_code"] = token.group(1)
    else:
        match = re.search(
            r"\b(?:zip|zipcode|zip code|pin|pincode|pin code|postal(?:\s*code)?)\s*"
            r"(?:is|:|=)?\s*(\d{5,6})\b",
            lowered,
        )
        if match:
            found["zip_code"] = match.group(1)
        else:
            # A bare 5/6-digit run that is not a year and not part of a mileage phrase.
            for candidate in re.finditer(r"(?<!\d)(\d{5,6})(?!\d)", lowered):
                value = candidate.group(1)
                window = lowered[max(0, candidate.start() - 18) : candidate.end() + 12]
                if re.search(r"mile|km|mileage|value|worth|\$|premium|cost", window):
                    continue
                found["zip_code"] = value
                break

    # ---- vehicle ---------------------------------------------------------- #
    make_pattern = "|".join(sorted((re.escape(m) for m in _MAKES), key=len, reverse=True))

    # "2021 Toyota Camry" / "Toyota Camry 2021"
    combo = re.search(rf"\b(19[89]\d|20[0-4]\d)\s+({make_pattern})\b\s*([a-z0-9-]+)?", lowered)
    if not combo:
        combo = re.search(rf"\b({make_pattern})\b\s*([a-z0-9-]+)?\s*(19[89]\d|20[0-4]\d)\b", lowered)
        if combo:
            found["vehicle_make"] = combo.group(1)
            if combo.group(2) and combo.group(2) not in {"car", "vehicle", "and", "the"}:
                found["vehicle_model"] = combo.group(2)
            found["vehicle_year"] = _num(combo.group(3))
    else:
        found["vehicle_year"] = _num(combo.group(1))
        found["vehicle_make"] = combo.group(2)
        if combo.group(3) and combo.group(3) not in {"car", "vehicle", "and", "the"}:
            found["vehicle_model"] = combo.group(3)

    if "vehicle_make" not in found:
        match = re.search(rf"\b({make_pattern})\b", lowered)
        if match:
            found["vehicle_make"] = match.group(1)

    if "vehicle_year" not in found:
        match = re.search(
            r"\b(?:year|model year|from|a)\s*[:=]?\s*(19[89]\d|20[0-4]\d)\b", lowered
        ) or re.search(r"\b(19[89]\d|20[0-4]\d)\b(?=\s|$|,|\.)", lowered)
        if match:
            year = _num(match.group(1))
            if year and year != found.get("age"):
                found["vehicle_year"] = year

    # ---- mileage ---------------------------------------------------------- #
    # Runs *after* vehicle parsing so a model year can be disqualified as a
    # mileage candidate: "I drive a 2021 Toyota" must not read as 2,021 miles.
    def _parse_mileage(raw: str) -> int | None:
        raw = raw.strip()
        if raw.endswith("k"):
            try:
                return int(float(raw[:-1].strip()) * 1000)
            except ValueError:
                return None
        return _num(raw)

    # Pass 1 — an explicit distance unit is unambiguous.
    match = re.search(r"\b(\d{1,3}(?:[,\s]?\d{3})*|\d+(?:\.\d+)?\s*k)\s*(?:miles?|mi|km)\b", lowered)
    if match:
        value = _parse_mileage(match.group(1))
        if value is not None:
            found["annual_mileage"] = value
    else:
        # Pass 2 — a context word, with year-shaped numbers rejected.
        for candidate in re.finditer(
            r"\b(?:mileage|drive|driving|travel|commute|cover)\D{0,18}?"
            r"(\d{1,3}(?:[,\s]?\d{3})*|\d+\s*k)\b",
            lowered,
        ):
            value = _parse_mileage(candidate.group(1))
            if value is None:
                continue
            if 1900 <= value <= 2049 or value == found.get("vehicle_year"):
                continue
            found["annual_mileage"] = value
            break

    # ---- licence tenure --------------------------------------------------- #
    match = re.search(
        r"\b(?:licen[cs]ed|driving|driven|had (?:my|a) licen[cs]e)\D{0,20}?(\d{1,2})\s*(?:years?|yrs?)\b",
        lowered,
    ) or re.search(r"\b(\d{1,2})\s*(?:years?|yrs?)\D{0,15}?(?:licen[cs]e|driving experience)\b", lowered)
    if match:
        found["years_licensed"] = _num(match.group(1))

    # ---- claims ------------------------------------------------------------ #
    if re.search(r"\b(?:no|zero|never had (?:any)?|0)\s*(?:prior\s*)?(?:claims?|accidents?)\b", lowered):
        found["prior_claims"] = 0
    else:
        match = re.search(r"\b(\d{1,2})\s*(?:prior\s*)?(?:claims?|accidents?)\b", lowered)
        if match:
            found["prior_claims"] = _num(match.group(1))

    # ---- vehicle value ------------------------------------------------------ #
    match = re.search(r"(?:worth|value[d]?(?:\s+at)?|costs?)\D{0,10}\$?\s*(\d[\d,]{2,9})\b", lowered)
    if match:
        found["vehicle_value"] = _num(match.group(1))

    # ---- categorical -------------------------------------------------------- #
    for keyword, value in _MARITAL.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            found["marital_status"] = value
            break

    for keyword, value in _COVERAGE.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            found["coverage_preference"] = value
            break

    # ---- name --------------------------------------------------------------- #
    # The lead-in is case-insensitive ("My name is"), but the captured name must
    # keep its original capitalisation so it can be distinguished from ordinary
    # words — hence a scoped inline flag rather than re.IGNORECASE on the whole.
    match = re.search(
        r"(?i:\b(?:my name is|i am|i'm|this is|name\s*[:=]))\s+"
        r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})\b",
        text,
    )
    if match:
        candidate = match.group(1).strip()
        if not re.search(r"\d", candidate) and candidate.lower() not in {"looking", "interested"}:
            found["full_name"] = candidate

    return normalise_payload(found)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_EXTRACTION_SYSTEM = """You extract structured insurance-application data.

Return ONLY fields explicitly stated or unambiguously implied by the user's \
message. Never guess, never infer from stereotypes, and leave anything unstated \
as null.

Notes:
- `USER_ZIP_90210` is a redacted postal-code token; the digits are the code.
- Tokens like [PHONE_REDACTED] or [AADHAAR_REDACTED] are removed identifiers — \
ignore them entirely.
- Mileage is per year. "12k" means 12000.
- coverage_preference must be exactly one of: basic, balanced, comprehensive."""

_QUESTION_TEMPLATES: dict[str, str] = {
    "full_name": "your full name as it appears on your licence",
    "age": "your age",
    "zip_code": "your ZIP or PIN code",
    "vehicle_year": "the model year of your vehicle",
    "vehicle_make": "the make of your vehicle (e.g. Toyota)",
    "annual_mileage": "roughly how many miles you drive per year",
}

_FIELD_ORDER = ("full_name", "age", "zip_code", "vehicle_make", "vehicle_year", "annual_mileage")


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class IntakeAgent:
    """Collects and structures applicant details across conversational turns."""

    name = "Intake Agent"

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)
        return self._llm

    # ------------------------------------------------------------------ #
    def extract(self, raw_text: str, sanitized_text: str | None = None) -> dict[str, Any]:
        """
        Merge deterministic and LLM extraction.

        Deterministic values win on conflict — the regex layer sees trusted
        local text, while the LLM sees sanitised text and could be influenced
        by injected content.
        """
        found = extract_deterministic(raw_text)

        llm = self.llm
        if is_offline(llm):
            return found

        missing = [f for f in ExtractedEntities.model_fields if f not in found]
        if not missing:
            return found

        try:
            structured = llm.with_structured_output(ExtractedEntities)
            result = structured.invoke(
                [
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": sanitized_text or raw_text},
                ]
            )
        except Exception as exc:
            logger.warning("LLM extraction failed (%s) — deterministic only.", exc)
            return found

        if result is None:
            return found

        payload = result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else {}
        for key, value in payload.items():
            if key not in found and value not in (None, "", []):
                found[key] = value

        return normalise_payload(found)

    # ------------------------------------------------------------------ #
    def update_state(self, state: OnboardingState, raw_text: str, sanitized_text: str) -> dict[str, Any]:
        """Extract from a turn and merge into ``state.collected``."""
        found = self.extract(raw_text, sanitized_text)
        if found:
            state.collected.update(found)
            state.log("intake_extracted", fields=sorted(found.keys()))
        return found

    # ------------------------------------------------------------------ #
    def next_question(self, state: OnboardingState) -> str:
        """Ask for the highest-priority missing field, acknowledging progress."""
        missing = [f for f in _FIELD_ORDER if f not in state.collected]
        missing += [f for f in REQUIRED_FIELDS if f not in state.collected and f not in missing]

        if not missing:
            return "Thanks — I have everything I need. Let me validate your details."

        target = missing[0]
        question = _QUESTION_TEMPLATES.get(target, f"your {target.replace('_', ' ')}")

        captured = len(REQUIRED_FIELDS) - len([f for f in REQUIRED_FIELDS if f not in state.collected])
        if captured == 0:
            return f"To get started, could you tell me {question}?"
        if len(missing) == 1:
            return f"Almost there — last thing: what is {question}?"
        return f"Got it. Next, could you tell me {question}?"

    # ------------------------------------------------------------------ #
    def acknowledge(self, found: dict[str, Any]) -> str:
        """Short confirmation of what was just understood."""
        if not found:
            return ""
        labels = {
            "full_name": "name",
            "age": "age",
            "zip_code": "location",
            "vehicle_year": "vehicle year",
            "vehicle_make": "vehicle make",
            "vehicle_model": "model",
            "annual_mileage": "annual mileage",
            "years_licensed": "licence history",
            "prior_claims": "claims history",
            "marital_status": "marital status",
            "coverage_preference": "coverage preference",
            "vehicle_value": "vehicle value",
            "occupation": "occupation",
        }
        captured = [labels.get(k, k.replace("_", " ")) for k in found]
        if len(captured) == 1:
            return f"Noted your {captured[0]}."
        return f"Noted your {', '.join(captured[:-1])} and {captured[-1]}."


intake_agent = IntakeAgent()
