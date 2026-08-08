"""
security/validations.py — Deterministic schema + business-rule enforcement.

This is the *hard* gate. Nothing reaches underwriting, the risk engine, or the
policy issuer until it has survived this module. An LLM may propose values;
only this file decides whether they are admissible.

Two enforcement layers
----------------------
1. **Schema layer** (Pydantic v2) — types, ranges, formats. Structurally
   impossible values cannot exist in an :class:`ApplicantProfile` instance.
2. **Anomaly layer** — cross-field plausibility checks that are individually
   valid but jointly suspicious (e.g. a 19-year-old with 25 years of driving
   history, or 96,000 annual miles on a personal auto policy).

The public entry point :func:`validate_profile` never raises: it returns a
:class:`ValidationReport` describing exactly what must be re-entered.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from config import settings

__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "ApplicantProfile",
    "AadhaarRequest",
    "OTPRequest",
    "validate_profile",
    "normalise_payload",
    "REQUIRED_FIELDS",
]

CURRENT_YEAR = date.today().year

REQUIRED_FIELDS: tuple[str, ...] = (
    "full_name",
    "age",
    "zip_code",
    "vehicle_year",
    "vehicle_make",
    "annual_mileage",
)


# --------------------------------------------------------------------------- #
# Issue reporting
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    ERROR = "error"      # schema violation — cannot proceed
    ANOMALY = "anomaly"  # plausible-but-suspicious — re-entry required
    WARNING = "warning"  # informational — proceeds, surfaced to underwriter


class ValidationIssue(BaseModel):
    """A single actionable finding."""

    model_config = ConfigDict(frozen=True)

    field_name: str
    code: str
    message: str
    severity: Severity = Severity.ERROR
    received: Any = None
    remediation: str | None = None

    def render(self) -> str:
        icon = {Severity.ERROR: "❌", Severity.ANOMALY: "⚠️", Severity.WARNING: "ℹ️"}[self.severity]
        text = f"{icon} **{self.field_name}** — {self.message}"
        if self.remediation:
            text += f" _{self.remediation}_"
        return text


class ValidationReport(BaseModel):
    """Aggregate outcome of a validation pass."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    issues: list[ValidationIssue] = Field(default_factory=list)
    profile: "ApplicantProfile | None" = None
    missing_fields: list[str] = Field(default_factory=list)

    # -- severity views --------------------------------------------------- #
    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def anomalies(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ANOMALY]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def blocking(self) -> list[ValidationIssue]:
        """Errors *and* anomalies both force a re-entry loop."""
        return self.errors + self.anomalies

    @property
    def is_valid(self) -> bool:
        """True only when a profile was built and nothing blocking remains."""
        return self.profile is not None and not self.blocking and not self.missing_fields

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields

    def render(self) -> str:
        if self.is_valid:
            return "✅ All checks passed."
        lines = [issue.render() for issue in self.blocking]
        if self.missing_fields:
            lines.append("📝 Still needed: " + ", ".join(f"`{f}`" for f in self.missing_fields))
        return "\n\n".join(lines) if lines else "✅ All checks passed."

    def add(self, **kwargs: Any) -> None:
        self.issues.append(ValidationIssue(**kwargs))


# --------------------------------------------------------------------------- #
# Coercion helpers — tolerate human phrasing without weakening the schema
# --------------------------------------------------------------------------- #
_INT_RE = re.compile(r"-?\d[\d,_ ]*")


def _to_int(value: Any) -> Any:
    """'12,000 miles' -> 12000. Returns the original on failure so Pydantic reports it."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        match = _INT_RE.search(value.replace("k", "000").replace("K", "000"))
        if match:
            try:
                return int(re.sub(r"[,_ ]", "", match.group(0)))
            except ValueError:
                return value
    return value


def normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce loose/LLM-extracted values into schema-ready primitives."""
    out = dict(payload or {})
    for key in ("age", "vehicle_year", "annual_mileage", "years_licensed", "prior_claims"):
        if key in out:
            out[key] = _to_int(out[key])
    for key in ("full_name", "vehicle_make", "vehicle_model", "zip_code", "occupation"):
        if key in out and isinstance(out[key], str):
            out[key] = out[key].strip()
    if isinstance(out.get("zip_code"), int):
        out["zip_code"] = str(out["zip_code"])
    # Drop keys that are present-but-empty so they register as "missing"
    # rather than as a type error.
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


# --------------------------------------------------------------------------- #
# Core schema
# --------------------------------------------------------------------------- #
class ApplicantProfile(BaseModel):
    """
    Strictly-typed applicant record.

    An instance of this class is, by construction, admissible to underwriting.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    # -- identity ---------------------------------------------------------- #
    full_name: str = Field(min_length=2, max_length=80)
    age: int = Field(ge=settings.min_age, le=settings.max_age)
    occupation: str | None = Field(default=None, max_length=60)
    marital_status: Literal["single", "married", "divorced", "widowed"] | None = None

    # -- location ---------------------------------------------------------- #
    zip_code: str = Field(min_length=5, max_length=10)
    region: Literal["US", "IN"] = "US"

    # -- vehicle ----------------------------------------------------------- #
    vehicle_year: int = Field(ge=settings.min_vehicle_year, le=settings.max_vehicle_year)
    vehicle_make: str = Field(min_length=1, max_length=40)
    vehicle_model: str | None = Field(default=None, max_length=40)
    vehicle_value: int | None = Field(default=None, ge=500, le=5_000_000)

    # -- usage / history --------------------------------------------------- #
    annual_mileage: int = Field(ge=settings.min_annual_mileage, le=settings.max_annual_mileage)
    years_licensed: int | None = Field(default=None, ge=0, le=85)
    prior_claims: int = Field(default=0, ge=0, le=20)
    coverage_preference: Literal["basic", "balanced", "comprehensive"] = "balanced"

    # ---------------------- field validators ------------------------------ #
    @field_validator("full_name")
    @classmethod
    def _name_shape(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-. ]{1,79}", v):
            raise ValueError("must contain only letters, spaces, apostrophes, hyphens or periods")
        return " ".join(part.capitalize() for part in v.split())

    @field_validator("zip_code")
    @classmethod
    def _zip_shape(cls, v: str) -> str:
        v = v.strip().replace(" ", "")
        if re.fullmatch(r"\d{5}(-\d{4})?", v):          # US ZIP / ZIP+4
            return v
        if re.fullmatch(r"[1-9]\d{5}", v):              # Indian PIN
            return v
        raise ValueError("must be a 5-digit US ZIP (or ZIP+4) or a 6-digit Indian PIN")

    @field_validator("vehicle_make", "vehicle_model")
    @classmethod
    def _vehicle_text(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-. ]{0,39}", v):
            raise ValueError("contains unsupported characters")
        return v.strip().title()

    # ---------------------- cross-field derivation ------------------------ #
    @model_validator(mode="after")
    def _derive_region(self) -> "ApplicantProfile":
        # `region` is derived from ZIP shape, not trusted from input.
        object.__setattr__(self, "region", "IN" if len(self.zip_code) == 6 else "US")
        return self

    # ---------------------- convenience ----------------------------------- #
    @property
    def vehicle_age(self) -> int:
        return max(0, CURRENT_YEAR - self.vehicle_year)

    @property
    def display_vehicle(self) -> str:
        parts = [str(self.vehicle_year), self.vehicle_make]
        if self.vehicle_model:
            parts.append(self.vehicle_model)
        return " ".join(parts)

    def underwriting_payload(self) -> dict[str, Any]:
        """PII-free projection safe to hand to underwriting tools and prompts."""
        return {
            "age": self.age,
            "region": self.region,
            "zip_token": f"USER_ZIP_{self.zip_code}",
            "vehicle_year": self.vehicle_year,
            "vehicle_age": self.vehicle_age,
            "vehicle_make": self.vehicle_make,
            "annual_mileage": self.annual_mileage,
            "years_licensed": self.years_licensed,
            "prior_claims": self.prior_claims,
            "coverage_preference": self.coverage_preference,
        }


ValidationReport.model_rebuild()


# --------------------------------------------------------------------------- #
# KYC schemas
# --------------------------------------------------------------------------- #
class AadhaarRequest(BaseModel):
    """12-digit Aadhaar, format-enforced before it ever hits the KYC backend."""

    model_config = ConfigDict(str_strip_whitespace=True)

    aadhaar_number: str

    @field_validator("aadhaar_number")
    @classmethod
    def _shape(cls, v: str) -> str:
        digits = re.sub(r"[\s-]", "", v or "")
        if not digits.isdigit():
            raise ValueError("Aadhaar must contain digits only")
        if len(digits) != settings.aadhaar_length:
            raise ValueError(f"Aadhaar must be exactly {settings.aadhaar_length} digits")
        if digits[0] in "01":
            raise ValueError("Aadhaar cannot begin with 0 or 1")
        return digits


class OTPRequest(BaseModel):
    """6-digit numeric OTP."""

    model_config = ConfigDict(str_strip_whitespace=True)

    txn_id: str = Field(min_length=6, max_length=64)
    otp: str

    @field_validator("otp")
    @classmethod
    def _shape(cls, v: str) -> str:
        digits = re.sub(r"[\s-]", "", v or "")
        if not re.fullmatch(r"\d{6}", digits):
            raise ValueError("OTP must be exactly 6 digits")
        return digits


# --------------------------------------------------------------------------- #
# Anomaly layer
# --------------------------------------------------------------------------- #
_FIELD_GUIDANCE: dict[str, str] = {
    "age": f"Enter an age between {settings.min_age} and {settings.max_age}.",
    "vehicle_year": f"Enter a year between {settings.min_vehicle_year} and {settings.max_vehicle_year}.",
    "annual_mileage": (
        f"Enter mileage between {settings.min_annual_mileage:,} and "
        f"{settings.max_annual_mileage:,} miles."
    ),
    "zip_code": "Enter a 5-digit US ZIP or a 6-digit Indian PIN code.",
    "full_name": "Enter your full legal name as it appears on your ID.",
    "vehicle_make": "Enter the manufacturer, e.g. Toyota or Maruti Suzuki.",
}


def _detect_anomalies(profile: ApplicantProfile, report: ValidationReport) -> None:
    """Cross-field plausibility checks on an already schema-valid profile."""

    # Driving history cannot predate legal licensing age (~15 at the earliest).
    if profile.years_licensed is not None and profile.years_licensed > profile.age - 15:
        report.add(
            field_name="years_licensed",
            code="LICENSE_HISTORY_EXCEEDS_AGE",
            message=(
                f"{profile.years_licensed} years of licensed driving is impossible "
                f"at age {profile.age}."
            ),
            severity=Severity.ANOMALY,
            received=profile.years_licensed,
            remediation=f"Maximum plausible value is {max(0, profile.age - 15)}.",
        )

    # Personal-auto mileage ceiling — beyond this it is a commercial exposure.
    if profile.annual_mileage > 50_000:
        report.add(
            field_name="annual_mileage",
            code="MILEAGE_COMMERCIAL_RANGE",
            message=(
                f"{profile.annual_mileage:,} miles/year exceeds the personal-auto "
                "threshold of 50,000 and indicates commercial use."
            ),
            severity=Severity.ANOMALY,
            received=profile.annual_mileage,
            remediation="Confirm the figure, or apply for a commercial policy instead.",
        )

    # Vintage vehicles are in-range but need a specialty endorsement.
    if profile.vehicle_age >= 25:
        report.add(
            field_name="vehicle_year",
            code="VINTAGE_VEHICLE",
            message=f"A {profile.vehicle_age}-year-old vehicle may require classic-car cover.",
            severity=Severity.WARNING,
            received=profile.vehicle_year,
        )

    # Claims frequency relative to exposure.
    if profile.prior_claims >= 4:
        report.add(
            field_name="prior_claims",
            code="HIGH_CLAIM_FREQUENCY",
            message=f"{profile.prior_claims} prior claims materially restricts eligibility.",
            severity=Severity.WARNING,
            received=profile.prior_claims,
        )

    # Very low mileage on a new vehicle — often a typo (1,200 vs 12,000).
    if profile.annual_mileage < 1_000 and profile.vehicle_age <= 3:
        report.add(
            field_name="annual_mileage",
            code="IMPLAUSIBLY_LOW_MILEAGE",
            message=(
                f"{profile.annual_mileage:,} miles/year on a {profile.vehicle_age}-year-old "
                "vehicle is unusually low."
            ),
            severity=Severity.ANOMALY,
            received=profile.annual_mileage,
            remediation="Confirm the figure — a common typo is dropping a trailing zero.",
        )

    # Youngest drivers on high-value vehicles are a known adverse-selection signal.
    if profile.age < 21 and (profile.vehicle_value or 0) > 60_000:
        report.add(
            field_name="vehicle_value",
            code="YOUNG_DRIVER_HIGH_VALUE",
            message="High-value vehicle for a driver under 21 requires manual underwriting.",
            severity=Severity.WARNING,
            received=profile.vehicle_value,
        )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def validate_profile(payload: dict[str, Any]) -> ValidationReport:
    """
    Validate a (possibly partial, possibly LLM-extracted) applicant payload.

    Never raises. Returns a :class:`ValidationReport` where
    ``is_valid`` is ``True`` only if the payload is complete, schema-clean and
    anomaly-free.
    """
    report = ValidationReport()
    data = normalise_payload(payload)

    # 1. Completeness — report missing fields without treating them as errors.
    report.missing_fields = [f for f in REQUIRED_FIELDS if f not in data]
    if report.missing_fields:
        return report

    # 2. Schema layer.
    try:
        profile = ApplicantProfile(**data)
    except ValidationError as exc:
        for err in exc.errors():
            field_name = str(err["loc"][0]) if err["loc"] else "payload"
            report.add(
                field_name=field_name,
                code=err["type"].upper(),
                message=_humanise(field_name, err),
                severity=Severity.ERROR,
                received=err.get("input"),
                remediation=_FIELD_GUIDANCE.get(field_name),
            )
        return report

    # 3. Anomaly layer.
    report.profile = profile
    _detect_anomalies(profile, report)
    return report


def _humanise(field_name: str, err: dict[str, Any]) -> str:
    """Turn a Pydantic error dict into copy an applicant can act on."""
    etype = err["type"]
    got = err.get("input")
    if etype in {"greater_than_equal", "less_than_equal"}:
        bounds = {
            "age": (settings.min_age, settings.max_age),
            "vehicle_year": (settings.min_vehicle_year, settings.max_vehicle_year),
            "annual_mileage": (settings.min_annual_mileage, settings.max_annual_mileage),
        }.get(field_name)
        if bounds:
            # Years are never thousands-separated; counts always are.
            fmt = (lambda n: str(n)) if field_name == "vehicle_year" else (lambda n: f"{n:,}")
            return f"{got!r} is outside the permitted range {fmt(bounds[0])}–{fmt(bounds[1])}."
    if etype in {"int_parsing", "int_type"}:
        return f"{got!r} is not a whole number."
    if etype == "string_too_short":
        return f"{got!r} is too short."
    if etype == "literal_error":
        return f"{got!r} is not one of the accepted options ({err.get('ctx', {}).get('expected', '')})."
    if etype == "missing":
        return "This field is required."
    return err.get("msg", "Invalid value.").replace("Value error, ", "")
