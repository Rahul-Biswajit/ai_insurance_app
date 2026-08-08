"""
backend/mock_apis.py — Deterministic mock service layer.

Simulates six upstream systems an insurer would really integrate with::

    GET  /api/v1/crm/customer              -> get_crm_customer()
    POST /api/v1/underwriting/mvr-check    -> mvr_check()
    POST /api/v1/underwriting/geo-hazard   -> geo_hazard_check()
    POST /api/v1/kyc/aadhaar/request-otp   -> request_aadhaar_otp()
    POST /api/v1/kyc/aadhaar/verify-otp    -> verify_aadhaar_otp()
    POST /api/v1/policies/issue            -> issue_policy()

Determinism
-----------
Every response is derived from a SHA-256 seed of the request payload, so the
same applicant always produces the same MVR record and the same geo-hazard
profile across runs, processes and machines. (Python's builtin ``hash()`` is
salted per process and is deliberately *not* used.)

Each function returns an :class:`ApiResponse` envelope mirroring a real HTTP
result — status code, body, latency and a request id — so swapping these for
``httpx`` calls later is a body-for-body substitution. Async variants are
provided for every endpoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "ApiResponse",
    "get_crm_customer",
    "mvr_check",
    "geo_hazard_check",
    "request_aadhaar_otp",
    "verify_aadhaar_otp",
    "issue_policy",
    "reset_kyc_store",
    "aget_crm_customer",
    "amvr_check",
    "ageo_hazard_check",
    "arequest_aadhaar_otp",
    "averify_aadhaar_otp",
    "aissue_policy",
]


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #
@dataclass
class ApiResponse:
    """HTTP-shaped result envelope."""

    status_code: int
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    endpoint: str = ""
    latency_ms: float = 0.0
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def unwrap(self) -> dict[str, Any]:
        """Return the body, raising on a non-2xx status."""
        if not self.ok:
            raise RuntimeError(f"{self.endpoint} failed [{self.status_code}]: {self.error}")
        return self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def _seed(*parts: Any) -> int:
    """Stable cross-process integer seed from arbitrary inputs."""
    payload = "|".join(str(p).strip().lower() for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _spread(seed: int, offset: int, low: int, high: int) -> int:
    """
    Deterministically pick an integer in ``[low, high]`` from *seed*.

    The offset is mixed through SHA-256 rather than used as a bit-shift:
    shifting one seed by nearby offsets and taking a small modulus produces
    correlated draws (four "random" violations all landing on the same type),
    whereas re-hashing decorrelates every stream.
    """
    if high <= low:
        return low
    mixed = int(hashlib.sha256(f"{seed}:{offset}".encode()).hexdigest()[:12], 16)
    return low + (mixed % (high - low + 1))


def _latency() -> None:
    if settings.mock_api_latency_ms > 0:
        time.sleep(settings.mock_api_latency_ms / 1000.0)


def _timed(endpoint: str):
    """Decorator adding latency simulation, timing and uniform error capture."""

    def wrapper(fn):
        def inner(*args: Any, **kwargs: Any) -> ApiResponse:
            start = time.perf_counter()
            _latency()
            try:
                response = fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("%s raised", endpoint)
                response = ApiResponse(status_code=500, error=f"Internal error: {exc}")
            response.endpoint = endpoint
            response.latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return response

        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner

    return wrapper


# =========================================================================== #
# 1. CRM  —  GET /api/v1/crm/customer
# =========================================================================== #
_CRM_RECORDS: dict[str, dict[str, Any]] = {
    "CUST-1001": {
        "customer_id": "CUST-1001",
        "full_name": "Ananya Sharma",
        "age": 34,
        "zip_code": "560001",
        "vehicle_year": 2021,
        "vehicle_make": "Toyota",
        "vehicle_model": "Camry",
        "vehicle_value": 32000,
        "annual_mileage": 11000,
        "years_licensed": 13,
        "prior_claims": 0,
        "occupation": "Software Architect",
        "marital_status": "married",
        "coverage_preference": "comprehensive",
        "tenure_years": 4,
        "loyalty_tier": "Gold",
    },
    "CUST-1002": {
        "customer_id": "CUST-1002",
        "full_name": "Marcus Bell",
        "age": 23,
        "zip_code": "90210",
        "vehicle_year": 2018,
        "vehicle_make": "Honda",
        "vehicle_model": "Civic",
        "vehicle_value": 17500,
        "annual_mileage": 19500,
        "years_licensed": 5,
        "prior_claims": 2,
        "occupation": "Delivery Driver",
        "marital_status": "single",
        "coverage_preference": "basic",
        "tenure_years": 1,
        "loyalty_tier": "Standard",
    },
    "CUST-1003": {
        "customer_id": "CUST-1003",
        "full_name": "Diane Okafor",
        "age": 57,
        "zip_code": "10001",
        "vehicle_year": 2023,
        "vehicle_make": "Volvo",
        "vehicle_model": "Xc60",
        "vehicle_value": 54000,
        "annual_mileage": 7200,
        "years_licensed": 36,
        "prior_claims": 0,
        "occupation": "Cardiologist",
        "marital_status": "married",
        "coverage_preference": "comprehensive",
        "tenure_years": 11,
        "loyalty_tier": "Platinum",
    },
}


@_timed("GET /api/v1/crm/customer")
def get_crm_customer(customer_id: str) -> ApiResponse:
    """Pre-fill an existing customer profile from the CRM of record."""
    key = (customer_id or "").strip().upper()
    if not key:
        return ApiResponse(status_code=400, error="customer_id is required")

    record = _CRM_RECORDS.get(key)
    if record is None:
        return ApiResponse(status_code=404, error=f"No CRM record for {key}")

    return ApiResponse(
        status_code=200,
        data={
            **record,
            "source": "crm",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def list_crm_customers() -> list[dict[str, str]]:
    """Directory of demo profiles for the UI picker."""
    return [
        {
            "customer_id": rec["customer_id"],
            "label": f"{rec['customer_id']} — {rec['full_name']}, {rec['age']}, "
            f"{rec['vehicle_year']} {rec['vehicle_make']}",
        }
        for rec in _CRM_RECORDS.values()
    ]


# =========================================================================== #
# 2. Underwriting  —  POST /api/v1/underwriting/mvr-check
# =========================================================================== #
_VIOLATION_TYPES = (
    "Speeding 1-15 over",
    "Speeding 16-25 over",
    "Failure to yield",
    "Improper lane change",
    "Running red light",
    "Distracted driving",
)


@_timed("POST /api/v1/underwriting/mvr-check")
def mvr_check(
    subject_key: str,
    age: int,
    years_licensed: int | None = None,
    declared_claims: int = 0,
) -> ApiResponse:
    """
    Motor Vehicle Record lookup.

    Returns accident/violation counts and a licence standing. The declared
    claim count anchors the record so the mock stays coherent with what the
    applicant told us, while the DMV may still surface *undeclared* events —
    exactly the discrepancy real underwriting looks for.
    """
    if age is None or age < 0:
        return ApiResponse(status_code=422, error="A valid age is required")

    seed = _seed(subject_key, age, years_licensed or 0)
    exposure = max(1, (years_licensed if years_licensed is not None else max(1, age - 18)))

    # Younger, less-experienced drivers carry more DMV events.
    if age < 25:
        accident_ceiling, violation_ceiling = 3, 4
    elif age < 40:
        accident_ceiling, violation_ceiling = 2, 3
    elif age < 65:
        accident_ceiling, violation_ceiling = 1, 2
    else:
        accident_ceiling, violation_ceiling = 2, 2

    dmv_accidents = _spread(seed, 4, 0, accident_ceiling)
    dmv_violations = _spread(seed, 12, 0, violation_ceiling)

    # The record is the union of declared and discovered history.
    accidents = max(int(declared_claims or 0), dmv_accidents)
    undisclosed = max(0, accidents - int(declared_claims or 0))

    violations = [
        {
            "type": _VIOLATION_TYPES[_spread(seed, 1000 + idx, 0, len(_VIOLATION_TYPES) - 1)],
            "years_ago": _spread(seed, 2000 + idx, 1, 5),
        }
        for idx in range(dmv_violations)
    ]

    dui = age < 30 and _spread(seed, 40, 0, 24) == 0  # ~4% among under-30s
    licence_status = "SUSPENDED" if dui else ("VALID" if exposure >= 1 else "PROVISIONAL")

    return ApiResponse(
        status_code=200,
        data={
            "accidents_5yr": accidents,
            "violations_5yr": dmv_violations,
            "violation_detail": violations,
            "dui_flag": dui,
            "licence_status": licence_status,
            "years_licensed": exposure,
            "undisclosed_incidents": undisclosed,
            "clean_record": accidents == 0 and dmv_violations == 0 and not dui,
            "source": "mvr",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        },
    )


# =========================================================================== #
# 3. Underwriting  —  POST /api/v1/underwriting/geo-hazard
# =========================================================================== #
# Curated real-world anchors; everything else is derived deterministically.
_GEO_ANCHORS: dict[str, dict[str, Any]] = {
    "90210": {"name": "Beverly Hills, CA", "theft": 62, "flood": 18, "hail": 8, "traffic": 78, "cat": 55},
    "10001": {"name": "New York, NY", "theft": 71, "flood": 44, "hail": 12, "traffic": 95, "cat": 40},
    "60601": {"name": "Chicago, IL", "theft": 68, "flood": 35, "hail": 55, "traffic": 82, "cat": 45},
    "33101": {"name": "Miami, FL", "theft": 74, "flood": 88, "hail": 30, "traffic": 80, "cat": 92},
    "94105": {"name": "San Francisco, CA", "theft": 79, "flood": 26, "hail": 5, "traffic": 88, "cat": 68},
    "73301": {"name": "Austin, TX", "theft": 48, "flood": 52, "hail": 72, "traffic": 66, "cat": 58},
    "560001": {"name": "Bengaluru, KA", "theft": 55, "flood": 61, "hail": 10, "traffic": 91, "cat": 34},
    "400001": {"name": "Mumbai, MH", "theft": 64, "flood": 86, "hail": 8, "traffic": 96, "cat": 62},
    "110001": {"name": "New Delhi, DL", "theft": 72, "flood": 40, "hail": 18, "traffic": 94, "cat": 44},
    "600001": {"name": "Chennai, TN", "theft": 51, "flood": 78, "hail": 6, "traffic": 84, "cat": 70},
}

_WEIGHTS = {"theft": 0.22, "flood": 0.26, "hail": 0.14, "traffic": 0.20, "cat": 0.18}


@_timed("POST /api/v1/underwriting/geo-hazard")
def geo_hazard_check(zip_code: str) -> ApiResponse:
    """Location hazard profile for a ZIP / PIN code (0 = benign, 100 = severe)."""
    code = re.sub(r"\s", "", zip_code or "")
    if not re.fullmatch(r"\d{5}(-\d{4})?|\d{6}", code):
        return ApiResponse(status_code=422, error=f"Malformed postal code: {zip_code!r}")

    base = code.split("-")[0]
    anchor = _GEO_ANCHORS.get(base)

    if anchor:
        hazards = {k: v for k, v in anchor.items() if k != "name"}
        locality = anchor["name"]
    else:
        seed = _seed("geo", base)
        hazards = {
            "theft": _spread(seed, 0, 15, 85),
            "flood": _spread(seed, 6, 5, 90),
            "hail": _spread(seed, 12, 3, 80),
            "traffic": _spread(seed, 18, 20, 95),
            "cat": _spread(seed, 24, 10, 85),
        }
        region = "IN" if len(base) == 6 else "US"
        locality = f"{'PIN' if region == 'IN' else 'ZIP'} {base}"

    composite = round(sum(hazards[k] * w for k, w in _WEIGHTS.items()), 1)

    if composite >= 70:
        band, multiplier = "SEVERE", 1.28
    elif composite >= 55:
        band, multiplier = "ELEVATED", 1.14
    elif composite >= 40:
        band, multiplier = "MODERATE", 1.0
    else:
        band, multiplier = "BENIGN", 0.92

    dominant = max(hazards, key=lambda k: hazards[k])
    labels = {
        "theft": "Vehicle theft density",
        "flood": "Flood exposure",
        "hail": "Hail & storm frequency",
        "traffic": "Traffic accident density",
        "cat": "Catastrophe / wildfire exposure",
    }

    return ApiResponse(
        status_code=200,
        data={
            "postal_code": base,
            "locality": locality,
            "hazards": hazards,
            "composite_hazard": composite,
            "hazard_band": band,
            "premium_multiplier": multiplier,
            "dominant_hazard": labels[dominant],
            "dominant_hazard_key": dominant,
            "source": "geo-hazard",
        },
    )


# =========================================================================== #
# 4/5. KYC  —  Aadhaar OTP request + verification
# =========================================================================== #
@dataclass
class _OTPSession:
    txn_id: str
    aadhaar_last4: str
    aadhaar_hash: str
    expires_at: datetime
    attempts: int = 0
    verified: bool = False


_OTP_STORE: dict[str, _OTPSession] = {}
_OTP_LOCK = threading.Lock()


def reset_kyc_store() -> None:
    """Clear all in-flight OTP sessions (used on 'start over')."""
    with _OTP_LOCK:
        _OTP_STORE.clear()


@_timed("POST /api/v1/kyc/aadhaar/request-otp")
def request_aadhaar_otp(aadhaar_number: str) -> ApiResponse:
    """
    Trigger an Aadhaar e-KYC OTP.

    Validates the 12-digit format and returns a ``txn_id``. The raw Aadhaar is
    never retained — only a salted hash and the last four digits, mirroring
    UIDAI storage-minimisation rules.
    """
    digits = re.sub(r"[\s-]", "", aadhaar_number or "")

    if not digits.isdigit():
        return ApiResponse(status_code=422, error="Aadhaar must contain digits only")
    if len(digits) != settings.aadhaar_length:
        return ApiResponse(
            status_code=422,
            error=f"Aadhaar must be exactly {settings.aadhaar_length} digits (got {len(digits)})",
        )
    if digits[0] in "01":
        return ApiResponse(status_code=422, error="Aadhaar cannot begin with 0 or 1")

    txn_id = f"txn_{secrets.token_hex(8)}"
    session = _OTPSession(
        txn_id=txn_id,
        aadhaar_last4=digits[-4:],
        aadhaar_hash=hashlib.sha256(f"aegis::{digits}".encode()).hexdigest(),
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=settings.otp_ttl_seconds),
    )
    with _OTP_LOCK:
        _OTP_STORE[txn_id] = session

    return ApiResponse(
        status_code=200,
        data={
            "txn_id": txn_id,
            "masked_aadhaar": f"XXXX XXXX {digits[-4:]}",
            "otp_sent_to": f"+91 XXXXX X{digits[-4:-1]}",
            "expires_in_seconds": settings.otp_ttl_seconds,
            "max_attempts": settings.max_otp_attempts,
            "message": "OTP dispatched to the mobile number registered with UIDAI.",
            "hint": f"Demo environment — use {settings.mock_otp}.",
        },
    )


@_timed("POST /api/v1/kyc/aadhaar/verify-otp")
def verify_aadhaar_otp(txn_id: str, otp: str) -> ApiResponse:
    """Verify the OTP for a transaction and mint a KYC token."""
    code = re.sub(r"[\s-]", "", otp or "")

    with _OTP_LOCK:
        session = _OTP_STORE.get(txn_id or "")

        if session is None:
            return ApiResponse(status_code=404, error="Unknown or expired transaction id")
        if session.verified:
            return ApiResponse(status_code=409, error="This transaction is already verified")
        if datetime.now(timezone.utc) > session.expires_at:
            _OTP_STORE.pop(txn_id, None)
            return ApiResponse(status_code=410, error="OTP expired — please request a new one")
        if not re.fullmatch(r"\d{6}", code):
            return ApiResponse(status_code=422, error="OTP must be exactly 6 digits")

        session.attempts += 1
        if session.attempts > settings.max_otp_attempts:
            _OTP_STORE.pop(txn_id, None)
            return ApiResponse(status_code=429, error="Too many attempts — transaction locked")

        if code != settings.mock_otp:
            remaining = settings.max_otp_attempts - session.attempts
            return ApiResponse(
                status_code=401,
                error=f"Incorrect OTP. {remaining} attempt(s) remaining.",
                data={"attempts_remaining": max(0, remaining)},
            )

        session.verified = True

    return ApiResponse(
        status_code=200,
        data={
            "verified": True,
            "kyc_token": f"kyc_{secrets.token_urlsafe(18)}",
            "masked_aadhaar": f"XXXX XXXX {session.aadhaar_last4}",
            "verification_level": "AADHAAR_OTP_L2",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "issuer": "UIDAI (mock)",
        },
    )


# =========================================================================== #
# 6. Issuance  —  POST /api/v1/policies/issue
# =========================================================================== #
@_timed("POST /api/v1/policies/issue")
def issue_policy(
    kyc_token: str,
    policy_id: str,
    policy_name: str,
    holder_name: str,
    monthly_premium: float,
    deductible: int,
    coverage_limit: int,
    risk_tier: str = "Standard",
) -> ApiResponse:
    """Bind the policy and return the executed contract payload."""
    if not kyc_token or not kyc_token.startswith("kyc_"):
        return ApiResponse(status_code=403, error="A valid KYC token is required before issuance")
    if not policy_id:
        return ApiResponse(status_code=422, error="policy_id is required")
    if monthly_premium <= 0:
        return ApiResponse(status_code=422, error="monthly_premium must be positive")

    now = datetime.now(timezone.utc)
    serial = _seed(kyc_token, policy_id) % 900000 + 100000
    contract_id = f"AEG-{now.year}-{policy_id.upper()[:6]}-{serial}"

    return ApiResponse(
        status_code=201,
        data={
            "policy_number": contract_id,
            "status": "ACTIVE",
            "product_id": policy_id,
            "product_name": policy_name,
            "policyholder": holder_name,
            "risk_tier": risk_tier,
            "monthly_premium": round(monthly_premium, 2),
            "annual_premium": round(monthly_premium * 12, 2),
            "deductible": deductible,
            "coverage_limit": coverage_limit,
            "effective_date": now.date().isoformat(),
            "expiry_date": (now + timedelta(days=365)).date().isoformat(),
            "issued_at": now.isoformat(),
            "underwriter": "AegisAI Automated Underwriting v2.4",
            "free_look_period_days": 15,
            "documents": [
                {"type": "SCHEDULE", "name": f"{contract_id}-schedule.pdf"},
                {"type": "TERMS", "name": f"{contract_id}-terms.pdf"},
                {"type": "KYC_CERT", "name": f"{contract_id}-kyc.pdf"},
            ],
        },
    )


# =========================================================================== #
# Async facades
# =========================================================================== #
async def _to_thread(fn, *args: Any, **kwargs: Any) -> ApiResponse:
    return await asyncio.to_thread(fn, *args, **kwargs)


async def aget_crm_customer(customer_id: str) -> ApiResponse:
    return await _to_thread(get_crm_customer, customer_id)


async def amvr_check(
    subject_key: str, age: int, years_licensed: int | None = None, declared_claims: int = 0
) -> ApiResponse:
    return await _to_thread(mvr_check, subject_key, age, years_licensed, declared_claims)


async def ageo_hazard_check(zip_code: str) -> ApiResponse:
    return await _to_thread(geo_hazard_check, zip_code)


async def arequest_aadhaar_otp(aadhaar_number: str) -> ApiResponse:
    return await _to_thread(request_aadhaar_otp, aadhaar_number)


async def averify_aadhaar_otp(txn_id: str, otp: str) -> ApiResponse:
    return await _to_thread(verify_aadhaar_otp, txn_id, otp)


async def aissue_policy(**kwargs: Any) -> ApiResponse:
    return await _to_thread(issue_policy, **kwargs)
