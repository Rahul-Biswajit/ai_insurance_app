"""
agents/state.py — The single source of truth passed between agents.

Two representations of the same data:

* :class:`OnboardingState` — a Pydantic model used by the deterministic
  orchestrator. Validated, typed, and safe to mutate through helpers.
* :class:`GraphState` — a ``TypedDict`` mirror consumed by LangGraph when it is
  installed (LangGraph requires a dict-like channel schema).

``OnboardingState.to_graph()`` / ``from_graph()`` convert between them, so the
same node functions serve both execution engines.

PII discipline
--------------
Raw identifiers live only in :attr:`OnboardingState.vault` and the private
KYC fields. Anything handed to a prompt goes through
``OnboardingState.llm_context()``, which emits tokens only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from security.pii_firewall import PIIVault
from security.validations import ApplicantProfile

__all__ = [
    "Stage",
    "ChatMessage",
    "RiskFactor",
    "RiskAssessment",
    "PolicyMatch",
    "KYCState",
    "IssuedPolicy",
    "OnboardingState",
    "GraphState",
    "STAGE_ORDER",
    "STAGE_LABELS",
]


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
class Stage(str, Enum):
    """Deterministic pipeline stages. The orchestrator only moves forward."""

    GREETING = "greeting"
    INTAKE = "intake"
    VALIDATION = "validation"
    RISK = "risk"
    RECOMMENDATION = "recommendation"
    INSPECTION = "inspection"
    KYC = "kyc"
    ISSUANCE = "issuance"
    COMPLETE = "complete"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.GREETING,
    Stage.INTAKE,
    Stage.VALIDATION,
    Stage.RISK,
    Stage.RECOMMENDATION,
    Stage.INSPECTION,
    Stage.KYC,
    Stage.ISSUANCE,
    Stage.COMPLETE,
)

STAGE_LABELS: dict[Stage, str] = {
    Stage.GREETING: "Welcome",
    Stage.INTAKE: "Collecting details",
    Stage.VALIDATION: "Validating",
    Stage.RISK: "Assessing risk",
    Stage.RECOMMENDATION: "Matching policies",
    Stage.INSPECTION: "Reviewing options",
    Stage.KYC: "Identity verification",
    Stage.ISSUANCE: "Issuing policy",
    Stage.COMPLETE: "Complete",
}


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    """One turn of dialogue, annotated with what the security layer did."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: str
    stage: Stage | None = None
    agent: str | None = None
    blocked: bool = False
    block_category: str | None = None
    pii_redacted: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
class RiskFactor(BaseModel):
    """A single contributor to the composite score."""

    model_config = ConfigDict(frozen=True)

    name: str
    impact: float           # signed points applied to the 0-100 safety score
    detail: str
    category: Literal["driver", "vehicle", "usage", "history", "location"] = "driver"

    @property
    def direction(self) -> str:
        return "favourable" if self.impact >= 0 else "adverse"


class RiskAssessment(BaseModel):
    """Output of the Risk Assessment Agent."""

    model_config = ConfigDict(extra="ignore")

    score: float = Field(ge=0, le=100)          # higher == safer
    tier: str
    class_code: str
    base_discount: float = Field(ge=0, le=40)   # percent
    premium_multiplier: float = Field(gt=0)
    primary_factor: str
    factors: list[RiskFactor] = Field(default_factory=list)
    summary: str = ""
    mvr: dict[str, Any] = Field(default_factory=dict)
    geo: dict[str, Any] = Field(default_factory=dict)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tier_label(self) -> str:
        return f"{self.tier} - Class {self.class_code}"

    @property
    def favourable(self) -> list[RiskFactor]:
        return sorted([f for f in self.factors if f.impact >= 0], key=lambda f: -f.impact)

    @property
    def adverse(self) -> list[RiskFactor]:
        return sorted([f for f in self.factors if f.impact < 0], key=lambda f: f.impact)


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
class PolicyMatch(BaseModel):
    """A scored policy recommendation with its five radar dimensions."""

    model_config = ConfigDict(extra="ignore")

    policy_id: str
    name: str
    insurer: str
    category: str
    tagline: str

    match_score: float = Field(ge=0, le=100)
    semantic_score: float = 0.0
    risk_match: float = Field(ge=0, le=100)

    # radar axes
    affordability: float
    coverage_breadth: float
    flexibility: float
    perks: float

    monthly_premium: float
    annual_premium: float
    deductible: int
    coverage_limit: int
    rationale: str = ""
    eligible: bool = True
    ineligibility_reason: str | None = None

    def radar_values(self) -> list[float]:
        """Ordered to match the radar indicator axes in the dashboard."""
        return [
            round(self.affordability, 1),
            round(self.coverage_breadth, 1),
            round(self.flexibility, 1),
            round(self.perks, 1),
            round(self.risk_match, 1),
        ]


# --------------------------------------------------------------------------- #
# KYC & issuance
# --------------------------------------------------------------------------- #
class KYCState(BaseModel):
    """e-KYC progress. The raw Aadhaar never lands here — only the mask."""

    model_config = ConfigDict(extra="ignore")

    aadhaar_masked: str | None = None
    txn_id: str | None = None
    otp_sent: bool = False
    verified: bool = False
    kyc_token: str | None = None
    verification_level: str | None = None
    attempts: int = 0
    last_error: str | None = None
    verified_at: datetime | None = None


class IssuedPolicy(BaseModel):
    """The bound contract returned by the issuance API."""

    model_config = ConfigDict(extra="ignore")

    policy_number: str
    status: str
    product_id: str
    product_name: str
    policyholder: str
    risk_tier: str
    monthly_premium: float
    annual_premium: float
    deductible: int
    coverage_limit: int
    effective_date: str
    expiry_date: str
    issued_at: str
    underwriter: str
    free_look_period_days: int = 15
    documents: list[dict[str, str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Master state
# --------------------------------------------------------------------------- #
class OnboardingState(BaseModel):
    """Everything the workflow knows about one applicant session."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    session_id: str
    stage: Stage = Stage.GREETING

    # intake
    collected: dict[str, Any] = Field(default_factory=dict)
    profile: ApplicantProfile | None = None
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    # downstream artefacts
    risk: RiskAssessment | None = None
    recommendations: list[PolicyMatch] = Field(default_factory=list)
    selected_policy_id: str | None = None
    selected_deductible: int | None = None

    kyc: KYCState = Field(default_factory=KYCState)
    issued: IssuedPolicy | None = None

    # conversation & security
    messages: list[ChatMessage] = Field(default_factory=list)
    vault: PIIVault = Field(default_factory=PIIVault)
    guardrail_blocks: int = 0
    audit: list[dict[str, Any]] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------ #
    # Conversation helpers
    # ------------------------------------------------------------------ #
    def add_message(self, role: str, content: str, **kwargs: Any) -> ChatMessage:
        message = ChatMessage(role=role, content=content, stage=self.stage, **kwargs)
        self.messages.append(message)
        return message

    def log(self, event: str, **details: Any) -> None:
        self.audit.append(
            {
                "event": event,
                "stage": self.stage.value,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **details,
            }
        )

    # ------------------------------------------------------------------ #
    # Stage helpers
    # ------------------------------------------------------------------ #
    def advance_to(self, stage: Stage) -> None:
        """Move forward only — the pipeline never silently regresses."""
        if STAGE_ORDER.index(stage) > STAGE_ORDER.index(self.stage):
            self.log("stage_advance", to=stage.value)
            self.stage = stage

    def reached(self, stage: Stage) -> bool:
        return STAGE_ORDER.index(self.stage) >= STAGE_ORDER.index(stage)

    @property
    def progress(self) -> float:
        return STAGE_ORDER.index(self.stage) / (len(STAGE_ORDER) - 1)

    # ------------------------------------------------------------------ #
    # Derived views
    # ------------------------------------------------------------------ #
    @property
    def selected_match(self) -> PolicyMatch | None:
        for match in self.recommendations:
            if match.policy_id == self.selected_policy_id:
                return match
        # A re-run of underwriting can drop the previously selected product from
        # the shortlist; fall back to the current best rather than nothing.
        return self.recommendations[0] if self.recommendations else None

    @property
    def holder_name(self) -> str:
        return self.profile.full_name if self.profile else "Applicant"

    def llm_context(self) -> dict[str, Any]:
        """PII-free snapshot safe to interpolate into a prompt."""
        context: dict[str, Any] = {"stage": self.stage.value}
        if self.profile:
            context["applicant"] = self.profile.underwriting_payload()
        if self.risk:
            context["risk"] = {
                "score": self.risk.score,
                "tier": self.risk.tier_label,
                "primary_factor": self.risk.primary_factor,
                "discount_pct": self.risk.base_discount,
            }
        if self.recommendations:
            context["recommended"] = [
                {
                    "id": m.policy_id,
                    "name": m.name,
                    "monthly_premium": m.monthly_premium,
                    "match": m.match_score,
                }
                for m in self.recommendations
            ]
        if self.kyc.verified:
            context["kyc"] = {"verified": True, "level": self.kyc.verification_level}
        return context

    # ------------------------------------------------------------------ #
    # LangGraph interop
    # ------------------------------------------------------------------ #
    def to_graph(self) -> "GraphState":
        return {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "collected": dict(self.collected),
            "profile": self.profile.model_dump() if self.profile else None,
            "validation_issues": list(self.validation_issues),
            "missing_fields": list(self.missing_fields),
            "risk": self.risk.model_dump(mode="json") if self.risk else None,
            "recommendations": [m.model_dump(mode="json") for m in self.recommendations],
            "selected_policy_id": self.selected_policy_id,
            "selected_deductible": self.selected_deductible,
            "kyc": self.kyc.model_dump(mode="json"),
            "issued": self.issued.model_dump(mode="json") if self.issued else None,
            "messages": [m.model_dump(mode="json") for m in self.messages],
            "guardrail_blocks": self.guardrail_blocks,
        }

    @classmethod
    def from_graph(cls, data: "GraphState", vault: PIIVault | None = None) -> "OnboardingState":
        state = cls(session_id=data["session_id"], vault=vault or PIIVault())
        state.stage = Stage(data.get("stage", Stage.GREETING.value))
        state.collected = dict(data.get("collected") or {})
        if data.get("profile"):
            state.profile = ApplicantProfile(**data["profile"])
        state.validation_issues = list(data.get("validation_issues") or [])
        state.missing_fields = list(data.get("missing_fields") or [])
        if data.get("risk"):
            state.risk = RiskAssessment(**data["risk"])
        state.recommendations = [PolicyMatch(**m) for m in (data.get("recommendations") or [])]
        state.selected_policy_id = data.get("selected_policy_id")
        state.selected_deductible = data.get("selected_deductible")
        if data.get("kyc"):
            state.kyc = KYCState(**data["kyc"])
        if data.get("issued"):
            state.issued = IssuedPolicy(**data["issued"])
        state.messages = [ChatMessage(**m) for m in (data.get("messages") or [])]
        state.guardrail_blocks = int(data.get("guardrail_blocks") or 0)
        return state


class GraphState(TypedDict, total=False):
    """Dict-shaped mirror of :class:`OnboardingState` for LangGraph channels."""

    session_id: str
    stage: str
    collected: dict[str, Any]
    profile: dict[str, Any] | None
    validation_issues: list[dict[str, Any]]
    missing_fields: list[str]
    risk: dict[str, Any] | None
    recommendations: list[dict[str, Any]]
    selected_policy_id: str | None
    selected_deductible: int | None
    kyc: dict[str, Any]
    issued: dict[str, Any] | None
    messages: list[dict[str, Any]]
    guardrail_blocks: int
