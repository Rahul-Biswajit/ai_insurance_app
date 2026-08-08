"""
orchestrator/workflow.py — Deterministic state machine coordinating the agents.

Every turn follows the same fixed path::

    raw input
      -> PII firewall           (redact before anything sees the text)
      -> guardrails input rail  (block injection / jailbreak / off-topic)
      -> route by stage         (deterministic — never model-decided)
      -> agent node
      -> guardrails output rail (compliance + groundedness)
      -> reply

The *routing* is deliberately not delegated to an LLM. A model deciding "should
I run KYC now?" is a model that can be talked into skipping it. Stage
transitions are gated on validated artefacts existing in state, so the only way
to reach issuance is to genuinely pass every prior stage.

``build_langgraph()`` exposes the same node functions as a LangGraph
``StateGraph`` when that package is installed; the deterministic engine below
remains the default and is always available.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from agents.compliance_agent import compliance_agent
from agents.intake_agent import intake_agent
from agents.qa_agent import qa_agent
from agents.recommendation_agent import recommendation_agent
from agents.risk_agent import risk_agent
from agents.state import OnboardingState, Stage
from backend.mock_apis import get_crm_customer, reset_kyc_store
from config import settings
from security.guardrails_runtime import RailDecision, get_shield
from security.pii_firewall import firewall
from security.validations import validate_profile

logger = logging.getLogger(__name__)

__all__ = ["TurnResult", "OnboardingWorkflow", "workflow", "build_langgraph"]


# --------------------------------------------------------------------------- #
# Turn result
# --------------------------------------------------------------------------- #
@dataclass
class TurnResult:
    """Everything the UI needs to render one exchange."""

    reply: str
    stage_before: Stage
    stage_after: Stage
    agent: str = "Orchestrator"
    blocked: bool = False
    block_category: str | None = None
    pii_kinds: list[str] = field(default_factory=list)
    extracted: dict[str, Any] = field(default_factory=dict)
    validation_blocking: list[str] = field(default_factory=list)
    advanced: bool = False

    @property
    def stage_changed(self) -> bool:
        return self.stage_before is not self.stage_after


# --------------------------------------------------------------------------- #
# Intent heuristics
# --------------------------------------------------------------------------- #
_QUESTION_LEAD = re.compile(
    r"^\s*(what|which|why|how|when|where|who|does|do|can|could|is|are|will|would|should|"
    r"tell me|explain|compare|show me)\b",
    re.IGNORECASE,
)

_KYC_INTENT = re.compile(
    r"\b(verify|verification|kyc|aadhaar|otp|identity)\b", re.IGNORECASE
)

_ISSUE_INTENT = re.compile(
    r"\b(issue|buy|purchase|confirm|proceed|finali[sz]e|go ahead|activate|take it|"
    r"sign up|enroll|i'?ll take)\b",
    re.IGNORECASE,
)

_RESTART_INTENT = re.compile(r"\b(start over|restart|reset|begin again|clear)\b", re.IGNORECASE)


def _looks_like_question(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.endswith("?") or bool(_QUESTION_LEAD.match(stripped))


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
class OnboardingWorkflow:
    """The deterministic orchestrator."""

    def __init__(self) -> None:
        self.shield = get_shield()

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #
    @staticmethod
    def new_state(session_id: str | None = None) -> OnboardingState:
        state = OnboardingState(session_id=session_id or f"sess_{uuid.uuid4().hex[:10]}")
        state.add_message(
            "assistant",
            "Welcome to **AegisAI**. I'll guide you through insurance onboarding — "
            "collecting your details, assessing risk, matching policies, and verifying "
            "your identity.\n\nTo get started, could you tell me your full name and age?",
            agent="Orchestrator",
        )
        state.advance_to(Stage.INTAKE)
        return state

    @staticmethod
    def reset() -> OnboardingState:
        reset_kyc_store()
        return OnboardingWorkflow.new_state()

    # ------------------------------------------------------------------ #
    # CRM prefill
    # ------------------------------------------------------------------ #
    def prefill_from_crm(self, state: OnboardingState, customer_id: str) -> tuple[bool, str]:
        """Populate intake from the CRM of record."""
        response = get_crm_customer(customer_id)
        if not response.ok:
            return False, f"❌ {response.error}"

        fields = (
            "full_name", "age", "zip_code", "vehicle_year", "vehicle_make", "vehicle_model",
            "vehicle_value", "annual_mileage", "years_licensed", "prior_claims",
            "occupation", "marital_status", "coverage_preference",
        )
        state.collected.update(
            {key: response.data[key] for key in fields if key in response.data}
        )
        state.log("crm_prefill", customer_id=customer_id)

        result = self._run_pipeline(state)
        message = (
            f"✅ Loaded **{response.data['full_name']}** from CRM "
            f"({response.data.get('loyalty_tier', 'Standard')} tier, "
            f"{response.data.get('tenure_years', 0)} years with us)."
        )
        return True, f"{message}\n\n{result}"

    # ------------------------------------------------------------------ #
    # Main turn handler
    # ------------------------------------------------------------------ #
    def handle_message(self, state: OnboardingState, user_text: str) -> TurnResult:
        stage_before = state.stage

        if not (user_text or "").strip():
            return TurnResult(
                reply="Could you say a little more?",
                stage_before=stage_before,
                stage_after=state.stage,
            )

        # ---- 1. PII firewall -------------------------------------------- #
        known_terms = [state.profile.full_name] if state.profile else []
        sanitisation = (
            firewall.sanitize(user_text, vault=state.vault, extra_terms=known_terms)
            if settings.enable_pii_firewall
            else None
        )
        sanitized_text = sanitisation.sanitized if sanitisation else user_text
        pii_kinds = sanitisation.kinds if sanitisation else []

        state.add_message("user", user_text, pii_redacted=pii_kinds)

        # ---- 2. Input rails --------------------------------------------- #
        decision: RailDecision = self.shield.check_input(sanitized_text)
        if decision.blocked:
            state.guardrail_blocks += 1
            reply = decision.safe_response or "I can't help with that request."
            state.add_message(
                "assistant",
                reply,
                agent="Guardrails",
                blocked=True,
                block_category=decision.category.value,
            )
            state.log("guardrail_block", category=decision.category.value, rail="input")
            return TurnResult(
                reply=reply,
                stage_before=stage_before,
                stage_after=state.stage,
                agent="Guardrails",
                blocked=True,
                block_category=decision.category.value,
                pii_kinds=pii_kinds,
            )

        # ---- 3. Route ---------------------------------------------------- #
        result = self._route(state, user_text, sanitized_text, stage_before, pii_kinds)

        # ---- 4. Output rails --------------------------------------------- #
        safe_reply, output_decision = self.shield.safe_output(result.reply)
        if output_decision.blocked:
            state.guardrail_blocks += 1
            result.reply = safe_reply
            result.blocked = True
            result.block_category = output_decision.category.value
            state.log("guardrail_block", category=output_decision.category.value, rail="output")

        state.add_message(
            "assistant",
            result.reply,
            agent=result.agent,
            blocked=result.blocked,
            block_category=result.block_category,
        )
        result.stage_after = state.stage
        return result

    # ------------------------------------------------------------------ #
    def _route(
        self,
        state: OnboardingState,
        raw_text: str,
        sanitized_text: str,
        stage_before: Stage,
        pii_kinds: list[str],
    ) -> TurnResult:
        """Deterministic stage-based routing."""

        def make(reply: str, agent: str, **kwargs: Any) -> TurnResult:
            return TurnResult(
                reply=reply,
                stage_before=stage_before,
                stage_after=state.stage,
                agent=agent,
                pii_kinds=pii_kinds,
                **kwargs,
            )

        if _RESTART_INTENT.search(raw_text):
            return make(
                "Use the **Start over** button in the sidebar to clear this session and begin again.",
                "Orchestrator",
            )

        # Identity / issuance intents are handled by the dedicated panels so the
        # applicant never types an Aadhaar into the chat transcript.
        if _KYC_INTENT.search(raw_text):
            if state.kyc.verified:
                return make(
                    f"Your identity is already verified ({state.kyc.aadhaar_masked}). "
                    "You can issue your policy from the **Verification & Issuance** panel.",
                    "Verification & Compliance Agent",
                )
            if state.reached(Stage.RECOMMENDATION):
                return make(
                    "Identity verification happens in the **Verification & Issuance** panel on "
                    "the right — enter your 12-digit Aadhaar there and I'll send an OTP. "
                    "Please don't type it into the chat.",
                    "Verification & Compliance Agent",
                )
            return make(
                "We'll verify your identity once your policies are matched. "
                "Let's finish your details first.",
                "Verification & Compliance Agent",
            )

        if _ISSUE_INTENT.search(raw_text) and state.reached(Stage.RECOMMENDATION):
            if not state.kyc.verified:
                return make(
                    "Before issuing, I need to verify your identity. Head to the "
                    "**Verification & Issuance** panel to complete Aadhaar e-KYC.",
                    "Verification & Compliance Agent",
                )
            return make(
                "You're verified and ready. Press **Issue Policy** in the "
                "**Verification & Issuance** panel to bind your cover.",
                "Verification & Compliance Agent",
            )

        # Intake extraction runs on every turn — an applicant may correct a
        # detail long after the initial collection.
        extracted = intake_agent.update_state(state, raw_text, sanitized_text)

        question = _looks_like_question(raw_text)

        # A question with no new data, once we have products, goes to Q&A.
        if question and not extracted and state.recommendations:
            answer = qa_agent.run(state, sanitized_text)
            return make(answer, "Policy Q&A Agent")

        # Still collecting.
        if not state.reached(Stage.RISK):
            pipeline_reply = self._run_pipeline(state)
            acknowledgement = intake_agent.acknowledge(extracted)
            reply = f"{acknowledgement} {pipeline_reply}".strip() if acknowledgement else pipeline_reply
            return make(
                reply,
                "Intake Agent",
                extracted=extracted,
                validation_blocking=[i["field"] for i in state.validation_issues],
                advanced=state.reached(Stage.RISK),
            )

        # Post-assessment: a correction re-runs underwriting.
        if extracted:
            pipeline_reply = self._run_pipeline(state, force=True)
            return make(
                f"Updated — I've re-run your assessment.\n\n{pipeline_reply}",
                "Intake Agent",
                extracted=extracted,
                advanced=True,
            )

        # Anything else with products on screen is a Q&A turn.
        if state.recommendations:
            answer = qa_agent.run(state, sanitized_text)
            return make(answer, "Policy Q&A Agent")

        return make(self._run_pipeline(state), "Orchestrator")

    # ------------------------------------------------------------------ #
    # Pipeline: validate -> risk -> recommend
    # ------------------------------------------------------------------ #
    def _run_pipeline(self, state: OnboardingState, force: bool = False) -> str:
        """Advance as far as the collected data allows; return a status message."""
        report = validate_profile(state.collected)

        state.missing_fields = list(report.missing_fields)
        state.validation_issues = [
            {
                "field": issue.field_name,
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity.value,
                "remediation": issue.remediation,
            }
            for issue in report.issues
        ]

        if report.missing_fields:
            state.advance_to(Stage.INTAKE)
            return intake_agent.next_question(state)

        state.advance_to(Stage.VALIDATION)

        if report.blocking:
            state.log("validation_blocked", issues=[i.code for i in report.blocking])
            problems = "\n\n".join(issue.render() for issue in report.blocking)
            return (
                f"I need to correct a few things before I can continue:\n\n{problems}\n\n"
                "Could you re-enter those?"
            )

        assert report.profile is not None
        previous = state.profile
        state.profile = report.profile

        unchanged = (
            previous is not None
            and previous.model_dump() == report.profile.model_dump()
            and state.risk is not None
        )
        if unchanged and not force:
            return self._summary(state)

        # ---- risk ------------------------------------------------------- #
        state.advance_to(Stage.RISK)
        assessment = risk_agent.run(state)
        if assessment is None:
            return "I couldn't complete the risk assessment. Please check your details."

        # ---- recommendations -------------------------------------------- #
        state.advance_to(Stage.RECOMMENDATION)
        matches = recommendation_agent.run(state)
        if not matches:
            return (
                f"Your risk assessment is complete ({assessment.tier_label}, "
                f"{assessment.score:.0f}/100), but no catalog product currently matches "
                "your profile. A licensed agent can look at specialist options."
            )

        state.advance_to(Stage.INSPECTION)
        return self._summary(state)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _summary(state: OnboardingState) -> str:
        """Post-assessment status message."""
        risk = state.risk
        matches = state.recommendations
        if risk is None or not matches:
            return "Your details are validated."

        top = matches[0]
        lines = [
            f"**Assessment complete.** You're rated **{risk.tier_label}** at "
            f"**{risk.score:.0f}/100**, with a **{risk.base_discount:.0f}%** base discount.",
            "",
            risk.summary,
            "",
            f"I've matched **{len(matches)} policies**. The strongest fit is "
            f"**{top.name}** at **{settings.currency_symbol}{top.monthly_premium:,.0f}/month** "
            f"({top.match_score:.0f}% match).",
            "",
            "Explore them in the dashboard — adjust the deductible, compare coverage, "
            "and ask me anything about the terms.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # KYC / issuance passthrough (called by the UI panels)
    # ------------------------------------------------------------------ #
    def start_kyc(self, state: OnboardingState, aadhaar: str) -> tuple[bool, str]:
        if not state.reached(Stage.RECOMMENDATION):
            return False, "❌ Complete your profile and policy matching before verification."
        state.advance_to(Stage.KYC)
        return compliance_agent.start_kyc(state, aadhaar)

    def verify_kyc(self, state: OnboardingState, otp: str) -> tuple[bool, str]:
        return compliance_agent.verify_kyc(state, otp)

    def issue_policy(self, state: OnboardingState) -> tuple[bool, str]:
        ok, message = compliance_agent.issue(state)
        if ok:
            state.advance_to(Stage.ISSUANCE)
            state.advance_to(Stage.COMPLETE)
        return ok, message

    def ask(self, state: OnboardingState, question: str) -> str:
        """Direct Q&A entry point for UI affordances (e.g. suggestion chips)."""
        return qa_agent.run(state, question)


workflow = OnboardingWorkflow()


# --------------------------------------------------------------------------- #
# Optional LangGraph compilation
# --------------------------------------------------------------------------- #
def build_langgraph():  # pragma: no cover - exercised only when langgraph present
    """
    Compile the same pipeline as a LangGraph ``StateGraph``.

    Returns the compiled graph, or ``None`` when ``langgraph`` is not installed.
    The deterministic engine above is the default execution path; this exists so
    the workflow can be dropped into a LangGraph deployment unchanged.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:
        logger.info("langgraph unavailable (%s) — deterministic engine only.", exc)
        return None

    from agents.state import GraphState

    def _load(data: GraphState) -> OnboardingState:
        return OnboardingState.from_graph(data)

    def node_validate(data: GraphState) -> GraphState:
        state = _load(data)
        report = validate_profile(state.collected)
        state.missing_fields = list(report.missing_fields)
        state.validation_issues = [
            {"field": i.field_name, "code": i.code, "message": i.message, "severity": i.severity.value}
            for i in report.issues
        ]
        if report.is_valid and report.profile:
            state.profile = report.profile
            state.advance_to(Stage.VALIDATION)
        return state.to_graph()

    def node_risk(data: GraphState) -> GraphState:
        state = _load(data)
        if state.profile:
            state.advance_to(Stage.RISK)
            risk_agent.run(state)
        return state.to_graph()

    def node_recommend(data: GraphState) -> GraphState:
        state = _load(data)
        if state.profile:
            state.advance_to(Stage.RECOMMENDATION)
            recommendation_agent.run(state)
            state.advance_to(Stage.INSPECTION)
        return state.to_graph()

    def route_after_validation(data: GraphState) -> str:
        state = _load(data)
        blocking = [i for i in state.validation_issues if i["severity"] in {"error", "anomaly"}]
        if state.missing_fields or blocking or state.profile is None:
            return "incomplete"
        return "assess"

    graph = StateGraph(GraphState)
    graph.add_node("validate", node_validate)
    graph.add_node("risk", node_risk)
    graph.add_node("recommend", node_recommend)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges(
        "validate", route_after_validation, {"assess": "risk", "incomplete": END}
    )
    graph.add_edge("risk", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()
