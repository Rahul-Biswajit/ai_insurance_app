"""
agents/qa_agent.py — Grounded policy Q&A.

Answers customer questions about catalog products using retrieved policy
documents as the *only* permitted source. Two answering modes:

* **LLM mode** — a strict grounded prompt; the response is then re-checked by
  the output rails (binding language + numeric groundedness) before display.
* **Deterministic mode** — intent classification over the retrieved policy
  fields. Used when no model is configured, and as the fallback whenever the
  LLM output fails its grounding check.

Because the deterministic answers are assembled directly from catalog fields,
they are grounded by construction and can never invent a limit or premium.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.state import OnboardingState
from backend.vector_store import Policy, get_policy, get_vector_store
from config import settings
from llm_factory import get_llm, is_offline
from security.guardrails_runtime import get_shield

logger = logging.getLogger(__name__)

__all__ = ["QAAgent", "qa_agent"]


_QA_SYSTEM = """You are AegisAI's policy specialist answering a customer's question.

ABSOLUTE RULES:
1. Use ONLY the policy context provided below. If the answer is not in the \
context, say you do not have that detail and offer to connect a licensed agent.
2. Never guarantee coverage, approval, claim payment or price. Say what a \
policy "is designed to cover", not what it "will cover".
3. Every figure you state — premium, limit, deductible, percentage — must \
appear verbatim in the context.
4. Always name the policy you are describing.
5. Two to four sentences. Plain language, no jargon dumps.

POLICY CONTEXT:
{evidence}

APPLICANT CONTEXT (may be empty):
{applicant}"""


# --------------------------------------------------------------------------- #
# Intent detection for the deterministic path
# --------------------------------------------------------------------------- #
_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("deductible", ("deductible", "excess", "out of pocket", "out-of-pocket")),
    ("premium", ("premium", "cost", "price", "how much", "monthly", "annually", "per month", "pay")),
    ("limit", ("limit", "maximum", "how much cover", "coverage limit", "payout", "cap")),
    (
        "exclusion",
        (
            "exclusion", "excluded", "not covered", "isn't covered", "isnt covered",
            "not cover", "doesn't cover", "does not cover", "won't cover", "wont cover",
            "left out", "loophole", "fine print",
        ),
    ),
    ("roadside", ("roadside", "towing", "tow", "breakdown", "stranded")),
    ("rental", ("rental", "courtesy car", "loaner", "replacement car")),
    ("addon", ("add-on", "addon", "rider", "endorsement", "optional", "extra")),
    ("compare", ("compare", "difference", "versus", " vs ", "better", "which one", "instead")),
    ("eligibility", ("eligible", "qualify", "can i get", "am i allowed", "requirement")),
    ("features", ("cover", "include", "benefit", "feature", "what do i get", "perks")),
)


def _detect_intent(question: str) -> str:
    lowered = f" {(question or '').lower()} "
    for intent, keywords in _INTENTS:
        if any(keyword in lowered for keyword in keywords):
            return intent
    return "features"


def _money(value: float) -> str:
    return f"{settings.currency_symbol}{value:,.0f}"


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class QAAgent:
    """Answers grounded questions about the policy catalog."""

    name = "Policy Q&A Agent"

    def __init__(self, llm: Any | None = None, store: Any | None = None) -> None:
        self._llm = llm
        self._store = store

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = get_llm(temperature=0.1)
        return self._llm

    @property
    def store(self) -> Any:
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, state: OnboardingState | None = None) -> list[Policy]:
        """
        Pick the policies to ground against.

        The currently-selected product and the shortlist are always included —
        a question like "what's the deductible?" refers to what is on screen,
        which pure semantic retrieval would miss.
        """
        policies: list[Policy] = []
        seen: set[str] = set()

        def _add(policy: Policy | None) -> None:
            if policy and policy.id not in seen:
                seen.add(policy.id)
                policies.append(policy)

        hits = self.store.query(question, k=settings.rag_top_k)
        lowered = (question or "").lower()

        # An explicit product mention outranks everything else on screen.
        for hit in hits:
            if hit.policy.name.lower() in lowered or hit.policy.id.lower() in lowered:
                _add(hit.policy)

        if state:
            _add(get_policy(state.selected_policy_id or ""))
            for match in state.recommendations[:3]:
                _add(get_policy(match.policy_id))

        for hit in hits:
            _add(hit.policy)

        return policies[: settings.rag_top_k]

    # ------------------------------------------------------------------ #
    def answer(self, question: str, state: OnboardingState | None = None) -> tuple[str, list[Policy]]:
        """
        Answer *question*, returning ``(answer_text, cited_policies)``.

        The answer has already passed the output rails.
        """
        policies = self.retrieve(question, state)
        if not policies:
            return (
                "I don't have any policy information loaded yet. Once we've matched "
                "products to your profile I can answer questions about them.",
                [],
            )

        evidence = self.store.documents_for([p.id for p in policies])
        evidence += self._computed_evidence(state)
        shield = get_shield()

        llm = self.llm
        if not is_offline(llm):
            drafted = self._answer_with_llm(question, evidence, state)
            if drafted:
                safe, decision = shield.safe_output(drafted, evidence)
                if decision.allowed:
                    return safe, policies
                logger.info("Q&A output rail blocked (%s) — using deterministic answer.", decision.category)

        deterministic = self._answer_deterministic(question, policies, state)
        safe, _ = shield.safe_output(deterministic, evidence)
        return safe, policies

    # ------------------------------------------------------------------ #
    @staticmethod
    def _computed_evidence(state: OnboardingState | None) -> str:
        """
        Quote figures the system derived for *this* applicant.

        Personalised premiums are computed by the pricing engine, so they never
        appear in the static catalog text. Without this block the groundedness
        rail would flag a correct quote as a hallucination — the evidence set
        has to cover everything the system legitimately knows, not just what it
        retrieved.
        """
        if state is None:
            return ""

        lines: list[str] = []
        if state.recommendations:
            lines.append("\n\n### Computed quotes for this applicant")
            for match in state.recommendations:
                lines.append(
                    f"- {match.name} ({match.policy_id}): "
                    f"{_money(match.monthly_premium)} per month, "
                    f"{_money(match.annual_premium)} per year, "
                    f"deductible {_money(match.deductible)}, "
                    f"coverage limit {_money(match.coverage_limit)}, "
                    f"match score {match.match_score}, "
                    f"risk match {match.risk_match}, "
                    f"affordability {match.affordability}."
                )
        if state.selected_deductible:
            lines.append(f"- Currently selected deductible: {_money(state.selected_deductible)}.")
        if state.risk:
            lines.append(
                f"- Risk score {state.risk.score} of 100, tier {state.risk.tier_label}, "
                f"base discount {state.risk.base_discount} percent, "
                f"premium multiplier {state.risk.premium_multiplier}."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _answer_with_llm(
        self, question: str, evidence: str, state: OnboardingState | None
    ) -> str | None:
        applicant = ""
        if state:
            context = state.llm_context()
            applicant = str({k: v for k, v in context.items() if k in {"applicant", "risk"}})

        try:
            response = self.llm.invoke(
                [
                    {
                        "role": "system",
                        "content": _QA_SYSTEM.format(evidence=evidence, applicant=applicant or "none"),
                    },
                    {"role": "user", "content": question},
                ]
            )
            return (getattr(response, "content", "") or "").strip() or None
        except Exception as exc:
            logger.warning("Q&A LLM call failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    def _answer_deterministic(
        self, question: str, policies: list[Policy], state: OnboardingState | None
    ) -> str:
        """Assemble an answer directly from catalog fields — grounded by construction."""
        intent = _detect_intent(question)
        primary = policies[0]

        # Prefer the selected product when the question is not policy-specific.
        if state and state.selected_policy_id:
            selected = get_policy(state.selected_policy_id)
            if selected and not any(
                p.name.lower() in (question or "").lower() for p in policies
            ):
                primary = selected

        match = None
        if state:
            match = next(
                (m for m in state.recommendations if m.policy_id == primary.id), None
            )

        if intent == "deductible":
            deductible = state.selected_deductible if state and state.selected_deductible else primary.default_deductible
            return (
                f"**{primary.name}** carries a standard deductible of "
                f"{_money(primary.default_deductible)}"
                + (
                    f", and you currently have {_money(deductible)} selected"
                    if deductible != primary.default_deductible
                    else ""
                )
                + ". That is the amount you pay on a covered claim before the insurer "
                "contributes. Raising it lowers your monthly premium; lowering it raises the premium."
            )

        if intent == "premium":
            if match:
                return (
                    f"**{primary.name}** is quoted at {_money(match.monthly_premium)} per month "
                    f"({_money(match.annual_premium)} per year) for your profile, at a "
                    f"{_money(match.deductible)} deductible. That figure already reflects your "
                    "risk assessment and any discount you qualify for."
                )
            return (
                f"**{primary.name}** starts at {_money(primary.base_monthly)} per month before "
                "risk rating. Your personalised figure appears once your risk assessment completes."
            )

        if intent == "limit":
            return (
                f"**{primary.name}** is written with a total coverage limit of "
                f"{_money(primary.coverage_limit)}. Within that limit, cover is allocated "
                f"{primary.allocation.liability}% liability, {primary.allocation.collision}% collision, "
                f"{primary.allocation.comprehensive}% comprehensive, {primary.allocation.medical}% medical "
                f"and {primary.allocation.roadside}% roadside."
            )

        if intent == "exclusion":
            bullets = "\n".join(f"- {item}" for item in primary.exclusions)
            return f"**{primary.name}** does not cover the following:\n\n{bullets}\n\nAnything outside these exclusions is assessed against the policy wording at claim time."

        if intent == "roadside":
            has_roadside = any(
                "roadside" in f.lower() or "breakdown" in f.lower() or "tow" in f.lower()
                for f in primary.features
            )
            if has_roadside:
                detail = next(
                    f for f in primary.features
                    if "roadside" in f.lower() or "breakdown" in f.lower() or "tow" in f.lower()
                )
                return (
                    f"Yes — **{primary.name}** includes roadside support: {detail}. "
                    f"Roadside accounts for {primary.allocation.roadside}% of the coverage allocation."
                )
            addon = next((a for a in primary.add_ons if "roadside" in a.lower()), None)
            return (
                f"**{primary.name}** does not include roadside assistance as standard"
                + (f", but it is available as an add-on: {addon}." if addon else ".")
            )

        if intent == "rental":
            detail = next(
                (f for f in primary.features if "rental" in f.lower() or "loaner" in f.lower()), None
            )
            if detail:
                return f"**{primary.name}** includes rental cover: {detail}."
            return (
                f"Rental reimbursement is not listed in the **{primary.name}** feature set. "
                "I can check another product in your shortlist if that matters to you."
            )

        if intent == "addon":
            if primary.add_ons:
                bullets = "\n".join(f"- {item}" for item in primary.add_ons)
                return f"**{primary.name}** offers these optional add-ons:\n\n{bullets}"
            return f"**{primary.name}** has no optional add-ons listed — its cover is fixed as written."

        if intent == "compare" and len(policies) >= 2:
            a, b = policies[0], policies[1]
            lines = [f"Comparing **{a.name}** and **{b.name}**:", ""]
            lines.append(
                f"- **Base premium**: {_money(a.base_monthly)}/mo vs {_money(b.base_monthly)}/mo"
            )
            lines.append(
                f"- **Coverage limit**: {_money(a.coverage_limit)} vs {_money(b.coverage_limit)}"
            )
            lines.append(
                f"- **Standard deductible**: {_money(a.default_deductible)} vs {_money(b.default_deductible)}"
            )
            lines.append(
                f"- **Coverage breadth**: {a.coverage_breadth}/100 vs {b.coverage_breadth}/100"
            )
            lines.append(f"- **Perks**: {a.perks}/100 vs {b.perks}/100")
            lines.append("")
            lines.append(
                f"{a.name} suits {', '.join(a.ideal_tiers)} profiles; "
                f"{b.name} suits {', '.join(b.ideal_tiers)} profiles."
            )
            return "\n".join(lines)

        if intent == "eligibility":
            return (
                f"**{primary.name}** is written for drivers aged {primary.min_age}-{primary.max_age}, "
                f"up to {primary.max_annual_mileage:,} miles per year, on vehicles up to "
                f"{primary.max_vehicle_age} years old. It is designed for "
                f"{', '.join(primary.ideal_tiers)} risk profiles."
            )

        # Default: features
        bullets = "\n".join(f"- {item}" for item in primary.features[:6])
        return (
            f"**{primary.name}** ({primary.category}) — {primary.tagline}\n\n"
            f"Included as standard:\n\n{bullets}\n\n"
            f"Total limit {_money(primary.coverage_limit)} with a "
            f"{_money(primary.default_deductible)} standard deductible."
        )

    # ------------------------------------------------------------------ #
    def run(self, state: OnboardingState, question: str) -> str:
        """Orchestrator entry point."""
        answer, policies = self.answer(question, state)
        state.log("qa_answered", intent=_detect_intent(question), cited=[p.id for p in policies])
        return answer


qa_agent = QAAgent()
