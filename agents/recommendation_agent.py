"""
agents/recommendation_agent.py — RAG policy matching + pricing engine.

Pipeline
--------
1. Build a semantic query from the validated profile and risk assessment.
2. Retrieve candidates from the vector store (Chroma or TF-IDF).
3. Score every catalog product with a **multi-factor weighted formula** whose
   weights shift with the applicant's stated coverage preference.
4. Filter hard ineligibility (age, mileage, vehicle age) — these are
   underwriting rules, not preferences, so they cannot be out-voted by a good
   semantic score.
5. Return the top N with a generated rationale.

This module also owns the pricing engine (:func:`quote_premium`,
:func:`claim_outcome`) used by the dashboard charts, so premium maths lives in
exactly one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents.state import OnboardingState, PolicyMatch, RiskAssessment
from backend.vector_store import POLICY_CATALOG, Policy, get_vector_store
from config import settings
from security.validations import ApplicantProfile

logger = logging.getLogger(__name__)

__all__ = [
    "RecommendationAgent",
    "recommendation_agent",
    "quote_premium",
    "claim_outcome",
    "premium_curve",
    "ScoreBreakdown",
]

# Premium falls as the deductible rises; the exponent controls how steeply.
DEDUCTIBLE_EXPONENT = 0.28


# --------------------------------------------------------------------------- #
# Pricing engine
# --------------------------------------------------------------------------- #
def _deductible_factor(policy: Policy, deductible: int) -> float:
    """Relative price of *deductible* versus the product's standard deductible."""
    if deductible <= 0:
        deductible = policy.default_deductible
    return (policy.default_deductible / deductible) ** DEDUCTIBLE_EXPONENT


def _mileage_factor(policy: Policy, mileage: int) -> float:
    """Usage-based products price mileage explicitly; others load it mildly."""
    if policy.category == "Usage Based":
        # Base rate plus per-mile charge, expressed as a multiplier of base.
        per_mile_monthly = (mileage * 0.061) / 12.0
        return 1.0 + (per_mile_monthly / policy.base_monthly)
    if mileage >= 25_000:
        return 1.12
    if mileage >= 15_000:
        return 1.06
    if mileage < 6_000:
        return 0.94
    return 1.0


def quote_premium(
    policy: Policy,
    profile: ApplicantProfile,
    risk: RiskAssessment | None,
    deductible: int | None = None,
) -> float:
    """
    Monthly premium for a product/applicant/deductible combination.

    ``base × risk multiplier × deductible factor × mileage factor × (1 − discount)``
    """
    deductible = deductible or policy.default_deductible
    multiplier = risk.premium_multiplier if risk else 1.0
    discount = (risk.base_discount / 100.0) if risk else 0.0

    premium = (
        policy.base_monthly
        * multiplier
        * _deductible_factor(policy, deductible)
        * _mileage_factor(policy, profile.annual_mileage)
        * (1.0 - discount)
    )
    return round(max(premium, 12.0), 2)


def premium_curve(
    policy: Policy,
    profile: ApplicantProfile,
    risk: RiskAssessment | None,
    deductibles: tuple[int, ...] | None = None,
) -> list[tuple[int, float]]:
    """``[(deductible, monthly premium), ...]`` for the Zone C line chart."""
    options = deductibles or settings.deductible_options
    return [(d, quote_premium(policy, profile, risk, d)) for d in options]


def claim_outcome(policy: Policy, deductible: int, claim_amount: int) -> tuple[float, float, float]:
    """
    Split a claim into ``(you_pay, insurer_pays, uncovered)``.

    The applicant always pays the deductible first; the insurer then covers the
    remainder up to the product's coverage limit.
    """
    you_pay = float(min(deductible, claim_amount))
    remainder = max(0.0, claim_amount - you_pay)
    insurer_pays = float(min(remainder, policy.coverage_limit))
    uncovered = max(0.0, remainder - insurer_pays)
    return you_pay, insurer_pays, uncovered


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class ScoreBreakdown:
    """Transparent record of how a match score was assembled."""

    semantic: float
    risk_match: float
    affordability: float
    coverage: float
    flexibility: float
    perks: float
    weights: dict[str, float]
    total: float


# Weight profiles keyed by the applicant's stated coverage preference.
_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "basic": {
        "semantic": 0.20,
        "risk_match": 0.22,
        "affordability": 0.34,
        "coverage": 0.10,
        "flexibility": 0.09,
        "perks": 0.05,
    },
    "balanced": {
        "semantic": 0.22,
        "risk_match": 0.26,
        "affordability": 0.20,
        "coverage": 0.18,
        "flexibility": 0.08,
        "perks": 0.06,
    },
    "comprehensive": {
        "semantic": 0.20,
        "risk_match": 0.24,
        "affordability": 0.08,
        "coverage": 0.32,
        "flexibility": 0.06,
        "perks": 0.10,
    },
}


def _risk_match_score(policy: Policy, profile: ApplicantProfile, risk: RiskAssessment | None) -> float:
    """
    0-100 measure of how well the product's underwriting envelope fits the
    applicant — tier alignment plus headroom on age, mileage and vehicle age.
    """
    score = 50.0

    if risk:
        if risk.tier in policy.ideal_tiers:
            score += 28.0
        else:
            # Partial credit for an adjacent band.
            order = ["High Risk", "Elevated Risk", "Standard", "Preferred", "Low Risk"]
            try:
                distance = min(
                    abs(order.index(risk.tier) - order.index(t))
                    for t in policy.ideal_tiers
                    if t in order
                )
                score += max(-18.0, 14.0 - distance * 14.0)
            except (ValueError, TypeError):
                score -= 5.0

    # Age headroom.
    if policy.min_age <= profile.age <= policy.max_age:
        span = max(1, policy.max_age - policy.min_age)
        centre = (policy.min_age + policy.max_age) / 2.0
        score += 10.0 * (1.0 - min(1.0, abs(profile.age - centre) / (span / 2.0)))
    else:
        score -= 30.0

    # Mileage headroom.
    if profile.annual_mileage <= policy.max_annual_mileage:
        utilisation = profile.annual_mileage / max(1, policy.max_annual_mileage)
        score += 8.0 * (1.0 - min(1.0, utilisation))
    else:
        score -= 25.0

    # Vehicle age.
    if profile.vehicle_age <= policy.max_vehicle_age:
        score += 4.0
    else:
        score -= 20.0

    return round(max(0.0, min(100.0, score)), 1)


def _affordability_fit(policy: Policy, premium: float, budget_anchor: float) -> float:
    """
    0-100 for how the quoted premium sits against a personalised anchor.

    Uses the *quoted* premium rather than the product's static affordability
    score, because risk loading and deductible choice can move a "cheap"
    product above an "expensive" one for a given applicant.
    """
    if premium <= 0:
        return 0.0
    ratio = premium / max(budget_anchor, 1.0)
    if ratio <= 0.6:
        return 100.0
    if ratio >= 2.0:
        return 5.0
    # Linear decay between the two anchors.
    return round(max(5.0, 100.0 - (ratio - 0.6) * (95.0 / 1.4)), 1)


def _eligibility(policy: Policy, profile: ApplicantProfile) -> tuple[bool, str | None]:
    """Hard underwriting gates. A failure cannot be offset by a good score."""
    if profile.age < policy.min_age:
        return False, f"Requires drivers aged {policy.min_age}+"
    if profile.age > policy.max_age:
        return False, f"Available to drivers up to age {policy.max_age}"
    if profile.annual_mileage > policy.max_annual_mileage:
        return False, f"Mileage cap is {policy.max_annual_mileage:,} miles/year"
    if profile.vehicle_age > policy.max_vehicle_age:
        return False, f"Vehicles must be under {policy.max_vehicle_age} years old"
    return True, None


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class RecommendationAgent:
    """Retrieves, scores and ranks catalog products for a validated applicant."""

    name = "Policy Recommendation Agent"

    def __init__(self, store: Any | None = None) -> None:
        self._store = store

    @property
    def store(self) -> Any:
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_query(profile: ApplicantProfile, risk: RiskAssessment | None) -> str:
        """Natural-language query embedding the applicant's situation."""
        parts = [
            f"{profile.age} year old driver",
            f"{profile.display_vehicle}",
            f"{profile.annual_mileage} miles per year",
            f"{profile.coverage_preference} coverage",
        ]
        if profile.annual_mileage < 7_000:
            parts.append("low mileage usage based telematics")
        if profile.annual_mileage > 20_000:
            parts.append("high mileage commuter highway")
        if profile.age < 25:
            parts.append("young driver first time")
        if profile.age >= 55:
            parts.append("senior mature experienced driver")
        if profile.vehicle_age >= 25:
            parts.append("classic vintage collector agreed value")
        make = profile.vehicle_make.lower()
        if make in {"tesla", "rivian", "lucid", "polestar", "byd"}:
            parts.append("electric vehicle battery charger EV")
        if risk:
            parts.append(f"{risk.tier} risk tier")
            if risk.adverse:
                parts.append(risk.adverse[0].name)
        return ", ".join(parts)

    # ------------------------------------------------------------------ #
    def recommend(
        self,
        profile: ApplicantProfile,
        risk: RiskAssessment | None,
        top_n: int | None = None,
        include_ineligible: bool = False,
    ) -> list[PolicyMatch]:
        """Return the ranked shortlist of matches."""
        top_n = top_n or settings.recommendation_count

        query = self.build_query(profile, risk)
        hits = self.store.query(query, k=len(POLICY_CATALOG))
        semantic_by_id = {hit.policy.id: hit.score for hit in hits}

        # Normalise semantic scores to 0-100 within this result set so the
        # weighting stays comparable across backends (cosine ranges differ).
        raw_scores = [s for s in semantic_by_id.values() if s is not None]
        top_score = max(raw_scores) if raw_scores else 0.0

        weights = _WEIGHT_PROFILES.get(profile.coverage_preference, _WEIGHT_PROFILES["balanced"])

        # Personalised budget anchor: the median quoted premium across eligible
        # products, so "affordable" means affordable *for this applicant*.
        premiums = {p.id: quote_premium(p, profile, risk) for p in POLICY_CATALOG}
        ordered = sorted(premiums.values())
        budget_anchor = ordered[len(ordered) // 2] if ordered else 100.0
        if profile.coverage_preference == "basic":
            budget_anchor *= 0.75
        elif profile.coverage_preference == "comprehensive":
            budget_anchor *= 1.35

        matches: list[PolicyMatch] = []
        for policy in POLICY_CATALOG:
            eligible, reason = _eligibility(policy, profile)
            if not eligible and not include_ineligible:
                continue

            premium = premiums[policy.id]
            raw_semantic = semantic_by_id.get(policy.id, 0.0) or 0.0
            semantic = round((raw_semantic / top_score) * 100.0, 1) if top_score > 0 else 50.0
            risk_match = _risk_match_score(policy, profile, risk)
            affordability = _affordability_fit(policy, premium, budget_anchor)

            total = (
                semantic * weights["semantic"]
                + risk_match * weights["risk_match"]
                + affordability * weights["affordability"]
                + policy.coverage_breadth * weights["coverage"]
                + policy.flexibility * weights["flexibility"]
                + policy.perks * weights["perks"]
            )
            if not eligible:
                total *= 0.35  # keep it visible in the catalog, never at the top

            match = PolicyMatch(
                policy_id=policy.id,
                name=policy.name,
                insurer=policy.insurer,
                category=policy.category,
                tagline=policy.tagline,
                match_score=round(min(100.0, total), 1),
                semantic_score=semantic,
                risk_match=risk_match,
                affordability=affordability,
                coverage_breadth=float(policy.coverage_breadth),
                flexibility=float(policy.flexibility),
                perks=float(policy.perks),
                monthly_premium=premium,
                annual_premium=round(premium * 12, 2),
                deductible=policy.default_deductible,
                coverage_limit=policy.coverage_limit,
                eligible=eligible,
                ineligibility_reason=reason,
            )
            match.rationale = self._rationale(policy, match, profile, risk, weights)
            matches.append(match)

        matches.sort(key=lambda m: (-m.match_score, m.monthly_premium))
        return matches[:top_n] if not include_ineligible else matches

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rationale(
        policy: Policy,
        match: PolicyMatch,
        profile: ApplicantProfile,
        risk: RiskAssessment | None,
        weights: dict[str, float],
    ) -> str:
        """Grounded, template-generated explanation — no LLM, no hallucination."""
        if not match.eligible:
            return f"Not available for this profile — {match.ineligibility_reason}."

        reasons: list[str] = []

        if risk and risk.tier in policy.ideal_tiers:
            reasons.append(f"designed for {risk.tier} profiles like yours")

        if match.affordability >= 75:
            reasons.append(f"priced below your comparison set at ${match.monthly_premium:.0f}/mo")
        elif match.coverage_breadth >= 85:
            reasons.append(f"carries a ${policy.coverage_limit:,} limit")

        headroom = policy.max_annual_mileage - profile.annual_mileage
        if 0 < headroom < 4_000:
            reasons.append(f"your {profile.annual_mileage:,} miles sits just inside its cap")
        elif policy.category == "Usage Based" and profile.annual_mileage < 8_000:
            reasons.append(f"your {profile.annual_mileage:,} miles/year suits per-mile pricing")

        if policy.category == "Electric Vehicle":
            reasons.append("covers the traction battery and home charger")
        elif policy.category == "Young Driver" and profile.age < 27:
            reasons.append("telematics discounts reduce the young-driver loading")
        elif policy.category == "Senior" and profile.age >= 55:
            reasons.append("includes the mature-driver credit")

        if match.perks >= 85:
            reasons.append("top-tier benefits package")

        if not reasons:
            reasons.append(f"a balanced fit at {match.match_score:.0f}% overall match")

        lead = ", ".join(reasons[:3])
        return f"{policy.name} — {lead[0].upper()}{lead[1:]}."

    # ------------------------------------------------------------------ #
    def catalog_with_fit(
        self, profile: ApplicantProfile, risk: RiskAssessment | None
    ) -> list[PolicyMatch]:
        """All 12 products scored for the Zone D expandable catalog."""
        return self.recommend(profile, risk, top_n=len(POLICY_CATALOG), include_ineligible=True)

    # ------------------------------------------------------------------ #
    def run(self, state: OnboardingState) -> list[PolicyMatch]:
        """Orchestrator entry point."""
        if state.profile is None:
            logger.warning("Recommendation requested without a validated profile.")
            return []

        matches = self.recommend(state.profile, state.risk)
        state.recommendations = matches

        # Re-running underwriting can change the shortlist. Keep the applicant's
        # choice when it survives; otherwise re-anchor on the new best match so
        # nothing downstream is left pointing at a dropped product.
        if matches:
            available = {m.policy_id for m in matches}
            if state.selected_policy_id not in available:
                state.selected_policy_id = matches[0].policy_id
                state.selected_deductible = matches[0].deductible
        state.log(
            "policies_matched",
            count=len(matches),
            top=[f"{m.name} ({m.match_score})" for m in matches],
            backend=getattr(self.store, "backend", "unknown"),
        )
        return matches


recommendation_agent = RecommendationAgent()
