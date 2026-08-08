"""
agents/risk_agent.py — Underwriting & risk evaluation (tool node).

This agent is deliberately **not** an LLM decision-maker. Scores that affect
pricing and eligibility are computed by a transparent, auditable formula; the
model is used only to phrase the explanation, and even that has a deterministic
fallback.

Scoring convention
------------------
``score`` is a 0-100 **safety** score — higher is better. 85+ maps to
"Low Risk - Class A+". Every adjustment is recorded as a
:class:`~agents.state.RiskFactor` so the dashboard can show exactly why a
number came out the way it did.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.state import OnboardingState, RiskAssessment, RiskFactor
from backend.mock_apis import geo_hazard_check, mvr_check
from config import settings, tier_for_score
from llm_factory import get_llm, is_offline
from security.validations import ApplicantProfile

logger = logging.getLogger(__name__)

__all__ = ["RiskAgent", "risk_agent"]

BASE_SCORE = 70.0


# --------------------------------------------------------------------------- #
# Factor tables
# --------------------------------------------------------------------------- #
def _age_factor(age: int) -> RiskFactor:
    if age < 21:
        impact, detail = -18.0, "Drivers under 21 have the highest claim frequency of any band."
    elif age < 25:
        impact, detail = -12.0, "Drivers aged 21-24 remain in an elevated-frequency band."
    elif age < 30:
        impact, detail = -3.0, "Drivers aged 25-29 are near the portfolio average."
    elif age < 60:
        impact, detail = 8.0, "Ages 30-59 are the lowest-frequency band in the portfolio."
    elif age < 70:
        impact, detail = 4.0, "Ages 60-69 retain a better-than-average claim profile."
    elif age < 80:
        impact, detail = -4.0, "Claim severity begins to rise after age 70."
    else:
        impact, detail = -12.0, "Drivers over 80 show materially higher severity."
    return RiskFactor(name=f"Age {age}", impact=impact, detail=detail, category="driver")


def _experience_factor(years: int | None, age: int) -> RiskFactor:
    effective = years if years is not None else max(0, age - 18)
    if effective >= 15:
        impact, detail = 7.0, f"{effective} years of licensed driving is a strong stability signal."
    elif effective >= 10:
        impact, detail = 5.0, f"{effective} years of licensed driving is above average."
    elif effective >= 5:
        impact, detail = 2.0, f"{effective} years of experience is around the portfolio average."
    elif effective >= 2:
        impact, detail = -4.0, f"Only {effective} years of licensed driving."
    else:
        impact, detail = -10.0, "Newly licensed drivers carry the highest first-year loss ratio."
    return RiskFactor(
        name=f"{effective} years licensed", impact=impact, detail=detail, category="driver"
    )


def _mileage_factor(mileage: int) -> RiskFactor:
    if mileage < 5_000:
        impact, detail = 8.0, f"{mileage:,} miles/year is very low exposure."
    elif mileage < 10_000:
        impact, detail = 5.0, f"{mileage:,} miles/year is below the national average."
    elif mileage < 15_000:
        impact, detail = 1.0, f"{mileage:,} miles/year is close to average exposure."
    elif mileage < 25_000:
        impact, detail = -5.0, f"{mileage:,} miles/year is above-average road exposure."
    else:
        impact, detail = -11.0, f"{mileage:,} miles/year is high-exposure commuting."
    return RiskFactor(
        name=f"{mileage:,} miles/year", impact=impact, detail=detail, category="usage"
    )


def _vehicle_factor(vehicle_age: int, display: str) -> RiskFactor:
    if vehicle_age <= 3:
        impact, detail = 4.0, "Modern safety systems reduce both frequency and severity."
    elif vehicle_age <= 9:
        impact, detail = 1.0, "Vehicle retains most current-generation safety equipment."
    elif vehicle_age <= 19:
        impact, detail = -3.0, "Older vehicles lack current driver-assistance systems."
    else:
        impact, detail = -6.0, "Parts availability and passive safety are materially dated."
    return RiskFactor(
        name=f"{display} ({vehicle_age}y old)", impact=impact, detail=detail, category="vehicle"
    )


def _geo_factor(geo: dict[str, Any]) -> RiskFactor:
    band = geo.get("hazard_band", "MODERATE")
    impact = {"BENIGN": 6.0, "MODERATE": 0.0, "ELEVATED": -6.0, "SEVERE": -12.0}.get(band, 0.0)
    return RiskFactor(
        name=f"{geo.get('locality', 'Location')} — {band.title()}",
        impact=impact,
        detail=(
            f"Composite hazard {geo.get('composite_hazard', 0)}/100. "
            f"Dominant exposure: {geo.get('dominant_hazard', 'unknown')}."
        ),
        category="location",
    )


def _history_factors(mvr: dict[str, Any]) -> list[RiskFactor]:
    factors: list[RiskFactor] = []

    accidents = int(mvr.get("accidents_5yr", 0))
    if accidents:
        factors.append(
            RiskFactor(
                name=f"{accidents} at-fault accident{'s' if accidents > 1 else ''} (5 yr)",
                impact=max(-27.0, -9.0 * accidents),
                detail="Prior accident frequency is the strongest single predictor of future loss.",
                category="history",
            )
        )
    else:
        factors.append(
            RiskFactor(
                name="No accidents on record",
                impact=6.0,
                detail="A clean 5-year accident history qualifies for the safe-driver credit.",
                category="history",
            )
        )

    violations = int(mvr.get("violations_5yr", 0))
    if violations:
        factors.append(
            RiskFactor(
                name=f"{violations} moving violation{'s' if violations > 1 else ''}",
                impact=max(-16.0, -4.0 * violations),
                detail="Moving violations correlate with elevated claim frequency.",
                category="history",
            )
        )

    if mvr.get("dui_flag"):
        factors.append(
            RiskFactor(
                name="DUI conviction on record",
                impact=-25.0,
                detail="A DUI triggers mandatory high-risk classification and manual review.",
                category="history",
            )
        )

    if mvr.get("licence_status") == "SUSPENDED":
        factors.append(
            RiskFactor(
                name="Licence currently suspended",
                impact=-15.0,
                detail="Suspended licences are ineligible for standard-market products.",
                category="history",
            )
        )

    undisclosed = int(mvr.get("undisclosed_incidents", 0))
    if undisclosed:
        factors.append(
            RiskFactor(
                name=f"{undisclosed} undisclosed incident{'s' if undisclosed > 1 else ''}",
                impact=-6.0 * undisclosed,
                detail="DMV records show incidents not declared on the application.",
                category="history",
            )
        )

    return factors


# --------------------------------------------------------------------------- #
# Discount & pricing
# --------------------------------------------------------------------------- #
_TIER_DISCOUNT = {"Low Risk": 15.0, "Preferred": 10.0, "Standard": 5.0, "Elevated Risk": 0.0, "High Risk": 0.0}


def _compute_discount(tier: str, profile: ApplicantProfile, mvr: dict[str, Any]) -> float:
    discount = _TIER_DISCOUNT.get(tier, 0.0)
    if discount <= 0:
        return 0.0
    if mvr.get("clean_record"):
        discount += 5.0
    if profile.annual_mileage < 8_000:
        discount += 3.0
    if (profile.years_licensed or 0) >= 20:
        discount += 2.0
    return round(min(discount, 25.0), 1)


def _premium_multiplier(score: float, geo_multiplier: float) -> float:
    """Map the safety score onto a rating multiplier, then apply geo loading."""
    base = 1.60 - (score / 100.0) * 0.70   # score 100 -> 0.90, score 0 -> 1.60
    return round(max(0.75, base) * geo_multiplier, 4)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
_SUMMARY_SYSTEM = """You are an underwriting analyst writing a two-sentence risk \
summary for the applicant to read.

Rules:
- Use ONLY the supplied factors and score. Invent nothing.
- Never guarantee coverage, approval or pricing.
- Be plain and non-judgemental; name the single biggest driver and one lever \
the applicant can actually influence.
- Two sentences maximum."""


class RiskAgent:
    """Calls underwriting services and produces the composite risk assessment."""

    name = "Risk Assessment Agent"

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = get_llm(temperature=0.2)
        return self._llm

    # ------------------------------------------------------------------ #
    def assess(self, profile: ApplicantProfile) -> RiskAssessment:
        """Run the full underwriting pass for a validated profile."""
        subject_key = f"{profile.full_name}|{profile.zip_code}|{profile.vehicle_year}"

        mvr_response = mvr_check(
            subject_key=subject_key,
            age=profile.age,
            years_licensed=profile.years_licensed,
            declared_claims=profile.prior_claims,
        )
        geo_response = geo_hazard_check(profile.zip_code)

        mvr = mvr_response.data if mvr_response.ok else {}
        geo = geo_response.data if geo_response.ok else {}

        if not mvr_response.ok:
            logger.warning("MVR check failed: %s", mvr_response.error)
        if not geo_response.ok:
            logger.warning("Geo-hazard check failed: %s", geo_response.error)

        factors: list[RiskFactor] = [
            _age_factor(profile.age),
            _experience_factor(profile.years_licensed, profile.age),
            _mileage_factor(profile.annual_mileage),
            _vehicle_factor(profile.vehicle_age, profile.display_vehicle),
        ]
        factors.extend(_history_factors(mvr))
        if geo:
            factors.append(_geo_factor(geo))

        raw_score = BASE_SCORE + sum(f.impact for f in factors)
        score = round(max(0.0, min(100.0, raw_score)), 1)
        tier, class_code = tier_for_score(score)

        primary = max(factors, key=lambda f: abs(f.impact))
        discount = _compute_discount(tier, profile, mvr)
        multiplier = _premium_multiplier(score, float(geo.get("premium_multiplier", 1.0)))

        assessment = RiskAssessment(
            score=score,
            tier=tier,
            class_code=class_code,
            base_discount=discount,
            premium_multiplier=multiplier,
            primary_factor=primary.name,
            factors=factors,
            mvr=mvr,
            geo=geo,
        )
        assessment.summary = self._summarise(profile, assessment)
        return assessment

    # ------------------------------------------------------------------ #
    def _summarise(self, profile: ApplicantProfile, assessment: RiskAssessment) -> str:
        """LLM-phrased summary with a deterministic template fallback."""
        deterministic = self._template_summary(profile, assessment)

        llm = self.llm
        if is_offline(llm):
            return deterministic

        factor_lines = "\n".join(
            f"- {f.name}: {f.impact:+.0f} points ({f.detail})" for f in assessment.factors
        )
        prompt = (
            f"Risk score: {assessment.score}/100 ({assessment.tier_label}).\n"
            f"Base discount: {assessment.base_discount}%.\n"
            f"Applicant: {profile.underwriting_payload()}\n"
            f"Factors:\n{factor_lines}"
        )

        try:
            response = llm.invoke(
                [{"role": "system", "content": _SUMMARY_SYSTEM}, {"role": "user", "content": prompt}]
            )
            text = getattr(response, "content", "").strip()
            return text or deterministic
        except Exception as exc:
            logger.warning("Risk summary generation failed (%s) — using template.", exc)
            return deterministic

    @staticmethod
    def _template_summary(profile: ApplicantProfile, assessment: RiskAssessment) -> str:
        favourable = assessment.favourable
        adverse = assessment.adverse

        lead = (
            f"Your profile scores {assessment.score:.0f}/100, placing you in the "
            f"{assessment.tier_label} band"
        )
        lead += (
            f" with a {assessment.base_discount:.0f}% base discount."
            if assessment.base_discount
            else "."
        )

        if favourable:
            best = favourable[0]
            lead += f" The strongest positive is {best.name.lower()} ({best.impact:+.0f} points)."

        if adverse:
            worst = adverse[0]
            lever = {
                "usage": "Reducing annual mileage would lift your score at renewal.",
                "history": "Your score improves automatically as incidents age out of the 5-year window.",
                "vehicle": "A newer vehicle with current safety systems would improve this.",
                "location": "This is a location-based loading and is outside your control.",
                "driver": "This factor improves with additional claim-free driving experience.",
            }.get(worst.category, "")
            lead += f" The largest drag is {worst.name.lower()} ({worst.impact:+.0f}). {lever}"

        return lead.strip()

    # ------------------------------------------------------------------ #
    def run(self, state: OnboardingState) -> RiskAssessment | None:
        """Orchestrator entry point — assess and write into state."""
        if state.profile is None:
            logger.warning("Risk assessment requested without a validated profile.")
            return None

        assessment = self.assess(state.profile)
        state.risk = assessment
        state.log(
            "risk_assessed",
            score=assessment.score,
            tier=assessment.tier_label,
            primary_factor=assessment.primary_factor,
        )
        return assessment


risk_agent = RiskAgent()
