"""
security/guardrails_runtime.py — The active AI security shield.

Every LLM interaction is wrapped::

    user text ─► PII firewall ─► check_input()  ─► agent ─► check_output() ─► UI

Two interchangeable engines implement the same :class:`RailDecision` contract:

* **NeMo engine** — loads ``security/guardrails/{config.yml,rails.co}`` through
  ``nemoguardrails.LLMRails``. Semantic, embedding-based intent matching.
* **Deterministic engine** — a dependency-free mirror of the same Colang rules
  built from curated regex signatures. Always available, always on.

The deterministic engine is not merely a stub: it runs *first* in both modes as
a fast pre-filter, because a known-signature injection should never cost an LLM
round-trip. NeMo then adds semantic coverage for novel phrasings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Pattern

from config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "RailCategory",
    "RailDecision",
    "GuardrailsShield",
    "get_shield",
    "SAFE_RESPONSES",
]


# --------------------------------------------------------------------------- #
# Decision contract
# --------------------------------------------------------------------------- #
class RailCategory(str, Enum):
    ALLOWED = "allowed"
    # input rails
    INSTRUCTION_OVERRIDE = "instruction_override"
    PROMPT_EXTRACTION = "prompt_extraction"
    PERSONA_JAILBREAK = "persona_jailbreak"
    ENCODING_SMUGGLE = "encoding_smuggle"
    UNDERWRITING_MANIPULATION = "underwriting_manipulation"
    OFF_TOPIC = "off_topic"
    LEGAL_MEDICAL = "legal_medical"
    HARMFUL = "harmful"
    # output rails
    BINDING_LANGUAGE = "binding_language"
    SYSTEM_LEAK = "system_leak"
    PII_LEAK = "pii_leak"
    UNGROUNDED = "ungrounded"


@dataclass
class RailDecision:
    """Result of a single rail evaluation."""

    allowed: bool
    rail: str                      # "input" | "output"
    category: RailCategory = RailCategory.ALLOWED
    reason: str = ""
    safe_response: str | None = None
    engine: str = "deterministic"
    matched: str | None = None
    score: float | None = None
    text: str = ""

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def as_log(self) -> dict[str, Any]:
        return {
            "rail": self.rail,
            "allowed": self.allowed,
            "category": self.category.value,
            "engine": self.engine,
            "reason": self.reason,
            "matched": self.matched,
        }


# --------------------------------------------------------------------------- #
# Safe fallback responses — mirror the `define bot ...` blocks in rails.co
# --------------------------------------------------------------------------- #
SAFE_RESPONSES: dict[RailCategory, str] = {
    RailCategory.INSTRUCTION_OVERRIDE: (
        "I can't set aside my operating rules. I'm AegisAI, and I'm here specifically to "
        "help you complete your insurance onboarding. Shall we continue with your quote?"
    ),
    RailCategory.PROMPT_EXTRACTION: (
        "I'm not able to share my internal instructions or configuration. I can tell you "
        "exactly what I do, though: I collect your intake details, assess risk, match "
        "policies from our catalog, and guide you through e-KYC. Where would you like to "
        "pick up?"
    ),
    RailCategory.PERSONA_JAILBREAK: (
        "I'll stay in my one role — your insurance onboarding assistant. That's the only "
        "mode I operate in. What can I help you with on your policy?"
    ),
    RailCategory.ENCODING_SMUGGLE: (
        "I can't execute instructions embedded in encoded or quoted content. If you have "
        "a question about your policy or application, ask me directly and I'll help."
    ),
    RailCategory.UNDERWRITING_MANIPULATION: (
        "Risk scores and eligibility are produced by our underwriting engine from verified "
        "data, and I can't alter them. I can explain exactly which factors drive your score "
        "and which ones you're able to improve. Would that help?"
    ),
    RailCategory.OFF_TOPIC: (
        "That's outside what I can help with — I'm limited to insurance onboarding. Let's "
        "get back to your policy: would you like to review your recommended plans or "
        "continue your application?"
    ),
    RailCategory.LEGAL_MEDICAL: (
        "I can't offer legal or medical advice. What I can do is explain what your policy "
        "covers and how claims are handled. Would you like me to walk through your coverage?"
    ),
    RailCategory.HARMFUL: (
        "I'm not able to help with that. I'm here for your insurance onboarding — let me "
        "know how I can help with your application."
    ),
    RailCategory.BINDING_LANGUAGE: (
        "Let me restate that more precisely: I can describe what a policy is designed to "
        "cover, but only your issued policy document determines actual coverage, and claims "
        "are assessed individually. Would you like me to pull up the specific coverage terms?"
    ),
    RailCategory.SYSTEM_LEAK: (
        "I'm not able to share internal configuration. Let's continue with your application "
        "— what would you like to know about your policy options?"
    ),
    RailCategory.PII_LEAK: (
        "I've withheld that response because it contained unmasked personal identifiers. "
        "Your details are stored securely and only ever shown in masked form."
    ),
    RailCategory.UNGROUNDED: (
        "I don't have that detail in the approved policy documents, so I won't guess. I can "
        "connect you with a licensed agent, or answer a different question from the catalog "
        "information I do have."
    ),
}


# --------------------------------------------------------------------------- #
# Deterministic signature engine
# --------------------------------------------------------------------------- #
def _compile(patterns: Iterable[str]) -> list[Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_INPUT_SIGNATURES: list[tuple[RailCategory, list[Pattern[str]]]] = [
    (
        RailCategory.INSTRUCTION_OVERRIDE,
        _compile(
            [
                r"\b(ignore|disregard|forget|discard|override|bypass|reset|erase)\b[\w\s,'-]{0,30}"
                r"\b(previous|prior|earlier|above|initial|original|all|any|your)\b"
                r"[\w\s,'-]{0,20}\b(instruction|prompt|rule|guardrail|direction|context|"
                r"constraint|restriction|guideline|polic(?:y|ies)|training)",
                r"\b(new|updated|revised)\s+(instructions?|rules?|directives?|system\s+prompt)\s*(are|is|:)",
                r"\bfrom now on\b[\w\s,'-]{0,40}\b(you|your)\b[\w\s,'-]{0,30}\b(no|without|ignore|free)\b",
                r"\bstop\s+(being|acting\s+as|behaving\s+like)\b",
                r"\b(disable|turn\s+off|remove|drop)\b[\w\s,'-]{0,20}\b(guardrail|safety|filter|"
                r"restriction|moderation|censorship)",
                r"\bdo\s+(anything|whatever)\s+(now|i\s+say)\b",
            ]
        ),
    ),
    (
        RailCategory.PROMPT_EXTRACTION,
        _compile(
            [
                r"\b(what|show|reveal|print|output|repeat|display|tell\s+me|give\s+me|list)\b"
                r"[\w\s,'-]{0,25}\b(your|the)\b[\w\s,'-]{0,15}"
                r"\b(system\s+prompt|initial\s+prompt|original\s+prompt|developer\s+message|"
                r"instructions?|configuration|config|guardrails?|rules?|prompt\s+template|"
                r"tool\s+(schema|definition)s?|function\s+definitions?)",
                r"\brepeat\b[\w\s,'-]{0,20}\b(text|words?|everything|content)\b[\w\s,'-]{0,15}\babove\b",
                r"\beverything\s+(between|inside|within)\b[\w\s,'-]{0,20}\b(tags?|delimiters?|brackets?)",
                r"\bwhat\s+(were|was)\s+you\s+(told|instructed|programmed)\b",
                r"\bverbatim\b[\w\s,'-]{0,25}\b(prompt|instruction)",
                r"\b(begin|start)\s+your\s+(reply|response|answer)\s+with\b",
            ]
        ),
    ),
    (
        RailCategory.PERSONA_JAILBREAK,
        _compile(
            [
                r"\byou\s+are\s+now\b[\w\s,'-]{0,20}\b(dan|stan|dude|unrestricted|unfiltered|jailbroken)\b",
                r"\b(dan|developer|god|sudo|root|admin|debug|maintenance)\s+mode\b",
                r"\b(pretend|act|behave|roleplay|role-play|simulate)\b[\w\s,'-]{0,25}"
                r"\b(no|without|zero)\s+(rules?|filters?|restrictions?|guardrails?|limits?|ethics)",
                r"\b(pretend|act)\s+(to\s+be|as)\s+an?\s+(unrestricted|unfiltered|uncensored|evil)\b",
                r"\bhypothetical(?:ly)?\b[\w\s,'-]{0,30}\b(no|without)\s+(rules?|restrictions?|guardrails?)",
                r"\bjailbr(?:eak|oken)\b",
                r"\bopposite\s+day\b",
                r"\byour\s+(alter\s+ego|evil\s+twin)\b",
            ]
        ),
    ),
    (
        RailCategory.ENCODING_SMUGGLE,
        _compile(
            [
                r"</?(?:system|assistant|developer|im_start|im_end|instruction)>",
                r"\[\s*(?:system|inst|instruction)\s*\]",
                r"\b(decode|decrypt|un-?base64|rot-?13|from\s+hex)\b[\w\s,'-]{0,30}"
                r"\b(then|and)\b[\w\s,'-]{0,15}\b(follow|execute|obey|do|run)\b",
                r"\btranslate\b[\w\s,'-]{0,30}\b(then|and)\b[\w\s,'-]{0,15}\b(follow|execute|obey|do)\b",
                r"\bexecute\b[\w\s,'-]{0,20}\b(instructions?|commands?|code)\b[\w\s,'-]{0,20}"
                r"\b(inside|within|between|in\s+the)\b",
                r"#{3,}\s*(?:end\s+of\s+|new\s+)?(?:system|instruction|prompt)",
            ]
        ),
    ),
    (
        RailCategory.UNDERWRITING_MANIPULATION,
        _compile(
            [
                r"\b(set|change|make|force|override|adjust|bump|raise|fix)\b[\w\s,'-]{0,25}"
                r"\b(risk\s+score|risk\s+tier|premium|discount|rating)\b[\w\s,'-]{0,20}"
                r"\b(to|as|at)\b",
                r"\bmark\s+me\s+as\b[\w\s,'-]{0,20}\b(low\s+risk|safe|preferred|class\s+a)\b",
                r"\b(skip|bypass|avoid|ignore|forget|without)\b[\w\s,'-]{0,20}"
                r"\b(kyc|verification|identity\s+check|aadhaar|otp|underwriting|mvr|"
                r"background\s+check)\b",
                r"\b(issue|approve|bind|activate)\b[\w\s,'-]{0,25}\bwithout\b[\w\s,'-]{0,25}"
                r"\b(verif|check|kyc|document)",
                r"\b(fake|forged|false|dummy|someone\s+else'?s?)\b[\w\s,'-]{0,15}"
                r"\b(aadhaar|identity|id|document|licence|license|record)\b",
                r"\b(hide|conceal|omit|don'?t\s+report|leave\s+out)\b[\w\s,'-]{0,25}"
                r"\b(accident|violation|claim|ticket|dui)\b",
                r"\bmaximum\s+discount\b[\w\s,'-]{0,25}\bregardless\b",
            ]
        ),
    ),
    (
        RailCategory.LEGAL_MEDICAL,
        _compile(
            [
                r"\b(should|can)\s+i\s+sue\b",
                r"\bwhat\s+(medication|medicine|drug|dosage)\b[\w\s,'-]{0,15}\b(should|do)\s+i\b",
                r"\b(win|fight)\s+my\s+(court\s+case|lawsuit|legal\s+battle)\b",
                r"\bdiagnos(?:e|is)\b[\w\s,'-]{0,15}\bmy\b",
                r"\blegally\s+binding\b[\w\s,'-]{0,20}\bin\s+my\s+(state|country|jurisdiction)\b",
            ]
        ),
    ),
    (
        RailCategory.OFF_TOPIC,
        _compile(
            [
                r"\bwrite\s+(me\s+)?(a|an|some)?\s*(python|javascript|java|c\+\+|sql|bash|html|"
                r"react|code|script|program|function|regex)\b",
                r"\b(debug|fix)\b[\w\s,'-]{0,15}\b(my|this)\b[\w\s,'-]{0,15}\b(code|script|bug|app|"
                r"function|error)\b",
                r"\brecipes?\b",
                r"\bhow\s+(do\s+i|to)\s+(cook|bake|make)\b[\w\s,'-]{0,20}\b(pasta|bread|cake|"
                r"chicken|curry|dinner|breakfast|lunch)\b",
                r"\bwho\s+should\s+i\s+vote\s+for\b",
                r"\b(write|compose|create)\s+(me\s+)?(a|an)\s+(poem|song|story|essay|novel|screenplay|"
                r"haiku|rap|joke)\b",
                r"\b(my|the)\s+(math|physics|chemistry|history)\s+homework\b",
                r"\bwhat'?s?\s+the\s+(weather|score|capital\s+of)\b",
                r"\btell\s+me\s+a\s+joke\b",
                r"\b(bitcoin|crypto|stock)\s+(price|prediction|tip)\b",
            ]
        ),
    ),
    (
        RailCategory.HARMFUL,
        _compile(
            [
                r"\bhow\s+to\s+(make|build|synthesi[sz]e)\b[\w\s,'-]{0,20}\b(bomb|explosive|weapon|"
                r"poison|meth|virus|malware|ransomware)\b",
                r"\b(kill|harm|hurt|attack)\s+(myself|yourself|someone|people)\b",
                r"\b(hack|breach|ddos|exploit)\b[\w\s,'-]{0,20}\b(into|the|their|someone)\b",
            ]
        ),
    ),
]


# Onboarding vocabulary — presence of these terms vetoes a weak OFF_TOPIC match,
# preventing false positives such as "does my policy cover a python bite".
_IN_SCOPE_TERMS = _compile(
    [
        r"\b(insurance|policy|policies|premium|deductible|coverage|covered|claim|underwrit|"
        r"quote|risk\s+score|risk\s+tier|kyc|aadhaar|otp|liability|collision|comprehensive|"
        r"roadside|endorsement|rider|no.?claim|renewal|vehicle|car|driver|licen[cs]e|mileage|"
        r"discount|plan|coverage\s+limit|onboarding|verify|verification)\b"
    ]
)

_OUTPUT_SIGNATURES: list[tuple[RailCategory, list[Pattern[str]]]] = [
    (
        RailCategory.BINDING_LANGUAGE,
        _compile(
            [
                r"\bi\s+(guarantee|promise|assure\s+you|certify)\b",
                r"\b(guaranteed|promised)\s+(coverage|approval|payout|acceptance|rate)\b",
                r"\bcovers\s+(everything|all\s+(damages?|losses?|scenarios?|situations?|events?))\b",
                r"\b(you|this)\s+(are|is|will\s+be)\s+(definitely|certainly|absolutely|100%|fully)\s+"
                r"(covered|approved|protected|insured)\b",
                r"\byou\s+will\s+(definitely|certainly|always)\s+(be\s+)?(approved|covered|paid)\b",
                r"\bno\s+(exclusions?|exceptions?|limits?|conditions?)\s+(at\s+all|whatsoever|apply)\b",
                r"\b(zero|no)\s+risk\s+of\s+(denial|rejection|refusal)\b",
                r"\byour\s+claim\s+will\s+(be\s+)?(approved|paid|settled)\b",
                r"\b100%\s+(coverage|covered|guaranteed|protection)\b",
                r"\bnever\s+be\s+(denied|rejected|refused)\b",
            ]
        ),
    ),
    (
        RailCategory.SYSTEM_LEAK,
        _compile(
            [
                r"\bmy\s+system\s+prompt\s+(is|says|states)\b",
                r"\bi\s+was\s+instructed\s+to\b",
                r"\bmy\s+(instructions?|configuration|guardrails?)\s+(are|is|say|state)\b",
                r"</?(?:system|im_start|im_end)>",
                r"\byou\s+are\s+aegisai,?\s+a\s+licensed\s+insurance\s+onboarding\s+assistant\b",
            ]
        ),
    ),
    (
        RailCategory.PII_LEAK,
        _compile(
            [
                r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",   # unmasked Aadhaar
                r"(?<!\d)(?:\d[ -]?){15,16}(?!\d)",                  # card-length digit run
            ]
        ),
    ),
]

_GUARANTEE_SOFTENERS = _compile(
    [r"\bsubject\s+to\b", r"\bterms\s+and\s+conditions\b", r"\bpolicy\s+document\s+(governs|determines)\b"]
)


class DeterministicEngine:
    """Regex-signature mirror of the Colang rails. Zero dependencies, always on."""

    name = "deterministic"

    # -- input ------------------------------------------------------------ #
    def check_input(self, text: str) -> RailDecision:
        if not text or not text.strip():
            return RailDecision(allowed=True, rail="input", text=text)

        for category, patterns in _INPUT_SIGNATURES:
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                # Weak-signal veto: an off-topic hit inside a genuine insurance
                # question is a false positive.
                if category is RailCategory.OFF_TOPIC and self._is_in_scope(text):
                    continue
                return RailDecision(
                    allowed=False,
                    rail="input",
                    category=category,
                    reason=f"Matched {category.value} signature.",
                    safe_response=SAFE_RESPONSES[category],
                    engine=self.name,
                    matched=match.group(0)[:120],
                    text=text,
                )
        return RailDecision(allowed=True, rail="input", engine=self.name, text=text)

    @staticmethod
    def _is_in_scope(text: str) -> bool:
        return any(p.search(text) for p in _IN_SCOPE_TERMS)

    # -- output ----------------------------------------------------------- #
    def check_output(self, text: str, evidence: str = "") -> RailDecision:
        if not text or not text.strip():
            return RailDecision(allowed=True, rail="output", text=text)

        for category, patterns in _OUTPUT_SIGNATURES:
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                if category is RailCategory.BINDING_LANGUAGE and any(
                    s.search(text) for s in _GUARANTEE_SOFTENERS
                ):
                    continue
                return RailDecision(
                    allowed=False,
                    rail="output",
                    category=category,
                    reason=f"Output violated {category.value} rail.",
                    safe_response=SAFE_RESPONSES[category],
                    engine=self.name,
                    matched=match.group(0)[:120],
                    text=text,
                )

        if settings.strict_output_rails and evidence:
            grounding = self.grounding_score(text, evidence)
            if grounding < 0.7:
                return RailDecision(
                    allowed=False,
                    rail="output",
                    category=RailCategory.UNGROUNDED,
                    reason=f"Groundedness {grounding:.0%} below the 70% threshold.",
                    safe_response=SAFE_RESPONSES[RailCategory.UNGROUNDED],
                    engine=self.name,
                    score=grounding,
                    text=text,
                )

        return RailDecision(allowed=True, rail="output", engine=self.name, text=text)

    # -- groundedness ----------------------------------------------------- #
    @staticmethod
    def grounding_score(response: str, evidence: str) -> float:
        """
        Fraction of quantitative claims in *response* traceable to *evidence*.

        Insurance hallucinations are almost always numeric — an invented limit,
        premium or deductible. Checking every number in the answer against the
        retrieved context catches the class of error that actually matters,
        deterministically and without a second model call.
        """
        numbers = re.findall(r"\$?\s?\d[\d,]*(?:\.\d+)?%?", response or "")
        claims = []
        for raw in numbers:
            token = raw.strip().lstrip("$").strip()
            normalised = token.replace(",", "").rstrip("%")
            try:
                value = float(normalised)
            except ValueError:
                continue
            # Ignore trivia: small counts, years, list numbering.
            if value < 10 or 1900 <= value <= 2100:
                continue
            claims.append(normalised)

        if not claims:
            return 1.0

        evidence_numbers = {
            n.replace(",", "").rstrip("%")
            for n in re.findall(r"\d[\d,]*(?:\.\d+)?%?", evidence or "")
        }
        supported = sum(
            1
            for claim in claims
            if claim in evidence_numbers
            or claim.rstrip("0").rstrip(".") in {e.rstrip("0").rstrip(".") for e in evidence_numbers}
        )
        return supported / len(claims)


# --------------------------------------------------------------------------- #
# NeMo Guardrails engine
# --------------------------------------------------------------------------- #
class NeMoEngine:
    """Wraps ``nemoguardrails.LLMRails`` around the Colang configuration."""

    name = "nemo"

    def __init__(self, rails: Any) -> None:
        self.rails = rails

    @classmethod
    def try_load(cls) -> "NeMoEngine | None":
        try:
            from nemoguardrails import LLMRails, RailsConfig  # type: ignore
        except Exception as exc:
            logger.info("nemoguardrails unavailable (%s) — deterministic rails only.", exc)
            return None

        if not settings.guardrails_path.exists():
            logger.warning("Guardrails path missing: %s", settings.guardrails_path)
            return None

        try:
            config = RailsConfig.from_path(str(settings.guardrails_path))
            rails = LLMRails(config)
            rails.register_action(_action_check_binding_language, "check_binding_language")
            logger.info("NeMo Guardrails loaded from %s", settings.guardrails_path)
            return cls(rails)
        except Exception as exc:
            logger.warning("NeMo Guardrails failed to initialise: %s", exc)
            return None

    def check_input(self, text: str) -> RailDecision:
        try:
            result = self.rails.generate(messages=[{"role": "user", "content": text}])
        except Exception as exc:
            logger.warning("NeMo input rail error: %s", exc)
            return RailDecision(allowed=True, rail="input", engine=self.name, text=text)

        content = (result or {}).get("content", "") if isinstance(result, dict) else str(result)
        category = _category_from_response(content)
        if category is not None:
            return RailDecision(
                allowed=False,
                rail="input",
                category=category,
                reason="NeMo input rail triggered.",
                safe_response=content or SAFE_RESPONSES[category],
                engine=self.name,
                text=text,
            )
        return RailDecision(allowed=True, rail="input", engine=self.name, text=text)

    def check_output(self, text: str, evidence: str = "") -> RailDecision:
        # Output rails run inside `generate`; for post-hoc inspection of text
        # produced by our own agents, the deterministic engine is authoritative.
        return DeterministicEngine().check_output(text, evidence)


def _action_check_binding_language(text: str = "") -> bool:
    """Custom Colang action referenced by `check binding language` in rails.co."""
    for category, patterns in _OUTPUT_SIGNATURES:
        if category is not RailCategory.BINDING_LANGUAGE:
            continue
        return any(p.search(text or "") for p in patterns)
    return False


def _category_from_response(content: str) -> RailCategory | None:
    """Map a NeMo refusal back onto our taxonomy by matching the safe response."""
    if not content:
        return None
    normalised = content.strip().lower()
    for category, response in SAFE_RESPONSES.items():
        # Compare on a distinctive prefix; NeMo may reflow whitespace.
        if normalised.startswith(response.strip().lower()[:45]):
            return category
    return None


# --------------------------------------------------------------------------- #
# Shield — the public facade
# --------------------------------------------------------------------------- #
class GuardrailsShield:
    """
    Combined shield. Deterministic signatures first (fast, free, certain),
    then NeMo semantic rails for novel phrasings.
    """

    def __init__(self, use_nemo: bool = True) -> None:
        self.deterministic = DeterministicEngine()
        self.nemo = NeMoEngine.try_load() if (use_nemo and settings.enable_guardrails) else None
        self.audit_log: list[dict[str, Any]] = []

    # -- status ----------------------------------------------------------- #
    @property
    def engines(self) -> list[str]:
        names = ["deterministic"]
        if self.nemo:
            names.append("nemo")
        return names

    @property
    def status(self) -> str:
        if not settings.enable_guardrails:
            return "⛔ Guardrails DISABLED"
        return "🛡️ NeMo + deterministic rails" if self.nemo else "🛡️ Deterministic rails"

    # -- rails ------------------------------------------------------------ #
    def check_input(self, text: str) -> RailDecision:
        if not settings.enable_guardrails:
            return RailDecision(allowed=True, rail="input", engine="disabled", text=text)

        decision = self.deterministic.check_input(text)
        if decision.blocked:
            return self._record(decision)

        if self.nemo:
            decision = self.nemo.check_input(text)
        return self._record(decision)

    def check_output(self, text: str, evidence: str = "") -> RailDecision:
        if not settings.enable_guardrails:
            return RailDecision(allowed=True, rail="output", engine="disabled", text=text)
        return self._record(self.deterministic.check_output(text, evidence))

    def safe_output(self, text: str, evidence: str = "") -> tuple[str, RailDecision]:
        """Return ``(text_to_display, decision)`` — substituting on violation."""
        decision = self.check_output(text, evidence)
        if decision.blocked and decision.safe_response:
            return decision.safe_response, decision
        return text, decision

    # -- audit ------------------------------------------------------------ #
    def _record(self, decision: RailDecision) -> RailDecision:
        if decision.blocked:
            self.audit_log.append(decision.as_log())
            logger.warning("Guardrail block: %s", decision.as_log())
        return decision

    @property
    def block_count(self) -> int:
        return len(self.audit_log)


_shield: GuardrailsShield | None = None


def get_shield() -> GuardrailsShield:
    """Process-wide singleton (NeMo init is expensive)."""
    global _shield
    if _shield is None:
        _shield = GuardrailsShield()
    return _shield
