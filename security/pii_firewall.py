"""
security/pii_firewall.py — Outbound PII redaction middleware.

Position in the stack::

    User input ──► PIIFirewall.sanitize ──► Guardrails ──► LLM
                        │
                        └──► PIIVault (in-process, never serialised to the model)

Contract
--------
* No raw Aadhaar / phone / email / PAN / card / street address ever reaches an
  LLM prompt.
* Signal-bearing values are *tokenised* rather than deleted, so downstream
  reasoning still works: ``90210`` becomes ``USER_ZIP_90210`` — the model can
  reason about the region without holding a re-identifiable address.
* Every redaction is reversible in-process via :class:`PIIVault`, so the UI and
  the mock KYC backend still receive exact values.

The firewall is deterministic and dependency-free (pure ``re``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Pattern

__all__ = [
    "PIIFinding",
    "PIIVault",
    "SanitizationResult",
    "PIIFirewall",
    "firewall",
    "sanitize",
    "mask_aadhaar",
    "verhoeff_is_valid",
]


# --------------------------------------------------------------------------- #
# Verhoeff checksum (the real UIDAI Aadhaar check digit algorithm)
# --------------------------------------------------------------------------- #
_D_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_is_valid(number: str) -> bool:
    """Validate a numeric string against the Verhoeff checksum (UIDAI spec)."""
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        return False
    check = 0
    for idx, char in enumerate(reversed(digits)):
        check = _D_TABLE[check][_P_TABLE[idx % 8][int(char)]]
    return check == 0


def mask_aadhaar(aadhaar: str) -> str:
    """``123456789012`` -> ``XXXX XXXX 9012`` (UIDAI-compliant display form)."""
    digits = re.sub(r"\D", "", aadhaar or "")
    if len(digits) != 12:
        return "XXXX XXXX XXXX"
    return f"XXXX XXXX {digits[-4:]}"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PIIFinding:
    """One redaction event."""

    kind: str
    original: str
    token: str
    start: int
    end: int

    @property
    def preview(self) -> str:
        """Safe-to-log preview of what was caught."""
        if self.kind == "aadhaar":
            return mask_aadhaar(self.original)
        if len(self.original) <= 4:
            return "*" * len(self.original)
        return f"{self.original[:2]}{'*' * (len(self.original) - 4)}{self.original[-2:]}"


@dataclass
class PIIVault:
    """
    In-process token → raw-value store.

    Lives only in Streamlit session state. It is never embedded, never logged,
    and never sent to a model.
    """

    _store: dict[str, str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def put(self, kind: str, value: str, token: str) -> None:
        self._store[token] = value
        self._counts[kind] = self._counts.get(kind, 0) + 1

    def get(self, token: str) -> str | None:
        return self._store.get(token)

    def restore(self, text: str) -> str:
        """Re-inject raw values for trusted sinks (UI display, KYC backend)."""
        out = text or ""
        # Longest token first so `USER_ZIP_110001` is not clipped by a prefix.
        for token in sorted(self._store, key=len, reverse=True):
            out = out.replace(token, self._store[token])
        return out

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def clear(self) -> None:
        self._store.clear()
        self._counts.clear()


@dataclass
class SanitizationResult:
    """Outcome of a sanitisation pass."""

    original: str
    sanitized: str
    findings: list[PIIFinding] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.findings)

    @property
    def kinds(self) -> list[str]:
        seen: list[str] = []
        for finding in self.findings:
            if finding.kind not in seen:
                seen.append(finding.kind)
        return seen

    def summary(self) -> str:
        if not self.findings:
            return "No PII detected."
        parts = [f"{kind} x{sum(1 for f in self.findings if f.kind == kind)}" for kind in self.kinds]
        return "Redacted: " + ", ".join(parts)


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
# Order is significant: the most specific / longest digit runs are consumed
# first so a 12-digit Aadhaar is never mistaken for a phone number, and a
# 10-digit phone is never mistaken for a PIN code.
_RULES: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "aadhaar",
        re.compile(r"(?<![\d-])([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})(?![\d-])"),
    ),
    (
        "card",
        re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    ),
    (
        "email",
        re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,63}\b"),
    ),
    (
        "pan",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "phone",
        re.compile(
            r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?"
            r"(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}"
            r"(?![\w.])"
        ),
    ),
    (
        "phone_in",
        re.compile(r"(?<![\w.])(?:\+91[\s-]?|0)?[6-9]\d{9}(?![\w.])"),
    ),
    (
        "dob",
        re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b"),
    ),
    (
        "street",
        # The house number must be adjacent to the street-type word, separated
        # only by whitespace and up to three plain words. A looser gap matches
        # ordinary intake phrasing — "I'm 57, ZIP 10001, I drive a 2023 Volvo"
        # reads as "<number> ... drive" and gets redacted as an address.
        re.compile(
            r"\b\d{1,5}\s+"
            # Street names are proper nouns; function words never appear between
            # a house number and its street type. Excluding them stops
            # "57 and I drive ..." from reading as an address.
            r"(?:(?!(?:and|or|the|an?|my|i|you|we|he|she|it|they|to|of|for|at|in|on|"
            r"about|per|every|with|from|by)\b)[A-Za-z][\w'-]*\s+){0,2}"
            r"(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|"
            r"boulevard|blvd|court|ct|way|nagar|colony|marg|sector|"
            r"apartment|apt|flat|block)\b\.?"
            r"(?:\s*(?:#|no\.?|apt\.?|unit)\s*[\w-]{1,8})?",
            re.IGNORECASE,
        ),
    ),
)

# ZIP / PIN handled separately: tokenised (signal-preserving), not erased.
# A trailing distance unit means it is mileage, not a postal code.
_ZIP_RULE: Pattern[str] = re.compile(
    r"(?<!\d)(\d{6}|\d{5}(?:-\d{4})?)(?!\d)(?!\s*(?:miles?|mi|kms?|km)\b)", re.IGNORECASE
)

_LABELS: dict[str, str] = {
    "aadhaar": "[AADHAAR_REDACTED]",
    "card": "[CARD_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "pan": "[PAN_REDACTED]",
    "ssn": "[SSN_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "phone_in": "[PHONE_REDACTED]",
    "dob": "[DOB_REDACTED]",
    "street": "[ADDRESS_REDACTED]",
    "name": "[NAME_REDACTED]",
}


def _luhn_ok(number: str) -> bool:
    """Guard the broad card regex so ordinary digit runs are not eaten."""
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    if len(digits) < 13:
        return False
    total = 0
    for idx, digit in enumerate(digits):
        if idx % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# --------------------------------------------------------------------------- #
# Firewall
# --------------------------------------------------------------------------- #
class PIIFirewall:
    """Stateless redactor; state lives in the caller-supplied :class:`PIIVault`."""

    def __init__(self, preserve_zip: bool = True) -> None:
        self.preserve_zip = preserve_zip

    # -- public API ------------------------------------------------------- #
    def sanitize(
        self,
        text: str,
        vault: PIIVault | None = None,
        extra_terms: Iterable[str] = (),
    ) -> SanitizationResult:
        """
        Redact *text* and return the model-safe version.

        ``extra_terms`` lets the caller scrub known-sensitive literals that no
        regex can generalise — typically the applicant's own full name pulled
        from validated state.
        """
        if not text:
            return SanitizationResult(original=text or "", sanitized=text or "")

        vault = vault if vault is not None else PIIVault()
        findings: list[PIIFinding] = []
        working = text

        for kind, pattern in _RULES:
            working = self._apply(working, kind, pattern, vault, findings)

        working = self._apply_literals(working, extra_terms, vault, findings)

        if self.preserve_zip:
            working = self._apply_zip(working, vault, findings)

        return SanitizationResult(original=text, sanitized=working, findings=findings)

    # -- internals -------------------------------------------------------- #
    def _apply(
        self,
        text: str,
        kind: str,
        pattern: Pattern[str],
        vault: PIIVault,
        findings: list[PIIFinding],
    ) -> str:
        label = _LABELS[kind]

        def _sub(match: re.Match[str]) -> str:
            raw = match.group(0)
            # The permissive card pattern must pass Luhn or it is left alone
            # for the phone / ZIP rules to classify correctly.
            if kind == "card" and not _luhn_ok(raw):
                return raw
            vault.put(kind, raw, label)
            findings.append(
                PIIFinding(kind=kind, original=raw, token=label, start=match.start(), end=match.end())
            )
            return label

        return pattern.sub(_sub, text)

    def _apply_literals(
        self,
        text: str,
        terms: Iterable[str],
        vault: PIIVault,
        findings: list[PIIFinding],
    ) -> str:
        out = text
        for term in terms:
            term = (term or "").strip()
            if len(term) < 3:
                continue
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if pattern.search(out):
                vault.put("name", term, _LABELS["name"])
                findings.append(
                    PIIFinding(kind="name", original=term, token=_LABELS["name"], start=-1, end=-1)
                )
                out = pattern.sub(_LABELS["name"], out)
        return out

    def _apply_zip(
        self,
        text: str,
        vault: PIIVault,
        findings: list[PIIFinding],
    ) -> str:
        def _sub(match: re.Match[str]) -> str:
            raw = match.group(0)
            token = f"USER_ZIP_{raw.replace('-', '_')}"
            vault.put("zip", raw, token)
            findings.append(
                PIIFinding(kind="zip", original=raw, token=token, start=match.start(), end=match.end())
            )
            return token

        return _ZIP_RULE.sub(_sub, text)


# Process-wide default instance.
firewall = PIIFirewall()


def sanitize(
    text: str,
    vault: PIIVault | None = None,
    extra_terms: Iterable[str] = (),
) -> SanitizationResult:
    """Module-level convenience wrapper around the default firewall."""
    return firewall.sanitize(text, vault=vault, extra_terms=extra_terms)


def scrub_for_logging(text: str) -> str:
    """Aggressive one-way scrub for log lines (no vault, no restoration)."""
    result = PIIFirewall(preserve_zip=False).sanitize(text)
    return result.sanitized


_REDACTOR: Callable[[str], str] = scrub_for_logging
