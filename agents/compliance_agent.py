"""
agents/compliance_agent.py — e-KYC verification, contract generation, issuance.

Responsibilities
----------------
* Validate Aadhaar input **before** it reaches the KYC backend (schema layer).
* Drive the OTP request / verify handshake and track attempts.
* Refuse issuance unless a verified KYC token exists — the one gate no
  conversational path can talk its way around.
* Produce the final contract summary as Markdown and as a real PDF.

The PDF writer is a minimal, dependency-free PDF 1.4 emitter. Using it rather
than ReportLab keeps the download working on a bare ``pip install streamlit``
environment, which matters because the contract is the deliverable the user
walks away with.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agents.state import IssuedPolicy, OnboardingState
from backend.mock_apis import issue_policy, request_aadhaar_otp, verify_aadhaar_otp
from backend.vector_store import get_policy
from config import settings
from security.pii_firewall import mask_aadhaar, verhoeff_is_valid
from security.validations import AadhaarRequest, OTPRequest

logger = logging.getLogger(__name__)

__all__ = ["ComplianceAgent", "compliance_agent", "build_pdf"]


# --------------------------------------------------------------------------- #
# Minimal PDF writer
# --------------------------------------------------------------------------- #
_PAGE_WIDTH, _PAGE_HEIGHT = 595, 842      # A4 in points
_MARGIN_X, _TOP_Y = 56, 786
_LINE_HEIGHT = 15.5
_MAX_LINES = int((_TOP_Y - 60) / _LINE_HEIGHT)


def _escape(text: str) -> str:
    """Escape a string for a PDF literal, dropping non-Latin-1 characters."""
    out = (text or "").replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return out.encode("latin-1", "replace").decode("latin-1")


def _content_stream(lines: list[tuple[str, str, int]]) -> bytes:
    """Build a page content stream from ``(text, font_key, size)`` tuples."""
    parts = ["BT", f"1 0 0 1 {_MARGIN_X} {_TOP_Y} Tm", f"{_LINE_HEIGHT} TL"]
    current: tuple[str, int] | None = None
    for text, font, size in lines:
        if (font, size) != current:
            parts.append(f"/{font} {size} Tf")
            current = (font, size)
        parts.append(f"({_escape(text)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", "replace")


def build_pdf(lines: list[tuple[str, str, int]]) -> bytes:
    """
    Emit a multi-page PDF 1.4 document.

    ``lines`` is a flat list of ``(text, font_key, size)`` where ``font_key`` is
    ``"F1"`` (Helvetica) or ``"F2"`` (Helvetica-Bold).
    """
    pages = [lines[i : i + _MAX_LINES] for i in range(0, max(1, len(lines)), _MAX_LINES)] or [[]]

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-indexed object number

    # Reserve 1 (catalog) and 2 (pages tree); fill them once ids are known.
    objects.append(b"")  # 1 catalog
    objects.append(b"")  # 2 pages

    font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_ids: list[int] = []
    for page_lines in pages:
        stream = _content_stream(page_lines)
        content_id = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        page_id = add(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents %d 0 R "
            b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> >>"
            % (_PAGE_WIDTH, _PAGE_HEIGHT, content_id, font_regular, font_bold)
        )
        page_ids.append(page_id)

    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return bytes(out)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ComplianceAgent:
    """Owns identity verification and policy issuance."""

    name = "Verification & Compliance Agent"

    # ------------------------------------------------------------------ #
    # KYC
    # ------------------------------------------------------------------ #
    def start_kyc(self, state: OnboardingState, aadhaar_number: str) -> tuple[bool, str]:
        """Validate the Aadhaar and dispatch an OTP."""
        try:
            request = AadhaarRequest(aadhaar_number=aadhaar_number)
        except Exception as exc:
            message = self._first_error(exc)
            state.kyc.last_error = message
            state.log("kyc_format_rejected", reason=message)
            return False, f"❌ {message}"

        digits = request.aadhaar_number

        # Checksum is advisory: UIDAI numbers satisfy Verhoeff, but the demo
        # accepts well-formed test numbers so the flow stays exercisable.
        checksum_ok = verhoeff_is_valid(digits)

        response = request_aadhaar_otp(digits)
        if not response.ok:
            state.kyc.last_error = response.error
            state.log("kyc_otp_request_failed", reason=response.error)
            return False, f"❌ {response.error}"

        state.kyc.aadhaar_masked = response.get("masked_aadhaar") or mask_aadhaar(digits)
        state.kyc.txn_id = response.get("txn_id")
        state.kyc.otp_sent = True
        state.kyc.attempts = 0
        state.kyc.last_error = None
        state.vault.put("aadhaar", digits, "[AADHAAR_REDACTED]")
        state.log("kyc_otp_sent", txn_id=state.kyc.txn_id, checksum_valid=checksum_ok)

        note = "" if checksum_ok else "\n\n_Note: this number does not satisfy the UIDAI checksum — accepted in demo mode._"
        return True, (
            f"✅ OTP sent to the mobile registered against **{state.kyc.aadhaar_masked}**.\n\n"
            f"Enter the 6-digit code to complete verification. "
            f"_Demo environment — use `{settings.mock_otp}`._{note}"
        )

    # ------------------------------------------------------------------ #
    def verify_kyc(self, state: OnboardingState, otp: str) -> tuple[bool, str]:
        """Verify the OTP and mint the KYC token."""
        if not state.kyc.txn_id:
            return False, "❌ No verification in progress. Please request an OTP first."

        try:
            request = OTPRequest(txn_id=state.kyc.txn_id, otp=otp)
        except Exception as exc:
            message = self._first_error(exc)
            state.kyc.last_error = message
            return False, f"❌ {message}"

        response = verify_aadhaar_otp(request.txn_id, request.otp)
        state.kyc.attempts += 1

        if not response.ok:
            state.kyc.last_error = response.error
            state.log("kyc_verify_failed", status=response.status_code, reason=response.error)
            if response.status_code in (410, 429):
                state.kyc.otp_sent = False
                state.kyc.txn_id = None
            return False, f"❌ {response.error}"

        state.kyc.verified = True
        state.kyc.kyc_token = response.get("kyc_token")
        state.kyc.verification_level = response.get("verification_level")
        state.kyc.verified_at = datetime.now(timezone.utc)
        state.kyc.last_error = None
        state.log("kyc_verified", level=state.kyc.verification_level)

        return True, (
            f"✅ **Identity verified** — {state.kyc.aadhaar_masked} "
            f"({state.kyc.verification_level}). You can now issue your policy."
        )

    # ------------------------------------------------------------------ #
    # Issuance
    # ------------------------------------------------------------------ #
    def issue(self, state: OnboardingState) -> tuple[bool, str]:
        """Bind the selected policy. Requires a verified KYC token."""
        if not state.kyc.verified or not state.kyc.kyc_token:
            return False, "❌ Identity verification must be completed before issuance."

        match = state.selected_match
        if match is None:
            return False, "❌ No policy selected."
        if state.profile is None:
            return False, "❌ Applicant profile is incomplete."
        if not match.eligible:
            return False, f"❌ {match.name} is not available for this profile — {match.ineligibility_reason}."

        policy = get_policy(match.policy_id)
        deductible = state.selected_deductible or match.deductible

        # Re-quote at the selected deductible so the bound premium matches the
        # figure the applicant last saw on the inspector.
        premium = match.monthly_premium
        if policy is not None and deductible != match.deductible:
            from agents.recommendation_agent import quote_premium

            premium = quote_premium(policy, state.profile, state.risk, deductible)

        response = issue_policy(
            kyc_token=state.kyc.kyc_token,
            policy_id=match.policy_id,
            policy_name=match.name,
            holder_name=state.profile.full_name,
            monthly_premium=premium,
            deductible=deductible,
            coverage_limit=match.coverage_limit,
            risk_tier=state.risk.tier_label if state.risk else "Standard",
        )

        if not response.ok:
            state.log("issuance_failed", reason=response.error)
            return False, f"❌ Issuance failed: {response.error}"

        state.issued = IssuedPolicy(**response.data)
        state.log("policy_issued", policy_number=state.issued.policy_number)
        return True, (
            f"🎉 **Policy {state.issued.policy_number} is now ACTIVE.**\n\n"
            f"{state.issued.product_name} — {settings.currency_symbol}{premium:,.2f}/month, "
            f"effective {state.issued.effective_date} through {state.issued.expiry_date}."
        )

    # ------------------------------------------------------------------ #
    # Contract rendering
    # ------------------------------------------------------------------ #
    def contract_markdown(self, state: OnboardingState) -> str:
        """Human-readable contract summary for on-screen display."""
        issued = state.issued
        if issued is None:
            return "_No policy has been issued yet._"

        policy = get_policy(issued.product_id)
        risk = state.risk
        symbol = settings.currency_symbol

        lines = [
            f"## Certificate of Insurance — {issued.policy_number}",
            "",
            f"**Status:** {issued.status}  |  **Underwriter:** {issued.underwriter}",
            "",
            "### Policyholder",
            f"- **Name:** {issued.policyholder}",
            f"- **Aadhaar (masked):** {state.kyc.aadhaar_masked or 'N/A'}",
            f"- **Verification:** {state.kyc.verification_level or 'N/A'}",
        ]
        if state.profile:
            lines += [
                f"- **Age:** {state.profile.age}",
                f"- **Postal code:** {state.profile.zip_code}",
                f"- **Insured vehicle:** {state.profile.display_vehicle}",
                f"- **Declared annual mileage:** {state.profile.annual_mileage:,} miles",
            ]

        lines += [
            "",
            "### Cover",
            f"- **Product:** {issued.product_name} ({issued.product_id})",
            f"- **Coverage limit:** {symbol}{issued.coverage_limit:,}",
            f"- **Deductible:** {symbol}{issued.deductible:,}",
            f"- **Monthly premium:** {symbol}{issued.monthly_premium:,.2f}",
            f"- **Annual premium:** {symbol}{issued.annual_premium:,.2f}",
            f"- **Effective:** {issued.effective_date} to {issued.expiry_date}",
            f"- **Free-look period:** {issued.free_look_period_days} days",
        ]

        if risk:
            lines += [
                "",
                "### Underwriting",
                f"- **Risk tier:** {risk.tier_label}",
                f"- **Risk score:** {risk.score:.0f}/100",
                f"- **Applied discount:** {risk.base_discount:.0f}%",
                f"- **Primary rating factor:** {risk.primary_factor}",
            ]

        if policy:
            lines += ["", "### Included cover"]
            lines += [f"- {feature}" for feature in policy.features]
            lines += ["", "### Exclusions"]
            lines += [f"- {exclusion}" for exclusion in policy.exclusions]

        lines += [
            "",
            "---",
            "",
            "_This certificate summarises the cover bound under the policy number shown. "
            "Coverage is governed by the full policy wording; claims are assessed individually "
            "against those terms. Generated by AegisAI Automated Underwriting in a demonstration "
            "environment — not a real contract of insurance._",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def contract_pdf(self, state: OnboardingState) -> bytes:
        """Render the contract summary as a downloadable PDF."""
        issued = state.issued
        if issued is None:
            return build_pdf([("No policy has been issued.", "F1", 11)])

        policy = get_policy(issued.product_id)
        risk = state.risk
        symbol = settings.currency_symbol
        lines: list[tuple[str, str, int]] = []

        def heading(text: str) -> None:
            lines.append(("", "F1", 11))
            lines.append((text, "F2", 13))

        def row(label: str, value: str) -> None:
            lines.append((f"{label}: {value}", "F1", 11))

        lines.append(("AEGIS AI - CERTIFICATE OF INSURANCE", "F2", 17))
        lines.append((f"Policy Number: {issued.policy_number}", "F2", 12))
        lines.append((f"Status: {issued.status}   Issued: {issued.issued_at[:19]}Z", "F1", 10))
        lines.append(("-" * 78, "F1", 10))

        heading("POLICYHOLDER")
        row("Name", issued.policyholder)
        row("Aadhaar (masked)", state.kyc.aadhaar_masked or "N/A")
        row("Verification level", state.kyc.verification_level or "N/A")
        if state.profile:
            row("Age", str(state.profile.age))
            row("Postal code", state.profile.zip_code)
            row("Insured vehicle", state.profile.display_vehicle)
            row("Declared annual mileage", f"{state.profile.annual_mileage:,} miles")

        heading("COVER")
        row("Product", f"{issued.product_name} ({issued.product_id})")
        row("Coverage limit", f"{symbol}{issued.coverage_limit:,}")
        row("Deductible", f"{symbol}{issued.deductible:,}")
        row("Monthly premium", f"{symbol}{issued.monthly_premium:,.2f}")
        row("Annual premium", f"{symbol}{issued.annual_premium:,.2f}")
        row("Effective period", f"{issued.effective_date} to {issued.expiry_date}")
        row("Free-look period", f"{issued.free_look_period_days} days")
        row("Underwriter", issued.underwriter)

        if risk:
            heading("UNDERWRITING SUMMARY")
            row("Risk tier", risk.tier_label)
            row("Risk score", f"{risk.score:.0f}/100")
            row("Applied discount", f"{risk.base_discount:.0f}%")
            row("Primary rating factor", risk.primary_factor)

        if policy:
            heading("INCLUDED COVER")
            for feature in policy.features:
                lines.append((f"  - {feature}", "F1", 10))
            heading("EXCLUSIONS")
            for exclusion in policy.exclusions:
                lines.append((f"  - {exclusion}", "F1", 10))

        lines.append(("", "F1", 10))
        lines.append(("-" * 78, "F1", 10))
        for text in (
            "This certificate summarises the cover bound under the policy number shown.",
            "Coverage is governed by the full policy wording; claims are assessed",
            "individually against those terms. Generated by AegisAI Automated",
            "Underwriting in a demonstration environment - not a real contract of insurance.",
        ):
            lines.append((text, "F1", 9))

        return build_pdf(lines)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_error(exc: Exception) -> str:
        """Pull a readable message out of a Pydantic ValidationError."""
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                first = errors()[0]
                return str(first.get("msg", exc)).replace("Value error, ", "")
            except (IndexError, KeyError, TypeError):
                pass
        return str(exc)


compliance_agent = ComplianceAgent()
