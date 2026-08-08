"""Mock service backends: underwriting, CRM, KYC, issuance, and the RAG store."""

from backend.mock_apis import (
    ApiResponse,
    geo_hazard_check,
    get_crm_customer,
    issue_policy,
    mvr_check,
    request_aadhaar_otp,
    verify_aadhaar_otp,
)

__all__ = [
    "ApiResponse",
    "get_crm_customer",
    "mvr_check",
    "geo_hazard_check",
    "request_aadhaar_otp",
    "verify_aadhaar_otp",
    "issue_policy",
]
