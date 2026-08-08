# MASTER PROMPT: SECURE AI-POWERED INSURANCE ONBOARDING & RISK DASHBOARD

## SYSTEM OVERVIEW & OBJECTIVE
You are an expert AI Systems Architect, Security Specialist, and Lead Python Engineer. Your task is to build a complete, production-grade, secure multi-agent AI Insurance Onboarding and Policy Recommendation application using Streamlit, LangGraph (or Python state machine logic), Apache ECharts (`streamlit-echarts`), RAG, and NeMo Guardrails.

The system automates traditional insurance onboarding through a hybrid conversational/form interface, evaluates risk profiles, matches curated policies from a vector store, renders interactive risk and scenario charts, executes mock Aadhaar e-KYC OTP verification, and issues finalized policy contracts—all while protected by robust semantic guardrails and programmatic validations.

---

## TECH STACK & REQUIREMENTS
1. **Frontend:** Streamlit (`app.py`), `streamlit-echarts` (Apache ECharts integration), `pandas`.
2. **Agentic Orchestration:** `langgraph` or a strongly-typed Python State Machine (`Pydantic`, `TypedDict`).
3. **LLM Engine & Abstraction:** LangChain / `init_chat_model` with a Factory Pattern supporting `base_url`, custom API keys, and model fallbacks.
4. **AI Guardrails & Security:** `nemoguardrails` (NeMo Guardrails with Colang `.co` files), PII Masking/Redaction Middleware.
5. **Vector DB / RAG:** ChromaDB or in-memory vector index storing mock policy documents with semantic retrieval.
6. **Data Validation & Logic:** `pydantic` schemas, custom anomaly detection, and range validation logic.
7. **Mock Backend APIs:** Python async/sync functions simulating REST/gRPC endpoints for Underwriting, CRM, MVR, Geo-Hazard, Aadhaar e-KYC, and Policy Issuance.

---

## ARCHITECTURAL COMPONENTS TO IMPLEMENT

### 1. LLM Factory (`llm_factory.py`)
Implement a unified factory function `get_llm()` using `init_chat_model`:
- Accept `model_name` (default: `openai:gpt-4o-mini`), `temperature`, `base_url`, `api_key`, and `fallback_model_name` (default: `openai:gpt-4o`).
- Wrap the primary model with `RunnableWithFallbacks` to handle rate limits or API failures.
- Support custom `base_url` for vLLM, LM Studio, or local OpenAI-compatible endpoints.

### 2. NeMo Guardrails Middleware (`security/guardrails/`)
Sit NeMo Guardrails as an active security shield between the UI and LLM Agents:
- **Input Rails (`rails.co` & `config.yml`):** Intercept and block prompt injections, jailbreak attempts, and system prompt extraction attacks (e.g., "Ignore previous instructions..."). Return safe fallback responses.
- **Topical Rails:** Detect off-topic queries (e.g., writing code, recipes, political chat) and steer the user back to insurance onboarding.
- **Output Rails:** Intercept generated LLM outputs to enforce hallucination/groundedness checks and block non-compliant binding language (e.g., "I guarantee this policy covers everything").

### 3. Programmatic & Business Logic Validations (`security/validations.py`)
Implement hard, deterministic validation rules that run before data hits underwriting tools:
- **Pydantic Type & Schema Enforcement:** Strict typing for age, vehicle year, annual mileage, and ZIP codes.
- **Business Rule Anomaly Detection:** 
  - Age must be between `18` and `100`.
  - Vehicle year must be between `1990` and `Current Year + 1`.
  - Annual mileage must be between `100` and `100,000` miles.
  - ZIP/PIN codes must match valid format rules.
- Flag invalid entries or anomalies and request re-entry before passing payload downstream.

### 4. PII Firewall Middleware (`security/pii_firewall.py`)
- Regex and tokenization logic to sanitize input before LLM prompting.
- Redact Aadhaar/National IDs, phone numbers, and street addresses into tokens (e.g., `[Aadhaar Redacted]`, `USER_ZIP_90210`).
- Ensure no raw PII reaches LLM prompt context while retaining structured values in secure state memory.

### 5. Multi-Agent Architecture (`agents/`)
Build 5 specialized agents coordinated by a **Deterministic Orchestrator**:
- **User Intake Agent (LLM):** Manages natural chat, performs structured entity extraction using Pydantic, and updates state.
- **Risk Assessment Agent (Tool Node):** Receives validated state, calls mock MVR and Geo-Hazard APIs, calculates numerical risk score (0–100) and tier, and writes semantic risk summaries.
- **Policy Recommendation Agent (RAG Node):** Embeds profile parameters, queries vector store, scores matches using a multi-factor weighting formula, and returns top 2–3 policies.
- **Policy Q&A Agent (LLM):** Answers customer questions about catalog items or policy comparisons using grounded context from RAG.
- **Verification & Compliance Agent (Tool/Validation Node):** Validates KYC data, generates final PDF contract summaries, and finalizes issuance.

### 6. Mock API Suite (`backend/mock_apis.py`)
Implement deterministic mock functions for:
- `GET /api/v1/crm/customer`: Pre-fills existing user profiles.
- `POST /api/v1/underwriting/mvr-check`: Returns accident and violation counts.
- `POST /api/v1/underwriting/geo-hazard`: Returns location risk scores by ZIP.
- `POST /api/v1/kyc/aadhaar/request-otp`: Validates 12-digit format, returns `txn_id`.
- `POST /api/v1/kyc/aadhaar/verify-otp`: Validates mock OTP (accepts `123456`), returns verified status token.
- `POST /api/v1/policies/issue`: Returns active Policy ID and digital contract payload.

---

## FRONTEND UI & DASHBOARD SPECS (`app.py`)

Layout must feature a dual-view workspace (Sidebar Chatbot + Main Panel Dashboard):

### Zone A: Global Risk Summary Header (Rendered Once)
- Streamlit metrics showing **Risk Tier** (*Low Risk - Class A+*), **Risk Score** (*85/100*), **Base Discount** (*15%*), and **Primary Factor**.

### Zone B: Policy Match & Apache ECharts Radar Chart
- `st_echarts` Radar Chart comparing Top Picks across 5 axes: *Affordability, Coverage Breadth, Flexibility, Perks, Risk Match*.
- Policy selection radio card allowing dynamic policy inspection.

### Zone C: Dynamic Policy Inspector (ECharts)
1. **Interactive Deductible Slider:** Modifies deductible ($250, $500, $1000, $2000).
2. **ECharts Smooth Line/Area Chart:** Animates premium curve with a vertical indicator line showing calculated monthly cost.
3. **ECharts Donut Chart:** Visualizes coverage allocation (Liability, Collision, Comprehensive, Medical, Roadside).
4. **ECharts Stacked Horizontal Bar Chart:** Simulates real-world claim outcomes ($3k, $18k, $35k) showing *You Pay (Deductible)* vs *Insurance Covers*.

### Zone D: Expandable Catalog & Hard KYC Modal
- Expandable table showing all 12 catalog policies tagged with personalized *Risk Match Fit %*.
- Hard KYC Section with Aadhaar 12-digit input, "Send OTP" button, mock OTP entry box (`123456`), and instant verification badge.
- Policy Issuance button displaying finalized contract summary.

---

## DIRECTORY STRUCTURE TO GENERATE
Generate the project according to the following file layout:

```text
ai_insurance_app/
├── app.py                      # Main Streamlit Dashboard & UI
├── requirements.txt            # All required dependencies
├── README.md                   # Setup and execution instructions
├── config.py                   # System configuration & environment variables
├── llm_factory.py              # Agnostic LLM Factory with fallbacks
├── security/
│   ├── pii_firewall.py        # PII masking and redaction middleware
│   ├── validations.py          # Pydantic schemas & business logic validators
│   └── guardrails/
│       ├── config.yml          # NeMo Guardrails configuration
│       └── rails.co            # Colang prompt injection & topical rules
├── backend/
│   ├── mock_apis.py            # Underwriting, KYC, and CRM mock APIs
│   └── vector_store.py         # Policy document loader & RAG retriever
├── agents/
│   ├── state.py                # Global OnboardingState schema
│   ├── intake_agent.py         # Conversational intake & Pydantic parser
│   ├── risk_agent.py           # Underwriting & risk evaluation agent
│   ├── recommendation_agent.py # RAG policy matching agent
│   ├── qa_agent.py             # Grounded policy Q&A agent
│   └── compliance_agent.py     # KYC & policy issuance agent
└── orchestrator/
    └── workflow.py             # Deterministic state machine / LangGraph workflow