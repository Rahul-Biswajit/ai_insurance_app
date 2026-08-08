"""
app.py — AegisAI dashboard.

Dual-view workspace:

    Sidebar          Conversational agent + live security telemetry
    Zone A           Global risk summary header
    Zone B           Policy match radar (ECharts) + selection cards
    Zone C           Dynamic policy inspector — deductible slider, premium
                     curve, coverage donut, claim-outcome bars
    Zone D           Full catalog with personalised fit, e-KYC, issuance

All premium maths lives in ``agents.recommendation_agent`` — this module only
renders. Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from agents.compliance_agent import compliance_agent
from agents.recommendation_agent import claim_outcome, quote_premium, recommendation_agent
from agents.state import OnboardingState, PolicyMatch, STAGE_LABELS, STAGE_ORDER, Stage
from backend.mock_apis import list_crm_customers
from backend.vector_store import get_policy, get_vector_store
from config import settings
from llm_factory import describe_llm, get_llm
from orchestrator.workflow import workflow
from security.guardrails_runtime import get_shield

logging.basicConfig(level=logging.INFO if settings.debug else logging.WARNING)

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=settings.app_title,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primary": "#5B8DEF",
    "accent": "#7C5CFF",
    "success": "#2FD4A7",
    "warning": "#F5A623",
    "danger": "#FF6B6B",
    "muted": "#8A94A6",
    "grid": "rgba(138,148,166,0.18)",
    "text": "#C9D1E0",
}

SERIES_COLORS = ["#5B8DEF", "#7C5CFF", "#2FD4A7", "#F5A623", "#FF6B6B", "#4ECDC4"]

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
      [data-testid="stMetricValue"] { font-size: 1.65rem; }
      [data-testid="stMetricLabel"] { font-size: .78rem; letter-spacing: .04em;
                                       text-transform: uppercase; opacity: .72; }
      .aegis-hero { padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1.1rem;
                    background: linear-gradient(105deg, rgba(91,141,239,.16), rgba(124,92,255,.09));
                    border: 1px solid rgba(91,141,239,.28); }
      .aegis-hero h1 { margin: 0; font-size: 1.5rem; letter-spacing: -.01em; }
      .aegis-hero p  { margin: .3rem 0 0; opacity: .72; font-size: .9rem; }
      .pill { display:inline-block; padding:.16rem .6rem; border-radius:999px;
              font-size:.72rem; font-weight:600; margin-right:.35rem; }
      .pill-ok   { background: rgba(47,212,167,.16); color:#2FD4A7; }
      .pill-warn { background: rgba(245,166,35,.16); color:#F5A623; }
      .pill-bad  { background: rgba(255,107,107,.16); color:#FF6B6B; }
      .pill-info { background: rgba(91,141,239,.16); color:#5B8DEF; }
      .rationale { font-size:.86rem; opacity:.78; line-height:1.45; margin:.1rem 0 .5rem; }
      .stChatMessage { padding: .35rem 0; }
      div[data-testid="stExpander"] details { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def boot() -> None:
    if "state" not in st.session_state:
        st.session_state.state = workflow.new_state()
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "kyc_notice" not in st.session_state:
        st.session_state.kyc_notice = None


boot()
state: OnboardingState = st.session_state.state


def money(value: float, decimals: int = 0) -> str:
    return f"{settings.currency_symbol}{value:,.{decimals}f}"


def md_safe(text: str) -> str:
    """
    Escape currency for ``st.markdown``.

    Streamlit renders ``$…$`` as LaTeX, so two dollar amounts on one line
    silently swallow the text between them ("$81/mo · $500 deductible" loses
    both symbols). Escaping is only correct at display time — never apply it to
    text destined for a download, a chart label, or a model prompt.
    """
    return (text or "").replace("$", r"\$")


def md_money(value: float, decimals: int = 0) -> str:
    """Markdown-safe currency string."""
    return md_safe(money(value, decimals))


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #
_BASE_TOOLTIP = {"backgroundColor": "rgba(20,24,34,.94)", "borderColor": "rgba(91,141,239,.35)",
                 "textStyle": {"color": "#E6EAF2"}}


def radar_option(matches: list[PolicyMatch]) -> dict[str, Any]:
    """Zone B — multi-policy comparison across the five match dimensions."""
    indicators = [
        {"name": "Affordability", "max": 100},
        {"name": "Coverage Breadth", "max": 100},
        {"name": "Flexibility", "max": 100},
        {"name": "Perks", "max": 100},
        {"name": "Risk Match", "max": 100},
    ]
    return {
        "backgroundColor": "transparent",
        "tooltip": {**_BASE_TOOLTIP, "trigger": "item"},
        "legend": {
            "bottom": 0,
            "itemGap": 18,
            "textStyle": {"color": PALETTE["text"], "fontSize": 11},
            "data": [m.name for m in matches],
        },
        "radar": {
            "indicator": indicators,
            "shape": "polygon",
            "radius": "66%",
            "center": ["50%", "48%"],
            "splitNumber": 4,
            "axisName": {"color": PALETTE["text"], "fontSize": 11},
            "splitLine": {"lineStyle": {"color": PALETTE["grid"]}},
            "splitArea": {"areaStyle": {"color": ["rgba(91,141,239,.03)", "rgba(124,92,255,.05)"]}},
            "axisLine": {"lineStyle": {"color": PALETTE["grid"]}},
        },
        "series": [
            {
                "type": "radar",
                "symbolSize": 5,
                "emphasis": {"focus": "series", "lineStyle": {"width": 3}},
                "data": [
                    {
                        "value": match.radar_values(),
                        "name": match.name,
                        "lineStyle": {"width": 2, "color": SERIES_COLORS[i % len(SERIES_COLORS)]},
                        "itemStyle": {"color": SERIES_COLORS[i % len(SERIES_COLORS)]},
                        "areaStyle": {"opacity": 0.16},
                    }
                    for i, match in enumerate(matches)
                ],
            }
        ],
        "animationDuration": 900,
        "animationEasing": "cubicOut",
    }


def premium_curve_option(policy: Any, selected_deductible: int) -> dict[str, Any]:
    """Zone C — smooth premium curve with an indicator line at the selection."""
    if state.profile is None:
        return {}

    # Fine-grained sweep so the curve reads as continuous, not four points.
    steps = list(range(250, 2001, 50))
    premiums = [quote_premium(policy, state.profile, state.risk, d) for d in steps]
    selected_premium = quote_premium(policy, state.profile, state.risk, selected_deductible)

    return {
        "backgroundColor": "transparent",
        "tooltip": {**_BASE_TOOLTIP, "trigger": "axis"},
        "grid": {"left": 58, "right": 26, "top": 34, "bottom": 44},
        "xAxis": {
            "type": "category",
            "data": [f"${d:,}" for d in steps],
            "name": "Deductible",
            "nameLocation": "middle",
            "nameGap": 30,
            "nameTextStyle": {"color": PALETTE["muted"], "fontSize": 11},
            "axisLabel": {
                "color": PALETTE["muted"],
                "fontSize": 10,
                "interval": 4,
            },
            "axisLine": {"lineStyle": {"color": PALETTE["grid"]}},
            "axisTick": {"show": False},
            "boundaryGap": False,
        },
        "yAxis": {
            "type": "value",
            "name": "Monthly premium",
            "nameTextStyle": {"color": PALETTE["muted"], "fontSize": 11},
            "axisLabel": {"color": PALETTE["muted"], "fontSize": 10, "formatter": "${value}"},
            "splitLine": {"lineStyle": {"color": PALETTE["grid"]}},
        },
        "series": [
            {
                "type": "line",
                "name": "Monthly premium",
                "data": [round(p, 2) for p in premiums],
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"width": 3, "color": PALETTE["primary"]},
                "areaStyle": {
                    "opacity": 0.28,
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "rgba(91,141,239,.55)"},
                            {"offset": 1, "color": "rgba(91,141,239,.02)"},
                        ],
                    },
                },
                "markLine": {
                    "symbol": ["none", "none"],
                    "label": {
                        "formatter": f"{money(selected_premium, 2)}/mo",
                        "color": PALETTE["success"],
                        "fontWeight": "bold",
                        "position": "insideEndTop",
                    },
                    "lineStyle": {"color": PALETTE["success"], "width": 2, "type": "dashed"},
                    "data": [{"xAxis": f"${selected_deductible:,}"}],
                },
                "markPoint": {
                    "symbol": "circle",
                    "symbolSize": 12,
                    "itemStyle": {"color": PALETTE["success"], "borderColor": "#fff", "borderWidth": 2},
                    "label": {"show": False},
                    "data": [{"coord": [f"${selected_deductible:,}", round(selected_premium, 2)]}],
                },
            }
        ],
        "animationDuration": 700,
    }


def donut_option(policy: Any) -> dict[str, Any]:
    """Zone C — coverage allocation by component."""
    limit = policy.coverage_limit
    data = [
        {
            "name": name,
            "value": round(limit * pct / 100),
            "itemStyle": {"color": SERIES_COLORS[i % len(SERIES_COLORS)]},
        }
        for i, (name, pct) in enumerate(policy.allocation.as_pairs())
    ]
    return {
        "backgroundColor": "transparent",
        "tooltip": {
            **_BASE_TOOLTIP,
            "trigger": "item",
            "formatter": "{b}: ${c} ({d}%)",
        },
        "legend": {
            "orient": "vertical",
            "right": 4,
            "top": "middle",
            "itemGap": 10,
            "textStyle": {"color": PALETTE["text"], "fontSize": 11},
        },
        "series": [
            {
                "type": "pie",
                "radius": ["46%", "72%"],
                "center": ["36%", "50%"],
                "avoidLabelOverlap": True,
                "itemStyle": {"borderRadius": 6, "borderColor": "rgba(0,0,0,0)", "borderWidth": 2},
                "label": {"show": False},
                "emphasis": {
                    "label": {
                        "show": True,
                        "fontSize": 13,
                        "fontWeight": "bold",
                        "color": PALETTE["text"],
                        "formatter": "{b}\n{d}%",
                    },
                    "scaleSize": 8,
                },
                "labelLine": {"show": False},
                "data": data,
            }
        ],
        "animationDuration": 800,
    }


def claim_outcome_option(policy: Any, deductible: int) -> dict[str, Any]:
    """Zone D/C — stacked horizontal bars for three real-world claim sizes."""
    scenarios = settings.claim_scenarios
    you, insurer, uncovered = [], [], []
    for amount in scenarios:
        y, i, u = claim_outcome(policy, deductible, amount)
        you.append(y)
        insurer.append(i)
        uncovered.append(u)

    series = [
        {
            "name": "You pay (deductible)",
            "type": "bar",
            "stack": "claim",
            "data": you,
            "itemStyle": {"color": PALETTE["warning"], "borderRadius": [4, 0, 0, 4]},
            "label": {
                "show": True,
                "position": "insideLeft",
                "color": "#1a1d26",
                "fontWeight": "bold",
                "fontSize": 11,
                "formatter": "${c}",
            },
        },
        {
            "name": "Insurance covers",
            "type": "bar",
            "stack": "claim",
            "data": insurer,
            "itemStyle": {"color": PALETTE["success"], "borderRadius": [0, 4, 4, 0]},
            "label": {
                "show": True,
                "position": "insideRight",
                "color": "#0f1419",
                "fontWeight": "bold",
                "fontSize": 11,
                "formatter": "${c}",
            },
        },
    ]
    if any(uncovered):
        series[1]["itemStyle"]["borderRadius"] = [0, 0, 0, 0]
        series.append(
            {
                "name": "Above policy limit",
                "type": "bar",
                "stack": "claim",
                "data": uncovered,
                "itemStyle": {"color": PALETTE["danger"], "borderRadius": [0, 4, 4, 0]},
                "label": {"show": True, "position": "insideRight", "fontSize": 11, "formatter": "${c}"},
            }
        )

    return {
        "backgroundColor": "transparent",
        "tooltip": {**_BASE_TOOLTIP, "trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "textStyle": {"color": PALETTE["text"], "fontSize": 11}},
        "grid": {"left": 84, "right": 30, "top": 42, "bottom": 26},
        "xAxis": {
            "type": "value",
            "axisLabel": {"color": PALETTE["muted"], "fontSize": 10, "formatter": "${value}"},
            "splitLine": {"lineStyle": {"color": PALETTE["grid"]}},
        },
        "yAxis": {
            "type": "category",
            "data": [f"${amount:,} claim" for amount in scenarios],
            "axisLabel": {"color": PALETTE["text"], "fontSize": 11},
            "axisLine": {"lineStyle": {"color": PALETTE["grid"]}},
            "axisTick": {"show": False},
        },
        "series": series,
        "animationDuration": 800,
    }


def risk_factor_option(state: OnboardingState) -> dict[str, Any]:
    """Diverging bar of every scoring factor — the 'why' behind the score."""
    factors = sorted(state.risk.factors, key=lambda f: f.impact)
    return {
        "backgroundColor": "transparent",
        "tooltip": {**_BASE_TOOLTIP, "trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 190, "right": 40, "top": 12, "bottom": 28},
        "xAxis": {
            "type": "value",
            "axisLabel": {"color": PALETTE["muted"], "fontSize": 10},
            "splitLine": {"lineStyle": {"color": PALETTE["grid"]}},
        },
        "yAxis": {
            "type": "category",
            "data": [f.name for f in factors],
            "axisLabel": {"color": PALETTE["text"], "fontSize": 10, "width": 180, "overflow": "truncate"},
            "axisLine": {"lineStyle": {"color": PALETTE["grid"]}},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "data": [
                    {
                        "value": f.impact,
                        "itemStyle": {
                            "color": PALETTE["success"] if f.impact >= 0 else PALETTE["danger"],
                            "borderRadius": [0, 4, 4, 0] if f.impact >= 0 else [4, 0, 0, 4],
                        },
                    }
                    for f in factors
                ],
                "barWidth": "58%",
                "label": {
                    "show": True,
                    "position": "right",
                    "color": PALETTE["text"],
                    "fontSize": 10,
                    "formatter": "{c}",
                },
            }
        ],
        "animationDuration": 700,
    }


# --------------------------------------------------------------------------- #
# Sidebar — conversational agent + security telemetry
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🛡️ AegisAI Assistant")

        stage_index = STAGE_ORDER.index(state.stage)
        st.progress(state.progress, text=f"**{STAGE_LABELS[state.stage]}** · step {stage_index + 1}/{len(STAGE_ORDER)}")

        chat_box = st.container(height=430)
        with chat_box:
            for message in state.messages:
                avatar = "🛡️" if message.role == "assistant" else "🧑"
                if message.blocked:
                    avatar = "🚫"
                with st.chat_message(message.role, avatar=avatar):
                    st.markdown(md_safe(message.content))
                    tags = []
                    if message.blocked:
                        tags.append(f'<span class="pill pill-bad">blocked · {message.block_category}</span>')
                    if message.pii_redacted:
                        tags.append(
                            f'<span class="pill pill-warn">PII redacted · {", ".join(message.pii_redacted)}</span>'
                        )
                    if message.agent and message.role == "assistant" and not message.blocked:
                        tags.append(f'<span class="pill pill-info">{message.agent}</span>')
                    if tags:
                        st.markdown("".join(tags), unsafe_allow_html=True)

        prompt = st.chat_input("Tell me about yourself, or ask about a policy…")
        if st.session_state.pending_question:
            prompt = st.session_state.pending_question
            st.session_state.pending_question = None

        if prompt:
            workflow.handle_message(state, prompt)
            st.rerun()

        st.divider()

        with st.expander("⚡ Quick start", expanded=not state.reached(Stage.RISK)):
            st.caption("Load a CRM profile to skip manual intake.")
            options = list_crm_customers()
            choice = st.selectbox(
                "Existing customer",
                options=[o["customer_id"] for o in options],
                format_func=lambda cid: next(o["label"] for o in options if o["customer_id"] == cid),
                key="crm_pick",
                label_visibility="collapsed",
            )
            if st.button("Load from CRM", width="stretch"):
                ok, message = workflow.prefill_from_crm(state, choice)
                state.add_message("assistant", message, agent="CRM Connector")
                st.rerun()

        with st.expander("🔐 Security posture", expanded=False):
            shield = get_shield()
            st.markdown(f"**Guardrails:** {shield.status}")
            st.markdown(
                f"**PII firewall:** {'🟢 active' if settings.enable_pii_firewall else '🔴 disabled'}"
            )
            st.markdown(f"**Output rails:** {'🟢 strict' if settings.strict_output_rails else '🟡 relaxed'}")
            st.markdown(f"**Vector backend:** `{get_vector_store().backend}`")
            st.markdown(f"**LLM:** `{describe_llm(get_llm())}`")

            col_a, col_b = st.columns(2)
            col_a.metric("Rails triggered", state.guardrail_blocks)
            col_b.metric("PII items vaulted", sum(state.vault.counts.values()))

            if state.vault.counts:
                st.caption("Redacted: " + ", ".join(f"{k} ×{v}" for k, v in state.vault.counts.items()))

            if shield.audit_log:
                st.caption("Recent blocks")
                for entry in shield.audit_log[-4:][::-1]:
                    st.markdown(
                        f'<span class="pill pill-bad">{entry["category"]}</span>'
                        f'<span style="font-size:.74rem;opacity:.6">{entry["rail"]} rail</span>',
                        unsafe_allow_html=True,
                    )

        with st.expander("🧾 Audit trail", expanded=False):
            if state.audit:
                st.dataframe(
                    pd.DataFrame(state.audit[::-1]),
                    hide_index=True,
                    width="stretch",
                    height=230,
                )
            else:
                st.caption("No events yet.")

        if st.button("🔄 Start over", width="stretch"):
            st.session_state.state = workflow.reset()
            st.session_state.kyc_notice = None
            st.rerun()


# --------------------------------------------------------------------------- #
# Zone A — global risk summary header
# --------------------------------------------------------------------------- #
def render_zone_a() -> None:
    risk = state.risk
    st.markdown(
        f"""
        <div class="aegis-hero">
          <h1>🛡️ AegisAI — Insurance Onboarding &amp; Risk Intelligence</h1>
          <p>Multi-agent underwriting with NeMo-style guardrails, PII firewalling and grounded RAG.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if risk is None:
        st.info(
            "👋 **Let's get started.** Chat with the assistant in the sidebar, or load a demo "
            "profile from **Quick start**. Your risk summary and policy matches appear here."
        )
        if state.missing_fields:
            st.caption("Still needed: " + ", ".join(f"`{f}`" for f in state.missing_fields))
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Tier", risk.tier_label, help="Underwriting classification band.")
    col2.metric(
        "Risk Score",
        f"{risk.score:.0f}/100",
        delta=f"{risk.score - 70:+.0f} vs baseline",
        help="Composite safety score — higher is safer.",
    )
    col3.metric("Base Discount", f"{risk.base_discount:.0f}%", help="Applied to every quoted premium.")
    col4.metric("Primary Factor", risk.primary_factor, help="Largest single contributor to the score.")

    if risk.summary:
        st.caption(md_safe(risk.summary))


# --------------------------------------------------------------------------- #
# Zone B — radar + policy selection
# --------------------------------------------------------------------------- #
def render_zone_b() -> None:
    matches = state.recommendations
    st.markdown("### 📊 Policy Match Analysis")

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st_echarts(radar_option(matches), height="430px", key="radar")

    with right:
        st.markdown("**Your top picks**")
        labels = {
            m.policy_id: f"{m.name} · {money(m.monthly_premium)}/mo · {m.match_score:.0f}% match"
            for m in matches
        }  # radio labels are plain text, not markdown — no escaping needed
        current = state.selected_policy_id or matches[0].policy_id
        if current not in labels:
            current = matches[0].policy_id

        chosen = st.radio(
            "Select a policy to inspect",
            options=list(labels.keys()),
            format_func=lambda pid: labels[pid],
            index=list(labels.keys()).index(current),
            key="policy_pick",
            label_visibility="collapsed",
        )
        if chosen != state.selected_policy_id:
            state.selected_policy_id = chosen
            selected = next(m for m in matches if m.policy_id == chosen)
            state.selected_deductible = selected.deductible
            st.rerun()

        match = next(m for m in matches if m.policy_id == chosen)
        st.markdown(f'<div class="rationale">{md_safe(match.rationale)}</div>', unsafe_allow_html=True)

        badge = (
            "pill-ok" if match.match_score >= 75 else "pill-warn" if match.match_score >= 60 else "pill-bad"
        )
        st.markdown(
            f'<span class="pill {badge}">{match.match_score:.0f}% overall match</span>'
            f'<span class="pill pill-info">{match.category}</span>'
            f'<span class="pill pill-info">{match.insurer}</span>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Monthly", money(match.monthly_premium, 2))
        c2.metric("Coverage limit", money(match.coverage_limit))

        st.caption("Ask the assistant about any of these")
        chips = st.columns(2)
        prompts = [
            ("What's covered?", f"What does {match.name} cover?"),
            ("Exclusions", f"What is not covered by {match.name}?"),
            ("Compare top 2", f"Compare {matches[0].name} and {matches[min(1, len(matches)-1)].name}"),
            ("Why this one?", f"Why was {match.name} recommended for me?"),
        ]
        for i, (label, question) in enumerate(prompts):
            if chips[i % 2].button(label, key=f"chip_{i}", width="stretch"):
                st.session_state.pending_question = question
                st.rerun()


# --------------------------------------------------------------------------- #
# Zone C — dynamic policy inspector
# --------------------------------------------------------------------------- #
def render_zone_c() -> None:
    match = state.selected_match
    if match is None or state.profile is None:
        return
    policy = get_policy(match.policy_id)
    if policy is None:
        return

    st.markdown(f"### 🔬 Policy Inspector — {policy.name}")

    deductible = st.select_slider(
        "**Deductible** — what you pay before cover starts",
        options=list(settings.deductible_options),
        value=state.selected_deductible or policy.default_deductible,
        format_func=lambda d: money(d),
        key="deductible_slider",
    )
    if deductible != state.selected_deductible:
        state.selected_deductible = deductible
        st.rerun()

    monthly = quote_premium(policy, state.profile, state.risk, deductible)
    baseline = quote_premium(policy, state.profile, state.risk, policy.default_deductible)
    delta = monthly - baseline

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Monthly premium",
        money(monthly, 2),
        delta=f"{delta:+,.2f} vs standard" if abs(delta) > 0.01 else None,
        delta_color="inverse",
    )
    m2.metric("Annual premium", money(monthly * 12, 2))
    m3.metric("Your deductible", money(deductible))
    m4.metric("Coverage limit", money(policy.coverage_limit))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Premium vs deductible**")
        st_echarts(premium_curve_option(policy, deductible), height="320px", key="premium_curve")
    with right:
        st.markdown("**Coverage allocation**")
        st_echarts(donut_option(policy), height="320px", key="donut")

    st.markdown("**What happens when you claim**")
    st.caption(
        f"With a {md_money(deductible)} deductible on {policy.name}, "
        f"here is how three real-world claim sizes split."
    )
    st_echarts(claim_outcome_option(policy, deductible), height="290px", key="claims")

    with st.expander("📋 Full terms — features, add-ons and exclusions"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Included**")
            for feature in policy.features:
                st.markdown(f"- {md_safe(feature)}")
        with c2:
            st.markdown("**Optional add-ons**")
            if policy.add_ons:
                for addon in policy.add_ons:
                    st.markdown(f"- {md_safe(addon)}")
            else:
                st.caption("None available.")
        with c3:
            st.markdown("**Exclusions**")
            for exclusion in policy.exclusions:
                st.markdown(f"- {md_safe(exclusion)}")


# --------------------------------------------------------------------------- #
# Zone D — catalog, KYC, issuance
# --------------------------------------------------------------------------- #
def render_catalog() -> None:
    rows = recommendation_agent.catalog_with_fit(state.profile, state.risk)
    with st.expander(f"📚 Full catalog — all {len(rows)} products with your personalised fit"):
        frame = pd.DataFrame(
            [
                {
                    "Policy": row.name,
                    "Category": row.category,
                    "Fit %": round(row.match_score, 1),
                    "Monthly": row.monthly_premium,
                    "Deductible": row.deductible,
                    "Limit": row.coverage_limit,
                    "Coverage": row.coverage_breadth,
                    "Perks": row.perks,
                    "Eligible": "✅" if row.eligible else "🚫",
                    "Note": row.ineligibility_reason or row.tagline,
                }
                for row in rows
            ]
        )
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Fit %": st.column_config.ProgressColumn(
                    "Risk Match Fit %", min_value=0, max_value=100, format="%.0f%%"
                ),
                "Monthly": st.column_config.NumberColumn("Monthly", format="$%.2f"),
                "Deductible": st.column_config.NumberColumn("Deductible", format="$%d"),
                "Limit": st.column_config.NumberColumn("Limit", format="$%d"),
                "Coverage": st.column_config.ProgressColumn(
                    "Coverage", min_value=0, max_value=100, format="%d"
                ),
                "Perks": st.column_config.ProgressColumn("Perks", min_value=0, max_value=100, format="%d"),
            },
        )


def render_kyc_and_issuance() -> None:
    st.markdown("### 🔐 Verification & Issuance")

    if state.issued is not None:
        render_issued_contract()
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Step 1 — Aadhaar e-KYC**")

            if state.kyc.verified:
                st.success(
                    f"✅ Verified — {state.kyc.aadhaar_masked} · {state.kyc.verification_level}"
                )
            else:
                aadhaar = st.text_input(
                    "12-digit Aadhaar number",
                    max_chars=14,
                    placeholder="2234 5678 9012",
                    key="aadhaar_input",
                    disabled=state.kyc.otp_sent,
                    help="Never sent to any language model. Stored masked; only a salted hash is retained.",
                )
                if not state.kyc.otp_sent:
                    if st.button("📲 Send OTP", type="primary", width="stretch"):
                        ok, message = workflow.start_kyc(state, aadhaar)
                        st.session_state.kyc_notice = ("success" if ok else "error", message)
                        st.rerun()
                else:
                    st.info(f"OTP sent to the mobile registered against **{state.kyc.aadhaar_masked}**")
                    otp = st.text_input(
                        "6-digit OTP", max_chars=6, placeholder=settings.mock_otp, key="otp_input"
                    )
                    c1, c2 = st.columns([2, 1])
                    if c1.button("✅ Verify OTP", type="primary", width="stretch"):
                        ok, message = workflow.verify_kyc(state, otp)
                        st.session_state.kyc_notice = ("success" if ok else "error", message)
                        st.rerun()
                    if c2.button("↺ Restart", width="stretch"):
                        state.kyc.otp_sent = False
                        state.kyc.txn_id = None
                        st.session_state.kyc_notice = None
                        st.rerun()
                    st.caption(f"Demo environment — the OTP is `{settings.mock_otp}`.")

            notice = st.session_state.kyc_notice
            if notice:
                kind, message = notice
                (st.success if kind == "success" else st.error)(message)

    with right:
        with st.container(border=True):
            st.markdown("**Step 2 — Issue policy**")
            match = state.selected_match
            if match is None:
                st.caption("Select a policy first.")
                return

            deductible = state.selected_deductible or match.deductible
            policy = get_policy(match.policy_id)
            premium = (
                quote_premium(policy, state.profile, state.risk, deductible)
                if policy and state.profile
                else match.monthly_premium
            )

            st.markdown(
                f"**{match.name}**  \n"
                f"{md_money(premium, 2)}/month · {md_money(deductible)} deductible · "
                f"{md_money(match.coverage_limit)} limit"
            )

            if not state.kyc.verified:
                st.warning("Complete identity verification to enable issuance.")
                st.button("📜 Issue Policy", disabled=True, width="stretch")
            else:
                if st.button("📜 Issue Policy", type="primary", width="stretch"):
                    ok, message = workflow.issue_policy(state)
                    state.add_message("assistant", message, agent="Verification & Compliance Agent")
                    st.session_state.kyc_notice = None
                    if not ok:
                        st.error(message)
                    st.rerun()


def render_issued_contract() -> None:
    issued = state.issued
    st.success(f"🎉 **Policy {issued.policy_number} is ACTIVE**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policy number", issued.policy_number)
    c2.metric("Monthly premium", money(issued.monthly_premium, 2))
    c3.metric("Effective", issued.effective_date)
    c4.metric("Expires", issued.expiry_date)

    tab_summary, tab_docs = st.tabs(["📄 Contract summary", "📎 Documents"])

    with tab_summary:
        # Escaped for display only — the download below keeps the raw Markdown.
        st.markdown(md_safe(compliance_agent.contract_markdown(state)))

    with tab_docs:
        st.download_button(
            "⬇️ Download certificate (PDF)",
            data=compliance_agent.contract_pdf(state),
            file_name=f"{issued.policy_number}-certificate.pdf",
            mime="application/pdf",
            type="primary",
        )
        st.download_button(
            "⬇️ Download summary (Markdown)",
            data=compliance_agent.contract_markdown(state),
            file_name=f"{issued.policy_number}-summary.md",
            mime="text/markdown",
        )
        st.caption("Bundled documents")
        for document in issued.documents:
            st.markdown(f"- `{document['name']}` — {document['type']}")


# --------------------------------------------------------------------------- #
# Risk detail
# --------------------------------------------------------------------------- #
def render_risk_detail() -> None:
    with st.expander("🔎 How your risk score was calculated"):
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            st_echarts(risk_factor_option(state), height="330px", key="factors")
        with right:
            risk = state.risk
            st.markdown("**Underwriting data**")
            if risk.mvr:
                st.markdown(
                    f"- Accidents (5 yr): **{risk.mvr.get('accidents_5yr', 0)}**\n"
                    f"- Violations (5 yr): **{risk.mvr.get('violations_5yr', 0)}**\n"
                    f"- Licence: **{risk.mvr.get('licence_status', 'N/A')}**\n"
                    f"- Clean record: **{'yes' if risk.mvr.get('clean_record') else 'no'}**"
                )
            if risk.geo:
                st.markdown(
                    f"- Location: **{risk.geo.get('locality', 'N/A')}**\n"
                    f"- Hazard band: **{risk.geo.get('hazard_band', 'N/A')}** "
                    f"({risk.geo.get('composite_hazard', 0)}/100)\n"
                    f"- Dominant exposure: **{risk.geo.get('dominant_hazard', 'N/A')}**"
                )
            st.caption(
                f"Premium multiplier **{risk.premium_multiplier:.3f}×** is applied to every "
                "base rate before your discount."
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    render_sidebar()
    render_zone_a()

    if state.validation_issues:
        blocking = [i for i in state.validation_issues if i["severity"] in {"error", "anomaly"}]
        if blocking:
            with st.container(border=True):
                st.markdown("#### ⚠️ Corrections needed")
                for issue in blocking:
                    icon = "❌" if issue["severity"] == "error" else "⚠️"
                    remediation = f" _{issue['remediation']}_" if issue.get("remediation") else ""
                    st.markdown(f"{icon} **{issue['field']}** — {issue['message']}{remediation}")

    if not state.recommendations:
        return

    render_risk_detail()
    st.divider()
    render_zone_b()
    st.divider()
    render_zone_c()
    st.divider()
    render_catalog()
    render_kyc_and_issuance()


main()
