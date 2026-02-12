"""
Mortgage Loan Verification vertical UI.
"""

import json
import streamlit as st
from pathlib import Path

from ui.components.hero import render_hero
from ui.components.steps import render_steps
from ui.components.score_card import render_score_card
from ui.components.chat import render_chat
from ui.components.results_common import (
    risk_color, tooltip, _val,
    run_async, create_gauge, create_risk_factors_chart,
    render_skeleton_loader, render_error,
)


AGENT_CARDS_HTML = """
<div style="padding: 0;">
    <div class="agent-card"><span class="agent-icon">📄</span><span class="agent-name">Document Intelligence</span><span class="agent-status">Nemotron</span></div>
    <div class="agent-card"><span class="agent-icon">💰</span><span class="agent-name">Income Verifier</span><span class="agent-status">NIM LLM</span></div>
    <div class="agent-card"><span class="agent-icon">🏠</span><span class="agent-name">Property Valuator</span><span class="agent-status">NIM LLM</span></div>
    <div class="agent-card"><span class="agent-icon">🔍</span><span class="agent-name">Inconsistency Detector</span><span class="agent-status">NIM LLM</span></div>
    <div class="agent-card"><span class="agent-icon">📊</span><span class="agent-name">Risk Scorer</span><span class="agent-status">Ensemble</span></div>
    <div class="agent-card"><span class="agent-icon">📝</span><span class="agent-name">Report Writer</span><span class="agent-status">NIM LLM</span></div>
</div>
"""


def render_sidebar_settings():
    """Render mortgage-specific sidebar settings."""
    with st.expander("AI Agents", expanded=False):
        st.markdown(AGENT_CARDS_HTML, unsafe_allow_html=True)


def _render_flags(flags, title="Flags"):
    """Render a list of flag items."""
    if not flags:
        st.markdown(f"""
        <div class="clean-scan">
            <div style="font-size: 32px; margin-bottom: 8px;">&#10003;</div>
            <div style="font-size: 15px; font-weight: 700; color: #51cf66; margin-bottom: 4px;">Clean Scan</div>
            <div style="font-size: 12px; color: #7a7a90;">No {title.lower()} detected</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for flag in flags:
        sev = flag.get("severity", "medium")
        desc = flag.get("description", "N/A")
        typ = flag.get("type", "N/A")
        conf = flag.get("confidence", 0)
        st.markdown(f"""
        <div class="finding-item">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                <span class="severity-pill severity-{sev}">{sev}</span>
                <span class="finding-title">{desc}</span>
            </div>
            <div class="finding-detail">{typ.replace('_', ' ').title()} &bull; {conf:.0%} confidence</div>
        </div>
        """, unsafe_allow_html=True)


def render_results(r):
    """Render mortgage analysis results."""
    incon_count = len(r.inconsistencies.get("inconsistencies", []))
    income_flags = len(r.income_analysis.get("flags", []))
    property_flags = len(r.property_analysis.get("flags", []))
    confidence = r.scoring_details.get("confidence", 0) if r.scoring_details else 0
    income_match = r.income_analysis.get("income_match_percentage", 0)

    render_score_card(
        score=r.risk_score,
        risk_level=r.risk_level,
        stats={
            "Income": (f"{income_match}%", risk_color('high') if income_match < 70 else risk_color('low')),
            "Issues": (str(incon_count), risk_color('high') if incon_count > 2 else risk_color('low')),
            "Income Flags": (str(income_flags), risk_color('high') if income_flags > 1 else risk_color('low')),
            "Property Flags": (str(property_flags), risk_color('high') if property_flags > 0 else risk_color('low')),
        },
        confidence=confidence,
        recommendation=r.recommendation,
        score_label="Risk Score",
    )

    # Gauge + Risk chart
    col_g, col_f = st.columns([1, 2])
    with col_g:
        st.plotly_chart(create_gauge(r.risk_score, r.risk_level), use_container_width=True)
    with col_f:
        if r.scoring_details:
            fig = create_risk_factors_chart(r.scoring_details)
            if fig:
                st.markdown(f'<div class="section-title" style="font-size: 13px;">Risk Factor Breakdown {tooltip("Each verification dimension scored 0-100.")}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

    # Tabs
    st.markdown("---")
    tab_report, tab_income, tab_property, tab_issues = st.tabs([
        "Report",
        f"Income ({income_flags})",
        f"Property ({property_flags})",
        f"Inconsistencies ({incon_count})",
    ])

    with tab_report:
        if r.narrative:
            st.markdown(f'<div class="narrative-text">{r.narrative}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 30px; color: #55556a; font-size: 12px;">No report generated.</div>', unsafe_allow_html=True)

    with tab_income:
        income = r.income_analysis
        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{_val(income.get('stated_income', 'N/A'))}</div><div class="stat-label-custom">Stated Income</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{_val(income.get('verified_income', 'N/A'))}</div><div class="stat-label-custom">Verified Income</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number" style="color: {'#51cf66' if income_match >= 80 else '#ff9f43' if income_match >= 60 else '#ff4d4f'};">{income_match}%</div><div class="stat-label-custom">Match</div></div>
        </div>
        """, unsafe_allow_html=True)
        if income.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{income["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(income.get("flags", []), "income issues")

    with tab_property:
        prop = r.property_analysis
        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{_val(prop.get('stated_value', 'N/A'))}</div><div class="stat-label-custom">Stated Value</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{_val(prop.get('estimated_fair_value', 'N/A'))}</div><div class="stat-label-custom">Fair Value</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{_val(prop.get('ltv_ratio', 'N/A'))}</div><div class="stat-label-custom">LTV Ratio</div></div>
        </div>
        """, unsafe_allow_html=True)
        if prop.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{prop["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(prop.get("flags", []), "property issues")

    with tab_issues:
        inconsistencies = r.inconsistencies.get("inconsistencies", [])
        _render_flags(inconsistencies, "inconsistencies")



def _build_chat_context(r):
    income = r.income_analysis
    prop = r.property_analysis
    incon_list = r.inconsistencies.get("inconsistencies", [])
    return f"""APPLICATION DATA: {json.dumps(r.application_data, indent=2, default=str)[:2000]}

RISK SCORE: {r.risk_score}/100 ({r.risk_level})
RECOMMENDATION: {r.recommendation}

INCOME ANALYSIS:
- Stated: {income.get('stated_income', 'N/A')}, Verified: {income.get('verified_income', 'N/A')}, Match: {income.get('income_match_percentage', 0)}%
- Flags: {len(income.get('flags', []))}
- Summary: {income.get('summary', 'N/A')}

PROPERTY ANALYSIS:
- Stated Value: {prop.get('stated_value', 'N/A')}, Fair Value: {prop.get('estimated_fair_value', 'N/A')}
- LTV: {prop.get('ltv_ratio', 'N/A')}, Flags: {len(prop.get('flags', []))}
- Summary: {prop.get('summary', 'N/A')}

INCONSISTENCIES: {len(incon_list)} found

NARRATIVE:
{r.narrative[:1200] if r.narrative else 'N/A'}"""


def render():
    """Main entry point for the mortgage verification vertical."""
    render_hero("MortgageLens", "AI", "Verify Every Application")

    uploaded_file = st.file_uploader(
        "Drop mortgage documents (PDF, image, or JSON)",
        type=["pdf", "png", "jpg", "jpeg", "json"],
        label_visibility="collapsed",
        key="mortgage_upload",
    )

    has_result = "mortgage_result" in st.session_state
    step = 3 if has_result else (2 if uploaded_file else 1)
    render_steps(step, labels=["Upload", "Verify", "Report"])

    input_path = None
    if uploaded_file:
        temp_path = Path(f"/tmp/{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_path = str(temp_path)

    if input_path:
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("Verify Application", type="primary", use_container_width=True)

        if analyze_clicked:
            progress = st.empty()
            progress.markdown(render_skeleton_loader("Verifying documents...", "5 AI agents checking application"), unsafe_allow_html=True)

            try:
                import core.nim_client as _nc
                _nc._nim_client = None
                from mortgage_lens import MortgageLensAI
                detector = MortgageLensAI()
                result = run_async(detector.analyze(input_path))
                st.session_state.mortgage_result = result
                st.session_state.mortgage_chat = []
                progress.empty()
            except Exception as e:
                progress.empty()
                st.markdown(render_error(str(e)), unsafe_allow_html=True)
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    if "mortgage_result" in st.session_state:
        r = st.session_state.mortgage_result
        st.markdown('<div style="height: 1px; background: var(--border); margin: 12px 0;"></div>', unsafe_allow_html=True)
        render_results(r)
        render_chat(
            context_text=_build_chat_context(r),
            session_key="mortgage_chat",
            persona="expert mortgage fraud analyst",
            vertical_name="MortgageLens AI",
            avatar="🏠",
            suggestions=[
                ("Income Issues", "What income verification issues were found and how severe are they?"),
                ("Property Risks", "Explain any property valuation concerns in this application"),
                ("Overall Risk", "Give me a summary of the overall fraud risk for this mortgage application"),
            ],
        )
