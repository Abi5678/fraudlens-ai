"""
Medical Insurance Fraud Detection vertical UI.
Detects upcoding, unbundling, services not rendered, duplicate billing,
and clinical inconsistencies in medical insurance claims.
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
    render_skeleton_loader, render_error, user_facing_error,
)


# ---------------------------------------------------------------------------
# Agent cards for sidebar
# ---------------------------------------------------------------------------
AGENT_CARDS_HTML = """
<div style="padding: 0;">
    <div class="agent-card"><span class="agent-icon">📄</span><span class="agent-name">Document Intelligence</span><span class="agent-status">Nemotron-Parse</span></div>
    <div class="agent-card"><span class="agent-icon">💊</span><span class="agent-name">Billing Integrity</span><span class="agent-status">NIM Llama 3.3</span></div>
    <div class="agent-card"><span class="agent-icon">🩺</span><span class="agent-name">Clinical Consistency</span><span class="agent-status">NIM Llama 3.3</span></div>
    <div class="agent-card"><span class="agent-icon">📋</span><span class="agent-name">Eligibility & Duplicates</span><span class="agent-status">NIM Llama 3.3</span></div>
    <div class="agent-card"><span class="agent-icon">🔍</span><span class="agent-name">Inconsistency Detector</span><span class="agent-status">NIM Llama 3.3</span></div>
    <div class="agent-card"><span class="agent-icon">📊</span><span class="agent-name">Risk Scorer</span><span class="agent-status">Weighted Ensemble</span></div>
    <div class="agent-card"><span class="agent-icon">📝</span><span class="agent-name">Report Writer</span><span class="agent-status">NIM Llama 3.3</span></div>
</div>
"""


def render_sidebar_settings():
    """Render medical insurance sidebar settings."""
    with st.expander("AI Agents", expanded=False):
        st.markdown(AGENT_CARDS_HTML, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        codes = flag.get("codes_involved", [])
        codes_html = ""
        if codes:
            codes_html = " &bull; Codes: " + ", ".join(f"<code>{c}</code>" for c in codes[:5])
        st.markdown(f"""
        <div class="finding-item">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                <span class="severity-pill severity-{sev}">{sev}</span>
                <span class="finding-title">{desc}</span>
            </div>
            <div class="finding-detail">{typ.replace('_', ' ').title()} &bull; {conf:.0%} confidence{codes_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_results(r):
    """Render medical claim analysis results."""
    billing_flags = len(r.billing_analysis.get("flags", []))
    clinical_flags = len(r.clinical_analysis.get("flags", []))
    eligibility_flags = len(r.eligibility_analysis.get("flags", []))
    incon_count = len(r.inconsistencies.get("inconsistencies", []))
    confidence = r.scoring_details.get("confidence", 0) if r.scoring_details else 0

    # Denial risk
    denial = r.clinical_analysis.get("denial_risk", {})
    denial_likely = denial.get("likely_denied", False) if denial else False

    render_score_card(
        score=r.risk_score,
        risk_level=r.risk_level,
        stats={
            "Billing": (str(billing_flags), risk_color('high') if billing_flags > 0 else risk_color('low')),
            "Clinical": (str(clinical_flags), risk_color('high') if clinical_flags > 0 else risk_color('low')),
            "Eligibility": (str(eligibility_flags), risk_color('high') if eligibility_flags > 0 else risk_color('low')),
            "Denial Risk": ("Yes" if denial_likely else "No", risk_color('critical') if denial_likely else risk_color('low')),
        },
        confidence=confidence,
        recommendation=r.recommendation,
        score_label="Fraud Score",
    )

    # Gauge + Risk chart
    col_g, col_f = st.columns([1, 2])
    with col_g:
        st.plotly_chart(create_gauge(r.risk_score, r.risk_level), use_container_width=True)
    with col_f:
        if r.scoring_details:
            fig = create_risk_factors_chart(r.scoring_details)
            if fig:
                st.markdown(f'<div class="section-title" style="font-size: 13px;">Risk Factor Breakdown {tooltip("Each analysis dimension scored 0-100.")}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

    # Tabs
    st.markdown("---")
    tab_report, tab_billing, tab_clinical, tab_eligibility, tab_issues = st.tabs([
        "Report",
        f"Billing ({billing_flags})",
        f"Clinical ({clinical_flags})",
        f"Eligibility ({eligibility_flags})",
        f"Inconsistencies ({incon_count})",
    ])

    with tab_report:
        if r.narrative:
            st.markdown(f'<div class="narrative-text">{r.narrative}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 30px; color: #55556a; font-size: 12px;">No report generated.</div>', unsafe_allow_html=True)

    with tab_billing:
        billing = r.billing_analysis
        codes = billing.get("codes_analyzed", {})
        cpt_codes = codes.get("cpt_codes", [])
        icd_codes = codes.get("icd10_codes", [])
        total_billed = codes.get("total_billed", "N/A")
        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{len(cpt_codes)}</div><div class="stat-label-custom">CPT Codes</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{len(icd_codes)}</div><div class="stat-label-custom">ICD-10 Codes</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{_val(total_billed)}</div><div class="stat-label-custom">Total Billed</div></div>
        </div>
        """, unsafe_allow_html=True)
        if cpt_codes:
            st.markdown(f'<div style="font-size: 11px; color: #7a7a90; margin-bottom: 6px;">CPT: {", ".join(str(c) for c in cpt_codes[:10])}</div>', unsafe_allow_html=True)
        if icd_codes:
            st.markdown(f'<div style="font-size: 11px; color: #7a7a90; margin-bottom: 12px;">ICD-10: {", ".join(str(c) for c in icd_codes[:10])}</div>', unsafe_allow_html=True)
        if billing.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{billing["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(billing.get("flags", []), "billing issues")

    with tab_clinical:
        clinical = r.clinical_analysis
        clin_summary = clinical.get("clinical_summary", {})
        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div class="stat-card" style="flex: 1;"><div class="stat-number" style="font-size: 13px;">{_val(clin_summary.get('primary_diagnosis', 'N/A'))}</div><div class="stat-label-custom">Primary Dx</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number" style="color: {'#51cf66' if clin_summary.get('procedures_justified') else '#ff4d4f'};">{'Yes' if clin_summary.get('procedures_justified') else 'No'}</div><div class="stat-label-custom">Procedures Justified</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number" style="color: {'#51cf66' if clin_summary.get('treatment_appropriate') else '#ff4d4f'};">{'Yes' if clin_summary.get('treatment_appropriate') else 'No'}</div><div class="stat-label-custom">Treatment Appropriate</div></div>
        </div>
        """, unsafe_allow_html=True)
        # Denial risk
        if denial and denial.get("denial_reasons"):
            reasons = denial["denial_reasons"]
            reasons_html = "".join(f'<li style="margin-bottom: 3px;">{r}</li>' for r in reasons[:5])
            st.markdown(f"""
            <div class="dark-card" style="border-color: rgba(255,77,79,0.4); margin-bottom: 12px; padding: 12px;">
                <div style="font-size: 12px; font-weight: 700; color: #ff4d4f; margin-bottom: 6px;">Denial Risk Factors</div>
                <ul style="font-size: 11px; color: #b0b0c0; margin: 0; padding-left: 16px;">{reasons_html}</ul>
            </div>
            """, unsafe_allow_html=True)
        if clinical.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{clinical["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(clinical.get("flags", []), "clinical issues")

    with tab_eligibility:
        elig = r.eligibility_analysis
        dup = elig.get("duplicate_indicators", {})
        dup_count = dup.get("potential_duplicates", 0) if dup else 0
        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div class="stat-card" style="flex: 1;"><div class="stat-number" style="color: {'#51cf66' if elig.get('eligible') else '#ff4d4f'};">{'Yes' if elig.get('eligible') else 'No'}</div><div class="stat-label-custom">Eligible</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{dup_count}</div><div class="stat-label-custom">Duplicates</div></div>
        </div>
        """, unsafe_allow_html=True)
        if elig.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{elig["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(elig.get("flags", []), "eligibility issues")

    with tab_issues:
        inconsistencies = r.inconsistencies.get("inconsistencies", [])
        _render_flags(inconsistencies, "inconsistencies")


# ---------------------------------------------------------------------------
# Chat context
# ---------------------------------------------------------------------------

def _build_chat_context(r):
    billing = r.billing_analysis
    clinical = r.clinical_analysis
    elig = r.eligibility_analysis
    incon_list = r.inconsistencies.get("inconsistencies", [])
    return f"""CLAIM DATA: {json.dumps(r.claim_data, indent=2, default=str)[:2000]}

RISK SCORE: {r.risk_score}/100 ({r.risk_level})
RECOMMENDATION: {r.recommendation}

BILLING ANALYSIS:
- Verified: {billing.get('billing_verified', 'N/A')}
- Risk Score: {billing.get('risk_score', 0)}
- Flags: {len(billing.get('flags', []))}
- Summary: {billing.get('summary', 'N/A')}

CLINICAL CONSISTENCY:
- Consistent: {clinical.get('clinically_consistent', 'N/A')}
- Risk Score: {clinical.get('risk_score', 0)}
- Flags: {len(clinical.get('flags', []))}
- Denial Likely: {clinical.get('denial_risk', {}).get('likely_denied', 'N/A')}
- Summary: {clinical.get('summary', 'N/A')}

ELIGIBILITY:
- Eligible: {elig.get('eligible', 'N/A')}
- Risk Score: {elig.get('risk_score', 0)}
- Duplicates: {elig.get('duplicate_indicators', {}).get('potential_duplicates', 0)}
- Summary: {elig.get('summary', 'N/A')}

INCONSISTENCIES: {len(incon_list)} found

NARRATIVE:
{r.narrative[:1200] if r.narrative else 'N/A'}"""


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    """Main entry point for the medical insurance fraud detection vertical."""
    render_hero("MedClaim", "AI", "Detect Medical Billing Fraud")

    st.markdown("""
    <div style="font-size: 12px; color: #7a7a90; margin-top: -6px; margin-bottom: 10px; line-height: 1.5;">
    Upload medical claim documents including <b>CMS-1500</b>, <b>UB-04</b>, <b>EOBs</b>,
    clinical notes, or structured JSON with CPT/ICD-10 codes.
    </div>
    """, unsafe_allow_html=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    sample_claim_path = project_root / "demo_assets" / "sample_medical_claim.json"

    input_mode = st.radio(
        "Input",
        options=["Upload file", "Use sample claim"],
        horizontal=True,
        key="medical_input_mode",
        label_visibility="collapsed",
    )

    uploaded_file = None
    if input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "Drop medical claim documents (PDF, image, or JSON)",
            type=["pdf", "png", "jpg", "jpeg", "json"],
            label_visibility="collapsed",
            key="medical_upload",
        )

    use_sample = input_mode == "Use sample claim" and sample_claim_path.exists()

    has_result = "medical_result" in st.session_state
    has_input = bool(uploaded_file) or use_sample
    step = 3 if has_result else (2 if has_input else 1)
    render_steps(step, labels=["Upload", "Analyze", "Report"])

    input_path = None
    if uploaded_file:
        temp_path = Path(f"/tmp/{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_path = str(temp_path)
    elif use_sample:
        input_path = str(sample_claim_path)
        st.caption("Using demo sample claim (Jane Doe, Demo Medical Group). No real data.")

    if input_path:
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("Analyze Claim", type="primary", use_container_width=True)

        if analyze_clicked:
            progress = st.empty()
            progress.markdown(render_skeleton_loader("Analyzing medical claim...", "7 AI agents checking billing, clinical & eligibility"), unsafe_allow_html=True)

            try:
                import core.nim_client as _nc
                _nc._nim_client = None
                from medical_lens import MedicalClaimLensAI
                detector = MedicalClaimLensAI()
                result = run_async(detector.analyze(input_path))
                st.session_state.medical_result = result
                st.session_state.medical_chat = []
                progress.empty()
            except Exception as e:
                progress.empty()
                st.markdown(render_error(user_facing_error(e)), unsafe_allow_html=True)
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    if "medical_result" in st.session_state:
        r = st.session_state.medical_result
        st.markdown('<div style="height: 1px; background: var(--border); margin: 12px 0;"></div>', unsafe_allow_html=True)
        render_results(r)
        render_chat(
            context_text=_build_chat_context(r),
            session_key="medical_chat",
            persona="expert medical billing fraud analyst",
            vertical_name="MedClaim AI",
            avatar="🩺",
            suggestions=[
                ("Billing Issues", "What billing integrity issues were found? Any upcoding or unbundling?"),
                ("Clinical Review", "Are the treatments clinically justified based on the documentation?"),
                ("Denial Risk", "Would this claim likely be denied on audit? Why?"),
                ("Summary", "Give me a brief summary of all findings in this medical claim analysis"),
            ],
        )
