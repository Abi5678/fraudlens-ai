"""
Insurance Fraud Detection vertical — extracted from the original app.py.
All insurance-specific UI logic lives here.
"""

import json
import streamlit as st
from pathlib import Path

from ui.components.hero import render_hero
from ui.components.steps import render_steps
from ui.components.score_card import render_score_card
from ui.components.chat import render_chat
from ui.components.results_common import (
    risk_color, risk_icon, tooltip, _val,
    run_async, create_gauge, create_risk_factors_chart,
    create_network_graph, render_skeleton_loader, render_error,
)


# ---------------------------------------------------------------------------
# Agent cards for sidebar
# ---------------------------------------------------------------------------
AGENT_CARDS_HTML = """
<div style="padding: 0;">
    <div class="agent-card"><span class="agent-icon">📄</span><span class="agent-name">Document Intelligence</span><span class="agent-status">Nemotron</span></div>
    <div class="agent-card"><span class="agent-icon">🔍</span><span class="agent-name">Inconsistency Detector</span><span class="agent-status">NIM LLM</span></div>
    <div class="agent-card"><span class="agent-icon">🎯</span><span class="agent-name">Pattern Matcher</span><span class="agent-status">NeMo RAG</span></div>
    <div class="agent-card"><span class="agent-icon">📊</span><span class="agent-name">Risk Scorer</span><span class="agent-status">Ensemble</span></div>
    <div class="agent-card"><span class="agent-icon">📝</span><span class="agent-name">Narrative Writer</span><span class="agent-status">NIM LLM</span></div>
    <div class="agent-card"><span class="agent-icon">🕸️</span><span class="agent-name">Network Analyzer</span><span class="agent-status">cuGraph</span></div>
    <div class="agent-card"><span class="agent-icon">🖼️</span><span class="agent-name">Deepfake Detector</span><span class="agent-status">TensorRT</span></div>
</div>
"""


def render_sidebar_settings():
    """Render insurance-specific sidebar settings."""
    with st.expander("AI Agents", expanded=False):
        st.markdown(AGENT_CARDS_HTML, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Document details
# ---------------------------------------------------------------------------

def render_document_details(r):
    cd = r.claim_data
    claimant = cd.get("claimant", {})
    claim = cd.get("claim", {})
    vehicle = cd.get("vehicle", {})
    incident = cd.get("incident", {})

    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div class="section-title" style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 15px;">📋</span> Extracted Document Data
        </div>
        <div style="font-size: 11px; color: #7a7a90; margin-top: -8px; margin-bottom: 12px;">
            Verify these fields match the original claim document
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="dark-card" style="padding: 16px;">
            <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #76B900; margin-bottom: 10px;">Claimant</div>
            <div class="summary-row"><span class="summary-key">Name</span><span class="summary-value">{_val(claimant.get('name'))}</span></div>
            <div class="summary-row"><span class="summary-key">Address</span><span class="summary-value" style="font-size: 11px; max-width: 180px; text-align: right;">{_val(claimant.get('address'))}</span></div>
            <div class="summary-row"><span class="summary-key">Phone</span><span class="summary-value data-mono">{_val(claimant.get('phone'))}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        claim_amt = claim.get('amount', 0)
        claim_display = f"${claim_amt:,.2f}" if claim_amt else "--"
        st.markdown(f"""
        <div class="dark-card" style="padding: 16px;">
            <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #76B900; margin-bottom: 10px;">Claim Details</div>
            <div class="summary-row"><span class="summary-key">Claim #</span><span class="summary-value data-mono">{_val(claim.get('number'))}</span></div>
            <div class="summary-row"><span class="summary-key">Amount</span><span class="summary-value data-mono">{claim_display}</span></div>
            <div class="summary-row"><span class="summary-key">Type</span><span class="summary-value">{_val(claim.get('type'))}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        vehicle_parts = [_val(vehicle.get('year'), ''), _val(vehicle.get('make'), ''), _val(vehicle.get('model'), '')]
        vehicle_desc = ' '.join(p for p in vehicle_parts if p).strip() or "--"
        st.markdown(f"""
        <div class="dark-card" style="padding: 16px;">
            <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #76B900; margin-bottom: 10px;">Vehicle / Incident</div>
            <div class="summary-row"><span class="summary-key">Vehicle</span><span class="summary-value">{vehicle_desc}</span></div>
            <div class="summary-row"><span class="summary-key">VIN</span><span class="summary-value data-mono" style="font-size: 10px;">{_val(vehicle.get('vin'))}</span></div>
            <div class="summary-row"><span class="summary-key">Incident</span><span class="summary-value">{_val(incident.get('date'))}</span></div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Deepfake results
# ---------------------------------------------------------------------------

def render_deepfake_results(r):
    df = r.deepfake_analysis
    images = getattr(r, 'extracted_images', []) or []

    if not df and not images:
        return

    st.markdown("""
    <div style="margin-top: 4px; margin-bottom: 12px;">
        <div class="section-title" style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 15px;">🖼️</span> Image Authenticity Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_score, col_detail = st.columns([1, 2])

    with col_score:
        if df and df.get("status") == "success":
            score = df.get("manipulation_score", 0)
            sc = "#ff4d4f" if score >= 60 else "#ff9f43" if score >= 35 else "#51cf66"
            label = "HIGH RISK" if score >= 60 else "MODERATE" if score >= 35 else "LOW RISK"
            img_count = df.get("images_analyzed", 0)
            detections = df.get("detections", [])
            det_html = ""
            if detections:
                det_html = '<div style="margin-top: 8px;">' + ''.join(
                    f'<span style="display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 20px; font-size: 10px; font-weight: 600; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #b0b0c0;">{d.replace("_", " ").title()}</span>'
                    for d in detections[:6]
                ) + '</div>'

            st.markdown(f"""
            <div class="dark-card" style="text-align: center; padding: 20px;">
                <div style="font-size: 42px; font-weight: 800; color: {sc}; letter-spacing: -2px;">{score:.0f}</div>
                <div style="font-size: 9px; font-weight: 600; color: #55556a; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px;">Manipulation Score</div>
                <div style="margin-top: 8px;">
                    <span style="padding: 3px 10px; border-radius: 20px; font-size: 10px; font-weight: 700; background: {'rgba(255,77,79,0.12)' if score >= 60 else 'rgba(255,159,67,0.12)' if score >= 35 else 'rgba(81,207,102,0.12)'}; color: {sc};">{label}</span>
                </div>
                <div style="margin-top: 10px; font-size: 11px; color: #7a7a90;">{img_count} images analyzed</div>
                {det_html}
            </div>
            """, unsafe_allow_html=True)
        elif df and df.get("status") == "skipped":
            st.markdown("""
            <div class="dark-card" style="text-align: center; padding: 20px;">
                <div style="font-size: 11px; color: #55556a;">No images found for analysis</div>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        if images:
            jpeg_imgs = [p for p in images if p.lower().endswith(('.jpg', '.jpeg'))]
            other_imgs = [p for p in images if not p.lower().endswith(('.jpg', '.jpeg'))]
            ordered_imgs = (jpeg_imgs + other_imgs)[:12]
            n_cols = min(4, max(1, len(ordered_imgs)))

            st.markdown(
                f'<div style="font-size: 11px; font-weight: 600; color: #7a7a90; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">'
                f'Extracted Images ({len(images)}{" — showing " + str(len(ordered_imgs)) if len(ordered_imgs) < len(images) else ""})'
                f'</div>',
                unsafe_allow_html=True,
            )

            for row_start in range(0, len(ordered_imgs), n_cols):
                row_imgs = ordered_imgs[row_start:row_start + n_cols]
                img_cols = st.columns(n_cols)
                for j, img_path in enumerate(row_imgs):
                    with img_cols[j]:
                        try:
                            st.image(img_path, use_container_width=True)
                        except Exception:
                            st.markdown(
                                f'<div style="background: var(--bg-elevated); padding: 20px; border-radius: 8px; text-align: center; font-size: 10px; color: #55556a;">Image {row_start + j + 1}</div>',
                                unsafe_allow_html=True,
                            )

        individual = df.get("individual_results", []) if df else []
        if individual:
            st.markdown('<div style="font-size: 11px; font-weight: 600; color: #7a7a90; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 12px; margin-bottom: 6px;">Per-Image Results</div>', unsafe_allow_html=True)
            chips_html = ""
            for idx, res in enumerate(individual[:16]):
                s = res.get("score", 0)
                sc = "#ff4d4f" if s >= 60 else "#ff9f43" if s >= 35 else "#51cf66"
                path = res.get("path", f"Image {idx+1}")
                name = Path(path).name if path else f"Image {idx+1}"
                short = name[:18] + "…" if len(name) > 19 else name
                chips_html += f'<span style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 20px; font-size: 10px; color: #b0b0c0; white-space: nowrap;"><span style="overflow: hidden; text-overflow: ellipsis; max-width: 120px;">{short}</span><span style="font-weight: 700; color: {sc}; font-size: 11px;">{s:.0f}</span></span>\n'
            st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 6px;">{chips_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Full results panel
# ---------------------------------------------------------------------------

def render_results(r):
    rc = risk_color(r.risk_level)
    incon_count = len(r.inconsistencies.get("inconsistencies", []))
    pat_count = len(r.pattern_matches.get("matched_patterns", []))
    confidence = r.scoring_details.get("confidence", 0) if r.scoring_details else 0
    img_count = len(getattr(r, 'extracted_images', []) or [])

    # Score card
    render_score_card(
        score=r.fraud_score,
        risk_level=r.risk_level,
        stats={
            "Issues": (str(incon_count), risk_color('high') if incon_count > 2 else risk_color('low')),
            "Patterns": (str(pat_count), risk_color('high') if pat_count > 1 else risk_color('low')),
            "Ring": ("Yes" if r.fraud_ring_detected else "No", '#ff4d4f' if r.fraud_ring_detected else '#51cf66'),
            "Images": (str(img_count), None),
        },
        confidence=confidence,
        recommendation=r.recommendation,
        score_label="Fraud Score",
    )

    # Gauge + Risk chart
    col_g, col_f = st.columns([1, 2])
    with col_g:
        st.plotly_chart(create_gauge(r.fraud_score, r.risk_level), use_container_width=True)
    with col_f:
        if r.scoring_details:
            fig = create_risk_factors_chart(r.scoring_details)
            if fig:
                st.markdown(f'<div class="section-title" style="font-size: 13px;">Risk Factor Breakdown {tooltip("Each agent scores a dimension 0-100. The weighted sum produces the overall fraud score.")}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

    # Document details
    render_document_details(r)

    # Deepfake / image analysis
    render_deepfake_results(r)

    # Scoring methodology
    if r.scoring_details:
        risk_factors = r.scoring_details.get("risk_factors", [])
        reasoning = r.scoring_details.get("reasoning", "")

        with st.expander("Scoring Methodology", expanded=False):
            weight_map = {
                "Inconsistencies": ("Inconsistency Agent", "Detects contradictions in statements, dates, amounts"),
                "Fraud Pattern Match": ("Pattern Matcher (RAG)", "Matches against known fraud patterns via NeMo embeddings"),
                "Network/Ring Risk": ("Network Analyzer", "Identifies fraud ring indicators via graph analysis"),
                "Image Authenticity": ("Deepfake Detector", "Analyzes claim photos for AI manipulation or tampering"),
                "Claim Characteristics": ("Claim Scorer", "Evaluates inherent claim risk flags (amount, injury type)"),
            }

            for rf in risk_factors:
                rf_name = rf.get("name", "")
                rf_score = rf.get("score", 0)
                rf_weight = rf.get("weight", 0)
                rf_weighted = rf.get("weighted_score", rf_score * rf_weight)
                rf_evidence = rf.get("evidence", [])
                info = weight_map.get(rf_name, (rf_name, ""))
                sc = "#ff4d4f" if rf_score >= 60 else "#ff9f43" if rf_score >= 35 else "#51cf66"
                ev_html = ""
                real_ev = [str(e) for e in rf_evidence if e]
                if real_ev:
                    ev_html = '<div style="margin-top: 4px;">' + ''.join(f'<div style="font-size: 10px; color: #7a7a90;">• {e}</div>' for e in real_ev[:3]) + '</div>'
                st.markdown(f"""
                <div style="margin-bottom: 10px; padding: 10px 12px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div><span style="font-size: 12px; font-weight: 600; color: #f0f0f5;">{info[0]}</span><span style="font-size: 10px; color: #55556a; margin-left: 6px;">{rf_weight:.0%}</span></div>
                        <div><span style="font-size: 14px; font-weight: 700; color: {sc};">{rf_score:.0f}</span><span style="font-size: 10px; color: #55556a;">/100 → {rf_weighted:.1f}pts</span></div>
                    </div>
                    <div style="background: rgba(255,255,255,0.05); border-radius: 3px; height: 4px; overflow: hidden; margin: 6px 0;">
                        <div style="height: 100%; width: {rf_score}%; background: {sc}; border-radius: 3px;"></div>
                    </div>
                    <div style="font-size: 10px; color: #b0b0c0;">{info[1]}</div>
                    {ev_html}
                </div>
                """, unsafe_allow_html=True)
            if reasoning:
                st.markdown(f"""
                <div style="padding: 10px 12px; background: rgba(118,185,0,0.05); border-radius: 8px; border: 1px solid rgba(118,185,0,0.12);">
                    <div style="font-size: 10px; font-weight: 600; color: #76B900; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">AI Reasoning</div>
                    <div style="font-size: 12px; color: #b0b0c0; line-height: 1.5;">{reasoning}</div>
                </div>
                """, unsafe_allow_html=True)

    # Detailed analysis tabs
    st.markdown("---")
    tab_narrative, tab_incon, tab_pat, tab_net = st.tabs([
        "Investigation",
        f"Inconsistencies ({incon_count})",
        f"Patterns ({pat_count})",
        "Network",
    ])

    with tab_narrative:
        if r.narrative:
            st.markdown(f'<div class="narrative-text">{r.narrative}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 30px; color: #55556a; font-size: 12px;">No narrative generated.</div>', unsafe_allow_html=True)

    with tab_incon:
        inconsistencies = r.inconsistencies.get("inconsistencies", [])
        if not inconsistencies:
            st.markdown("""
            <div class="clean-scan">
                <div style="font-size: 32px; margin-bottom: 8px;">&#10003;</div>
                <div style="font-size: 15px; font-weight: 700; color: #51cf66; margin-bottom: 4px;">Clean Scan</div>
                <div style="font-size: 12px; color: #7a7a90;">No inconsistencies detected in this claim</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for inc in inconsistencies:
                sev = inc.get("severity", "medium")
                desc = inc.get("description", "N/A")
                typ = inc.get("type", "N/A")
                conf = inc.get("confidence", 0)
                st.markdown(f"""
                <div class="finding-item">
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                        <span class="severity-pill severity-{sev}">{sev}</span>
                        <span class="finding-title">{desc}</span>
                    </div>
                    <div class="finding-detail">{typ} &bull; {conf:.0%} confidence</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_pat:
        patterns = r.pattern_matches.get("matched_patterns", [])
        if not patterns:
            st.markdown("""
            <div class="clean-scan">
                <div style="font-size: 32px; margin-bottom: 8px;">&#10003;</div>
                <div style="font-size: 15px; font-weight: 700; color: #51cf66; margin-bottom: 4px;">Clean Scan</div>
                <div style="font-size: 12px; color: #7a7a90;">No known fraud patterns matched</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for pat in patterns:
                sev = pat.get("severity", "medium")
                name = pat.get("pattern_name", "Unknown")
                cat = pat.get("category", "N/A")
                sim = pat.get("similarity_score", 0)
                st.markdown(f"""
                <div class="finding-item">
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                        <span class="severity-pill severity-{sev}">{sev}</span>
                        <span class="finding-title">{name}</span>
                        <span style="margin-left: auto; font-size: 11px; font-weight: 700; color: #76B900;">{sim:.0%}</span>
                    </div>
                    <div class="finding-detail">Category: {cat}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_net:
        if r.network_analysis:
            summary = r.network_analysis.get("summary", "")
            analysis_text = r.network_analysis.get("analysis", "")
            ring = r.network_analysis.get("fraud_ring_detected", False)
            net_score = r.network_analysis.get("network_risk_score", 0)
            indicators = r.network_analysis.get("indicators", [])
            ring_c = "#ff4d4f" if ring else "#51cf66"
            st.markdown(f"""
            <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                <div class="stat-card" style="flex: 1;"><div class="stat-number" style="color: {ring_c}; font-size: 16px;">{"DETECTED" if ring else "NOT DETECTED"}</div><div class="stat-label-custom">Fraud Ring</div></div>
                <div class="stat-card" style="flex: 1;"><div class="stat-number">{net_score}</div><div class="stat-label-custom">Network Risk</div></div>
            </div>
            """, unsafe_allow_html=True)
            net_fig = create_network_graph(r.network_analysis)
            if net_fig:
                st.plotly_chart(net_fig, use_container_width=True)
            if summary:
                st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{summary}</div>', unsafe_allow_html=True)
            real_indicators = [i for i in indicators if i.get("name") != "_llm_summary"]
            if real_indicators:
                for ind in real_indicators:
                    s = ind.get("score", 0)
                    sc = "#ff4d4f" if s >= 60 else "#ff9f43" if s >= 35 else "#51cf66"
                    st.markdown(f"""
                    <div style="margin-bottom: 10px; padding: 10px 12px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-size: 12px; font-weight: 600; color: #f0f0f5;">{ind.get("name","")}</span>
                            <span style="font-size: 13px; font-weight: 700; color: {sc};">{s}/100</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); border-radius: 3px; height: 4px; overflow: hidden; margin-bottom: 6px;">
                            <div style="height: 100%; width: {s}%; background: {sc}; border-radius: 3px;"></div>
                        </div>
                        <div style="font-size: 11px; color: #b0b0c0; line-height: 1.4;">{ind.get("explanation","")}</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif analysis_text:
                with st.expander("Detailed Network Analysis"):
                    st.markdown(f'<div class="narrative-text">{analysis_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dark-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 24px; margin-bottom: 8px; opacity: 0.4;">&#128279;</div>
                <div style="font-size: 13px; font-weight: 600; color: #7a7a90;">Network Analysis Not Included</div>
                <div style="font-size: 11px; color: #55556a; margin-top: 4px;">Enable network analysis in the sidebar to detect fraud rings</div>
            </div>
            """, unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# Chat context builder
# ---------------------------------------------------------------------------

def _build_chat_context(r):
    """Build a clean text context string from insurance analysis results."""
    incon_list = r.inconsistencies.get("inconsistencies", [])
    if incon_list:
        incon_lines = [f"- [{inc.get('severity','?').upper()}] {inc.get('description','N/A')} (type: {inc.get('type','N/A')}, confidence: {inc.get('confidence',0):.0%})" for inc in incon_list]
        incon_summary = "\n".join(incon_lines)
    else:
        incon_summary = "No inconsistencies detected."

    patterns = r.pattern_matches.get("matched_patterns", [])
    if patterns:
        pat_lines = [f"- [{pat.get('severity','?').upper()}] {pat.get('pattern_name','Unknown')} (category: {pat.get('category','N/A')}, similarity: {pat.get('similarity_score',0):.0%})" for pat in patterns]
        pat_summary = "\n".join(pat_lines)
    else:
        pat_summary = "No fraud patterns matched."
    pat_score = r.pattern_matches.get("pattern_risk_score", "N/A")

    net = r.network_analysis or {}
    ring = net.get("fraud_ring_detected", False)
    net_score = net.get("network_risk_score", "N/A")
    net_text = net.get("summary", net.get("analysis", ""))[:500]
    net_summary = f"Fraud ring detected: {'Yes' if ring else 'No'}, Risk score: {net_score}\n{net_text}"

    df = r.deepfake_analysis or {}
    df_det = df.get("deepfake_detected", False)
    df_score = df.get("confidence_score", "N/A")
    df_summary = f"Deepfake detected: {'Yes' if df_det else 'No'}, Confidence: {df_score}"

    return f"""CLAIM DATA: {json.dumps(r.claim_data, indent=2, default=str)[:2000]}

FRAUD SCORE: {r.fraud_score}/100 ({r.risk_level})
RECOMMENDATION: {r.recommendation}

INCONSISTENCIES ({len(incon_list)} found):
{incon_summary}

PATTERN MATCHES ({len(patterns)} found, risk score: {pat_score}):
{pat_summary}

NETWORK ANALYSIS:
{net_summary}

DEEPFAKE ANALYSIS:
{df_summary}

NARRATIVE:
{r.narrative[:1200] if r.narrative else 'N/A'}"""


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    """Main entry point for the insurance fraud vertical."""
    render_hero("FraudLens", "AI", "See Through Every Claim")

    uploaded_file = st.file_uploader(
        "Drop a claim document (PDF, image, or JSON)",
        type=["pdf", "png", "jpg", "jpeg", "json"],
        label_visibility="collapsed",
        key="insurance_upload",
    )

    has_result = "insurance_result" in st.session_state
    step = 3 if has_result else (2 if uploaded_file else 1)
    render_steps(step)

    input_path = None
    if uploaded_file:
        temp_path = Path(f"/tmp/{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_path = str(temp_path)

    if input_path:
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("Analyze Claim", type="primary", use_container_width=True)

        if analyze_clicked:
            progress = st.empty()
            progress.markdown(render_skeleton_loader("Analyzing claim...", "7 AI agents working in parallel"), unsafe_allow_html=True)

            try:
                import core.nim_client as _nc
                _nc._nim_client = None
                from fraudlens import FraudLensAI
                detector = FraudLensAI()
                result = run_async(detector.analyze(input_path))
                st.session_state.insurance_result = result
                st.session_state.insurance_chat = []
                progress.empty()
            except Exception as e:
                progress.empty()
                st.markdown(render_error(str(e)), unsafe_allow_html=True)
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    if "insurance_result" in st.session_state:
        r = st.session_state.insurance_result
        st.markdown('<div style="height: 1px; background: var(--border); margin: 12px 0;"></div>', unsafe_allow_html=True)
        render_results(r)
        context = _build_chat_context(r)
        render_chat(
            context_text=context,
            session_key="insurance_chat",
            persona="expert insurance fraud analyst",
            vertical_name="FraudLens AI",
            suggestions=[
                ("Explain Risk Score", "Explain how the fraud score was calculated and what factors contributed most"),
                ("Summarize Findings", "Give me a brief summary of all findings in this analysis"),
                ("Top Red Flags", "What are the top red flags in this claim that an investigator should focus on?"),
                ("Draft Denial Letter", "Draft a professional claim denial letter based on these findings"),
            ],
        )
