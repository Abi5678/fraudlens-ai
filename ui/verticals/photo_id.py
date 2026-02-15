"""
Photo ID Verification vertical UI.
"""

import json
import streamlit as st
from pathlib import Path

from ui.components.hero import render_hero
from ui.components.steps import render_steps
from ui.components.score_card import render_score_card
from ui.components.chat import render_chat
from ui.components.results_common import (
    risk_color, tooltip,
    run_async, create_gauge,
    render_skeleton_loader, render_error,
)


AGENT_CARDS_HTML = """
<div style="padding: 0;">
    <div class="agent-card"><span class="agent-icon">📄</span><span class="agent-name">Document / OCR</span><span class="agent-status">NeMo OCR + Nano VL</span></div>
    <div class="agent-card"><span class="agent-icon">🎭</span><span class="agent-name">Deepfake + AI-Gen</span><span class="agent-status">NIM Nemotron 4</span></div>
    <div class="agent-card"><span class="agent-icon">📐</span><span class="agent-name">Template Matcher</span><span class="agent-status">NIM Llama 3.3</span></div>
    <div class="agent-card"><span class="agent-icon">🔬</span><span class="agent-name">Metadata Analyzer</span><span class="agent-status">NIM Llama 3.3</span></div>
    <div class="agent-card"><span class="agent-icon">🪪</span><span class="agent-name">ID Plausibility</span><span class="agent-status">Rules + NIM Vision</span></div>
    <div class="agent-card"><span class="agent-icon">👤</span><span class="agent-name">Face Verification</span><span class="agent-status">Nemotron Nano VL</span></div>
    <div class="agent-card"><span class="agent-icon">📊</span><span class="agent-name">Risk Scorer</span><span class="agent-status">Weighted Ensemble</span></div>
</div>
"""


def render_sidebar_settings():
    with st.expander("AI Agents", expanded=False):
        st.markdown(AGENT_CARDS_HTML, unsafe_allow_html=True)


def _render_flags(flags, title="issues"):
    if not flags:
        st.markdown(f"""
        <div class="clean-scan">
            <div style="font-size: 32px; margin-bottom: 8px;">&#10003;</div>
            <div style="font-size: 15px; font-weight: 700; color: #51cf66; margin-bottom: 4px;">Clean Scan</div>
            <div style="font-size: 12px; color: #7a7a90;">No {title} detected</div>
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
    confidence = r.scoring_details.get("confidence", 0) if r.scoring_details else 0
    df = r.deepfake_analysis or {}
    tmpl = r.template_analysis or {}
    meta = r.metadata_analysis or {}
    consistency = getattr(r, "consistency_analysis", None) or {}

    deepfake_score = df.get("manipulation_score", 0)
    template_score = tmpl.get("template_match_score", 0)
    meta_score = meta.get("risk_score", 0)
    consistency_score = consistency.get("risk_score", 0)

    render_score_card(
        score=r.authenticity_score,
        risk_level=r.risk_level,
        stats={
            "Deepfake": (f"{deepfake_score:.0f}", risk_color('high') if deepfake_score > 50 else risk_color('low')),
            "Template": (f"{template_score:.0f}%", risk_color('low') if template_score >= 70 else risk_color('high')),
            "Metadata": (f"{meta_score:.0f}", risk_color('high') if meta_score > 40 else risk_color('low')),
            "Plausibility": (f"{consistency_score:.0f}", risk_color('high') if consistency_score > 40 else risk_color('low')),
            "Images": (str(len(r.image_paths)), None),
        },
        confidence=confidence,
        recommendation=r.recommendation,
        score_label="Risk Score",
    )

    # Gauge
    col_g, col_info = st.columns([1, 2])
    with col_g:
        st.plotly_chart(create_gauge(r.authenticity_score, r.risk_level), use_container_width=True)
    with col_info:
        # Show uploaded images
        if r.image_paths:
            n_cols = min(3, len(r.image_paths))
            img_cols = st.columns(n_cols)
            for i, img_path in enumerate(r.image_paths[:3]):
                with img_cols[i % n_cols]:
                    try:
                        st.image(img_path, use_container_width=True)
                    except Exception:
                        st.markdown(f'<div style="background: var(--bg-elevated); padding: 20px; border-radius: 8px; text-align: center; font-size: 10px; color: #55556a;">Image {i+1}</div>', unsafe_allow_html=True)

    # Tabs
    fv = getattr(r, "face_verification", None) or {}
    tab_names = ["Report", "Face Analysis", "Template Match", "Metadata", "ID Plausibility"]
    if fv.get("performed"):
        tab_names.append("Face Match")
    st.markdown("---")
    tabs = st.tabs(tab_names)
    tab_report, tab_face, tab_template, tab_meta, tab_plausibility = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
    tab_face_match = tabs[5] if len(tabs) > 5 else None

    with tab_report:
        if r.narrative:
            st.markdown(f'<div class="narrative-text">{r.narrative}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 30px; color: #55556a; font-size: 12px;">No report generated.</div>', unsafe_allow_html=True)

    with tab_face:
        if df.get("status") == "success":
            sc = "#ff4d4f" if deepfake_score >= 60 else "#ff9f43" if deepfake_score >= 35 else "#51cf66"
            label = "HIGH RISK" if deepfake_score >= 60 else "MODERATE" if deepfake_score >= 35 else "LOW RISK"
            ai_gen = df.get("ai_generated_detected", False)
            ai_gen_score = df.get("ai_generated_score", 0)
            ai_gen_badge = ""
            if ai_gen or ai_gen_score >= 50:
                ai_gen_badge = f"""
                <div style="margin-top: 12px; padding: 10px 14px; background: rgba(255,77,79,0.12); border: 1px solid rgba(255,77,79,0.35); border-radius: 8px;">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #ff4d4f;">AI-Generated ID risk</div>
                    <div style="font-size: 13px; color: #f0f0f5;">Score: {ai_gen_score:.0f}/100 — {'Detected' if ai_gen else 'Possible synthetic portrait or document'}</div>
                </div>
                """
            st.markdown(f"""
            <div class="dark-card" style="text-align: center; padding: 20px;">
                <div style="font-size: 42px; font-weight: 800; color: {sc}; letter-spacing: -2px;">{deepfake_score:.0f}</div>
                <div style="font-size: 9px; font-weight: 600; color: #55556a; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px;">Manipulation / AI-generated score</div>
                <div style="margin-top: 8px;"><span style="padding: 3px 10px; border-radius: 20px; font-size: 10px; font-weight: 700; background: {'rgba(255,77,79,0.12)' if deepfake_score >= 60 else 'rgba(255,159,67,0.12)' if deepfake_score >= 35 else 'rgba(81,207,102,0.12)'}; color: {sc};">{label}</span></div>
                {ai_gen_badge}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 30px; color: #55556a; font-size: 12px;">No face analysis available.</div>', unsafe_allow_html=True)

    with tab_template:
        doc_type = tmpl.get("document_type", "unknown")
        jurisdiction = tmpl.get("issuing_jurisdiction", "unknown")
        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div class="stat-card" style="flex: 1;"><div class="stat-number" style="font-size: 16px;">{doc_type.replace('_', ' ').title()}</div><div class="stat-label-custom">Document Type</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number" style="font-size: 16px;">{jurisdiction}</div><div class="stat-label-custom">Jurisdiction</div></div>
            <div class="stat-card" style="flex: 1;"><div class="stat-number">{template_score}%</div><div class="stat-label-custom">Template Match</div></div>
        </div>
        """, unsafe_allow_html=True)
        if tmpl.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{tmpl["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(tmpl.get("flags", []), "template issues")

    with tab_meta:
        if meta.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{meta["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(meta.get("flags", []), "metadata issues")

    with tab_plausibility:
        st.markdown("""
        <div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">
        Rule-based checks: placeholder ID numbers, expired documents, generic dates, issue vs DOB, address red flags, and physical description vs photo.
        </div>
        """, unsafe_allow_html=True)
        if consistency.get("summary"):
            st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{consistency["summary"]}</div>', unsafe_allow_html=True)
        _render_flags(consistency.get("flags", []), "plausibility issues")
        if consistency.get("id_number_checked"):
            st.caption(f"ID number checked: {consistency.get('id_number_checked')}")
        if consistency.get("dates_checked"):
            dc = consistency["dates_checked"]
            st.caption(f"Dates: expiry={dc.get('expiry')}, DOB={dc.get('dob')}, issue={dc.get('issue')}")

    if tab_face_match and fv.get("performed"):
        with tab_face_match:
            same = fv.get("same_person", False)
            conf = fv.get("confidence", 0)
            color = "#51cf66" if same else "#ff4d4f"
            st.markdown(f"""
            <div class="dark-card" style="text-align: center; padding: 20px;">
                <div style="font-size: 28px; font-weight: 800; color: {color};">{"MATCH" if same else "NO MATCH"}</div>
                <div style="font-size: 10px; color: #55556a; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px;">Face verification (Nemotron Nano VL)</div>
                <div style="margin-top: 8px; font-size: 14px; color: #b0b0c0;">Confidence: {conf:.0f}%</div>
                <div style="margin-top: 12px; font-size: 12px; color: #7a7a90; text-align: left;">{fv.get("explanation", "")[:400]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Compare ID portrait with second uploaded image (e.g. selfie).")


def _build_chat_context(r):
    df = r.deepfake_analysis or {}
    tmpl = r.template_analysis or {}
    meta = r.metadata_analysis or {}
    consistency = getattr(r, "consistency_analysis", None) or {}
    fv = getattr(r, "face_verification", None) or {}
    fv_str = f"Same person: {fv.get('same_person')} ({fv.get('confidence', 0):.0f}%)" if fv.get("performed") else "Not performed"
    return f"""DOCUMENT DATA: {json.dumps(r.document_data, indent=2, default=str)[:1500]}

RISK SCORE: {r.authenticity_score}/100 ({r.risk_level})
RECOMMENDATION: {r.recommendation}

DEEPFAKE ANALYSIS: Score={df.get('manipulation_score', 'N/A')}, Status={df.get('status', 'N/A')}
TEMPLATE ANALYSIS: Type={tmpl.get('document_type', 'N/A')}, Match={tmpl.get('template_match_score', 'N/A')}%, Flags={len(tmpl.get('flags', []))}
METADATA ANALYSIS: Risk={meta.get('risk_score', 'N/A')}, Tampering={meta.get('tampering_detected', 'N/A')}
ID PLAUSIBILITY: Risk={consistency.get('risk_score', 'N/A')}, Flags={len(consistency.get('flags', []))}
FACE VERIFICATION: {fv_str}

NARRATIVE: {r.narrative[:1000] if r.narrative else 'N/A'}"""


def render():
    """Main entry point for the photo ID verification vertical."""
    render_hero("IDVerify", "AI", "Authenticate Every Identity")

    st.markdown(
        """
        <div style="font-size: 12px; color: #7a7a90; margin-top: -6px; margin-bottom: 10px; line-height: 1.5;">
        For stronger checks of <b>security features</b>, upload multiple shots:
        <b>(1)</b> straight-on, <b>(2)</b> angled/tilted (to reveal holograms/overlays),
        <b>(3)</b> close-up/macro (microprint/fine-line patterns),
        <b>(4)</b> UV light photo (UV elements). Raised text requires an angled-light close-up.
        </div>
        """,
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    sample_id_path = project_root / "demo_assets" / "sample_id_placeholder.png"

    input_mode = st.radio(
        "Input",
        options=["Upload images", "Use sample ID"],
        horizontal=True,
        key="photoid_input_mode",
        label_visibility="collapsed",
    )

    uploaded_files = None
    if input_mode == "Upload images":
        uploaded_files = st.file_uploader(
            "Drop photo ID images (PNG, JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            accept_multiple_files=True,
            key="photoid_upload",
        )

    use_sample = input_mode == "Use sample ID" and sample_id_path.exists()

    has_result = "photoid_result" in st.session_state
    has_input = bool(uploaded_files) or use_sample
    step = 3 if has_result else (2 if has_input else 1)
    render_steps(step, labels=["Upload", "Scan", "Verify"])

    image_paths = []
    if uploaded_files:
        for uf in uploaded_files:
            temp_path = Path(f"/tmp/{uf.name}")
            with open(temp_path, "wb") as f:
                f.write(uf.getbuffer())
            image_paths.append(str(temp_path))
    elif use_sample:
        image_paths = [str(sample_id_path)]
        st.caption("Using demo sample ID image. Not a real document.")

    if image_paths:
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("Verify ID", type="primary", use_container_width=True)

        if analyze_clicked:
            progress = st.empty()
            progress.markdown(render_skeleton_loader("Scanning ID...", "4 AI agents analyzing document"), unsafe_allow_html=True)

            try:
                import core.nim_client as _nc
                _nc._nim_client = None
                from id_verify import IDVerifyAI
                verifier = IDVerifyAI()
                result = run_async(verifier.analyze(image_paths))
                st.session_state.photoid_result = result
                st.session_state.photoid_chat = []
                progress.empty()
            except Exception as e:
                progress.empty()
                st.markdown(render_error(str(e)), unsafe_allow_html=True)
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    if "photoid_result" in st.session_state:
        r = st.session_state.photoid_result
        st.markdown('<div style="height: 1px; background: var(--border); margin: 12px 0;"></div>', unsafe_allow_html=True)
        render_results(r)
        render_chat(
            context_text=_build_chat_context(r),
            session_key="photoid_chat",
            persona="expert document forensics analyst",
            vertical_name="IDVerify AI",
            avatar="🪪",
            suggestions=[
                ("Face Analysis", "What did the deepfake detection find about the face photo?"),
                ("Document Validity", "Is this ID document format valid for the claimed jurisdiction?"),
                ("Tampering Signs", "Were there any signs of metadata tampering or image editing?"),
            ],
        )
