"""
AI Content Detection vertical UI.
"""

import json
import streamlit as st
from pathlib import Path

from ui.components.hero import render_hero
from ui.components.steps import render_steps
from ui.components.score_card import render_score_card
from ui.components.chat import render_chat
from ui.components.results_common import (
    risk_color,
    run_async, create_gauge,
    render_skeleton_loader, render_error,
)


AGENT_CARDS_HTML = """
<div style="padding: 0;">
    <div class="agent-card"><span class="agent-icon">📝</span><span class="agent-name">Text Gen Detector</span><span class="agent-status">NIM LLM</span></div>
    <div class="agent-card"><span class="agent-icon">🎨</span><span class="agent-name">Image Gen Detector</span><span class="agent-status">NIM LLM</span></div>
    <div class="agent-card"><span class="agent-icon">🔬</span><span class="agent-name">Metadata Analyzer</span><span class="agent-status">NIM LLM</span></div>
</div>
"""


def render_sidebar_settings():
    with st.expander("AI Agents", expanded=False):
        st.markdown(AGENT_CARDS_HTML, unsafe_allow_html=True)


def render_results(r):
    text_result = r.text_analysis or {}
    image_result = r.image_analysis or {}
    meta_result = r.metadata_analysis or {}

    text_prob = text_result.get("ai_probability", 0) * 100
    img_prob = image_result.get("overall_ai_probability", 0) * 100
    text_class = text_result.get("classification", "N/A")
    confidence = 0.8  # Simulated

    render_score_card(
        score=r.risk_score,
        risk_level=r.risk_level,
        stats={
            "Text AI": (f"{text_prob:.0f}%", risk_color('high') if text_prob > 60 else risk_color('low')),
            "Image AI": (f"{img_prob:.0f}%", risk_color('high') if img_prob > 60 else risk_color('low')),
            "Type": (r.content_type.title(), None),
            "Images": (str(len(r.image_paths)), None),
        },
        confidence=confidence,
        recommendation=r.recommendation,
        score_label="AI Score",
    )

    col_g, col_info = st.columns([1, 2])
    with col_g:
        st.plotly_chart(create_gauge(r.risk_score, r.risk_level), use_container_width=True)
    with col_info:
        # Classification card
        overall_class = "Likely AI" if r.ai_probability > 0.7 else "Possibly AI" if r.ai_probability > 0.4 else "Likely Human"
        class_color = "#ff4d4f" if r.ai_probability > 0.7 else "#ff9f43" if r.ai_probability > 0.4 else "#51cf66"
        st.markdown(f"""
        <div class="dark-card" style="text-align: center; padding: 24px;">
            <div style="font-size: 28px; font-weight: 800; color: {class_color}; letter-spacing: -1px;">{overall_class}</div>
            <div style="font-size: 9px; font-weight: 600; color: #55556a; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px;">Overall Classification</div>
            <div style="font-size: 14px; color: #b0b0c0; margin-top: 8px;">{r.ai_probability:.0%} AI probability</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs
    st.markdown("---")
    tabs = ["Overview"]
    if r.raw_text:
        tabs.append("Text Analysis")
    if r.image_paths:
        tabs.append("Image Analysis")
    if meta_result:
        tabs.append("Metadata")

    tab_list = st.tabs(tabs)
    tab_idx = 0

    # Overview tab
    with tab_list[tab_idx]:
        st.markdown(f"""
        <div style="font-size: 13px; color: #b0b0c0; line-height: 1.7; padding: 16px; background: var(--bg-elevated); border-radius: var(--radius-md); border: 1px solid var(--border);">
            <strong>Content Type:</strong> {r.content_type.title()}<br/>
            <strong>AI Probability:</strong> {r.ai_probability:.0%}<br/>
            <strong>Risk Level:</strong> {r.risk_level.upper()}<br/>
            <strong>Recommendation:</strong> {r.recommendation}
        </div>
        """, unsafe_allow_html=True)
    tab_idx += 1

    # Text Analysis tab
    if r.raw_text and tab_idx < len(tab_list):
        with tab_list[tab_idx]:
            classification = text_result.get("classification", "uncertain")
            indicators = text_result.get("indicators", [])

            cc = "#ff4d4f" if "ai" in classification else "#ff9f43" if "possibly" in classification else "#51cf66"
            st.markdown(f"""
            <div style="display: flex; gap: 12px; margin-bottom: 12px;">
                <div class="stat-card" style="flex: 1;"><div class="stat-number" style="color: {cc};">{text_prob:.0f}%</div><div class="stat-label-custom">AI Probability</div></div>
                <div class="stat-card" style="flex: 1;"><div class="stat-number" style="font-size: 16px; color: {cc};">{classification.replace('_', ' ').title()}</div><div class="stat-label-custom">Classification</div></div>
                <div class="stat-card" style="flex: 1;"><div class="stat-number">{len(indicators)}</div><div class="stat-label-custom">Indicators</div></div>
            </div>
            """, unsafe_allow_html=True)

            if text_result.get("summary"):
                st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{text_result["summary"]}</div>', unsafe_allow_html=True)

            for ind in indicators:
                sev = ind.get("severity", "medium")
                desc = ind.get("description", "N/A")
                typ = ind.get("type", "N/A")
                conf = ind.get("confidence", 0)
                st.markdown(f"""
                <div class="finding-item">
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                        <span class="severity-pill severity-{sev}">{sev}</span>
                        <span class="finding-title">{desc}</span>
                    </div>
                    <div class="finding-detail">{typ.replace('_', ' ').title()} &bull; {conf:.0%} confidence</div>
                </div>
                """, unsafe_allow_html=True)

            # Per-paragraph breakdown
            para_analysis = text_result.get("paragraph_analysis", [])
            if para_analysis:
                st.markdown('<div style="font-size: 11px; font-weight: 600; color: #7a7a90; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 16px; margin-bottom: 8px;">Per-Paragraph Breakdown</div>', unsafe_allow_html=True)
                for pa in para_analysis[:10]:
                    p_num = pa.get("paragraph_num", "?")
                    p_prob = pa.get("ai_probability", 0) * 100
                    p_class = pa.get("classification", "uncertain")
                    pc = "#ff4d4f" if p_prob > 70 else "#ff9f43" if p_prob > 40 else "#51cf66"
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 12px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); margin-bottom: 4px;">
                        <span style="font-size: 10px; color: #55556a; min-width: 20px;">P{p_num}</span>
                        <div style="flex: 1; background: rgba(255,255,255,0.05); border-radius: 3px; height: 4px; overflow: hidden;">
                            <div style="height: 100%; width: {p_prob}%; background: {pc}; border-radius: 3px;"></div>
                        </div>
                        <span style="font-size: 11px; font-weight: 700; color: {pc}; min-width: 35px; text-align: right;">{p_prob:.0f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
        tab_idx += 1

    # Image Analysis tab
    if r.image_paths and tab_idx < len(tab_list):
        with tab_list[tab_idx]:
            individual = image_result.get("individual_results", [])
            if individual:
                for img_res in individual:
                    fname = img_res.get("file", "unknown")
                    ai_p = img_res.get("ai_probability", 0) * 100
                    cls = img_res.get("classification", "uncertain")
                    ic = "#ff4d4f" if ai_p > 70 else "#ff9f43" if ai_p > 40 else "#51cf66"
                    st.markdown(f"""
                    <div class="finding-item">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                            <span class="finding-title">{fname}</span>
                            <span style="font-size: 12px; font-weight: 700; color: {ic};">{ai_p:.0f}% AI</span>
                        </div>
                        <div class="finding-detail">{cls.replace('_', ' ').title()}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size: 12px; color: #b0b0c0;">{image_result.get("summary", "No detailed results")}</div>', unsafe_allow_html=True)
        tab_idx += 1

    # Metadata tab
    if meta_result and tab_idx < len(tab_list):
        with tab_list[tab_idx]:
            if meta_result.get("summary"):
                st.markdown(f'<div style="font-size: 12px; color: #b0b0c0; margin-bottom: 12px; line-height: 1.5;">{meta_result["summary"]}</div>', unsafe_allow_html=True)
            flags = meta_result.get("flags", [])
            for flag in flags:
                sev = flag.get("severity", "medium")
                desc = flag.get("description", "N/A")
                st.markdown(f"""
                <div class="finding-item">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span class="severity-pill severity-{sev}">{sev}</span>
                        <span class="finding-title">{desc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def _build_chat_context(r):
    text_result = r.text_analysis or {}
    image_result = r.image_analysis or {}
    return f"""CONTENT TYPE: {r.content_type}
AI PROBABILITY: {r.ai_probability:.0%}
RISK SCORE: {r.risk_score}/100 ({r.risk_level})
RECOMMENDATION: {r.recommendation}

TEXT ANALYSIS: Classification={text_result.get('classification', 'N/A')}, AI Probability={text_result.get('ai_probability', 0):.0%}
- Indicators: {len(text_result.get('indicators', []))}
- Summary: {text_result.get('summary', 'N/A')[:500]}

IMAGE ANALYSIS: AI Probability={image_result.get('overall_ai_probability', 0):.0%}
- Images analyzed: {image_result.get('images_analyzed', 0)}
- Summary: {image_result.get('summary', 'N/A')[:500]}

TEXT (first 1000 chars):
{r.raw_text[:1000] if r.raw_text else 'No text provided'}"""


def render():
    """Main entry point for the AI content detection vertical."""
    render_hero("ContentScan", "AI", "Detect What's Real")

    st.markdown("""
    <div style="font-size: 12px; color: #7a7a90; margin-bottom: 12px;">
        Upload text or images to check for AI-generated content.
    </div>
    """, unsafe_allow_html=True)

    # Text input
    text_input = st.text_area(
        "Paste text to analyze",
        height=120,
        placeholder="Paste text content here to check for AI generation...",
        key="ai_text_input",
    )

    # Image upload
    uploaded_images = st.file_uploader(
        "Upload images to check",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        accept_multiple_files=True,
        key="ai_image_upload",
    )

    has_input = bool(text_input and text_input.strip()) or bool(uploaded_images)
    has_result = "ai_content_result" in st.session_state
    step = 3 if has_result else (2 if has_input else 1)
    render_steps(step, labels=["Upload", "Scan", "Report"])

    image_paths = []
    if uploaded_images:
        for uf in uploaded_images:
            temp_path = Path(f"/tmp/{uf.name}")
            with open(temp_path, "wb") as f:
                f.write(uf.getbuffer())
            image_paths.append(str(temp_path))

    if has_input:
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("Scan Content", type="primary", use_container_width=True)

        if analyze_clicked:
            progress = st.empty()
            progress.markdown(render_skeleton_loader("Scanning content...", "3 AI agents detecting generation patterns"), unsafe_allow_html=True)

            try:
                import core.nim_client as _nc
                _nc._nim_client = None
                from ai_detect import ContentScanAI
                scanner = ContentScanAI()
                result = run_async(scanner.analyze(
                    text=text_input.strip() if text_input else "",
                    image_paths=image_paths,
                ))
                st.session_state.ai_content_result = result
                st.session_state.ai_content_chat = []
                progress.empty()
            except Exception as e:
                progress.empty()
                st.markdown(render_error(str(e)), unsafe_allow_html=True)
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    if "ai_content_result" in st.session_state:
        r = st.session_state.ai_content_result
        st.markdown('<div style="height: 1px; background: var(--border); margin: 12px 0;"></div>', unsafe_allow_html=True)
        render_results(r)
        render_chat(
            context_text=_build_chat_context(r),
            session_key="ai_content_chat",
            persona="expert AI content detection analyst",
            vertical_name="ContentScan AI",
            avatar="🤖",
            suggestions=[
                ("Explain Score", "How was the AI detection score calculated?"),
                ("Which Parts?", "Which specific parts of the content appear most likely to be AI-generated?"),
                ("Confidence", "How confident is this analysis and what could improve accuracy?"),
            ],
        )
