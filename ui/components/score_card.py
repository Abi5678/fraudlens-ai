"""Shared score card component — the hero card with gauge, stats, and recommendation."""

import streamlit as st
from ui.components.results_common import risk_color, risk_icon


def render_score_card(score, risk_level, stats: dict, confidence: float,
                      recommendation: str, score_label="Fraud Score"):
    """Render the main score hero card.

    Args:
        score: Numeric score 0-100.
        risk_level: One of 'critical', 'high', 'medium', 'low'.
        stats: Dict of stat_name -> (value_str, color_override_or_None).
        confidence: 0.0 - 1.0.
        recommendation: Recommendation text.
        score_label: Label under the score number.
    """
    rc = risk_color(risk_level)
    ri = risk_icon(risk_level)

    stat_cards = ""
    for label, (val, clr) in stats.items():
        c = clr or "#f0f0f5"
        stat_cards += f"""
        <div class="stat-card">
            <div class="stat-number" style="color: {c};">{val}</div>
            <div class="stat-label-custom">{label}</div>
        </div>"""

    st.markdown(f"""
    <div class="dark-card-glow" style="padding: 28px;">
        <div style="display: flex; align-items: flex-start; gap: 32px; flex-wrap: wrap;">
            <div style="text-align: center; min-width: 140px;">
                <div class="score-value" style="color: {rc};">{score:.0f}</div>
                <div class="score-label">{score_label}</div>
                <div style="margin-top: 10px;"><span class="risk-badge risk-badge-{risk_level}">{ri} {risk_level.upper()}</span></div>
            </div>
            <div style="flex: 1; min-width: 280px;">
                <div class="stat-grid">{stat_cards}</div>
                <div style="margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 9px; font-weight: 600; color: #55556a; text-transform: uppercase; letter-spacing: 1px;">Confidence</span>
                        <span style="font-size: 12px; font-weight: 700; color: #f0f0f5;">{confidence:.0%}</span>
                    </div>
                    <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width: {confidence * 100:.0f}%;"></div></div>
                </div>
                <div style="margin-top: 12px; padding: 12px 14px; background: rgba(118,185,0,0.06); border: 1px solid rgba(118,185,0,0.15); border-radius: 10px;">
                    <div style="font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #76B900; margin-bottom: 4px;">Recommendation</div>
                    <div style="font-size: 13px; font-weight: 500; color: #f0f0f5;">{recommendation}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
