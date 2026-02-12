"""Shared hero header component."""

import streamlit as st


def render_hero(title="FraudLens", highlight="AI", tagline="See Through Every Claim", subtitle="Powered by NVIDIA NIM"):
    """Render the hero title bar. Customizable per vertical."""
    st.markdown(f"""
    <div style="padding: 8px 0 8px 0;">
        <p style="font-size: 32px; font-weight: 800; letter-spacing: -1.5px; color: #f0f0f5; margin: 0;">
            {title} <span style="color: #76B900;">{highlight}</span>
        </p>
        <p style="font-size: 14px; color: #9a9ab0; margin: 4px 0 0 0; font-weight: 400;">
            {tagline} <span style="color: #55556a;">&mdash;</span> <span style="font-size: 11px; color: #55556a;">{subtitle}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
