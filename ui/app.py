"""
FraudLens AI - Streamlit Dashboard
Professional-grade dark UI with NVIDIA Green accents
"""

import streamlit as st
import asyncio
import json
import os
import math
import plotly.graph_objects as go
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design System CSS
# ---------------------------------------------------------------------------
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161e;
    --bg-card-hover: #1c1c26;
    --bg-elevated: #1e1e28;
    --text-primary: #f0f0f5;
    --text-secondary: #b0b0c0;
    --text-tertiary: #7a7a90;
    --text-muted: #55556a;
    --nvidia-green: #76B900;
    --nvidia-green-light: #8fd400;
    --nvidia-green-glow: rgba(118, 185, 0, 0.25);
    --nvidia-green-subtle: rgba(118, 185, 0, 0.08);
    --blue: #4a9eff;
    --red: #ff4d4f;
    --red-glow: rgba(255, 77, 79, 0.15);
    --orange: #ff9f43;
    --orange-glow: rgba(255, 159, 67, 0.15);
    --yellow: #ffd43b;
    --yellow-glow: rgba(255, 212, 59, 0.15);
    --green: #51cf66;
    --green-glow: rgba(81, 207, 102, 0.15);
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --border-active: rgba(118, 185, 0, 0.4);
    --shadow-glow: 0 0 30px rgba(118, 185, 0, 0.1);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-pill: 980px;
}

html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
/* Preserve Material Symbols font for Streamlit's icon spans (expander arrows, checkboxes, etc.) */
[data-testid="stIconMaterial"],
span[translate="no"][class*="st-emotion-cache"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg-primary) !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 6rem !important; max-width: 1400px !important; }
.block-container [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 0 !important; }
[data-testid="stFileUploader"] { margin-bottom: -8px !important; }
.main hr { border-color: var(--border) !important; margin: 8px 0 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border) !important; width: 280px !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; padding-left: 1.2rem !important; padding-right: 1.2rem !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p { color: var(--text-secondary) !important; font-size: 13px !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 { font-size: 10px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1.2px !important; color: var(--text-muted) !important; margin-bottom: 8px !important; margin-top: 2px !important; }
section[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 12px 0 !important; }
section[data-testid="stSidebar"] .stCheckbox { padding: 2px 0 !important; }
section[data-testid="stSidebar"] .stCheckbox label span { color: var(--text-secondary) !important; font-size: 12px !important; }

/* Typography */
h1 { font-weight: 800 !important; letter-spacing: -1.5px !important; color: var(--text-primary) !important; }
h2 { font-weight: 700 !important; color: var(--text-primary) !important; }
h3 { font-weight: 600 !important; color: var(--text-primary) !important; font-size: 17px !important; }

/* Buttons */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--nvidia-green) 0%, var(--nvidia-green-light) 100%) !important;
    color: #000 !important; border: none !important; border-radius: var(--radius-sm) !important;
    padding: 12px 32px !important; font-size: 14px !important; font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--nvidia-green-glow) !important;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 25px rgba(118, 185, 0, 0.4) !important; }
.stButton > button { border-radius: var(--radius-sm) !important; padding: 10px 20px !important; font-size: 13px !important; font-weight: 500 !important; border: 1px solid var(--border) !important; background: var(--bg-card) !important; color: var(--text-primary) !important; }
.stButton > button:hover { border-color: var(--nvidia-green) !important; color: var(--nvidia-green) !important; background: var(--nvidia-green-subtle) !important; }
.stDownloadButton > button { background: var(--bg-elevated) !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }

/* Inputs & Chat */
.stTextInput > div > div > input, .stTextArea > div > div > textarea { border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; background: var(--bg-card) !important; color: var(--text-primary) !important; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: var(--nvidia-green) !important; box-shadow: 0 0 0 3px var(--nvidia-green-glow) !important; }
.stTextInput > label, .stSelectbox > label, .stTextArea > label, .stCheckbox > label, .stFileUploader > label { font-size: 12px !important; font-weight: 600 !important; color: var(--text-tertiary) !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }

/* Chat input - native integration */
[data-testid="stChatInput"] { border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; background: var(--bg-card) !important; }
[data-testid="stChatInput"] * { color: var(--text-primary) !important; caret-color: var(--text-primary) !important; }
[data-testid="stChatInput"] div { background: transparent !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: var(--text-primary) !important; border: none !important; caret-color: var(--text-primary) !important; font-size: 14px !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stChatInput"] button { background: var(--nvidia-green) !important; color: #000 !important; border: none !important; border-radius: var(--radius-sm) !important; }
[data-testid="stChatInput"] button svg { color: #000 !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--nvidia-green) !important; box-shadow: 0 0 0 3px var(--nvidia-green-glow) !important; }
/* Ensure the sticky chat input container has a solid background */
[data-testid="stBottom"] { background: var(--bg-primary) !important; border-top: 1px solid var(--border) !important; padding-top: 8px !important; }
[data-testid="stBottom"] > div { background: var(--bg-primary) !important; }
/* Chat messages - seamless dark theme */
[data-testid="stChatMessage"] { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; padding: 12px 16px !important; margin-bottom: 8px !important; color: var(--text-primary) !important; }
[data-testid="stChatMessage"] p { color: #e0e0ea !important; font-size: 14px !important; line-height: 1.7 !important; }
[data-testid="stChatMessage"] ol, [data-testid="stChatMessage"] ul { color: #e0e0ea !important; }
[data-testid="stChatMessage"] li { color: #d0d0dc !important; font-size: 14px !important; line-height: 1.7 !important; }
[data-testid="stChatMessage"] strong { color: var(--text-primary) !important; }
[data-testid="stChatMessage"] code { color: var(--nvidia-green) !important; }
[data-testid="stChatMessage"] a { color: var(--blue) !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color: #e0e0ea !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color: #e0e0ea !important; }
/* User messages - green tint */
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) { background: rgba(118,185,0,0.04) !important; border-color: rgba(118,185,0,0.12) !important; }
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) p { color: var(--text-primary) !important; }
/* Assistant messages - subtle branding */
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) { background: var(--bg-elevated) !important; border-color: var(--border) !important; }

/* File uploader */
[data-testid="stFileUploader"] { border: 1px dashed rgba(255,255,255,0.1) !important; border-radius: var(--radius-md) !important; padding: 24px !important; background: var(--bg-card) !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--nvidia-green) !important; background: var(--nvidia-green-subtle) !important; }
[data-testid="stFileUploader"] p, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small { color: var(--text-tertiary) !important; }
[data-testid="stFileUploadDropzone"] { background: var(--bg-card) !important; border-color: rgba(255,255,255,0.08) !important; color: var(--text-tertiary) !important; }
[data-testid="stFileUploadDropzone"] span, [data-testid="stFileUploadDropzone"] small, [data-testid="stFileUploadDropzone"] div { color: var(--text-tertiary) !important; }
[data-testid="stFileUploadDropzone"] button { background: var(--bg-elevated) !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stFileUploadDropzone"] svg { fill: var(--text-muted) !important; stroke: var(--text-muted) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: var(--radius-sm) !important; padding: 4px !important; gap: 2px !important; border: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { border-radius: 6px !important; padding: 8px 16px !important; font-size: 13px !important; font-weight: 500 !important; color: var(--text-tertiary) !important; border: none !important; background: transparent !important; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,0.04) !important; color: var(--text-primary) !important; }
.stTabs [aria-selected="true"] { background: var(--nvidia-green) !important; color: #000 !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 16px !important; }

/* Code */
.stCode, pre, code { background: var(--bg-elevated) !important; color: var(--text-secondary) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }

/* Hide branding & deploy button */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(0, 0, 0, 0) !important; }
[data-testid="stStatusWidget"], .stDeployButton, [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
hr { border: none !important; height: 1px !important; background: var(--border) !important; margin: 20px 0 !important; }

/* Expanders - modern Streamlit (emotion-cache based) */
[data-testid="stExpander"] details { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; }
[data-testid="stExpander"] summary { background: var(--bg-card) !important; border-radius: var(--radius-md) !important; font-size: 13px !important; font-weight: 600 !important; color: var(--text-primary) !important; padding: 12px 16px !important; }
[data-testid="stExpander"] summary:hover { background: var(--bg-card-hover) !important; }
[data-testid="stExpander"] summary span[data-testid="stExpanderToggleIcon"] { color: var(--text-muted) !important; }
[data-testid="stExpander"] div[data-testid="stExpanderDetails"] { background: var(--bg-card) !important; border-top: 1px solid var(--border) !important; color: var(--text-secondary) !important; padding: 16px !important; }

/* Custom components */
.dark-card { background: var(--bg-card); border-radius: var(--radius-lg); padding: 24px; border: 1px solid var(--border); margin-bottom: 16px; }
.dark-card:hover { border-color: var(--border-hover); }
.dark-card-glow { background: var(--bg-card); border-radius: var(--radius-lg); padding: 24px; border: 1px solid var(--border-active); margin-bottom: 16px; box-shadow: var(--shadow-glow); }
.score-value { font-size: 64px; font-weight: 800; letter-spacing: -3px; line-height: 1; }
.score-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin-top: 6px; }
.risk-badge { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: var(--radius-pill); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.risk-badge-critical { background: var(--red-glow); color: var(--red); border: 1px solid rgba(255,77,79,0.3); }
.risk-badge-high { background: var(--orange-glow); color: var(--orange); border: 1px solid rgba(255,159,67,0.3); }
.risk-badge-medium { background: var(--yellow-glow); color: var(--yellow); border: 1px solid rgba(255,212,59,0.3); }
.risk-badge-low { background: var(--green-glow); color: var(--green); border: 1px solid rgba(81,207,102,0.3); }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.stat-card { background: var(--bg-elevated); border-radius: var(--radius-md); padding: 14px 8px; text-align: center; border: 1px solid var(--border); min-width: 0; }
.stat-number { font-size: 24px; font-weight: 700; color: var(--text-primary); letter-spacing: -1px; line-height: 1; margin-bottom: 4px; }
.stat-label-custom { font-size: 8px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.confidence-bar-bg { background: rgba(255,255,255,0.06); border-radius: var(--radius-pill); height: 5px; overflow: hidden; margin-top: 5px; }
.confidence-bar-fill { height: 100%; border-radius: var(--radius-pill); background: linear-gradient(90deg, var(--nvidia-green), var(--nvidia-green-light)); }
.severity-pill { display: inline-block; padding: 3px 10px; border-radius: var(--radius-pill); font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; }
.severity-critical { background: var(--red-glow); color: var(--red); }
.severity-high { background: var(--orange-glow); color: var(--orange); }
.severity-medium { background: var(--yellow-glow); color: var(--yellow); }
.severity-low { background: var(--green-glow); color: var(--green); }
.finding-item { background: var(--bg-elevated); border-radius: var(--radius-sm); padding: 12px 16px; margin-bottom: 6px; border: 1px solid var(--border); }
.finding-item:hover { border-color: var(--border-hover); background: var(--bg-card-hover); }
.finding-title { font-size: 13px; font-weight: 500; color: var(--text-primary); }
.finding-detail { font-size: 11px; color: var(--text-tertiary); }
.narrative-text { font-size: 13px; line-height: 1.7; color: var(--text-secondary); background: var(--bg-elevated); border-radius: var(--radius-md); padding: 20px; border: 1px solid var(--border); white-space: pre-wrap; }
.section-title { font-size: 16px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.3px; margin-bottom: 12px; margin-top: 2px; }
.tech-badge { display: inline-block; padding: 3px 8px; border-radius: var(--radius-pill); font-size: 9px; font-weight: 600; letter-spacing: 0.5px; margin: 2px; background: var(--nvidia-green-subtle); color: var(--nvidia-green); border: 1px solid rgba(118,185,0,0.15); }
.status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 4px; }
.status-online { background: var(--green); box-shadow: 0 0 6px var(--green-glow); }
.status-idle { background: var(--yellow); }
.agent-card { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-radius: var(--radius-sm); background: var(--bg-card); border: 1px solid var(--border); margin-bottom: 4px; }
.agent-icon { font-size: 14px; width: 20px; text-align: center; }
.agent-name { font-size: 11px; font-weight: 500; color: var(--text-primary); }
.agent-status { font-size: 9px; color: var(--text-muted); margin-left: auto; }
.summary-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); }
.summary-row:last-child { border-bottom: none; }
.summary-key { font-size: 12px; color: var(--text-tertiary); font-weight: 500; }
.summary-value { font-size: 13px; color: var(--text-primary); font-weight: 600; }
.info-tooltip { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; background: rgba(255,255,255,0.06); color: var(--text-muted); font-size: 10px; font-weight: 700; cursor: help; margin-left: 6px; vertical-align: middle; }
.info-tooltip .tooltip-text { visibility: hidden; opacity: 0; position: absolute; z-index: 999; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); width: 240px; padding: 10px 12px; background: #1e1e28; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; font-size: 11px; font-weight: 400; color: var(--text-secondary); line-height: 1.5; box-shadow: 0 8px 30px rgba(0,0,0,0.5); pointer-events: none; }
.info-tooltip:hover .tooltip-text { visibility: visible; opacity: 1; }
.info-tooltip .tooltip-text::after { content: ''; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #1e1e28 transparent transparent transparent; }

/* Loading */
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
.loading-bar { height: 4px; border-radius: 2px; background: linear-gradient(90deg, transparent, var(--nvidia-green), transparent); background-size: 200% 100%; animation: shimmer 2s ease-in-out infinite; }

/* Plotly */
.stPlotlyChart { border-radius: var(--radius-md) !important; overflow: hidden !important; }

/* Data monospace */
.data-mono { font-family: 'JetBrains Mono', 'SF Mono', 'Cascadia Code', 'Consolas', monospace !important; font-size: 12px !important; letter-spacing: -0.3px; }

/* Chat chip buttons */
.chip-row .stButton > button { background: var(--bg-elevated) !important; border: 1px solid rgba(118,185,0,0.2) !important; color: var(--nvidia-green) !important; font-size: 11px !important; padding: 6px 12px !important; border-radius: var(--radius-pill) !important; font-weight: 500 !important; }
.chip-row .stButton > button:hover { background: rgba(118,185,0,0.08) !important; border-color: var(--nvidia-green) !important; }

/* Clean scan card */
.clean-scan { text-align: center; padding: 40px 20px; background: var(--bg-card); border: 1px solid rgba(81,207,102,0.15); border-radius: var(--radius-lg); }
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_async(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def risk_color(level):
    return {"critical": "#ff4d4f", "high": "#ff9f43", "medium": "#ffd43b", "low": "#51cf66"}.get(level, "#7a7a90")


def risk_icon(level):
    return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(level, "⚪")


def tooltip(text):
    return f'<span class="info-tooltip">?<span class="tooltip-text">{text}</span></span>'


def _val(v, fallback="--"):
    """Return display value or fallback for empty strings / None."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return fallback
    return str(v)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def create_gauge(score, level):
    c = risk_color(level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={'font': {'size': 44, 'color': c, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(0,0,0,0)", 'tickfont': {'size': 1, 'color': 'rgba(0,0,0,0)'}, 'showticklabels': False},
            'bar': {'color': c, 'thickness': 0.25},
            'bgcolor': 'rgba(255,255,255,0.03)', 'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(81,207,102,0.06)'},
                {'range': [25, 50], 'color': 'rgba(255,212,59,0.06)'},
                {'range': [50, 75], 'color': 'rgba(255,159,67,0.06)'},
                {'range': [75, 100], 'color': 'rgba(255,77,79,0.06)'},
            ],
            'threshold': {'line': {'color': c, 'width': 3}, 'thickness': 0.8, 'value': score},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family': 'Inter', 'color': '#f0f0f5'})
    for lbl, xp, clr in [("Low", 0.13, "#51cf66"), ("Medium", 0.37, "#ffd43b"), ("High", 0.63, "#ff9f43"), ("Critical", 0.87, "#ff4d4f")]:
        fig.add_annotation(x=xp, y=-0.05, text=lbl, showarrow=False, font=dict(size=9, color=clr, family="Inter"), xref="paper", yref="paper")
    return fig


def create_risk_factors_chart(scoring_details):
    factors = scoring_details.get("risk_factors", [])
    if not factors:
        return None
    names = [f["name"] for f in factors]
    scores = [f["score"] for f in factors]
    colors = []
    for s in scores:
        if s >= 75: colors.append("rgba(255,77,79,0.85)")
        elif s >= 50: colors.append("rgba(255,159,67,0.85)")
        elif s >= 25: colors.append("rgba(255,212,59,0.85)")
        else: colors.append("rgba(81,207,102,0.85)")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=scores, orientation='h', marker_color=colors, marker_line_width=0, text=[f'{s:.0f}' for s in scores], textposition='outside', textfont={'size': 11, 'family': 'Inter', 'color': '#b0b0c0'}))
    fig.update_layout(height=max(160, len(factors) * 44), margin=dict(l=10, r=50, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis={'range': [0, 110], 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.03)', 'zeroline': False, 'showticklabels': False}, yaxis={'autorange': 'reversed', 'tickfont': {'size': 11, 'family': 'Inter', 'color': '#7a7a90'}}, showlegend=False, bargap=0.35)
    return fig


def create_network_graph(network_analysis):
    """Radial network graph showing claim at center with risk indicator nodes."""
    indicators = network_analysis.get("indicators", [])
    real = [i for i in indicators if i.get("name") != "_llm_summary"]
    if not real:
        return None

    n = len(real)
    cx, cy = 0.5, 0.5
    radius = 0.35

    # Central claim node
    node_x, node_y = [cx], [cy]
    node_text = ["Claim"]
    node_hover = [f"Claim<br>Network Risk: {network_analysis.get('network_risk_score', 0)}/100"]
    net_score = network_analysis.get("network_risk_score", 0)
    center_color = "#ff4d4f" if net_score >= 60 else "#ff9f43" if net_score >= 35 else "#51cf66"
    node_color = [center_color]
    node_size = [44]
    edge_x, edge_y = [], []

    for i, ind in enumerate(real):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        node_x.append(x)
        node_y.append(y)

        s = ind.get("score", 0)
        sc = "#ff4d4f" if s >= 60 else "#ff9f43" if s >= 35 else "#51cf66"
        short_name = ind.get("name", "").replace("_", " ").replace("RISK", "").replace("FLAGS", "").strip().title()
        node_text.append(short_name)
        node_hover.append(f"{ind.get('name','')}<br>Score: {s}/100<br>{ind.get('explanation','')[:80]}")
        node_color.append(sc)
        node_size.append(22 + s * 0.2)

        edge_x.extend([cx, x, None])
        edge_y.extend([cy, y, None])

    fig = go.Figure()
    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.5, color='rgba(255,255,255,0.08)'),
        hoverinfo='none', showlegend=False,
    ))
    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=1.5, color='rgba(255,255,255,0.12)')),
        text=node_text,
        textposition="bottom center",
        textfont=dict(size=10, color="#b0b0c0", family="Inter"),
        hovertext=node_hover, hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        height=320, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.15, 1.1]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="font-size: 30px; margin-bottom: 4px;">🛡️</div>
            <div style="font-size: 18px; font-weight: 800; color: #f0f0f5; letter-spacing: -0.5px;">
                FraudLens <span style="color: #76B900;">AI</span>
            </div>
            <div style="font-size: 10px; color: #55556a; margin-top: 3px; letter-spacing: 1px; text-transform: uppercase;">
                Powered by NVIDIA NIM
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### System Status")
        st.markdown("""
        <div style="padding: 2px 0 6px 0;">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                <span class="status-dot status-online"></span>
                <span style="font-size: 11px; color: #b0b0c0;">NIM API</span>
                <span style="margin-left: auto; font-size: 10px; color: #51cf66; font-weight: 600;">Connected</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span class="status-dot status-online"></span>
                <span style="font-size: 11px; color: #b0b0c0;">NVIDIA API Key</span>
                <span style="margin-left: auto; font-size: 10px; color: #51cf66; font-weight: 600;">Configured</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### Modules")
        include_network = st.checkbox("Network Analysis", True, help="Detect fraud rings via graph analytics")
        include_deepfake = st.checkbox("Deepfake Detection", True, help="Analyze images for AI manipulation")

        st.markdown("---")

        st.markdown("### AI Agents")
        st.markdown("""
        <div style="padding: 0;">
            <div class="agent-card"><span class="agent-icon">📄</span><span class="agent-name">Document Intelligence</span><span class="agent-status">Nemotron</span></div>
            <div class="agent-card"><span class="agent-icon">🔍</span><span class="agent-name">Inconsistency Detector</span><span class="agent-status">NIM LLM</span></div>
            <div class="agent-card"><span class="agent-icon">🎯</span><span class="agent-name">Pattern Matcher</span><span class="agent-status">NeMo RAG</span></div>
            <div class="agent-card"><span class="agent-icon">📊</span><span class="agent-name">Risk Scorer</span><span class="agent-status">Ensemble</span></div>
            <div class="agent-card"><span class="agent-icon">📝</span><span class="agent-name">Narrative Writer</span><span class="agent-status">NIM LLM</span></div>
            <div class="agent-card"><span class="agent-icon">🕸️</span><span class="agent-name">Network Analyzer</span><span class="agent-status">cuGraph</span></div>
            <div class="agent-card"><span class="agent-icon">🖼️</span><span class="agent-name">Deepfake Detector</span><span class="agent-status">TensorRT</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 3px; padding: 2px 0 6px 0;">
            <span class="tech-badge">NIM</span><span class="tech-badge">Nemotron</span><span class="tech-badge">NeMo</span>
            <span class="tech-badge">cuGraph</span><span class="tech-badge">Milvus</span><span class="tech-badge">TensorRT</span>
        </div>
        <div style="text-align: center; padding: 8px 0 0 0;">
            <div style="font-size: 10px; color: #3a3a4a;">NVIDIA GTC 2026</div>
        </div>
        """, unsafe_allow_html=True)

    return include_network, include_deepfake


# ---------------------------------------------------------------------------
# Document Details Verification
# ---------------------------------------------------------------------------

def render_document_details(r):
    """Show extracted claim details so user can verify against original document."""
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
# Deepfake Analysis Panel
# ---------------------------------------------------------------------------

def render_deepfake_results(r):
    """Show deepfake / image authenticity results."""
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
            # Prioritise actual photos (JPEG) over document scans (PNG)
            jpeg_imgs = [p for p in images if p.lower().endswith(('.jpg', '.jpeg'))]
            other_imgs = [p for p in images if not p.lower().endswith(('.jpg', '.jpeg'))]
            # Show vehicle photos first, then other evidence — cap at 12
            ordered_imgs = (jpeg_imgs + other_imgs)[:12]
            n_cols = min(4, max(1, len(ordered_imgs)))

            st.markdown(
                f'<div style="font-size: 11px; font-weight: 600; color: #7a7a90; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">'
                f'Extracted Images ({len(images)}{" — showing " + str(len(ordered_imgs)) if len(ordered_imgs) < len(images) else ""})'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Render rows of 4 columns
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
                # Shorten name: show extension icon + truncated name
                short = name[:18] + "…" if len(name) > 19 else name
                chips_html += f'<span style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 20px; font-size: 10px; color: #b0b0c0; white-space: nowrap;"><span style="overflow: hidden; text-overflow: ellipsis; max-width: 120px;">{short}</span><span style="font-weight: 700; color: {sc}; font-size: 11px;">{s:.0f}</span></span>\n'
            st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 6px;">{chips_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_results(r):
    rc = risk_color(r.risk_level)
    ri = risk_icon(r.risk_level)
    incon_count = len(r.inconsistencies.get("inconsistencies", []))
    pat_count = len(r.pattern_matches.get("matched_patterns", []))
    confidence = r.scoring_details.get("confidence", 0) if r.scoring_details else 0
    has_deepfake = bool(r.deepfake_analysis and r.deepfake_analysis.get("status") == "success")
    img_count = len(getattr(r, 'extracted_images', []) or [])

    # Score card
    st.markdown(f"""
    <div class="dark-card-glow" style="padding: 28px;">
        <div style="display: flex; align-items: flex-start; gap: 32px; flex-wrap: wrap;">
            <div style="text-align: center; min-width: 140px;">
                <div class="score-value" style="color: {rc};">{r.fraud_score:.0f}</div>
                <div class="score-label">Fraud Score</div>
                <div style="margin-top: 10px;"><span class="risk-badge risk-badge-{r.risk_level}">{ri} {r.risk_level.upper()}</span></div>
            </div>
            <div style="flex: 1; min-width: 280px;">
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-number" style="color: {risk_color('high') if incon_count > 2 else risk_color('low')};">{incon_count}</div>
                        <div class="stat-label-custom">Issues</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" style="color: {risk_color('high') if pat_count > 1 else risk_color('low')};">{pat_count}</div>
                        <div class="stat-label-custom">Patterns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" style="color: {'#ff4d4f' if r.fraud_ring_detected else '#51cf66'};">{"Yes" if r.fraud_ring_detected else "No"}</div>
                        <div class="stat-label-custom">Ring</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{img_count}</div>
                        <div class="stat-label-custom">Images</div>
                    </div>
                </div>
                <div style="margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 9px; font-weight: 600; color: #55556a; text-transform: uppercase; letter-spacing: 1px;">Confidence</span>
                        <span style="font-size: 12px; font-weight: 700; color: #f0f0f5;">{confidence:.0%}</span>
                    </div>
                    <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width: {confidence * 100:.0f}%;"></div></div>
                </div>
                <div style="margin-top: 12px; padding: 12px 14px; background: rgba(118,185,0,0.06); border: 1px solid rgba(118,185,0,0.15); border-radius: 10px;">
                    <div style="font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #76B900; margin-bottom: 4px;">Recommendation</div>
                    <div style="font-size: 13px; font-weight: 500; color: #f0f0f5;">{r.recommendation}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    # Document details verification
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
            # Network graph visualization
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

    # Export
    st.markdown("---")
    col_dl, col_view, _ = st.columns([1, 1, 2])
    with col_dl:
        st.download_button("Download Report (JSON)", json.dumps(r.to_dict(), indent=2, default=str), "fraudlens_report.json", mime="application/json")
    with col_view:
        if st.button("View Raw JSON", use_container_width=True):
            st.session_state.show_json = not st.session_state.get("show_json", False)
    if st.session_state.get("show_json", False):
        st.json(r.to_dict())


# ---------------------------------------------------------------------------
# Chat (native Streamlit integration)
# ---------------------------------------------------------------------------

def render_chat():
    """Claims analyst chat, using native st.chat_message for seamless integration."""
    st.markdown("""
    <div style="margin-top: 8px; padding: 20px 24px 12px 24px; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(118,185,0,0.1); border: 1px solid rgba(118,185,0,0.2); display: flex; align-items: center; justify-content: center; font-size: 16px;">💬</div>
            <div>
                <div style="font-size: 14px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.3px;">Claims Analyst Chat</div>
                <div style="font-size: 11px; color: var(--text-muted);">Ask follow-up questions about the analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Quick-action suggestion chips (shown only when no messages yet)
    if not st.session_state.chat_history:
        st.markdown('<div style="font-size: 10px; color: #55556a; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-bottom: 6px;">Suggested Questions</div>', unsafe_allow_html=True)
        suggestions = [
            ("Explain Risk Score", "Explain how the fraud score was calculated and what factors contributed most"),
            ("Summarize Findings", "Give me a brief summary of all findings in this analysis"),
            ("Top Red Flags", "What are the top red flags in this claim that an investigator should focus on?"),
            ("Draft Denial Letter", "Draft a professional claim denial letter based on these findings"),
        ]
        chip_cols = st.columns(4)
        for i, (label, prompt_text) in enumerate(suggestions):
            with chip_cols[i]:
                if st.button(label, key=f"chip_{i}", use_container_width=True):
                    st.session_state.chat_chip_prompt = prompt_text
                    st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about this claim...")
    # Handle chip-triggered prompts
    if "chat_chip_prompt" in st.session_state:
        user_input = st.session_state.pop("chat_chip_prompt")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result = st.session_state.get("result")
        context = ""
        if result:
            # Build clean summaries instead of truncated JSON
            r = result

            # Inconsistencies summary
            incon_list = r.inconsistencies.get("inconsistencies", [])
            if incon_list:
                incon_lines = []
                for inc in incon_list:
                    incon_lines.append(f"- [{inc.get('severity','?').upper()}] {inc.get('description','N/A')} (type: {inc.get('type','N/A')}, confidence: {inc.get('confidence',0):.0%})")
                incon_summary = "\n".join(incon_lines)
            else:
                incon_summary = "No inconsistencies detected."

            # Pattern matches summary
            patterns = r.pattern_matches.get("matched_patterns", [])
            if patterns:
                pat_lines = []
                for pat in patterns:
                    pat_lines.append(f"- [{pat.get('severity','?').upper()}] {pat.get('pattern_name','Unknown')} (category: {pat.get('category','N/A')}, similarity: {pat.get('similarity_score',0):.0%})")
                pat_summary = "\n".join(pat_lines)
            else:
                pat_summary = "No fraud patterns matched."
            pat_score = r.pattern_matches.get("pattern_risk_score", "N/A")

            # Network analysis summary
            net = r.network_analysis or {}
            ring = net.get("fraud_ring_detected", False)
            net_score = net.get("network_risk_score", "N/A")
            net_text = net.get("summary", net.get("analysis", ""))[:500]
            net_summary = f"Fraud ring detected: {'Yes' if ring else 'No'}, Risk score: {net_score}\n{net_text}"

            # Deepfake summary
            df = r.deepfake_analysis or {}
            df_det = df.get("deepfake_detected", False)
            df_score = df.get("confidence_score", "N/A")
            df_summary = f"Deepfake detected: {'Yes' if df_det else 'No'}, Confidence: {df_score}"

            context = f"""CLAIM DATA: {json.dumps(r.claim_data, indent=2, default=str)[:2000]}

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

        messages = [
            {"role": "system", "content": f"""You are FraudLens AI, an expert insurance fraud analyst. You have analyzed a claim and the adjuster is asking follow-up questions.

Answer concisely and professionally. Reference specific findings when relevant. If asked about something not in the data, say so.

ANALYSIS CONTEXT:
{context}"""},
        ]
        for msg in st.session_state.chat_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        with st.chat_message("assistant", avatar="🛡️"):
            try:
                import core.nim_client as _nc
                if _nc._nim_client is None:
                    _nc._nim_client = _nc.NIMClient()
                response = run_async(_nc._nim_client.chat(messages=messages, temperature=0.3, max_tokens=500))
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                err_msg = "Connection error. Please try again."
                st.markdown(err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_steps(step: int):
    """Render a 3-step process bar. step: 1=Upload, 2=Analyze, 3=Review."""
    steps = [("1", "Upload"), ("2", "Analyze"), ("3", "Review")]
    items = ""
    for i, (num, label) in enumerate(steps):
        idx = i + 1
        if idx < step:
            # completed
            dot = f'<div style="width: 24px; height: 24px; border-radius: 50%; background: var(--nvidia-green); display: flex; align-items: center; justify-content: center; font-size: 12px; color: #0a0a0f; font-weight: 700;">&#10003;</div>'
            lbl = f'<span style="font-size: 10px; color: var(--nvidia-green); font-weight: 600; margin-top: 4px;">{label}</span>'
        elif idx == step:
            # active
            dot = f'<div style="width: 24px; height: 24px; border-radius: 50%; background: var(--nvidia-green); display: flex; align-items: center; justify-content: center; font-size: 11px; color: #0a0a0f; font-weight: 700;">{num}</div>'
            lbl = f'<span style="font-size: 10px; color: #f0f0f5; font-weight: 600; margin-top: 4px;">{label}</span>'
        else:
            # upcoming
            dot = f'<div style="width: 24px; height: 24px; border-radius: 50%; background: transparent; border: 1.5px solid rgba(255,255,255,0.12); display: flex; align-items: center; justify-content: center; font-size: 11px; color: #55556a; font-weight: 600;">{num}</div>'
            lbl = f'<span style="font-size: 10px; color: #55556a; font-weight: 500; margin-top: 4px;">{label}</span>'

        items += f'<div style="display: flex; flex-direction: column; align-items: center; gap: 0; min-width: 56px;">{dot}{lbl}</div>'
        if i < len(steps) - 1:
            completed_line = idx < step
            line_color = "var(--nvidia-green)" if completed_line else "rgba(255,255,255,0.08)"
            items += f'<div style="flex: 1; height: 2px; background: {line_color}; margin: 0 4px; align-self: flex-start; margin-top: 12px; max-width: 80px;"></div>'

    st.markdown(f'<div style="display: flex; align-items: flex-start; justify-content: center; gap: 0; margin-bottom: 12px;">{items}</div>', unsafe_allow_html=True)


def main():
    include_network, include_deepfake = render_sidebar()

    # Hero
    st.markdown("""
    <div style="padding: 8px 0 8px 0;">
        <p style="font-size: 32px; font-weight: 800; letter-spacing: -1.5px; color: #f0f0f5; margin: 0;">
            Fraud <span style="color: #76B900;">Detection</span>
        </p>
        <p style="font-size: 13px; color: #7a7a90; margin: 4px 0 0 0;">
            Multi-agent AI analysis powered by NVIDIA NIM, NeMo Retriever, and cuGraph
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Drop a claim document (PDF, image, or JSON)",
        type=["pdf", "png", "jpg", "jpeg", "json"],
        label_visibility="collapsed",
    )

    # Process steps — computed after upload so we know the state
    has_result = hasattr(st.session_state, 'result')
    step = 3 if has_result else (2 if uploaded_file else 1)
    render_steps(step)

    input_path = None
    if uploaded_file:
        temp_path = Path(f"/tmp/{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_path = str(temp_path)

    # Analyze button
    if input_path:
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("Analyze Claim", type="primary", use_container_width=True)

        if analyze_clicked:
            progress = st.empty()
            skel_bg = "linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.06), rgba(255,255,255,0.02))"
            progress.markdown(f"""
            <div style="padding: 0;">
                <!-- Skeleton score hero -->
                <div class="dark-card-glow" style="padding: 28px;">
                    <div style="display: flex; align-items: flex-start; gap: 32px;">
                        <div style="text-align: center; min-width: 140px;">
                            <div style="width: 80px; height: 64px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 2s ease-in-out infinite; border-radius: 8px; margin: 0 auto;"></div>
                            <div style="width: 60px; height: 10px; background: rgba(255,255,255,0.04); border-radius: 4px; margin: 12px auto 0;"></div>
                            <div style="width: 70px; height: 22px; background: rgba(255,255,255,0.03); border-radius: 20px; margin: 10px auto 0;"></div>
                        </div>
                        <div style="flex: 1;">
                            <div class="stat-grid">
                                <div class="stat-card" style="height: 58px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 2s ease-in-out infinite;"></div>
                                <div class="stat-card" style="height: 58px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 1.8s ease-in-out infinite;"></div>
                                <div class="stat-card" style="height: 58px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 2.2s ease-in-out infinite;"></div>
                                <div class="stat-card" style="height: 58px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 1.6s ease-in-out infinite;"></div>
                            </div>
                            <div style="height: 5px; background: rgba(255,255,255,0.04); border-radius: 3px; margin-top: 16px;"></div>
                            <div style="height: 48px; background: rgba(118,185,0,0.04); border: 1px solid rgba(118,185,0,0.1); border-radius: 10px; margin-top: 12px;"></div>
                        </div>
                    </div>
                </div>
                <!-- Skeleton chart row -->
                <div style="display: flex; gap: 16px; margin-top: 4px;">
                    <div style="flex: 1; height: 160px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 2.4s ease-in-out infinite; border-radius: 12px; border: 1px solid rgba(255,255,255,0.04);"></div>
                    <div style="flex: 2; height: 160px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 2.1s ease-in-out infinite; border-radius: 12px; border: 1px solid rgba(255,255,255,0.04);"></div>
                </div>
                <!-- Status -->
                <div style="text-align: center; margin-top: 20px;">
                    <div style="font-size: 14px; font-weight: 600; color: #f0f0f5;">Analyzing claim...</div>
                    <div style="font-size: 11px; color: #7a7a90; margin-top: 4px;">7 AI agents working in parallel</div>
                    <div style="max-width: 200px; margin: 12px auto;"><div class="loading-bar"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                import core.nim_client as _nc
                _nc._nim_client = None
                from fraudlens import FraudLensAI
                detector = FraudLensAI()
                result = run_async(detector.analyze(
                    input_path,
                    include_network=include_network,
                    include_deepfake=include_deepfake,
                ))
                st.session_state.result = result
                st.session_state.chat_history = []
                progress.empty()
            except Exception as e:
                progress.empty()
                st.markdown(f"""
                <div class="dark-card" style="border-color: rgba(255,77,79,0.4);">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">⚠️</span>
                        <div>
                            <div style="font-size: 14px; font-weight: 600; color: #f0f0f5;">Analysis Error</div>
                            <div style="font-size: 12px; color: #ff4d4f; margin-top: 2px;">{e}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    # Render results
    if hasattr(st.session_state, 'result'):
        st.markdown('<div style="height: 1px; background: var(--border); margin: 12px 0;"></div>', unsafe_allow_html=True)
        render_results(st.session_state.result)
        render_chat()


if __name__ == "__main__":
    main()
