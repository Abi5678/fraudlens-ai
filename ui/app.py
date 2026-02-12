"""
FraudLens AI Platform - Streamlit Dashboard
Multi-vertical document analysis powered by NVIDIA NIM

Verticals:
  - Insurance Fraud Detection
  - Mortgage Loan Verification
  - Photo ID Verification
"""

import streamlit as st
import streamlit.components.v1 as components
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

/* Sidebar brand header */
.fl-sidebar-header { text-align: center; padding: 14px 0 16px 0; }
.fl-logo-wrap {
    display: inline-flex; align-items: center; justify-content: center;
    width: 48px; height: 48px;
    background: linear-gradient(145deg, rgba(118,185,0,0.15) 0%, rgba(118,185,0,0.05) 100%);
    border: 1px solid rgba(118,185,0,0.25); border-radius: 12px;
    margin-bottom: 12px; box-shadow: 0 2px 12px rgba(118,185,0,0.12);
}
.fl-logo-icon { font-size: 26px; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.3)); }
.fl-brand-name { font-size: 20px; font-weight: 800; color: #f0f0f5; letter-spacing: -0.6px; line-height: 1.2; }
.fl-brand-highlight { color: var(--nvidia-green) !important; text-shadow: 0 0 20px rgba(118,185,0,0.35); }
.fl-brand-tagline { font-size: 9px; color: var(--text-muted); margin-top: 6px; letter-spacing: 1.8px; text-transform: uppercase; font-weight: 600; opacity: 0.85; }
.fl-sidebar-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); margin: 4px 0 14px 0; }
.fl-module-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 10px; }

/* Sidebar radio - card style */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 8px !important;
    display: flex !important;
    flex-direction: column !important;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    background: rgba(30,30,36,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    border-color: rgba(118,185,0,0.4) !important;
    background: rgba(118,185,0,0.06) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    border-color: var(--nvidia-green) !important;
    background: rgba(118,185,0,0.18) !important;
    box-shadow: 0 0 16px rgba(118,185,0,0.15), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

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

/* Chat input */
[data-testid="stChatInput"] { border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; background: var(--bg-card) !important; }
[data-testid="stChatInput"] * { color: var(--text-primary) !important; caret-color: var(--text-primary) !important; }
[data-testid="stChatInput"] div { background: transparent !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: var(--text-primary) !important; border: none !important; caret-color: var(--text-primary) !important; font-size: 14px !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stChatInput"] button { background: var(--nvidia-green) !important; color: #000 !important; border: none !important; border-radius: var(--radius-sm) !important; }
[data-testid="stChatInput"] button svg { color: #000 !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--nvidia-green) !important; box-shadow: 0 0 0 3px var(--nvidia-green-glow) !important; }
[data-testid="stBottom"] { background: var(--bg-primary) !important; border-top: 1px solid var(--border) !important; padding-top: 8px !important; }
[data-testid="stBottom"] > div { background: var(--bg-primary) !important; }
[data-testid="stChatMessage"] { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; padding: 12px 16px !important; margin-bottom: 8px !important; color: var(--text-primary) !important; }
[data-testid="stChatMessage"] p { color: #e0e0ea !important; font-size: 14px !important; line-height: 1.7 !important; }
[data-testid="stChatMessage"] ol, [data-testid="stChatMessage"] ul { color: #e0e0ea !important; }
[data-testid="stChatMessage"] li { color: #d0d0dc !important; font-size: 14px !important; line-height: 1.7 !important; }
[data-testid="stChatMessage"] strong { color: var(--text-primary) !important; }
[data-testid="stChatMessage"] code { color: var(--nvidia-green) !important; }
[data-testid="stChatMessage"] a { color: var(--blue) !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color: #e0e0ea !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color: #e0e0ea !important; }
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) { background: rgba(118,185,0,0.04) !important; border-color: rgba(118,185,0,0.12) !important; }
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) p { color: var(--text-primary) !important; }
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

/* Hide branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; pointer-events: none !important; }
[data-testid="stStatusWidget"], .stDeployButton, [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
hr { border: none !important; height: 1px !important; background: var(--border) !important; margin: 20px 0 !important; }

/* Hide default sidebar collapse/expand controls */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarNavCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}
section[data-testid="stSidebar"] {
    min-width: 280px !important;
    width: 280px !important;
    transition: margin-left 0.25s cubic-bezier(0.4, 0, 0.2, 1), visibility 0.25s !important;
}
/* Custom sidebar hidden state - controlled by JS */
section[data-testid="stSidebar"].fl-sidebar-hidden {
    margin-left: -280px !important;
    visibility: hidden !important;
}
section[data-testid="stSidebar"].fl-sidebar-visible {
    margin-left: 0 !important;
    visibility: visible !important;
}

/* Expanders */
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
# Vertical definitions
# ---------------------------------------------------------------------------

VERTICALS = {
    "Insurance Fraud": {
        "key": "insurance",
        "description": "Detect fraud in insurance claims",
    },
    "Mortgage Verify": {
        "key": "mortgage",
        "description": "Verify mortgage loan applications",
    },
    "Photo ID Check": {
        "key": "photo_id",
        "description": "Authenticate identity documents",
    },
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render sidebar with vertical selector and per-vertical settings."""
    with st.sidebar:
        # Logo & brand
        st.markdown("""
        <div class="fl-sidebar-header">
            <div class="fl-logo-wrap">
                <span class="fl-logo-icon">🛡️</span>
            </div>
            <div class="fl-brand-name">FraudLens <span class="fl-brand-highlight">AI</span></div>
            <div class="fl-brand-tagline">Multi-Vertical Platform</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="fl-sidebar-divider"></div>', unsafe_allow_html=True)

        # Vertical selector
        st.markdown('<div class="fl-module-label">Select Module</div>', unsafe_allow_html=True)
        selected = st.radio(
            "Module",
            options=list(VERTICALS.keys()),
            index=0,
            label_visibility="collapsed",
            key="vertical_selector",
        )

        vertical_key = VERTICALS[selected]["key"]

        st.markdown("---")

        # Per-vertical settings (AI Agents expanders)
        if vertical_key == "insurance":
            from ui.verticals.insurance import render_sidebar_settings
            render_sidebar_settings()
        elif vertical_key == "mortgage":
            from ui.verticals.mortgage import render_sidebar_settings
            render_sidebar_settings()
        elif vertical_key == "photo_id":
            from ui.verticals.photo_id import render_sidebar_settings
            render_sidebar_settings()

        # System status (collapsed expander)
        with st.expander("System Status", expanded=False):
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

        # Tech badges
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 3px; padding: 2px 0 6px 0;">
            <span class="tech-badge">NIM</span><span class="tech-badge">Nemotron</span><span class="tech-badge">NeMo</span>
            <span class="tech-badge">cuGraph</span><span class="tech-badge">Milvus</span><span class="tech-badge">TensorRT</span>
        </div>
        <div style="text-align: center; padding: 8px 0 0 0;">
            <div style="font-size: 10px; color: #3a3a4a;">NVIDIA GTC 2026</div>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar toggle button — inject into parent DOM via JS
        components.html("""
        <script>
        (function() {
            var doc = window.parent.document;
            if (doc.getElementById('fl-sidebar-toggle')) return;

            var style = doc.createElement('style');
            style.textContent = '#fl-sidebar-toggle{position:fixed;top:12px;left:292px;z-index:999999;width:36px;height:36px;display:flex;align-items:center;justify-content:center;background:#16161e;border:1px solid rgba(255,255,255,0.08);border-radius:8px;cursor:pointer;color:#7a7a90;transition:all .3s cubic-bezier(.4,0,.2,1);box-shadow:0 2px 8px rgba(0,0,0,0.3)}#fl-sidebar-toggle:hover{color:#76B900;border-color:rgba(118,185,0,0.4);background:#1e1e28;box-shadow:0 2px 12px rgba(118,185,0,0.15)}#fl-sidebar-toggle.fl-collapsed{left:12px}';
            doc.head.appendChild(style);

            var btn = doc.createElement('div');
            btn.id = 'fl-sidebar-toggle';
            btn.title = 'Toggle sidebar';
            btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>';
            doc.body.appendChild(btn);

            var hidden = false;
            btn.addEventListener('click', function() {
                var sb = doc.querySelector('section[data-testid="stSidebar"]');
                if (!sb) return;
                hidden = !hidden;
                if (hidden) {
                    sb.classList.add('fl-sidebar-hidden');
                    sb.classList.remove('fl-sidebar-visible');
                    btn.classList.add('fl-collapsed');
                } else {
                    sb.classList.remove('fl-sidebar-hidden');
                    sb.classList.add('fl-sidebar-visible');
                    btn.classList.remove('fl-collapsed');
                }
            });

            var sb = doc.querySelector('section[data-testid="stSidebar"]');
            if (sb) { sb.classList.add('fl-sidebar-visible'); sb.style.transform = 'none'; }
        })();
        </script>
        """, height=0)

    return vertical_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    vertical_key = render_sidebar()

    if vertical_key == "insurance":
        from ui.verticals.insurance import render as render_insurance
        render_insurance()

    elif vertical_key == "mortgage":
        from ui.verticals.mortgage import render as render_mortgage
        render_mortgage()

    elif vertical_key == "photo_id":
        from ui.verticals.photo_id import render as render_photo_id
        render_photo_id()


if __name__ == "__main__":
    main()
