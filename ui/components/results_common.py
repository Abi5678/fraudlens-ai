"""Shared utility functions and chart builders used across verticals."""

import asyncio
import math
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Async helpers
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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

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
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.5, color='rgba(255,255,255,0.08)'),
        hoverinfo='none', showlegend=False,
    ))
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
# Skeleton loader & error card
# ---------------------------------------------------------------------------

def render_skeleton_loader(message="Analyzing...", sub_message="AI agents working in parallel"):
    """Return HTML string for the skeleton loading animation."""
    import streamlit as st
    skel_bg = "linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.06), rgba(255,255,255,0.02))"
    return f"""
    <div style="padding: 0;">
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
        <div style="display: flex; gap: 16px; margin-top: 4px;">
            <div style="flex: 1; height: 160px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 2.4s ease-in-out infinite; border-radius: 12px; border: 1px solid rgba(255,255,255,0.04);"></div>
            <div style="flex: 2; height: 160px; background: {skel_bg}; background-size: 200% 100%; animation: shimmer 2.1s ease-in-out infinite; border-radius: 12px; border: 1px solid rgba(255,255,255,0.04);"></div>
        </div>
        <div style="text-align: center; margin-top: 20px;">
            <div style="font-size: 14px; font-weight: 600; color: #f0f0f5;">{message}</div>
            <div style="font-size: 11px; color: #7a7a90; margin-top: 4px;">{sub_message}</div>
            <div style="max-width: 200px; margin: 12px auto;"><div class="loading-bar"></div></div>
        </div>
    </div>
    """


def user_facing_error(exc: BaseException) -> str:
    """Turn API/network exceptions into a clear message for Hugging Face and local runs."""
    msg = str(exc).strip()
    lower = msg.lower()
    if not msg:
        msg = "Unknown error"
    if any(
        x in lower
        for x in (
            "network error",
            "axioserror",
            "connection",
            "timeout",
            "api key",
            "nvidia_api_key",
            "unreachable",
            "connection refused",
            "connection reset",
        )
    ) or "integrate.api.nvidia.com" in msg:
        return (
            "Cannot reach the AI service (NVIDIA NIM). "
            "On Hugging Face Spaces: add NVIDIA_API_KEY in Settings → Variables and secrets. "
            "If the key is set, the Space may be timing out or unable to reach integrate.api.nvidia.com."
        )
    return msg


def render_error(error_msg):
    """Return HTML string for an error card."""
    return f"""
    <div class="dark-card" style="border-color: rgba(255,77,79,0.4);">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px;">⚠️</span>
            <div>
                <div style="font-size: 14px; font-weight: 600; color: #f0f0f5;">Analysis Error</div>
                <div style="font-size: 12px; color: #ff4d4f; margin-top: 2px;">{error_msg}</div>
            </div>
        </div>
    </div>
    """
