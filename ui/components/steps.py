"""Shared process-step indicator component."""

import streamlit as st


def render_steps(step: int, labels=None):
    """Render a 3-step process bar.

    Args:
        step: Current step (1, 2, or 3).
        labels: Optional list of 3 label strings. Defaults to Upload/Analyze/Review.
    """
    if labels is None:
        labels = ["Upload", "Analyze", "Review"]
    steps = [(str(i + 1), lbl) for i, lbl in enumerate(labels)]
    items = ""
    for i, (num, label) in enumerate(steps):
        idx = i + 1
        if idx < step:
            dot = f'<div style="width: 24px; height: 24px; border-radius: 50%; background: var(--nvidia-green); display: flex; align-items: center; justify-content: center; font-size: 12px; color: #0a0a0f; font-weight: 700;">&#10003;</div>'
            lbl = f'<span style="font-size: 10px; color: var(--nvidia-green); font-weight: 600; margin-top: 4px;">{label}</span>'
        elif idx == step:
            dot = f'<div style="width: 24px; height: 24px; border-radius: 50%; background: var(--nvidia-green); display: flex; align-items: center; justify-content: center; font-size: 11px; color: #0a0a0f; font-weight: 700;">{num}</div>'
            lbl = f'<span style="font-size: 10px; color: #f0f0f5; font-weight: 600; margin-top: 4px;">{label}</span>'
        else:
            dot = f'<div style="width: 24px; height: 24px; border-radius: 50%; background: transparent; border: 1.5px solid rgba(255,255,255,0.12); display: flex; align-items: center; justify-content: center; font-size: 11px; color: #55556a; font-weight: 600;">{num}</div>'
            lbl = f'<span style="font-size: 10px; color: #55556a; font-weight: 500; margin-top: 4px;">{label}</span>'

        items += f'<div style="display: flex; flex-direction: column; align-items: center; gap: 0; min-width: 56px;">{dot}{lbl}</div>'
        if i < len(steps) - 1:
            completed_line = idx < step
            line_color = "var(--nvidia-green)" if completed_line else "rgba(255,255,255,0.08)"
            items += f'<div style="flex: 1; height: 2px; background: {line_color}; margin: 0 4px; align-self: flex-start; margin-top: 12px; max-width: 80px;"></div>'

    st.markdown(f'<div style="display: flex; align-items: flex-start; justify-content: center; gap: 0; margin-bottom: 12px;">{items}</div>', unsafe_allow_html=True)
