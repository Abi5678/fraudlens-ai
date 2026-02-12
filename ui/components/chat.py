"""Shared AI chat component — used across all verticals."""

import json
import streamlit as st
from ui.components.results_common import run_async


def render_chat(context_text: str, session_key: str = "chat_history",
                persona: str = "expert fraud analyst",
                vertical_name: str = "FraudLens AI",
                suggestions: list = None, avatar: str = "🛡️"):
    """Render a chat panel with AI Q&A about the analysis.

    Args:
        context_text: Pre-formatted analysis context string.
        session_key: Session state key for this vertical's chat history.
        persona: AI persona description.
        vertical_name: Display name for the AI assistant.
        suggestions: List of (label, prompt_text) tuples for suggestion chips.
        avatar: Emoji avatar for assistant messages.
    """
    st.markdown(f"""
    <div style="margin-top: 8px; padding: 20px 24px 12px 24px; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(118,185,0,0.1); border: 1px solid rgba(118,185,0,0.2); display: flex; align-items: center; justify-content: center; font-size: 16px;">💬</div>
            <div>
                <div style="font-size: 14px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.3px;">{vertical_name} Chat</div>
                <div style="font-size: 11px; color: var(--text-muted);">Ask follow-up questions about the analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if session_key not in st.session_state:
        st.session_state[session_key] = []

    history = st.session_state[session_key]

    # Suggestion chips
    if suggestions is None:
        suggestions = [
            ("Explain Score", "Explain how the risk score was calculated and what factors contributed most"),
            ("Summarize", "Give me a brief summary of all findings in this analysis"),
            ("Top Red Flags", "What are the top red flags that an investigator should focus on?"),
        ]

    if not history:
        st.markdown('<div style="font-size: 10px; color: #55556a; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-bottom: 6px;">Suggested Questions</div>', unsafe_allow_html=True)
        chip_cols = st.columns(min(4, len(suggestions)))
        for i, (label, prompt_text) in enumerate(suggestions):
            with chip_cols[i]:
                if st.button(label, key=f"chip_{session_key}_{i}", use_container_width=True):
                    st.session_state[f"{session_key}_chip"] = prompt_text
                    st.rerun()

    for msg in history:
        with st.chat_message(msg["role"], avatar=avatar if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about this analysis...")
    chip_key = f"{session_key}_chip"
    if chip_key in st.session_state:
        user_input = st.session_state.pop(chip_key)

    if user_input:
        history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        messages = [
            {"role": "system", "content": f"""You are {vertical_name}, an {persona}. You have analyzed a document and the user is asking follow-up questions.

Answer concisely and professionally. Reference specific findings when relevant. If asked about something not in the data, say so.

ANALYSIS CONTEXT:
{context_text}"""},
        ]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        with st.chat_message("assistant", avatar=avatar):
            try:
                import core.nim_client as _nc
                if _nc._nim_client is None:
                    _nc._nim_client = _nc.NIMClient()
                response = run_async(_nc._nim_client.chat(messages=messages, temperature=0.3, max_tokens=500))
                st.markdown(response)
                history.append({"role": "assistant", "content": response})
            except Exception:
                err_msg = "Connection error. Please try again."
                st.markdown(err_msg)
                history.append({"role": "assistant", "content": err_msg})
