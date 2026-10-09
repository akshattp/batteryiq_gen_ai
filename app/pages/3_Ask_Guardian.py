"""app/pages/3_Ask_Guardian.py - Conversational Assistant with Tools, RAG & CodeExec."""

from __future__ import annotations

import json
import sys
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

from guardian.genai.chat_agent import GuardianChatAgent
from guardian.genai.llm_client import get_llm_client
from guardian.genai.rag import RAGPipeline

if "app.ui_theme" in sys.modules:
    importlib.reload(sys.modules["app.ui_theme"])
else:
    import app.ui_theme
from app.ui_theme import inject_theme, render_header, render_footer

st.set_page_config(page_title="Ask Guardian | Assistant", page_icon="💬", layout="wide")

render_header(
    title="Ask the Guardian Assistant",
    subtitle="AI Diagnostic Copilot with Grounded FAISS Knowledge Retrieval & Sandboxed Pandas Execution",
    badge_text="COPILOT ONLINE",
    tag="WORKBENCH 03",
)


@st.cache_resource
def get_agent_instance() -> GuardianChatAgent:
    pq_path = Path("data/processed/battery_telemetry.parquet")
    csv_path = Path("data/processed/battery_telemetry.csv")
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    client = get_llm_client()
    rag = RAGPipeline()
    return GuardianChatAgent(telemetry_df=df, llm_client=client, rag_pipeline=rag)


agent = get_agent_instance()

# Session State for Conversation History
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI Battery Safety Guardian assistant. You can ask me technical questions about thermal runaway mechanisms, check real-time telemetry state, run what-if simulations, or ask me to write and execute pandas code on the telemetry dataset.",
            "tools_called": [],
            "code_traces": [],
        }
    ]

# Display Conversation History
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display Tool & Code Execution Trace if present
        if msg.get("tools_called"):
            with st.expander(f"🛠️ Tool Execution Trace ({len(msg['tools_called'])} calls)"):
                for t in msg["tools_called"]:
                    st.markdown(f"**Tool:** `{t.get('tool')}`")
                    st.json(t.get("args", {}))
                    if t.get("tool") == "run_analysis":
                        trace = t.get("result", {})
                        st.markdown(f"**Status:** `{trace.get('status')}` | **Attempts:** `{trace.get('attempts')}` | **Healed:** `{trace.get('healed')}`")
                        st.code(trace.get("code", ""), language="python")

        if msg.get("citations"):
            with st.expander("📚 Knowledge Base Citations"):
                for cit in msg["citations"]:
                    st.markdown(f"<span class='chip chip-info' style='margin: 3px;'>{cit}</span>", unsafe_allow_html=True)


# Quick Suggestion Chips for Technical Inquiries
st.markdown("<h4 class='h4-title' style='color: #FFD500; margin-top: 18px;'>💡 Suggested Technical Inquiries</h4>", unsafe_allow_html=True)
s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    if st.button("🔥 Thermal Runaway Mechanism", use_container_width=True):
        st.session_state["prefill_chat_query"] = "What are the chemical reaction stages of thermal runaway in high-voltage lithium cells?"
        st.rerun()
with s_col2:
    if st.button("📊 Query High-Temp Cycles", use_container_width=True):
        st.session_state["prefill_chat_query"] = "Execute pandas code to find all cycles where cell temperature exceeded 42°C."
        st.rerun()
with s_col3:
    if st.button("🛡️ Battery SOP Safeguards", use_container_width=True):
        st.session_state["prefill_chat_query"] = "What are the critical temperature limits and immediate technician SOPs?"
        st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Pre-fill query from Events page or chips if any
default_val = st.session_state.pop("prefill_chat_query", "")

user_input = st.chat_input("Ask a battery safety question or analysis request...", key="chat_input")
if not user_input and default_val:
    user_input = default_val

if user_input:
    # Append User Message
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Agent Response
    with st.spinner("Analyzing telemetry & consulting knowledge base..."):
        response = agent.chat(user_input)
        ans_text = response.get("answer", "")
        tools = response.get("tools_called", [])
        code_traces = response.get("code_traces", [])
        citations = []

        # Extract citations
        for t in tools:
            if t.get("tool") == "search_kb":
                passages = t.get("result", {}).get("passages", [])
                for p in passages:
                    citations.append(f"[{p.get('chunk_id')}] {p.get('section', '')}")

    # Append Assistant Message
    st.session_state["chat_messages"].append({
        "role": "assistant",
        "content": ans_text,
        "tools_called": tools,
        "code_traces": code_traces,
        "citations": citations,
    })

    with st.chat_message("assistant"):
        st.markdown(ans_text)
        if tools:
            with st.expander(f"🛠️ Tool Execution Trace ({len(tools)} calls)"):
                for t in tools:
                    st.markdown(f"**Tool:** `{t.get('tool')}`")
                    st.json(t.get("args", {}))
                    if t.get("tool") == "run_analysis":
                        trace = t.get("result", {})
                        st.markdown(f"**Status:** `{trace.get('status')}` | **Attempts:** `{trace.get('attempts')}` | **Healed:** `{trace.get('healed')}`")
                        st.code(trace.get("code", ""), language="python")
        if citations:
            with st.expander("📚 Knowledge Base Citations"):
                for cit in citations:
                    st.markdown(f"<span class='citation-tag'>{cit}</span>", unsafe_allow_html=True)

render_footer()
