# app.py
# Streamlit chat UI for the research assistant agent.
#
# What this does:
#   - Multi-turn chat interface (like ChatGPT)
#   - Upload your own documents for the agent to search
#   - Shows which tools the agent used for each answer
#   - Runs guardrails on every response (source check, uncertainty flag)
#   - Tracks cumulative cost in the sidebar
#
# Run with:  streamlit run app.py

import streamlit as st
import os
import sys
from pathlib import Path

# make sure our src/ and tools/ are importable
sys.path.insert(0, str(Path(__file__).parent))

from src.agent_core import ResearchAgent
from tools.document_search import DocumentSearchTool
from tools.web_search import WebSearchTool
from src.guardrails import ResponseGuardrails

# -- Page config --

st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🤖",
    layout="wide",
)

# -- Helper: build or rebuild the agent with current documents --

def init_agent():
    """Create the agent with both tools. Called once on first load."""
    doc_tool = DocumentSearchTool()
    web_tool = WebSearchTool(mode="simulated")

    # if user already uploaded docs, load them
    data_dir = Path("data")
    files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.pdf"))
    if files:
        doc_tool.load_documents([str(f) for f in files])

    agent = ResearchAgent(
        tools=[doc_tool.as_tool(), web_tool.as_tool()]
    )
    return agent, doc_tool


# -- Session state (persists across Streamlit reruns) --

if "agent" not in st.session_state:
    agent, doc_tool = init_agent()
    st.session_state.agent = agent
    st.session_state.doc_tool = doc_tool
    st.session_state.guardrails = ResponseGuardrails()
    st.session_state.messages = []       # chat history for display
    st.session_state.total_cost = 0.0

# -- Sidebar: document upload + controls --

with st.sidebar:
    st.title("Documents")

    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

    if uploaded_file and st.button("Process Document"):
        # save to data/
        save_path = Path("data") / uploaded_file.name
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # reload agent with the new file included
        with st.spinner("Loading document..."):
            agent, doc_tool = init_agent()
            # carry over existing conversation history
            agent.conversation_history = st.session_state.agent.conversation_history
            st.session_state.agent = agent
            st.session_state.doc_tool = doc_tool

        st.success(f"Loaded {uploaded_file.name}")

    # show which files are loaded
    data_dir = Path("data")
    loaded_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.pdf"))
    if loaded_files:
        st.caption(f"{len(loaded_files)} document(s) loaded:")
        for f in loaded_files:
            st.caption(f"  - {f.name}")

    st.divider()

    st.title("Cost")
    st.metric("Total Session Cost", f"${st.session_state.total_cost:.4f}")

    st.divider()

    if st.button("Reset Conversation"):
        st.session_state.agent.reset_conversation()
        st.session_state.messages = []
        st.rerun()

# -- Main chat area --

st.title("Research Assistant Agent")
st.caption("I'll search your documents or the web depending on your question.")

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # show tool badges under assistant messages
        if msg["role"] == "assistant" and msg.get("tools"):
            tool_labels = {
                "search_documents": "Documents",
                "search_web": "Web Search",
            }
            cols = st.columns(len(msg["tools"]))
            for i, tool_name in enumerate(msg["tools"]):
                label = tool_labels.get(tool_name, tool_name)
                cols[i].caption(f"🔧 {label}")

# chat input
if prompt := st.chat_input("Ask a question..."):
    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.query_with_history(prompt, verbose=False)

            # run guardrails
            guardrails = st.session_state.guardrails
            answer = guardrails.enhance_response(result["answer"], result["tools_used"])
            uncertainty = guardrails.detect_uncertainty(answer)
            verification = guardrails.verify_sources(answer, result["tools_used"])

        # display the answer
        st.markdown(answer)

        # show tool badges
        if result["tools_used"]:
            tool_labels = {
                "search_documents": "Documents",
                "search_web": "Web Search",
            }
            cols = st.columns(len(result["tools_used"]))
            for i, tool_name in enumerate(result["tools_used"]):
                label = tool_labels.get(tool_name, tool_name)
                cols[i].caption(f"🔧 {label}")

        # show uncertainty warning if detected
        if uncertainty["is_uncertain"]:
            st.warning("The agent expressed uncertainty. You may want to verify this answer.")

        # show confidence
        conf = verification["confidence"]
        if conf >= 0.9:
            st.caption(f"Confidence: High")
        elif conf >= 0.7:
            st.caption(f"Confidence: Medium")
        else:
            st.caption(f"Confidence: Low — no tools were used")

    # save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tools": result["tools_used"],
    })

# -- How to use (collapsible) --

with st.expander("How to use"):
    st.markdown("""
**This agent decides which tool to use based on your question:**

- **"What are my notes on X?"** → searches your uploaded documents
- **"What's the latest news about X?"** → searches the web
- **"How do recent trends compare to my notes?"** → uses both

**Try a multi-turn conversation:**
1. "What is RAG?"
2. "What are recent advances in it?" ← agent remembers "it" = RAG
3. "How does this compare to my notes?" ← agent uses both tools
    """)
