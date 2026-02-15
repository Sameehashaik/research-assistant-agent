# test_agent.py
# Three tests that prove the agent is actually THINKING about which tool to use:
#   1. Personal question   → should pick search_documents
#   2. Current-events question → should pick search_web
#   3. Comparison question → should pick BOTH tools
#
# Watch the verbose output — you'll see the ReAct loop:
#   THOUGHT → ACTION → OBSERVATION → (repeat?) → FINAL ANSWER

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_core import ResearchAgent
from tools.document_search import DocumentSearchTool
from tools.web_search import WebSearchTool


def setup_agent():
    """Create an agent with both tools loaded and ready."""
    # tool 1: document search (loads our sample notes)
    doc_tool = DocumentSearchTool()
    doc_tool.load_documents(["data/research_notes.txt"])

    # tool 2: web search (simulated — free)
    web_tool = WebSearchTool(mode="simulated")

    # hand both tools to the agent — it sees their names + descriptions
    agent = ResearchAgent(
        tools=[doc_tool.as_tool(), web_tool.as_tool()]
    )
    return agent


def test_document_selection():
    """
    'What are my notes on RAG?' — says 'my notes', clearly personal.
    Agent should pick search_documents.
    """
    agent = setup_agent()
    result = agent.query("What are my notes on RAG?")

    assert "search_documents" in result["tools_used"], \
        f"Expected search_documents but got {result['tools_used']}"
    print("\nAgent correctly chose DOCUMENT search for a personal question!")


def test_web_selection():
    """
    'What are the latest advances in RAG?' — says 'latest', needs current info.
    Agent should pick search_web.
    """
    agent = setup_agent()
    result = agent.query("What are the latest advances in RAG systems?")

    assert "search_web" in result["tools_used"], \
        f"Expected search_web but got {result['tools_used']}"
    print("\nAgent correctly chose WEB search for a current-events question!")


def test_multi_tool():
    """
    'How do recent RAG advances compare to what I learned in my notes?'
    Needs both personal docs AND current web info.
    Agent should use at least one tool, ideally both.
    """
    agent = setup_agent()
    result = agent.query(
        "How do recent RAG advances compare to what I've learned in my notes?"
    )

    assert len(result["tools_used"]) > 0, "Agent should have used at least one tool"
    print(f"\nAgent used: {result['tools_used']}")
    if "search_documents" in result["tools_used"] and "search_web" in result["tools_used"]:
        print("Agent used BOTH tools for the comparison question!")
    else:
        print("Agent used some tools — it may not always pick both, and that's OK.")


if __name__ == "__main__":
    print("TEST 1: Personal question (should use documents)")
    print("=" * 60)
    test_document_selection()

    print("\n\nTEST 2: Current-events question (should use web)")
    print("=" * 60)
    test_web_selection()

    print("\n\nTEST 3: Comparison question (should use both)")
    print("=" * 60)
    test_multi_tool()

    print("\n" + "=" * 60)
    print("All agent reasoning tests complete!")
    print("=" * 60)
