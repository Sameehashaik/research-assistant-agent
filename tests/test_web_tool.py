# test_web_tool.py
# Verifies the web search tool works in simulated mode.
# No API calls, no cost — just making sure the tool returns
# sensible results and the LangChain schema is wired up right.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.web_search import WebSearchTool


def test_simulated_search():
    """Check that simulated mode returns keyword-matched fake results."""
    tool = WebSearchTool(mode="simulated")

    # query about RAG — should hit the RAG branch
    result = tool.search("What are recent advances in RAG?")
    assert "RAG" in result
    assert "Source:" in result
    print("Simulated RAG search works")
    print(f"\nResult preview:\n{result[:200]}...")

    # query about news — should hit the news branch
    result2 = tool.search("latest AI news")
    assert "news" in result2.lower() or "latest" in result2.lower()
    print("\nSimulated news search works")

    # generic query — should hit the fallback branch
    result3 = tool.search("quantum computing basics")
    assert "quantum computing" in result3.lower()
    print("Simulated fallback search works")


def test_tool_schema():
    """Make sure the LangChain tool wrapper has the right name/description."""
    tool = WebSearchTool()
    langchain_tool = tool.as_tool()

    assert langchain_tool.name == "search_web"
    assert "current" in langchain_tool.description
    print("Web search tool schema correct")
    print(f"  Name: {langchain_tool.name}")
    print(f"  Description: {langchain_tool.description[:80]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Simulated Search")
    print("=" * 60)
    test_simulated_search()

    print("\n" + "=" * 60)
    print("TEST 2: Tool Schema")
    print("=" * 60)
    test_tool_schema()

    print("\n" + "=" * 60)
    print("Web search tool ready!")
    print("=" * 60)
