# test_conversation.py
# Proves the agent remembers previous turns in a conversation.
#
# The key test: we say "What is RAG?" then "What are recent advances in it?"
# Without memory, the agent has no idea what "it" means.
# With memory, it sees the previous turn and knows "it" = RAG.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_core import ResearchAgent
from tools.document_search import DocumentSearchTool
from tools.web_search import WebSearchTool


def setup_agent():
    """Create an agent with both tools loaded."""
    doc_tool = DocumentSearchTool()
    doc_tool.load_documents(["data/research_notes.txt"])
    web_tool = WebSearchTool(mode="simulated")

    return ResearchAgent(tools=[doc_tool.as_tool(), web_tool.as_tool()])


def test_multi_turn_conversation():
    """
    Three-turn conversation where each question builds on the previous one.
    The agent must remember context to answer correctly.
    """
    agent = setup_agent()

    print("Starting multi-turn conversation...\n")

    # Turn 1: straightforward question
    print("-" * 40)
    print("TURN 1")
    result1 = agent.query_with_history("What is RAG?")
    print(f"  Tools: {result1['tools_used']}")

    # Turn 2: "it" refers to RAG from Turn 1 — only works with memory
    print("\n" + "-" * 40)
    print("TURN 2 — 'it' should refer to RAG from Turn 1")
    result2 = agent.query_with_history("What are recent advances in it?")
    print(f"  Tools: {result2['tools_used']}")

    # Turn 3: "this" refers to the advances from Turn 2
    print("\n" + "-" * 40)
    print("TURN 3 — 'this' should refer to recent advances from Turn 2")
    result3 = agent.query_with_history("How does this compare to my notes?")
    print(f"  Tools: {result3['tools_used']}")

    # verify the agent tracked the conversation
    print("\n" + "-" * 40)
    print(agent.get_conversation_summary())
    assert len(agent.conversation_history) == 6  # 3 human + 3 AI messages
    print("Agent maintained context across all 3 turns!")


def test_reset():
    """After reset, the agent should lose all conversation context."""
    agent = setup_agent()

    # have a conversation
    agent.query_with_history("What is RAG?", verbose=False)
    assert len(agent.conversation_history) == 2

    # reset
    agent.reset_conversation()
    assert len(agent.conversation_history) == 0
    print("Reset clears conversation history")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Multi-turn conversation")
    print("=" * 60)
    test_multi_turn_conversation()

    print("\n" + "=" * 60)
    print("TEST 2: Reset")
    print("=" * 60)
    test_reset()

    print("\n" + "=" * 60)
    print("Conversation memory works!")
    print("=" * 60)
