# test_document_tool.py
# Smoke tests for the document search tool.
# Checks three things:
#   1. Tool schema is wired up correctly (name + description)
#   2. Graceful response when no docs are loaded
#   3. Actual load → embed → search round-trip works

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.document_search import DocumentSearchTool


def test_tool_creation():
    """Make sure the LangChain tool wrapper has the right name/description."""
    tool = DocumentSearchTool()
    langchain_tool = tool.as_tool()

    assert langchain_tool.name == "search_documents"
    assert "personal documents" in langchain_tool.description
    print("Tool schema created correctly")
    print(f"  Name: {langchain_tool.name}")
    print(f"  Description: {langchain_tool.description[:80]}...")


def test_no_documents():
    """If we search before loading anything, we should get a friendly message, not a crash."""
    tool = DocumentSearchTool()
    result = tool.search("anything")
    assert "No documents" in result
    print("Handles missing documents gracefully")


def test_search():
    """Full round-trip: load a file, embed it, search it, verify results make sense."""
    tool = DocumentSearchTool()

    print("Loading research_notes.txt...")
    tool.load_documents(["data/research_notes.txt"])

    print(f"Chunks created: {len(tool.chunks)}")
    print(f"Vectors stored: {tool.index.ntotal}")

    # broad topic search — should find our RAG notes
    result = tool.search("What is RAG?")
    assert "retrieval" in result.lower() or "generation" in result.lower()
    print(f"\nSearch for 'What is RAG?':")
    print(result[:400])

    # narrower search — just verify we got results back at all
    result2 = tool.search("chunk size")
    assert "Document Search Results" in result2
    print(f"Search for 'chunk size':")
    print(result2[:400])


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Tool Schema")
    print("=" * 60)
    test_tool_creation()

    print("\n" + "=" * 60)
    print("TEST 2: No Documents")
    print("=" * 60)
    test_no_documents()

    print("\n" + "=" * 60)
    print("TEST 3: Load & Search")
    print("=" * 60)
    test_search()

    print("\n" + "=" * 60)
    print("Document tool ready!")
    print("=" * 60)
