"""Test document search tool"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.document_search import DocumentSearchTool


def test_tool_creation():
    """Test that the tool schema is correct."""
    tool = DocumentSearchTool()
    langchain_tool = tool.as_tool()

    assert langchain_tool.name == "search_documents"
    assert "personal documents" in langchain_tool.description
    print("Tool schema created correctly")
    print(f"  Name: {langchain_tool.name}")
    print(f"  Description: {langchain_tool.description[:80]}...")


def test_no_documents():
    """Test behavior when no docs are loaded."""
    tool = DocumentSearchTool()
    result = tool.search("anything")
    assert "No documents" in result
    print("Handles missing documents gracefully")


def test_search():
    """Test loading docs and searching."""
    tool = DocumentSearchTool()

    print("Loading research_notes.txt...")
    tool.load_documents(["data/research_notes.txt"])

    print(f"Chunks created: {len(tool.chunks)}")
    print(f"Vectors stored: {tool.index.ntotal}")

    # Search for RAG info
    result = tool.search("What is RAG?")
    assert "retrieval" in result.lower() or "generation" in result.lower()
    print(f"\nSearch for 'What is RAG?':")
    print(result[:400])

    # Search for something specific
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
