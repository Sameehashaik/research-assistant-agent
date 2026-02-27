# test_guardrails.py
# All tests here are pure string checks — no API calls, no cost.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.guardrails import ResponseGuardrails


def test_source_detection():
    """Verify we can tell whether a response cites its sources."""
    guard = ResponseGuardrails()

    # response WITH a source URL → high confidence
    with_source = "RAG improves accuracy by 40%. Source: https://example.com"
    result1 = guard.verify_sources(with_source, ["search_web"])
    assert result1["has_sources"] is True
    assert result1["confidence"] > 0.8
    print("Detects responses WITH sources (confidence:", result1["confidence"], ")")

    # response without any source mention, no tools used → low confidence
    no_source = "RAG is a technique for AI."
    result2 = guard.verify_sources(no_source, [])
    assert result2["confidence"] < 0.6
    print("Detects responses WITHOUT sources (confidence:", result2["confidence"], ")")

    # tools were used but response doesn't cite → medium confidence
    tools_no_cite = "The answer is 42."
    result3 = guard.verify_sources(tools_no_cite, ["search_web"])
    assert result3["confidence"] == 0.7
    print("Detects tools-used-but-not-cited (confidence:", result3["confidence"], ")")


def test_uncertainty_detection():
    """Check that we catch hedging language."""
    guard = ResponseGuardrails()

    certain = "RAG combines retrieval with generation."
    result1 = guard.detect_uncertainty(certain)
    assert result1["is_uncertain"] is False
    print("Confident response detected correctly")

    uncertain = "I'm not sure about that, I couldn't find relevant information."
    result2 = guard.detect_uncertainty(uncertain)
    assert result2["is_uncertain"] is True
    print("Uncertain response detected correctly")


def test_response_enhancement():
    """If sources are missing, enhance_response should append them."""
    guard = ResponseGuardrails()

    # response that doesn't mention sources
    bare = "RAG is a powerful technique."
    tools = ["search_documents", "search_web"]

    enhanced = guard.enhance_response(bare, tools)
    assert "Sources:" in enhanced
    assert "your documents" in enhanced
    assert "web search" in enhanced
    print(f"Original:  {bare}")
    print(f"Enhanced:  {enhanced}")

    # response that already has sources → should be left alone
    already_cited = "RAG improves accuracy. Based on your documents and web search results."
    not_changed = guard.enhance_response(already_cited, tools)
    assert not_changed == already_cited
    print("Already-cited response left unchanged")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Source Detection")
    print("=" * 60)
    test_source_detection()

    print("\n" + "=" * 60)
    print("TEST 2: Uncertainty Detection")
    print("=" * 60)
    test_uncertainty_detection()

    print("\n" + "=" * 60)
    print("TEST 3: Response Enhancement")
    print("=" * 60)
    test_response_enhancement()

    print("\n" + "=" * 60)
    print("Guardrails working!")
    print("=" * 60)
