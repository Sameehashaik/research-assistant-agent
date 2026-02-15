# web_search.py
# The agent's second tool — searches the internet for current/external info.
#
# Two modes:
#   "simulated" — returns fake-but-realistic results so we can test the agent
#                 without paying for API calls. The agent can't tell the difference.
#   "tavily"    — real web search via the Tavily API (~$0.002/search).
#
# Why do we need this alongside document search?
#   - Documents = YOUR personal/static knowledge ("my notes on X")
#   - Web       = THE WORLD's current knowledge ("latest news about X")
#   The agent reads both tool descriptions and picks the right one per question.

from langchain_core.tools import Tool
import os
from typing import Optional


class WebSearchTool:
    """
    Web search tool the agent can invoke.
    In simulated mode it returns canned results keyed on query keywords.
    In tavily mode it hits the real Tavily search API.
    """

    def __init__(self, mode: str = "simulated"):
        """
        mode: "tavily" for real searches, "simulated" for free testing.
        Falls back to simulated if Tavily isn't set up.
        """
        self.mode = mode

        if mode == "tavily":
            try:
                from tavily import TavilyClient
                api_key = os.getenv("TAVILY_API_KEY")
                if not api_key:
                    raise ValueError("TAVILY_API_KEY not set")
                self.client = TavilyClient(api_key=api_key)
            except Exception:
                print("  Tavily not available, falling back to simulated mode")
                self.mode = "simulated"

    def search(self, query: str) -> str:
        """
        The function the agent calls.
        Routes to real or simulated search depending on self.mode.
        """
        if self.mode == "tavily":
            return self._search_tavily(query)
        return self._search_simulated(query)

    # -- Real web search --

    def _search_tavily(self, query: str) -> str:
        """Hit the Tavily API and format the top 3 results."""
        results = self.client.search(query, max_results=3)

        formatted = "Web Search Results:\n\n"
        for i, result in enumerate(results.get("results", []), 1):
            formatted += f"[{i}] {result['title']}\n"
            formatted += f"    {result['content']}\n"
            formatted += f"    Source: {result['url']}\n\n"

        return formatted

    # -- Simulated web search (free, for learning/testing) --

    def _search_simulated(self, query: str) -> str:
        """
        Return fake but realistic results based on keywords in the query.
        The agent treats these exactly like real results — it can't tell
        the difference, which is the whole point.
        """
        q = query.lower()

        # RAG-related queries
        if "rag" in q or "retrieval" in q:
            return """Web Search Results:

[1] Recent Advances in RAG Systems - AI Research Blog
    Recent improvements in RAG include hybrid search (combining dense and sparse
    retrieval), re-ranking strategies, and better chunking methods. Companies are
    seeing 40% improvement in answer accuracy.
    Source: https://airesearch.example.com/rag-advances-2024

[2] RAG vs Fine-tuning: When to Use Which - ML Journal
    RAG is preferred when you need up-to-date information, source attribution, or
    domain-specific knowledge without retraining. Fine-tuning better for style/tone.
    Source: https://mljournal.example.com/rag-vs-finetuning

[3] Production RAG at Scale - Tech Conference 2024
    Panel discussion on scaling RAG to millions of documents. Key insights: FAISS
    handles 50M vectors, prompt caching saves 90% on costs, hybrid search crucial.
    Source: https://techconf.example.com/rag-production
"""

        # news / latest queries
        if "news" in q or "latest" in q or "recent" in q:
            return """Web Search Results:

[1] Latest AI Developments This Week
    OpenAI announced improvements to GPT-4, Google released Gemini 1.5, Anthropic
    updated Claude with better reasoning. Competition intensifying.
    Source: https://ainews.example.com/weekly-update

[2] AI Regulation Updates
    EU AI Act implementation begins, US considering new AI safety guidelines,
    industry pushing for balanced regulation approach.
    Source: https://airegulation.example.com/updates
"""

        # fallback for any other query
        return f"""Web Search Results:

[1] Information about: {query}
    Current information from the web about {query}. This is a simulated result
    for testing the agent's web search capability.
    Source: https://example.com/search

[2] More on {query}
    Additional context and recent updates related to your query.
    Source: https://example2.com/info
"""

    def as_tool(self) -> Tool:
        """
        Package as a LangChain Tool.
        The description tells the agent: "use me for current/external info."
        Compare this to the document tool's description ("use me for personal notes")
        — that contrast is how the agent decides which tool fits each question.
        """
        return Tool(
            name="search_web",
            description=(
                "Search the internet for current, up-to-date information. "
                "Use this when the question asks about recent events, news, "
                "latest developments, or information not in personal documents. "
                "Input should be a search query."
            ),
            func=self.search
        )
