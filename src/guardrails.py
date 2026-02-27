# guardrails.py
# Safety checks we run on every agent response BEFORE showing it to the user.
#
# Three things we check:
#   1. Source verification — did the agent actually cite where its info came from?
#   2. Uncertainty detection — is the agent hedging or admitting it doesn't know?
#   3. Response enhancement — if sources are missing, tack on a note about which tools were used
#
# Why bother?
#   LLMs can "hallucinate" — confidently state things that are wrong.
#   These guardrails don't prevent that entirely, but they:
#     - Flag when the agent didn't use any tools (higher risk of making stuff up)
#     - Detect when the agent itself is uncertain
#     - Ensure every answer at least says where the info came from

from typing import Dict, List


class ResponseGuardrails:
    """Post-processing checks on agent responses."""

    def verify_sources(self, response: str, tools_used: List[str]) -> Dict:
        """
        Check whether the response cites its sources.

        Returns:
          has_sources  — True if the response mentions a URL or tool name
          cited_tools  — which tools were used
          confidence   — rough score: 0.9 if sourced, 0.5 if no tools at all
        """
        has_url = "http" in response or "Source:" in response
        has_tool_mention = any(
            word in response.lower()
            for word in ["documents", "web", "search", "notes"]
        )

        # simple heuristic: used tools + cited them = high confidence
        if tools_used and (has_url or has_tool_mention):
            confidence = 0.9
        elif tools_used:
            confidence = 0.7   # used tools but didn't explicitly cite
        else:
            confidence = 0.5   # no tools at all — could be guessing

        return {
            "has_sources": has_url or has_tool_mention,
            "cited_tools": tools_used,
            "confidence": confidence,
        }

    def detect_uncertainty(self, response: str) -> Dict:
        """
        Look for phrases that signal the agent isn't confident.
        If detected, the UI can show a warning or suggest the user verify.
        """
        uncertainty_phrases = [
            "not sure", "don't know", "cannot find",
            "unclear", "uncertain", "unable to",
            "don't have", "couldn't find", "no information",
        ]

        response_lower = response.lower()
        is_uncertain = any(phrase in response_lower for phrase in uncertainty_phrases)

        return {
            "is_uncertain": is_uncertain,
            "should_ask_clarification": is_uncertain,
        }

    def enhance_response(self, response: str, tools_used: List[str]) -> str:
        """
        If the agent forgot to mention its sources, append a note.
        This way the user always knows where the info came from.
        """
        verification = self.verify_sources(response, tools_used)

        if not verification["has_sources"] and tools_used:
            # map internal tool names to friendly labels
            friendly = {
                "search_documents": "your documents",
                "search_web": "web search",
            }
            source_labels = [friendly.get(t, t) for t in tools_used]
            response += f"\n\n*Sources: {', '.join(source_labels)}*"

        return response
