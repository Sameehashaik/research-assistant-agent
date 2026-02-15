# agent_core.py
# The BRAIN of the whole system — this is where the agent thinks and acts.
#
# How it works (the ReAct pattern):
#   1. THOUGHT  — LLM reads the question + tool descriptions, decides what to do
#   2. ACTION   — LLM calls a tool (search_documents or search_web)
#   3. OBSERVE  — LLM reads the tool's result
#   4. REPEAT   — if the LLM needs more info, it goes back to step 1
#   5. ANSWER   — once satisfied, it writes a final response
#
# We use LangChain's create_agent which handles that loop for us.
# All we do is hand it an LLM + a list of tools, and it wires up the
# think → act → observe cycle automatically.
#
# Why is this "agentic"?
#   RAG: question → ALWAYS search docs → answer  (no decisions)
#   Agent: question → THINK which tool → use it → maybe use another → answer
#
# Conversation memory (Phase 5):
#   Without memory: each question is independent — "What are advances in it?" fails
#                   because the agent doesn't know what "it" refers to.
#   With memory: we feed previous messages alongside the new question, so the agent
#                sees the full conversation and resolves pronouns/references.
#   We cap history at max_history messages to keep costs down (more messages = more tokens).

import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

# cost_tracker lives in project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import CostTracker


class ResearchAgent:
    """
    The research assistant agent.
    Give it tools, ask it questions, and it figures out which tools to use.
    Remembers conversation history so follow-up questions work naturally.
    """

    def __init__(self, tools: list, model_name: str = "gpt-4o-mini"):
        """
        tools: list of LangChain Tool objects (from .as_tool() on our tool classes)
        model_name: which OpenAI model does the thinking (gpt-4o-mini is cheap + smart enough)
        """
        self.tools = tools
        self.tracker = CostTracker(log_file="project2_costs.json")

        # conversation history — list of HumanMessage/AIMessage pairs
        # the agent sees these when reasoning so it can resolve "it", "that", etc.
        self.conversation_history = []
        self.max_history = 10  # keep last 10 exchanges to limit token cost

        # the LLM that does the reasoning — temperature=0 for consistent decisions
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # the system prompt teaches the agent HOW to think about tool selection
        self.system_prompt = (
            "You are a helpful research assistant with access to tools.\n\n"
            "DECISION RULES:\n"
            "1. If the question is about the user's personal notes or documents → use search_documents\n"
            "2. If the question asks about current/recent/latest information → use search_web\n"
            "3. If the question needs both personal context AND current info → use BOTH tools\n"
            "4. If unsure, start with one tool and use another if the first wasn't enough\n\n"
            "IMPORTANT:\n"
            "- Always mention which source(s) your answer came from\n"
            "- Combine information from multiple tools when you use more than one\n"
            "- Be honest if you don't have enough information\n"
            "- Think step-by-step before acting\n"
        )

        # create_agent wires up the full ReAct loop:
        #   LLM thinks → picks a tool → gets result → thinks again → ... → final answer
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    # -- Single query (no memory, used by tests) --

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        One-shot question — no conversation history.
        Good for testing individual tool selection.
        """
        result = self.agent.invoke({
            "messages": [HumanMessage(content=question)]
        })

        return self._parse_result(result, question, verbose)

    # -- Query with conversation memory --

    def query_with_history(self, question: str, verbose: bool = True) -> dict:
        """
        Ask a question WITH conversation context.

        The agent sees previous messages, so it can understand:
          Turn 1: "What is RAG?"
          Turn 2: "What are recent advances in it?"
                   ^ agent knows "it" = RAG because it sees Turn 1

        We trim history to self.max_history messages to keep costs reasonable.
        """
        # build the full message list: history + new question
        history_slice = self.conversation_history[-self.max_history:]
        messages = history_slice + [HumanMessage(content=question)]

        # run the agent with full context
        result = self.agent.invoke({"messages": messages})
        parsed = self._parse_result(result, question, verbose)

        # save this exchange to history for future turns
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=parsed["answer"]))

        return parsed

    def reset_conversation(self):
        """Clear all conversation history — fresh start."""
        self.conversation_history = []

    def get_conversation_summary(self) -> str:
        """Quick summary of how long the conversation is."""
        turns = len(self.conversation_history) // 2
        return f"Conversation: {turns} turns ({len(self.conversation_history)} messages)"

    # -- Internal helpers --

    def _parse_result(self, result, question, verbose) -> dict:
        """
        Extract answer, tools used, and reasoning steps from the agent's output.
        The agent returns a list of messages — we walk through them to find
        tool calls (AIMessage with tool_calls) and tool results (ToolMessage).
        """
        all_messages = result["messages"]
        final_answer = all_messages[-1].content if all_messages else "No answer generated."

        tools_used = []
        steps = []
        for msg in all_messages:
            # AIMessage with tool_calls = the agent decided to use a tool
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])
                    steps.append({
                        "action": tc["name"],
                        "input": tc["args"],
                    })
            # ToolMessage = the result that came back from a tool
            if msg.type == "tool":
                if steps:
                    steps[-1]["observation"] = msg.content[:200] + "..."

        if verbose:
            self._print_reasoning(question, steps, tools_used, final_answer)

        return {
            "answer": final_answer,
            "tools_used": tools_used,
            "steps": steps,
        }

    def _print_reasoning(self, question, steps, tools_used, answer):
        """Pretty-print the agent's thinking so we can see ReAct in action."""
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        if steps:
            print("\n--- Agent's ReAct Reasoning ---")
            for i, step in enumerate(steps, 1):
                print(f"\n  THOUGHT {i}: I should use '{step['action']}'")
                print(f"  ACTION  {i}: {step['action']}({step['input']})")
                if "observation" in step:
                    print(f"  OBSERVE {i}: {step['observation'][:120]}...")
        else:
            print("\n  (Agent answered directly without using tools)")

        print(f"\n--- Final Answer ---")
        print(f"  {answer[:300]}...")
        print(f"\n  Tools used: {tools_used}")
        print(f"{'='*60}")
