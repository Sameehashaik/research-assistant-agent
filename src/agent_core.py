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
    """

    def __init__(self, tools: list, model_name: str = "gpt-4o-mini"):
        """
        tools: list of LangChain Tool objects (from .as_tool() on our tool classes)
        model_name: which OpenAI model does the thinking (gpt-4o-mini is cheap + smart enough)
        """
        self.tools = tools
        self.tracker = CostTracker(log_file="project2_costs.json")

        # the LLM that does the reasoning — temperature=0 for consistent decisions
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # the system prompt teaches the agent HOW to think about tool selection
        # this is basically the agent's "personality + decision rules"
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
        # we don't have to write any of that loop ourselves
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        Ask the agent a question. It will:
          1. Read the question
          2. Decide which tool(s) to use
          3. Call those tools
          4. Synthesize results into a final answer

        Returns a dict with:
          answer      — the final text response
          tools_used  — list of tool names the agent chose (e.g. ["search_web"])
          steps       — the full thought/action/observation trace
        """
        # feed the question into the ReAct agent graph
        result = self.agent.invoke({
            "messages": [HumanMessage(content=question)]
        })

        # pull out the messages the agent produced during its reasoning
        all_messages = result["messages"]

        # the last message is the agent's final answer
        final_answer = all_messages[-1].content if all_messages else "No answer generated."

        # figure out which tools the agent decided to use by looking at tool call messages
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
        """Pretty-print the agent's thinking process so we can see ReAct in action."""
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
