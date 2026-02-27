# Research Assistant Agent

An agentic system that intelligently chooses between searching your personal documents and the web to answer questions.

## What Makes This "Agentic"?

Unlike simple RAG (which always searches docs), this agent:
- **Thinks** before acting — decides which tool fits each question
- **Chooses** the right tool — documents for personal info, web for current events
- **Combines** multiple sources — uses both tools when needed
- **Remembers** conversation context — handles follow-up questions naturally

## Architecture

```
User Question
    |
    v
Agent Core (ReAct pattern)
    |--- Reads instructions from markdown files
    |--- Thinks: "Which tool do I need?"
    |
    |--- search_documents (personal notes, uploaded files)
    |--- search_web (current info, news, external knowledge)
    |
    v
Guardrails (source check, uncertainty detection)
    |
    v
Response with sources
```

## How Tool Selection Works

The agent reads tool descriptions and matches them to the question:

| Question type | Tool chosen | Why |
|---|---|---|
| "What are my notes on RAG?" | `search_documents` | "my notes" = personal info |
| "What's the latest AI news?" | `search_web` | "latest" = current info |
| "How do trends compare to my notes?" | Both | needs personal + external |

## The ReAct Pattern

Every question goes through a think-act-observe loop:

```
THOUGHT: "This asks about 'latest' — I need current info from the web"
ACTION:  search_web("latest RAG advances")
OBSERVE: "Found: hybrid search, re-ranking strategies..."
THOUGHT: "I have enough to answer"
ANSWER:  "Recent advances in RAG include..."
```

## Project Structure

```
research-assistant-agent/
  instructions/           # agent behavior defined in markdown
    base_instructions.md           # general rules (how to respond, safety)
    research_agent_instructions.md # tool-specific rules (when to use which)
  src/
    agent_core.py         # ReAct agent — loads instructions, runs think/act loop
    guardrails.py         # source verification + uncertainty detection
  tools/
    document_search.py    # RAG pipeline: load → chunk → embed → FAISS search
    web_search.py         # web search (simulated or Tavily API)
  tests/
    test_agent.py         # proves agent picks the right tool per question
    test_conversation.py  # proves multi-turn memory works
    test_document_tool.py # document search round-trip
    test_web_tool.py      # web search simulated mode
    test_guardrails.py    # source + uncertainty checks
  app.py                  # Streamlit chat UI
  cost_tracker.py         # API cost logging
```

## Setup

```bash
git clone [https://github.com/YOUR_USERNAME/research-assistant-agent.git](https://github.com/Sameehashaik/research-assistant-agent.git)
cd research-assistant-agent

pip install -r requirements.txt

# add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# run the chat UI
streamlit run app.py
```

## Tech Stack

- **Agent framework:** LangChain `create_agent` (ReAct pattern)
- **LLM:** OpenAI GPT-4o-mini
- **Document search:** FAISS + OpenAI embeddings
- **Web search:** Tavily API or simulated mode
- **Guardrails:** Custom source verification
- **UI:** Streamlit

## Instruction Files

The agent's behavior is defined in editable markdown files, not hardcoded strings:

- `instructions/base_instructions.md` — general behavior rules
- `instructions/research_agent_instructions.md` — tool selection rules

To change how the agent thinks, edit those files. No code changes needed.

## Features

- Intelligent tool selection (documents vs web vs both)
- Multi-turn conversations with memory
- Source citation and verification
- Uncertainty detection
- Confidence scoring
- Cost tracking
- Chat interface with tool visibility

## Example Conversation

```
User: What is RAG?
Agent: [searches documents] Based on your notes, RAG combines retrieval and generation...
       Tools: search_documents

User: What are recent advances in it?
Agent: [searches web] Recent advances include hybrid search, re-ranking...
       Tools: search_web
       (Note: agent resolved "it" = RAG from previous turn)

User: How does this compare to my notes?
Agent: [searches both] Your notes cover the fundamentals, while recent advances include...
       Tools: search_documents, search_web
```
