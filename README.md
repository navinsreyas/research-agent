# Deep Research Agent

A production-grade autonomous research agent built with [LangGraph](https://github.com/langchain-ai/langgraph) and Claude. Give it a research question; it plans, searches, reads full articles, scores source credibility, critiques its own work, identifies knowledge gaps, and iterates — all while pausing for your feedback.

---

## The Problem It Solves

Most "AI search" tools return the same thing as a Google snippet — surface-level, single-pass, no depth. This agent mimics how a research analyst actually works:

1. **Decompose** the question into sub-queries
2. **Search** in parallel, read full articles (not just 200-char snippets)
3. **Score** sources by credibility (.edu, .gov, Reuters > random blogs)
4. **Draft** a synthesized answer, citing sources
5. **Critique** the draft and ask: *"What's missing?"*
6. **Loop** with targeted follow-up questions until quality threshold is met

The result is a progressively deeper answer across multiple iterations, not a one-shot summary.

---

## Architecture

```
┌─────────────┐
│  plan_node  │ ← Reads knowledge_gaps from previous iteration
└──────┬──────┘
       │  Send API (fan-out)
       ├─────────────────────────────────┐
       ▼                                 ▼
┌─────────────┐                ┌─────────────┐  ...×3 parallel workers
│execute_search│                │execute_search│
│  _query #1  │                │  _query #2  │
└──────┬──────┘                └──────┬──────┘
       │  operator.add (fan-in)        │
       └──────────────┬───────────────┘
                      ▼
             ┌─────────────────┐
             │ synthesize_node │ ← Scores sources, prioritizes credible ones
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │   *** PAUSE ***  │ ← User reviews draft here (HITL)
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │  critique_node  │ ← LLM scores quality (0.0–1.0)
             └────────┬────────┘
                      │
            ┌─────────┴──────────┐
       score > 0.85          score ≤ 0.85
            │                    │
           END           ┌───────▼───────┐
                         │  refine_node  │ ← Detective logic: finds gaps
                         └───────┬───────┘
                                 │  Loop back
                         ┌───────▼───────┐
                         │   plan_node   │ ← Now has knowledge_gaps
                         └───────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Graph framework | LangGraph | Native support for cyclic graphs, Send API, checkpointing |
| Parallel search | Send API (map-reduce) | 3x speedup — workers run simultaneously |
| State merging | `operator.add` on lists | Accumulates findings across iterations without losing history |
| Persistence | SQLite (`SqliteSaver`) | Durable checkpoints, no external database needed |
| Human-in-loop | `interrupt_before=["critique"]` | User reviews draft before quality evaluation |
| Scraping | Jina Reader API | Free, returns clean Markdown, handles JS/paywalls |
| Source scoring | Heuristic (whitelist) | Transparent, no ML needed, explainable to users |
| Caching | JSON disk cache | Prevents re-paying for identical searches across sessions |

---

## Features

- **Parallel execution** — 3 search queries run simultaneously via the Send API
- **Deep reading** — scrapes full articles (up to 25,000 chars) via Jina Reader, not just snippets
- **Source credibility scoring** — heuristic model: base 0.5 + authority +0.2 + freshness +0.2 + depth +0.1
- **Detective logic** — LLM identifies 2–3 knowledge gaps after each iteration; next iteration targets them
- **Human-in-the-loop steering** — pause after each draft; approve, quit, or redirect the research
- **SQLite persistence** — sessions survive restarts; full checkpoint history
- **Cost tracking** — tracks Claude tokens and Tavily API calls with estimated USD cost
- **Disk caching** — repeated searches are served from cache instantly
- **Circuit breaker** — forces completion after `max_iterations` to prevent infinite loops
- **Graceful degradation** — every failure mode has a fallback (scrape fail → snippet, critique fail → force pass)

---

## Tech Stack

| Tool | Role |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Cyclic graph orchestration, state management, checkpointing |
| [Claude (claude-sonnet-4-5)](https://www.anthropic.com/) | Planning, synthesis, critique, gap detection |
| [Tavily](https://tavily.com/) | Web search optimized for LLM workflows |
| [Jina Reader](https://jina.ai/reader/) | Full-article scraping (free tier, Markdown output) |
| [tenacity](https://tenacity.readthedocs.io/) | Exponential backoff retry for API calls |
| SQLite | Durable checkpoint storage |

---

## Quick Start

### Prerequisites

- Python 3.11+
- API keys for [Anthropic](https://console.anthropic.com/) and [Tavily](https://tavily.com/)

### Setup

```bash
git clone <repo-url>
cd "Research Automation/deep_research_agent"

pip install -r requirements.txt

# Create .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "TAVILY_API_KEY=your_key_here" >> .env

python run_agent.py
```

### Example Session

```
What would you like to research?
> How does Claude's extended thinking compare to OpenAI's o3?

[Iteration 0]
  Planning: Breaking into 3 sub-queries...
  Searching in parallel:
    ✓ execute_search_query: "Claude extended thinking benchmark performance"
    ✓ execute_search_query: "OpenAI o3 reasoning capabilities"
    ✓ execute_search_query: "LLM reasoning comparison 2025"
  Synthesizing: Scored 9 sources (top: 0.90, 0.85, 0.70)...
  Draft ready. Quality score: 0.72

[PAUSE] Review the draft above.
  Press Enter to approve, type feedback to steer, or 'q' to quit:
> Focus more on coding task benchmarks

[Iteration 1]
  Planning: Generating queries targeting knowledge gaps + your feedback...
  ...

Final Answer:
[comprehensive research report with citations]

=== Session Cost ===
  Tavily searches: 6 (3 cached)
  Jina scrapes: 6
  Claude tokens: 12,450 in / 2,100 out
  Estimated cost: $0.07
```

---

## Project Structure

```
Research Automation/
├── README.md
├── .gitignore
│
├── deep_research_agent/          # Source code
│   ├── run_agent.py              # CLI entry point — streaming, HITL, cost display
│   ├── graph.py                  # LangGraph compilation — nodes, edges, SQLite persistence
│   ├── state.py                  # ResearchState TypedDict — all 19 state fields
│   ├── requirements.txt
│   ├── nodes/
│   │   ├── real_nodes.py         # Production nodes (plan, search, synthesize, critique, refine)
│   │   └── mock_nodes.py         # Deterministic mock nodes for testing without API calls
│   └── utils/
│       ├── scraper.py            # Jina Reader wrapper — fetch full article content
│       ├── scoring.py            # Source credibility heuristic scorer
│       ├── cache.py              # @disk_cache decorator — JSON file-based caching
│       ├── tracker.py            # CostTracker singleton — token + API cost aggregation
│       └── logger.py             # RotatingFileHandler — logs/deep_research_agent.log
│
└── docs/                         # Private learning & planning notes (gitignored)
    └── FOR_ME.md                 # Deep-dive architecture guide with war stories
```

---

## State Design

The `ResearchState` TypedDict uses two patterns:

**Accumulative fields** (`Annotated[List, operator.add]`) — lists that *append* across iterations and parallel workers:
- `knowledge_base`, `search_queries`, `visited_urls`, `failed_queries`, `execution_log`, `knowledge_gaps`

**Overwriting fields** (plain types) — values that *replace* on each update:
- `current_draft`, `quality_score`, `current_plan`, `critique`, `user_feedback`

This distinction is what makes iterative research work: each iteration adds to the knowledge base rather than replacing it.

---

## The Detective Pattern

The agent's most interesting behavior emerges from a feedback loop between two nodes:

1. **`refine_node`** reads the current draft and calls the LLM with: *"What specific questions are NOT answered here?"* → stores 2–3 gaps in `knowledge_gaps`
2. **`plan_node`** reads `knowledge_gaps` next iteration and generates queries that target them directly

This creates self-directed follow-up research without any changes to the graph topology. The agent gets smarter each iteration.

---

## Source Credibility Scoring

Each source is scored 0.0–1.0 using three heuristics:

| Criterion | Bonus | Rationale |
|---|---|---|
| Authority domain (`.edu`, `.gov`, Reuters, Nature, etc.) | +0.2 | Domain reputation as proxy for editorial standards |
| Freshness (mentions 2025/2026) | +0.2 | Recent content is more relevant for tech topics |
| Depth (>5,000 characters) | +0.1 | Longer content correlates with comprehensive coverage |
| Base score | 0.5 | Neutral prior for unknown sources |

Sources are sorted by score before synthesis, so the LLM sees the most credible content first.

---

## Key Engineering Patterns

### Map-Reduce with Send API
```python
# plan_node returns Send objects → LangGraph fans out to parallel workers
return [Send("execute_search_query", {"query": q, "iteration": i}) for q in queries]
```
Three workers run simultaneously. Results merge automatically via `operator.add`.

### Human-in-the-Loop with Recursive Resume
```python
# Graph pauses BEFORE critique
graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["critique"])

# Recursive handler — each resume may trigger another pause
def handle_human_in_the_loop(graph, config, state):
    # Get user input, inject into state, resume
    for event in graph.stream(Command(resume=updated_state), config, stream_mode="updates"):
        handle_event(event)
    # Check if paused again (next iteration)
    snapshot = graph.get_state(config)
    if snapshot.next == ("critique",):
        handle_human_in_the_loop(graph, config, snapshot.values)
```

### Circuit Breaker
```python
if iteration >= max_iterations:
    return {"quality_score": 1.0, ...}  # Force END regardless of actual quality
```
Prevents infinite loops when the LLM is perpetually unsatisfied.

---

## Lessons Learned

**`operator.add` is the key insight.** Without it, every iteration would overwrite the knowledge base, and the agent would "forget" previous research. The `Annotated[List[str], operator.add]` annotation is what makes accumulation work.

**Parallel workers need `check_same_thread=False`.** SQLite's default threading model throws errors when multiple `execute_search_query` workers try to write checkpoints simultaneously. Setting `check_same_thread=False` on the connection is required.

**User steering via score manipulation, not new edges.** Instead of adding a new graph branch for user feedback, we set `quality_score=0.0` when feedback is present. This reuses the existing `refine` path and keeps the graph topology static.

**Nested f-strings can't contain backslashes (Python 3.11).** Extract complex expressions to variables before embedding them in f-strings.

---

---

*Built as a learning project exploring LangGraph, agentic loops, and production AI system design.*
