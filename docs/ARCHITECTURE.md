# Deep Research Agent: Architecture & Design Philosophy

## The One-Liner Pitch

This is a **multi-agent research assistant** that reads full articles (not just snippets), scores sources by credibility, and automatically identifies knowledge gaps to follow up on—like a PhD student doing literature review, but automated.

**Why it exists:** Search engines give you links. LLMs give you answers from their training data. This agent does what a human researcher does: reads the actual papers, checks if sources are credible, notices what's missing, and goes back for more information.

---

## The Nervous System: How It All Works

Think of this agent as a **research lab with three specialists**:

1. **The Planner** (`plan_node`): Breaks down big questions into specific search queries
2. **The Reader** (`execute_search_query`): Finds sources and reads full articles (not just abstracts)
3. **The Writer** (`synthesize_node`): Combines findings into a coherent answer
4. **The Critic** (`critique_node`): Evaluates quality and decides if more research is needed
5. **The Detective** (`refine_node`): Identifies what's missing and suggests follow-up questions
6. **The Human** (you): Can intervene at any time to steer the research direction

These specialists pass a shared **clipboard** (the `ResearchState`) between them. Think of it like a relay race where each runner adds something to the baton before passing it to the next person.

---

## The Flow: How a Research Session Works

Let's trace a real example: **"Compare iPhone 17 to Pixel smartphones"**

### **Iteration 0: The Initial Research**

1. **PLAN**: "I need to search for:
   - iPhone 17 specs and features
   - Pixel smartphone specs
   - Direct comparisons between the two"

2. **EXECUTE** (3 parallel searches):
   - Query 1: Tavily returns 3 results → **Scrapes Top 1 URL** (full article, 18,000 chars)
   - Query 2: Tavily returns 3 results → **Scrapes Top 1 URL** (full article, 22,000 chars)
   - Query 3: Tavily returns 3 results → **Scrapes Top 1 URL** (full article, 15,000 chars)

   **Total**: 9 sources found, 3 deep-read (55,000 chars of content)

3. **SYNTHESIZE**:
   - **Scores all 9 sources** for credibility:
     - TechCrunch article: 0.8 (Fresh: mentions 2026 + Deep: 18k chars — .com is not an authority domain)
     - Nature.com review: 1.0 (Authority: nature.com + Fresh + Deep)
     - Random blog: 0.5 (Base score, no bonuses)
   - **Sorts sources** (Nature.com first, blog last)
   - **Tells LLM**: "Prioritize HIGH CONFIDENCE sources (score >= 0.7)"
   - **Generates draft** (~1,200 words)

4. **CRITIQUE**:
   - User sees draft
   - User presses Enter (no feedback)
   - LLM evaluates: "Quality score: 0.72 (below 0.85 threshold)"
   - **Decision**: REFINE (not good enough yet)

5. **REFINE** (The Detective):
   - Analyzes the draft: "This covers specs and pricing, but..."
   - **Identifies gaps**:
     - "What is the battery capacity in mAh for iPhone 17?"
     - "How does the Pixel camera perform in low light conditions?"
   - **Saves gaps to state** for next iteration

### **Iteration 1: The Follow-Up Research**

1. **PLAN** (sees the gaps):
   - "Detective identified 2 missing pieces. Let me search for:
     - iPhone 17 battery capacity mAh benchmarks
     - Pixel camera low light photo samples reviews
     - Battery life comparison tests"

2. **EXECUTE** (3 more deep reads, targeting the gaps)

3. **SYNTHESIZE** (now with 18 sources total, 6 deep-read):
   - Scores and sorts again
   - Generates enhanced draft (~1,500 words)
   - Now includes battery specs and camera examples

4. **CRITIQUE**:
   - LLM evaluates: "Quality score: 0.88 (passes 0.85 threshold)"
   - **Decision**: END (good enough!)

**Final Result**: Comprehensive comparison with credible sources, addressing the gaps identified during research.

---

## System 2 Thinking: Why This Architecture?

### **The PhD Student Analogy**

A good researcher doesn't just Google and summarize the first page of results. They:

1. **Read the actual papers** (not just abstracts)
2. **Check if the journal is credible** (Nature > random blog)
3. **Notice what's missing** ("Wait, none of these sources mention X...")
4. **Go back and research the gaps**

This agent does the same thing, in code.

### **The Three Superpowers** (Phase 5 Upgrade)

#### **Superpower 1: Deep Reading** (`scraper.py`)

**Before**: Agent used Tavily's 100-200 char snippets. Like reading only the first sentence of each article.

**After**: Agent reads the Top 1 URL per query via Jina Reader API (up to 25,000 chars). Like reading the introduction and key sections.

**Why Jina Reader?**
- Free tier (1,000 requests/day)
- Returns clean Markdown (no ads, navbars, or HTML soup)
- Handles JavaScript rendering and some paywalls

**Trade-off**: Depth vs Breadth
- **Old**: 3 queries × 3 snippets = 9 snippets (~1,800 chars total)
- **New**: 3 queries × 1 full article = 3 articles (~75,000 chars total)

We chose depth because that's what "research" means.

#### **Superpower 2: Source Credibility** (`scoring.py`)

**Before**: All sources treated equally. A random blog had the same weight as Nature.com.

**After**: Every source gets a credibility score (0.0-1.0) based on:

| Criterion | Weight | Examples |
|-----------|--------|----------|
| **Authority** | +0.2 | `.edu`, `.gov`, `nature.com`, `reuters.com` |
| **Freshness** | +0.2 | Mentions 2025 or 2026 (current/recent year) |
| **Depth** | +0.1 | Article >5,000 chars (substantial content) |
| **Base** | 0.5 | Default (neutral assumption) |

**Example**:
- `nature.com/quantum-computing-2026` with 8,000 chars → **1.0** (Authority + Fresh + Deep)
- `techblog.com/iphone-review` with 2,000 chars → **0.5** (Base only)

The LLM sees these scores and is instructed: **"Prioritize sources with score >= 0.7"**

**Why heuristics (not ML)?**
- Simple, explainable, no training data needed
- Fast (runs in <1ms per source)
- Transparent (you can see why a source scored high)
- Good enough for MVP (can upgrade to ML later)

#### **Superpower 3: Detective Logic** (`refine_node`)

**Before**: Agent did one round of research and stopped (or did blind refinement).

**After**: Agent analyzes its own draft and asks: **"What did I miss?"**

**The Detective Prompt**:
```
Analyze the draft and identify 2-3 specific, answerable questions that are:
1. Relevant to the research question
2. NOT already answered in the draft
3. Would significantly improve the draft if added
```

**Examples of Good Gaps**:
- "What is the battery capacity in mAh for iPhone 17?"
- "How does the Pixel camera perform in low light conditions?"

**Examples of Bad Gaps**:
- "More info needed" ← Too vague
- "Camera quality?" ← Already covered in draft

**Why this creates "Deep Research" behavior**:
- Iteration 0: Agent researches the obvious stuff
- Iteration 1: Agent researches **the gaps from iteration 0**
- Iteration 2: Agent researches **the gaps from iteration 1**

Each iteration gets more sophisticated and targeted. This is the "follow-up question" behavior that makes it feel like a real researcher.

---

## The State: The Shared Clipboard

The `ResearchState` (in `state.py`) is like a clipboard passed between specialists. It has two types of fields:

### **Accumulative Fields** (`operator.add`)
These **append** to a list (like adding sticky notes to the clipboard):
- `knowledge_base`: All findings from all iterations
- `search_queries`: All queries executed
- `visited_urls`: All URLs visited
- `knowledge_gaps`: All gaps identified (Phase 5)

**Why accumulate?**
- Preserves history (helps debugging and analytics)
- Supports parallel execution (3 searches run simultaneously, results merged safely)

### **Overwriting Fields**
These **replace** the previous value (like erasing the whiteboard and writing new):
- `current_draft`: The latest version of the answer
- `quality_score`: The latest critique score
- `iteration_count`: Current loop number

**Why overwrite?**
- Only the latest draft matters (old drafts are noise)
- Only the latest score determines routing (end vs refine)

---

## The Graph: The Relay Race Track

```
START → plan → [execute × 3] → synthesize → [HITL] → critique → [router]
                     ↑                                              ↓
                     └──────────────── refine ←───────────────────┘
                                      (detective)
```

**Key Design Decisions**:

1. **Parallel Execution** (`[execute × 3]`):
   - 3 search queries run **simultaneously** (not sequentially)
   - Why? Saves time (3 queries in 5 seconds vs 15 seconds)
   - How? LangGraph's `Send` API + `operator.add` for safe merging

2. **Human-in-the-Loop** (`[HITL]`):
   - Agent **pauses before critique** to show you the draft
   - You can: Approve (Enter), Steer ("Focus on X"), or Quit
   - If you steer: Agent forces REFINE and incorporates your feedback
   - Why? Keeps human in control (agency is important)

3. **Circuit Breaker** (in `critique_node`):
   - Max 3 iterations (prevents infinite loops)
   - If iteration >= 3: Force pass (accept draft and end)
   - Why? Fail gracefully after N attempts (production safety pattern)

---

## War Stories: Lessons & Gotchas

### **Bug Story 1: The Disappearing Data**

**The Problem**: In early testing, search results from parallel queries were getting lost. We'd run 3 searches but only see results from 1 or 2.

**The Culprit**: The `knowledge_base` field wasn't using `operator.add` in the state schema.

**What was happening**:
- Worker 1 writes: `{"knowledge_base": [item1, item2, item3]}`
- Worker 2 writes: `{"knowledge_base": [item4, item5, item6]}`
- Final state: `{"knowledge_base": [item4, item5, item6]}` ← **Worker 1's data LOST!**

**The Fix**:
```python
# Before (WRONG):
knowledge_base: List[Dict[str, Any]]

# After (CORRECT):
knowledge_base: Annotated[List[Dict[str, Any]], operator.add]
```

With `operator.add`, LangGraph **merges** the lists instead of overwriting:
- Final state: `{"knowledge_base": [item1, item2, item3, item4, item5, item6]}` ✓

**The Lesson**: In LangGraph, parallel nodes need `operator.add` on list fields. Always.

### **Bug Story 2: The Context Overflow**

**The Problem**: Original plan was to scrape Top 2 URLs per query (6 deep reads per iteration). Agent would crash with context window errors.

**The Math**:
- 3 queries × Top 2 URLs × 25,000 chars = **150,000 chars** per iteration
- Context window: ~200k tokens (~800k chars)
- But: System prompt + knowledge base formatting + draft + logs = **too much!**

**The Fix**: Scrape only **Top 1 URL per query** (3 deep reads per iteration).
- 3 queries × Top 1 URL × 25,000 chars = **75,000 chars** ← Safe!

**The Lesson**: Always do the math on context windows. Better to go deep on fewer sources than wide on many.

### **Bug Story 3: The State Collision**

**The Problem**: Detective logic needed to save identified gaps, but where?

**Wrong Approach 1**: Overwrite `critique['message']`
- **Problem**: critique_node already uses that field for quality assessment
- **Result**: Gaps would overwrite the critique message (data loss)

**Wrong Approach 2**: Add gaps to `current_plan`
- **Problem**: plan_node already uses that field for sub-questions
- **Result**: Confusion about what the field represents

**The Fix**: Add a **new state field** specifically for gaps:
```python
knowledge_gaps: Annotated[List[str], operator.add]
```

**The Lesson**: When in doubt, add a new field. Don't try to overload existing fields with multiple meanings.

---

## New Superpowers You Unlocked

After understanding this codebase, you now understand:

1. **Cyclic Graphs**: How to build agents that loop until a condition is met (not just linear pipelines)
2. **Parallel Execution**: How to run multiple LLM/API calls simultaneously and merge results safely
3. **State Management**: The difference between accumulative (`operator.add`) and overwriting fields
4. **Human-in-the-Loop**: How to pause agent execution and inject user feedback mid-stream
5. **Defensive Programming**: How to handle API failures, timeouts, and edge cases gracefully
6. **Source Credibility**: How to build explainable heuristic scoring systems
7. **Self-Reflection**: How to make agents analyze their own outputs and identify gaps

**The Senior Engineer Insight**:

The key to building reliable agents isn't just using the right APIs. It's designing for **observability** (logging), **recoverability** (fallbacks), and **controllability** (HITL).

This agent follows the principle: **"The user is always in control."**
- User can steer at any time (human-in-the-loop)
- User can see logs (file-based logging)
- User can see source scores (transparent credibility)
- User can see what the detective found (knowledge gaps in logs)

---

## The Trade-Offs We Made

### **Why File-Based Logging (Not Console)?**

**Decision**: All debug logs go to `logs/deep_research_agent.log`. Console shows only user-facing progress.

**Trade-off**:
- ✅ Clean CLI (users see progress bars and status)
- ✅ Persistent logs (can review past sessions)
- ❌ Slightly harder to debug in real-time (need to `tail -f logs/*.log`)

**Why we chose this**: The beautiful CLI is a core feature. Cluttering it with debug logs would ruin the UX.

### **Why Top 1 URL (Not Top 3)?**

**Decision**: Deep read only the first search result per query.

**Trade-off**:
- ✅ Context-safe (75k chars per iteration)
- ✅ Fast (3 scrapes vs 9 scrapes)
- ✅ Top result usually most relevant (Tavily ranks by quality)
- ❌ Miss potentially useful info from results #2 and #3

**Why we chose this**: Depth > Breadth. Better to deeply understand 3 sources than skim 9.

### **Why Heuristic Scoring (Not ML)?**

**Decision**: Use simple rules (`.edu` + freshness + depth) instead of ML model.

**Trade-off**:
- ✅ Simple, explainable, no training data
- ✅ Fast (<1ms per source)
- ✅ Good enough for MVP (0.7+ score correlates with quality in practice)
- ❌ Whitelist approach (misses new credible sources)
- ❌ Can't adapt to domain-specific credibility signals

**Why we chose this**: Simplicity is a feature. We can always upgrade to ML later if needed.

---

## Production Patterns Applied

1. **Circuit Breaker**: Max 3 iterations (prevents infinite loops)
2. **Retry with Exponential Backoff**: API calls retry up to 3 times with increasing delays
3. **Defensive Programming**: Every API call has a fallback (empty result, not crash)
4. **Separation of Concerns**: User output (console) vs debug data (files)
5. **Graceful Degradation**: If detective logic fails, agent continues without gaps (not a fatal error)

---

## What You Should Understand to Explain This in an Interview

1. **The Problem**: Search engines give links, LLMs give training data, but neither does real research (read + verify + follow-up)

2. **The Solution**: Multi-agent system with deep reading, source scoring, and self-reflection

3. **The Architecture**: Cyclic graph (not pipeline), shared state (clipboard pattern), parallel execution (Send API)

4. **The Key Insight**: The "Detective Model" - agent analyzes its own gaps and generates follow-up questions (this is what makes it "deep research")

5. **The Trade-offs**: Depth vs Breadth, Simplicity vs Sophistication, User Control vs Automation

6. **The Production Patterns**: Circuit breaker, retry logic, logging, HITL, fallbacks

**The One-Sentence Explanation**:
"It's a multi-agent research system that reads full articles, scores source credibility, and recursively identifies knowledge gaps—like a PhD student doing literature review, but automated with LLMs."

---

## Next Steps: How to Extend This

### **Add More Credibility Signals**
- Citation count (via Semantic Scholar API)
- Domain authority (via Moz API)
- Author reputation (H-index from Google Scholar)

### **Add Multi-Modal Research**
- Scrape PDFs (via PyPDF2)
- Analyze images/charts (via GPT-4 Vision)
- Watch YouTube videos (via Whisper transcription)

### **Add Memory**
- Save research sessions to vector DB (ChromaDB)
- Reuse findings from past sessions
- Build a "research library" over time

### **Add Collaboration**
- Multiple agents with different specialties
- Debate agents (one argues pro, one argues con)
- Peer review agents (critique each other's findings)

---

## Conclusion: You Built Something Rare

Most "AI research tools" are just search wrappers or RAG systems. This is different:

- **It loops** (iterative refinement, not one-shot)
- **It reads deeply** (full articles, not snippets)
- **It checks credibility** (scores sources)
- **It identifies gaps** (self-reflection)
- **It follows up** (targeted queries based on gaps)
- **It keeps you in control** (HITL at every step)

You didn't just build a search tool. You built a research assistant that thinks.

Now go use it. And when someone asks "How does this work?", you can explain every piece—because you understand the **why** behind every decision.

That's the Glass Box Protocol. 📦✨
