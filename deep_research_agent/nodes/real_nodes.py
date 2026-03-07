"""
Deep Research Agent - Node Implementations

LangGraph nodes for the Deep Research Agent pipeline:
- plan_node: Decomposes research task into 3 targeted search queries
- execute_search_query: Map-reduce worker (one per query, runs in parallel)
- synthesize_node: Synthesizes a research draft from scored sources
- critique_node: Evaluates draft quality and applies user steering
- refine_node: Increments iteration and identifies knowledge gaps (detective logic)
"""

import json
import logging
import os
import time
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.types import Send

# Phase 5: Deep Research Upgrade imports
from utils.scraper import scrape_url
from utils.scoring import calculate_source_score

# Production Optimization: Caching and Cost Tracking
from utils.cache import disk_cache
from utils.tracker import CostTracker


@disk_cache
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def cached_tavily_search(query: str, max_results: int = 3) -> dict:
    """
    Tavily search with disk caching and exponential backoff retry.

    Why this function?
    - TavilyClient is not JSON-serializable, so we can't cache at the call site
    - By accepting only primitive args, the @disk_cache decorator can hash them
    - @retry handles transient API failures with exponential backoff (2s, 4s, 8s)
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"[cached_tavily_search] query='{query[:50]}...', max_results={max_results}")

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    start_time = time.time()
    try:
        result = tavily_client.search(query=query, max_results=max_results)
        duration = time.time() - start_time
        result_count = len(result.get("results", []))
        logger.info(f"[cached_tavily_search] SUCCESS: {result_count} results in {duration:.2f}s for '{query[:40]}...'")
        CostTracker.track_search(from_cache=False)
        return result
    except Exception as e:
        logger.warning(f"[cached_tavily_search] RETRY triggered: {type(e).__name__}: {str(e)[:100]}")
        raise


def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planning Node: Decomposes the research task into 3 targeted search queries.

    Incorporates user steering feedback and detective-identified knowledge gaps
    when generating queries, so subsequent iterations are progressively more targeted.
    """
    logger = logging.getLogger(__name__)

    task = state.get("task", "")
    iteration = state.get("iteration_count", 0)
    critique = state.get("critique", {})
    user_feedback = critique.get("feedback", "")
    knowledge_gaps = state.get("knowledge_gaps", [])

    logger.info(f"[plan_node] iteration={iteration}, task='{task[:50]}...'")

    if user_feedback:
        logger.info(f"[plan_node] User steering: '{user_feedback[:100]}...'")
    if knowledge_gaps:
        logger.info(f"[plan_node] Detective gaps: {len(knowledge_gaps)} to address")

    if not task:
        return {
            "current_plan": {
                "sub_questions": [],
                "strategy": "error",
                "error": "No task provided"
            },
            "execution_log": [{
                "node": "plan",
                "iteration": iteration,
                "action": "Planning failed: no task provided",
                "status": "error"
            }]
        }

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    has_feedback = critique.get("user_steering", False)

    # Build prompt: priority order is User Feedback > Knowledge Gaps > Standard
    if knowledge_gaps:
        gaps_text = "\n".join([f"- {gap}" for gap in knowledge_gaps[-3:]])
        base_prompt = f"""You are a research planning expert. Break down this research task into 3 specific, targeted search queries.

RESEARCH TASK:
{task}

KNOWLEDGE GAPS IDENTIFIED:
The following gaps were identified in the previous iteration. Your queries should address these:
{gaps_text}

YOUR TASK:
Generate exactly 3 search queries that will help answer this research question.
**PRIORITY:** Address the knowledge gaps listed above.

Return ONLY a JSON object with this structure:
{{"sub_questions": ["query1", "query2", "query3"]}}

No markdown, no explanations, just the JSON object."""

    elif has_feedback and user_feedback:
        base_prompt = f"""You are a research planning assistant.

Task: {task}
Iteration: {iteration}

IMPORTANT - USER STEERING:
The user reviewed the previous draft and provided this feedback:
"{user_feedback}"

You MUST generate queries that address this feedback. The user's instruction takes priority over everything else.
Focus your queries on the specific aspects the user mentioned.

Break this task into exactly 3 distinct search queries that:
1. Cover different aspects of the topic
2. Are specific enough to yield useful results
3. Don't overlap significantly
4. PRIORITIZE addressing the user's feedback above all else

Return ONLY a JSON object with this structure:
{{"sub_questions": ["query1", "query2", "query3"]}}

No markdown, no explanations, just the JSON object."""

    else:
        base_prompt = f"""You are a research planning expert. Break down this research task into 3 specific, targeted search queries.

RESEARCH TASK:
{task}

YOUR TASK:
Generate exactly 3 search queries that will help answer this research question.

Return ONLY a JSON object with this structure:
{{"sub_questions": ["query1", "query2", "query3"]}}

No markdown, no explanations, just the JSON object."""

    prompt = base_prompt

    try:
        start_time = time.time()
        response = llm.invoke(prompt)
        duration = time.time() - start_time
        logger.info(f"[plan_node] LLM responded in {duration:.2f}s")

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            CostTracker.track_llm(
                response.usage_metadata.get('input_tokens', 0),
                response.usage_metadata.get('output_tokens', 0)
            )

        response_text = response.content.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        plan_data = json.loads(response_text)

        if "sub_questions" not in plan_data or not isinstance(plan_data["sub_questions"], list):
            raise ValueError("Invalid plan structure: missing sub_questions list")

        logger.info(f"[plan_node] Generated {len(plan_data.get('sub_questions', []))} queries")
        return {
            "current_plan": {
                "sub_questions": plan_data["sub_questions"],
                "strategy": "llm_generated",
                "iteration": iteration
            },
            "execution_log": [{
                "node": "plan",
                "iteration": iteration,
                "action": f"Generated {len(plan_data['sub_questions'])} search queries",
                "status": "success"
            }]
        }

    except Exception as e:
        logger.error(f"[plan_node] Planning failed: {type(e).__name__}: {str(e)}", exc_info=True)
        return {
            "current_plan": {
                "sub_questions": [task],
                "strategy": "fallback",
                "error": str(e)
            },
            "execution_log": [{
                "node": "plan",
                "iteration": iteration,
                "action": "Planning failed, falling back to original task as query",
                "status": "error",
                "error": str(e)
            }]
        }


def execute_search_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map-Reduce Worker: Processes a single search query.

    Invoked in parallel by the Send API (one instance per query). Each worker:
    1. Searches Tavily for the query (cached + retried)
    2. Deep-reads the top result URL via Jina Reader (25k chars)
    3. Falls back to Tavily snippet if scraping fails
    4. Returns partial state that LangGraph merges via operator.add

    When 3 workers run in parallel, each returns partial lists and LangGraph
    fans them in: {"visited_urls": ["url1..3"]} + {"visited_urls": ["url4..6"]}

    LangGraph merges them using operator.add:
    Final state: {"visited_urls": ["url1", ..., "url9"]}

    = ["url1..9"]} merged into global state with no race conditions.
    """
    logger = logging.getLogger(__name__)

    query = state.get("query", "")
    iteration = state.get("iteration", 0)

    logger.info(f"[execute_search_query] query='{query[:50]}...', iteration={iteration}")

    knowledge_base = []
    visited_urls = []
    search_queries = []
    failed_queries = []

    try:
        results = cached_tavily_search(query, max_results=3)
        result_count = len(results.get("results", []))
        logger.info(f"[execute_search_query] {result_count} results returned")

        for idx, result in enumerate(results.get("results", [])):
            url = result.get("url", "")
            title = result.get("title", "")
            snippet = result.get("content", "")

            visited_urls.append(url)

            # Deep-read top result; fallback to snippet for results #2 and #3
            if idx == 0:
                logger.info(f"[execute_search_query] DEEP READING: {url[:80]}...")
                full_content = scrape_url(url)

                if full_content:
                    CostTracker.track_scrape(url)
                    logger.info(f"[execute_search_query] Read {len(full_content)} chars from {url[:50]}...")
                    knowledge_base.append({
                        "query": query,
                        "content": f"### SOURCE: {title}\n**URL:** {url}\n\n{full_content}",
                        "url": url,
                        "title": title,
                        "iteration": iteration,
                        "deep_read": True
                    })
                else:
                    logger.warning(f"[execute_search_query] Scraping failed, using snippet: {url[:50]}...")
                    knowledge_base.append({
                        "query": query,
                        "content": snippet,
                        "url": url,
                        "title": title,
                        "iteration": iteration,
                        "deep_read": False
                    })
            else:
                knowledge_base.append({
                    "query": query,
                    "content": snippet,
                    "url": url,
                    "title": title,
                    "iteration": iteration,
                    "deep_read": False
                })

        search_queries.append(query)

    except Exception as e:
        logger.error(f"[execute_search_query] Search failed after retries: {type(e).__name__}: {str(e)}", exc_info=True)
        failed_queries.append(query)
        knowledge_base.append({
            "query": query,
            "content": f"Search failed after 3 retry attempts: {str(e)}",
            "url": "",
            "title": "ERROR",
            "iteration": iteration,
            "error": True
        })

    return {
        "knowledge_base": knowledge_base,
        "visited_urls": visited_urls,
        "search_queries": search_queries,
        "failed_queries": failed_queries,
        "execution_log": [{
            "node": "execute_search_query",
            "iteration": iteration,
            "query": query,
            "action": f"Searched query '{query}', found {len(knowledge_base)} results",
            "status": "success" if len(search_queries) > 0 else "failed"
        }]
    }


def synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesis Node: Scores all accumulated sources by credibility, then writes
    a research draft that prioritizes high-quality sources.
    """
    logger = logging.getLogger(__name__)

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    task = state.get("task", "")
    knowledge_base = state.get("knowledge_base", [])
    iteration = state.get("iteration_count", 0)

    logger.info(f"[synthesize_node] iteration={iteration}, sources={len(knowledge_base)}")

    if not knowledge_base:
        return {
            "current_draft": "No knowledge available to synthesize. Search may have failed.",
            "execution_log": [{
                "node": "synthesize",
                "iteration": iteration,
                "action": "Synthesis skipped (no knowledge available)",
                "status": "error"
            }]
        }

    logger.info(f"[synthesize_node] Scoring {len(knowledge_base)} sources...")

    scored_items = []
    for item in knowledge_base:
        if item.get("error", False):
            continue
        score_data = calculate_source_score(item.get("url", ""), item.get("content", ""))
        scored_items.append({
            **item,
            "credibility_score": score_data["score"],
            "credibility_reasons": score_data["reasons"]
        })

    scored_items.sort(key=lambda x: x.get("credibility_score", 0.5), reverse=True)

    top_scores = [f"{item.get('credibility_score', 0):.2f}" for item in scored_items[:3]]
    logger.info(f"[synthesize_node] Top 3 credibility scores: {top_scores}")

    # Format knowledge base for prompt (with scores visible)
    formatted_knowledge = "\n\n---\n\n".join([
        f"**CREDIBILITY SCORE: {item.get('credibility_score', 0.5):.2f}** ({', '.join(item.get('credibility_reasons', []))})\n"
        f"Source: {item.get('title', 'Unknown')}\n"
        f"URL: {item.get('url', '')}\n"
        f"Query: {item.get('query', '')}\n"
        f"Content: {item.get('content', '')}"
        for item in scored_items
    ])

    prompt = f"""You are a research synthesis assistant. Write a concise, well-structured research answer.

Research Task: {task}

Research Findings (sources ordered by credibility score):
{formatted_knowledge}

Write a research answer that:
1. Directly answers the research task
2. Synthesizes information from multiple sources
3. Cites sources using URLs in parentheses
4. Is 2-3 paragraphs maximum
5. Prioritizes HIGH CONFIDENCE sources (score >= 0.7) for key claims
6. Acknowledges uncertainty if only low-confidence sources are available
"""

    try:
        start_time = time.time()
        response = llm.invoke(prompt)
        duration = time.time() - start_time
        draft = response.content
        logger.info(f"[synthesize_node] Draft generated in {duration:.2f}s ({len(draft)} chars)")

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            CostTracker.track_llm(
                response.usage_metadata.get('input_tokens', 0),
                response.usage_metadata.get('output_tokens', 0)
            )

        return {
            "current_draft": draft,
            "execution_log": [{
                "node": "synthesize",
                "iteration": iteration,
                "action": f"Synthesized draft from {len(knowledge_base)} sources",
                "draft_length": len(draft),
                "status": "success"
            }]
        }

    except Exception as e:
        logger.error(f"[synthesize_node] Synthesis failed: {type(e).__name__}: {str(e)}", exc_info=True)
        return {
            "current_draft": f"Synthesis failed. Raw research findings:\n\n{formatted_knowledge}",
            "execution_log": [{
                "node": "synthesize",
                "iteration": iteration,
                "action": "Synthesis failed, returning raw data",
                "status": "error",
                "error": str(e)
            }]
        }


def critique_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critique Node: Evaluates draft quality and routes the agent loop.

    Priority order:
    1. User steering feedback → force quality_score=0.0 (triggers REFINE)
    2. Circuit breaker (max_iterations reached) → force quality_score=1.0 (triggers END)
    3. LLM quality evaluation → returns 0.0–1.0 score

    User steering works by manipulating quality_score rather than adding new graph edges,
    keeping the graph topology static while still directing the agent's behavior.
    """
    logger = logging.getLogger(__name__)

    task = state.get("task", "")
    draft = state.get("current_draft", "")
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    user_feedback = state.get("user_feedback")

    logger.info(f"[critique_node] iteration={iteration}, has_user_feedback={user_feedback is not None}")

    if user_feedback:
        print(f"\n[CRITIQUE] USER STEERING DETECTED")
        print(f"   Feedback: \"{user_feedback}\"")
        print(f"   Forcing REFINE to apply steering instructions...")
        logger.info(f"[critique_node] User steering: '{user_feedback[:100]}'")

        return {
            "quality_score": 0.0,
            "critique": {
                "score": 0.0,
                "message": f"USER INTERVENTION: {user_feedback}",
                "feedback": user_feedback,
                "passed": False,
                "user_steering": True
            },
            "execution_log": [{
                "node": "critique",
                "iteration": iteration,
                "action": f"User steering applied: '{user_feedback}'",
                "status": "user_feedback",
                "forced_refine": True
            }],
            "user_feedback": None  # Clear after processing — each feedback applies once
        }

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    if iteration >= max_iterations:
        logger.warning(f"[critique_node] Circuit breaker: iteration={iteration} >= max={max_iterations}")
        return {
            "quality_score": 1.0,
            "critique": {
                "score": 1.0,
                "message": f"Max iterations ({max_iterations}) reached. Accepting current draft.",
                "passed": True,
                "circuit_breaker": True
            },
            "execution_log": [{
                "node": "critique",
                "iteration": iteration,
                "action": f"Circuit breaker activated ({iteration} >= {max_iterations})",
                "status": "circuit_breaker"
            }]
        }

    prompt = f"""You are a research quality evaluator. Score the following draft objectively.

Original Task: {task}

Draft to Evaluate:
{draft}

Score on a scale of 0.0 to 1.0:
1. Answers the task directly and completely (0.4 weight)
2. Cites credible sources with URLs (0.3 weight)
3. Is well-structured and clear (0.3 weight)

Guide:
- 0.0-0.5: Poor (missing info, no sources, confusing)
- 0.5-0.7: Acceptable (lacks depth or sources)
- 0.7-0.85: Good (solid, could be better)
- 0.85-1.0: Excellent (comprehensive, well-sourced)

Return ONLY: {{"score": 0.85, "message": "Brief explanation"}}
"""

    try:
        start_time = time.time()
        response = llm.invoke(prompt)
        duration = time.time() - start_time
        logger.info(f"[critique_node] LLM responded in {duration:.2f}s")

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            CostTracker.track_llm(
                response.usage_metadata.get('input_tokens', 0),
                response.usage_metadata.get('output_tokens', 0)
            )

        response_text = response.content.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        critique_data = json.loads(response_text)
        score = max(0.0, min(1.0, float(critique_data["score"])))
        message = critique_data.get("message", "No message provided")
        threshold = state.get("quality_threshold", 0.85)
        passed = score > threshold

        logger.info(f"[critique_node] score={score:.2f}, threshold={threshold:.2f}, passed={passed}")

        return {
            "quality_score": score,
            "critique": {"score": score, "message": message, "passed": passed},
            "execution_log": [{
                "node": "critique",
                "iteration": iteration,
                "action": f"Quality score: {score:.2f} (passed={passed})",
                "status": "success"
            }]
        }

    except Exception as e:
        # Force pass on error to avoid an infinite loop
        logger.error(f"[critique_node] Critique failed: {type(e).__name__}: {str(e)}", exc_info=True)
        return {
            "quality_score": 1.0,
            "critique": {
                "score": 1.0,
                "message": f"Critique failed: {str(e)}. Forcing pass.",
                "passed": True,
                "error": True
            },
            "execution_log": [{
                "node": "critique",
                "iteration": iteration,
                "action": "Critique failed, forcing pass",
                "status": "error",
                "error": str(e)
            }]
        }


def refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refine Node: Increments the iteration counter and runs Detective Logic.

    After each iteration, acts as a "peer reviewer" — reads the current draft,
    identifies 2–3 specific knowledge gaps (missing questions), and saves them to
    the knowledge_gaps state field. plan_node reads these gaps next iteration and
    generates targeted follow-up queries, producing progressively deeper research.
    """
    logger = logging.getLogger(__name__)

    current_iteration = state.get("iteration_count", 0)
    new_iteration = current_iteration + 1
    draft = state.get("current_draft", "")
    task = state.get("task", "")
    knowledge_base = state.get("knowledge_base", [])

    logger.info(f"[refine_node] {current_iteration} -> {new_iteration}")

    if not draft or len(draft) < 100:
        logger.warning(f"[refine_node] Draft too short ({len(draft)} chars), skipping gap detection")
        return {
            "iteration_count": new_iteration,
            "knowledge_gaps": [],
            "execution_log": [{
                "node": "refine",
                "iteration": current_iteration,
                "action": f"{current_iteration} → {new_iteration} (draft too short for gap detection)"
            }]
        }

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.2,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    kb_summary = "\n".join([
        f"- {item.get('title', 'Unknown')} ({item.get('url', '')[:50]}...)"
        for item in knowledge_base[:10]
    ])

    detective_prompt = f"""You are a Research Detective. Your job is to identify knowledge gaps in a research draft.

RESEARCH QUESTION:
{task}

CURRENT DRAFT:
{draft}

SOURCES CONSULTED SO FAR:
{kb_summary}

YOUR TASK:
Analyze the draft and identify 2-3 **specific, answerable questions** that are:
1. Relevant to the research question
2. NOT already answered in the draft
3. Would significantly improve the draft if added

Return ONLY a JSON object (no markdown, no explanations):
{{
  "gaps": [
    "Specific question 1?",
    "Specific question 2?",
    "Specific question 3?"
  ]
}}

EXAMPLES OF GOOD GAPS:
- "What is the battery capacity in mAh for iPhone 17?"
- "How does the Pixel camera perform in low light conditions?"
- "What are the long-term reliability statistics for both phones?"

EXAMPLES OF BAD GAPS (too vague):
- "More info needed"  ← Not specific
- "Camera quality?"  ← Already covered in draft
- "What about price?"  ← Already answered

Be specific and actionable. These gaps will be used to generate new search queries.
    """

    try:
        logger.info(f"[refine_node] Running detective logic...")
        start_time = time.time()
        response = llm.invoke(detective_prompt)
        duration = time.time() - start_time
        logger.info(f"[refine_node] Detective LLM responded in {duration:.2f}s")

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            CostTracker.track_llm(
                response.usage_metadata.get('input_tokens', 0),
                response.usage_metadata.get('output_tokens', 0)
            )

        response_text = response.content.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        gaps_data = json.loads(response_text)
        identified_gaps = gaps_data.get("gaps", [])
        logger.info(f"[refine_node] Identified {len(identified_gaps)} gaps: {identified_gaps}")

        return {
            "iteration_count": new_iteration,
            "knowledge_gaps": identified_gaps,
            "execution_log": [{
                "node": "refine",
                "iteration": current_iteration,
                "action": f"{current_iteration} → {new_iteration}, {len(identified_gaps)} gaps identified",
                "gaps": identified_gaps,
                "status": "success"
            }]
        }

    except Exception as e:
        logger.error(f"[refine_node] Detective logic failed: {type(e).__name__}: {str(e)}", exc_info=True)
        return {
            "iteration_count": new_iteration,
            "knowledge_gaps": [],
            "execution_log": [{
                "node": "refine",
                "iteration": current_iteration,
                "action": f"{current_iteration} → {new_iteration} (gap detection failed, continuing)",
                "error": str(e),
                "status": "fallback"
            }]
        }
