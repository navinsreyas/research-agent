"""
Deep Research Agent — interactive CLI runner.

Usage:
    cd deep_research_agent
    python run_agent.py

Flow:
    1. Validate API keys (fail-fast before any API calls)
    2. Accept research question from the user
    3. Stream graph execution with real-time node-by-node output
    4. Pause before each critique for human review and optional steering
    5. Resume recursively until the quality threshold is met or the user quits
    6. Print the final draft, sources, and session cost breakdown
"""

import logging
import os
import time
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

from graph import create_research_graph
from langgraph.types import Command
from utils.logger import setup_logger
from utils.tracker import CostTracker


def validate_environment() -> bool:
    """
    Check required API keys before starting. Fails fast with clear error messages.

    Returns:
        True if all keys are present, False otherwise.
    """
    logger = logging.getLogger(__name__)
    load_dotenv()

    required_keys = {
        "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
        "TAVILY_API_KEY": "https://app.tavily.com/home"
    }

    missing_keys = []
    for key, help_url in required_keys.items():
        value = os.getenv(key)
        if not value or value.strip() == "":
            missing_keys.append((key, help_url))
            print(f"[X] Missing: {key}")
            print(f"   -> Get it at: {help_url}")
            logger.error(f"[validate_environment] Missing: {key}")
        else:
            print(f"[OK] Found: {key} ({value[:10]}...)")
            logger.info(f"[validate_environment] Found: {key}")

    if missing_keys:
        print("\n[WARN] Cannot start — add missing keys to your .env file")
        return False

    print("[OK] Environment validation passed\n")
    logger.info("[validate_environment] Passed")
    return True


def create_initial_state(task: str) -> Dict[str, Any]:
    """
    Build the initial ResearchState for a new research session.

    Args:
        task: The research question from the user.

    Returns:
        Complete state dict matching the ResearchState schema.
    """
    return {
        # Accumulative (operator.add appends across iterations and parallel workers)
        "search_queries": [],
        "visited_urls": [],
        "failed_queries": [],
        "raw_search_results": [],
        "knowledge_base": [],
        "execution_log": [],
        "knowledge_gaps": [],

        # Overwriting (replaced on each node update)
        "task": task,
        "current_draft": "",
        "quality_score": 0.0,
        "current_plan": {},
        "critique": {},
        "user_feedback": None,

        # Control
        "next_action": "start",
        "iteration_count": 0,
        "max_iterations": 3,
        "quality_threshold": 0.85
    }


def print_node_update(node_name: str, update_data: Dict[str, Any], iteration: int) -> None:
    """Pretty-print the relevant output from a single node update."""
    if node_name == "plan":
        plan = update_data.get("current_plan", {})
        queries = plan.get("sub_questions", [])
        print(f"\n[PLAN] Iteration {iteration}")
        print(f"   Generated {len(queries)} search queries:")
        for i, q in enumerate(queries, 1):
            print(f"     {i}. {q}")
        print()

    elif node_name == "execute_search_query":
        exec_log = update_data.get("execution_log", [])
        if exec_log:
            query = exec_log[0].get("query", "Unknown")
            status = exec_log[0].get("status", "unknown")
            kb_size = len(update_data.get("knowledge_base", []))
            emoji = "[OK]" if status == "success" else "[X]"
            print(f"   [SEARCH] {emoji} '{query}' -> {kb_size} results")

    elif node_name == "synthesize":
        draft = update_data.get("current_draft", "")
        print(f"\n[SYNTHESIZE] Iteration {iteration}")
        print(f"   Draft length: {len(draft)} characters")
        print()

    elif node_name == "critique":
        score = update_data.get("quality_score", 0.0)
        critique = update_data.get("critique", {})
        message = critique.get("message", "No message")
        circuit_breaker = critique.get("circuit_breaker", False)

        print(f"\n[CRITIQUE] Iteration {iteration}")
        print(f"   Quality Score: {score:.2f}")
        print(f"   Threshold: 0.85")

        if circuit_breaker:
            print(f"   [WARN] Circuit Breaker Activated — max iterations reached")

        print(f"   Feedback: {message}")

        if score > 0.85:
            print(f"   -> Decision: ACCEPT (score > 0.85)")
        else:
            print(f"   -> Decision: REFINE (score <= 0.85)")
        print()

    elif node_name == "refine":
        new_iter = update_data.get("iteration_count", 0)
        print(f"\n[REFINE]")
        print(f"   Moving to iteration {new_iter}")
        print(f"   Looping back to planning...\n")


def stream_graph_execution(graph, initial_state: Dict[str, Any], config: Dict[str, Any]):
    """
    Stream graph execution with real-time node-by-node output.

    Uses stream_mode="updates" so each event contains only the state fields
    that changed. Non-dict events (routing signals from conditional edges) are
    skipped — they are internal LangGraph events, not node state updates.

    Returns:
        StateSnapshot from graph.get_state(config) after streaming completes.
    """
    logger = logging.getLogger(__name__)
    thread_id = config["configurable"]["thread_id"]
    logger.info(f"[session_start] thread_id={thread_id}, task='{initial_state['task'][:80]}...'")

    print("=" * 70)
    print("RESEARCH AGENT EXECUTION")
    print("=" * 70)

    parallel_searches: Dict[int, int] = {}

    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, update_data in event.items():
            if not isinstance(update_data, dict):
                continue

            exec_log = update_data.get("execution_log", [])
            iteration = exec_log[0].get("iteration", 0) if exec_log else 0

            logger.info(f"[node_execution] node={node_name}, iteration={iteration}")

            if node_name == "execute_search_query":
                parallel_searches[iteration] = parallel_searches.get(iteration, 0) + 1

            print_node_update(node_name, update_data, iteration)

    if parallel_searches:
        print("\n[SUMMARY] Parallel searches per iteration:")
        for iter_num, count in sorted(parallel_searches.items()):
            print(f"   Iteration {iter_num}: {count} searches ran in parallel")

    return graph.get_state(config)


def handle_human_in_the_loop(graph, config: Dict[str, Any], state_snapshot) -> Tuple[bool, Any]:
    """
    Handle the graph's pause at the critique node (interrupt_before=["critique"]).

    Shows the current draft to the user and offers three options:
      - Type feedback  → injects it as user_feedback, forcing a REFINE iteration
      - Press Enter    → approves the draft, continues to quality evaluation
      - Type 'q/quit'  → accepts the current draft immediately and exits

    Called recursively after each resume to handle multiple HITL cycles
    (one pause per iteration until the graph finishes or the user quits).

    Returns:
        (should_continue: bool, final_state: dict)
    """
    logger = logging.getLogger(__name__)

    if not state_snapshot.next or "critique" not in state_snapshot.next:
        return (False, state_snapshot.values)

    current_state = state_snapshot.values
    draft = current_state.get("current_draft", "")
    iteration = current_state.get("iteration_count", 0)

    logger.info(f"[hitl_pause] iteration={iteration}, draft_length={len(draft)}")

    print("\n" + "=" * 70)
    print("[PAUSE] HUMAN-IN-THE-LOOP: Review Draft")
    print("=" * 70)
    print(f"\nIteration: {iteration}  |  Draft: {len(draft)} characters\n")
    print("Draft Preview (first 500 chars):")
    print("-" * 70)
    print(draft[:500])
    if len(draft) > 500:
        print(f"\n... ({len(draft) - 500} more characters)")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("OPTIONS:")
    print("  Type feedback  — steer the research (e.g. 'Focus on battery life')")
    print("  Press Enter    — approve and continue to quality evaluation")
    print("  Type 'q'       — accept this draft and stop")
    print("=" * 70)

    user_input = input("\nYour input: ").strip()

    if user_input.lower() in ['q', 'quit']:
        logger.info(f"[hitl_quit] User quit at iteration {iteration}")
        print("\n[STOP] Accepted current draft.")
        return (False, current_state)

    resume_state = current_state.copy()
    if user_input:
        logger.info(f"[hitl_feedback] Steering: '{user_input[:100]}'")
        print(f"\n[RESUME] Applying steering: \"{user_input}\"")
        resume_state["user_feedback"] = user_input
    else:
        logger.info("[hitl_approve] Approved — continuing to critique")
        print("\n[RESUME] Approved — proceeding to quality evaluation")
        resume_state["user_feedback"] = None

    for event in graph.stream(Command(resume=resume_state), config=config, stream_mode="updates"):
        for node_name, update_data in event.items():
            if not isinstance(update_data, dict):
                continue
            exec_log = update_data.get("execution_log", [])
            iteration = exec_log[0].get("iteration", 0) if exec_log else 0
            print_node_update(node_name, update_data, iteration)

    return handle_human_in_the_loop(graph, config, graph.get_state(config))


def run_interactive_research_agent():
    """Run one full research session: validate → question → stream → HITL → results."""
    print("\n" + "=" * 70)
    print("DEEP RESEARCH AGENT - Interactive Runner")
    print("=" * 70)
    print("\nThis agent will:")
    print("  * Break your question into sub-questions")
    print("  * Search the web in parallel (3x queries)")
    print("  * Synthesize a research draft")
    print("  * Pause for your review (human-in-the-loop)")
    print("  * Iterate until quality threshold is met")
    print("\n" + "=" * 70)

    if not validate_environment():
        return

    print("\nWhat would you like to research?")
    print("Example: 'What are the latest developments in quantum computing?'")
    task = input("\nYour question: ").strip()

    if not task:
        print("[X] Empty question — please provide a research topic")
        return

    print(f"\n[OK] Research Task: {task}")

    graph = create_research_graph()
    thread_id = f"research_{int(time.time())}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50
    }

    logger = logging.getLogger(__name__)
    logger.info(f"[session_init] thread_id={thread_id}, task='{task[:80]}...'")
    print(f"[OK] Session ID: {thread_id}")

    initial_state = create_initial_state(task)
    start_time = time.time()

    state_snapshot = stream_graph_execution(graph, initial_state, config)
    _, final_state = handle_human_in_the_loop(graph, config, state_snapshot)

    duration = time.time() - start_time
    visited_urls = final_state.get("visited_urls", [])
    unique_urls = list(set(visited_urls))
    iterations = final_state.get("iteration_count", 0)
    quality = final_state.get("quality_score", 0.0)
    draft = final_state.get("current_draft", "")

    logger.info(f"[session_complete] duration={duration:.1f}s, iterations={iterations+1}, "
                f"quality={quality:.2f}, urls={len(unique_urls)}, draft_len={len(draft)}")

    print("\n" + "=" * 70)
    print("[SUCCESS] RESEARCH COMPLETE")
    print("=" * 70)

    print(f"\n[STATS]")
    print(f"   URLs visited:    {len(visited_urls)} ({len(unique_urls)} unique)")
    print(f"   Iterations:      {iterations + 1}")
    print(f"   Quality score:   {quality:.2f}")
    print(f"   Execution time:  {duration:.1f}s")

    print(f"\n[DRAFT] Final Research Answer:")
    print("=" * 70)
    print(draft)
    print("=" * 70)

    if unique_urls:
        print(f"\n[SOURCES] ({len(unique_urls)} URLs):")
        for i, url in enumerate(unique_urls[:10], 1):
            print(f"   {i}. {url}")
        if len(unique_urls) > 10:
            print(f"   ... and {len(unique_urls) - 10} more")

    CostTracker.display_costs()

    print("\n" + "=" * 70)
    print("Thank you for using Deep Research Agent!")
    print("=" * 70)


def main():
    """Entry point with top-level error handling."""
    logger = setup_logger(__name__)
    logger.info("DEEP RESEARCH AGENT SESSION STARTING")

    try:
        run_interactive_research_agent()
    except KeyboardInterrupt:
        logger.warning("[session_interrupted] Ctrl+C")
        print("\n\n[WARN] Interrupted — exiting gracefully")
    except Exception as e:
        logger.error(f"[session_error] {type(e).__name__}: {str(e)}", exc_info=True)
        print(f"\n[X] Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
