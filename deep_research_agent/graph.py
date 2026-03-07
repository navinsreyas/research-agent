"""
Graph Compilation Module for Deep Research Agent

This module compiles the LangGraph state graph with all nodes and edges,
including SQLite persistence for production use.

Production Features:
- Cyclic graph with conditional routing
- Parallel execution (map-reduce pattern)
- SQLite checkpointing (durable persistence)
- Human-in-the-loop pauses

Graph Flow:
    plan -> [execute_search_query (parallel)] -> synthesize -> **PAUSE** -> critique -> [conditional]
                                                                                         ├─> END (if quality > 0.85)
                                                                                         └─> refine -> [loop back to plan]
"""

from typing import Literal, Dict, Any
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Send
from state import ResearchState
from nodes import (
    plan_node,
    execute_search_query,
    synthesize_node,
    critique_node,
    refine_node
)


def should_continue(state: Dict[str, Any]) -> Literal["refine", "end"]:
    """
    Router function: Decides whether to continue iterating or finish.

    This is a conditional edge that implements the "System 2" loop logic.
    It checks the quality score from the critique node and decides:
    - If quality is good enough (> 0.85): End the process
    - If quality is insufficient: Refine and loop back

    Args:
        state: Current ResearchState containing quality_score

    Returns:
        "end": Terminate the graph (quality threshold met)
        "refine": Continue to refinement node (loop back for improvement)

    Why 0.85?
    This threshold ensures we only accept high-quality outputs. In production,
    this could be configurable via state['quality_threshold'].
    """
    quality_score = state.get("quality_score", 0.0)
    quality_threshold = state.get("quality_threshold", 0.85)

    if quality_score > quality_threshold:
        return "end"
    else:
        return "refine"


def create_research_graph():
    """
    Create and compile the research agent graph with parallel execution, SQLite persistence,
    and human-in-the-loop.

    Production Features:
    - Map-Reduce Pattern: plan_node returns Send objects → parallel execute_search_query workers
    - SQLite Persistence: SqliteSaver enables durable checkpointing (survives restarts)
    - Human-in-the-Loop: interrupt_before=["critique"] pauses for user review

    Graph Structure Explanation:
    1. Entry Point: "plan" - Every research cycle starts with planning
    2. Map-Reduce: Plan → [execute_search_query (parallel)] → Synthesize
       (Plan fans out to multiple workers, results fan in to synthesize)
    3. Human-in-the-Loop: Synthesize → **PAUSE** → Critique
       (User can review draft before quality evaluation)
    4. Conditional Edge: Critique → Router → (End OR Refine)
       (The router checks quality and decides whether to finish or loop)
    5. Loop Edge: Refine → Plan
       (If quality is low, we refine and start a new iteration)

    Returns:
        Compiled LangGraph with parallel execution, SQLite checkpointing, and human-in-the-loop
    """
    # Initialize the state graph with ResearchState schema
    workflow = StateGraph(ResearchState)

    # Add all nodes to the graph
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_search_query", execute_search_query)  # NEW: Map-reduce worker
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("refine", refine_node)

    # Set the entry point (where the graph starts)
    workflow.set_entry_point("plan")

    # Map-Reduce Edges using conditional_edges
    # Why conditional_edges instead of add_edge? Because plan_node returns Send objects
    # The conditional edge function receives state and returns the Send objects
    # LangGraph then routes to execute_search_query nodes in parallel
    def continue_to_search(state: Dict[str, Any]):
        """
        Route from plan to parallel search workers.

        Why this function? Conditional edges can return Send objects for map-reduce.
        Each Send creates an isolated execution of execute_search_query with one query.

        Why minimal state in Send? Only pass what's needed for that specific worker.
        The plan is already in global state from plan_node.
        """
        plan = state.get("current_plan", {})
        queries = plan.get("sub_questions", [])
        iteration = state.get("iteration_count", 0)

        return [
            Send(
                "execute_search_query",
                {
                    "query": query,
                    "iteration": iteration
                }
            )
            for query in queries
        ]

    workflow.add_conditional_edges("plan", continue_to_search)

    # Fan-In Edge: All parallel workers → synthesize
    workflow.add_edge("execute_search_query", "synthesize")

    # Linear edges (unchanged)
    workflow.add_edge("synthesize", "critique")

    # Add conditional edge from critique (the decision point)
    # This uses our router function to decide: end or refine?
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "end": END,      # If quality is good, finish
            "refine": "refine"  # If quality is poor, go to refine
        }
    )

    # Add loop edge (refine goes back to plan for another iteration)
    workflow.add_edge("refine", "plan")

    # ========================================================================
    # PRODUCTION PERSISTENCE: SQLite Checkpointing
    # ========================================================================
    #
    # Why SQLite instead of MemorySaver?
    # - PERSISTENCE: Sessions survive script restarts (data saved to disk)
    # - RECOVERY: Can resume interrupted research sessions
    # - AUDIT: Full history of state changes stored permanently
    # - PRODUCTION-READY: No external database needed (PostgreSQL optional for scale)
    #
    # Why check_same_thread=False?
    # - CRITICAL for parallel execution (map-reduce pattern)
    # - Multiple execute_search_query workers run simultaneously in different threads
    # - Without this flag, SQLite throws "objects created in a thread can only be used
    #   in that same thread" errors when parallel workers try to checkpoint
    # - Trade-off: Slightly less safe for concurrent writes, but LangGraph handles
    #   synchronization internally, so this is safe for our use case
    #
    # Database Location: checkpoints.sqlite (created in project root)
    # - Auto-created on first run
    # - Can be deleted to reset all checkpoints (fresh start)
    # - Can be inspected with `sqlite3 checkpoints.sqlite` for debugging
    #
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Compile with checkpointing and interrupts
    # Why interrupt_before=["critique"]? Pauses BEFORE critique node executes
    # This allows user to review the draft after synthesis
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["critique"]  # Pause BEFORE critique node executes
    )

    return graph


if __name__ == "__main__":
    graph = create_research_graph()
    print("Graph compiled successfully.")
    print("\nFlow: plan → [execute_search_query ×3 parallel] → synthesize → PAUSE → critique")
    print("             └── quality > 0.85 → END")
    print("             └── quality <= 0.85 → refine → plan (loop)")
