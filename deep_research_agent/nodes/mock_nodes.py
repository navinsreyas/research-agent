"""
Mock Nodes for Phase 2: The Mock Logic Loop

These nodes simulate the research cycle with predictable behaviors
to test loop mechanics and state accumulation.

Each node represents a step in the "System 2" thinking process:
Plan -> Search -> Synthesize -> Critique -> Refine -> Loop
"""

from typing import Dict, Any


def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planning Node: Creates a mock research plan.

    This simulates the LLM breaking down a complex query into sub-questions.

    Args:
        state: Current ResearchState

    Returns:
        dict: Partial state update with research plan and log entry
    """
    return {
        "current_plan": {
            "sub_questions": [
                "What is X?",
                "Why is X?"
            ],
            "strategy": "mock_search_strategy"
        },
        "execution_log": [{
            "node": "plan",
            "iteration": state.get("iteration_count", 0),
            "action": "Planning started"
        }]
    }


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search Node: Simulates web searches with iteration-specific URLs.

    CRUCIAL TEST: Returns different URLs based on iteration_count.
    This proves operator.add accumulates across iterations.

    - Iteration 0: Returns ["mock_url_1.com"]
    - Iteration 1: Returns ["mock_url_2.com"]

    Args:
        state: Current ResearchState

    Returns:
        dict: Partial state update with visited URLs and log entry
    """
    iteration = state.get("iteration_count", 0)

    # Different URL per iteration to test accumulation
    if iteration == 0:
        new_url = "mock_url_1.com"
    else:
        new_url = "mock_url_2.com"

    return {
        "visited_urls": [new_url],
        "execution_log": [{
            "node": "search",
            "iteration": iteration,
            "action": f"Searched and visited {new_url}"
        }]
    }


def synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesis Node: Creates a draft from accumulated knowledge.

    Simulates the LLM writing a research document based on findings.
    Each iteration produces a different draft version.

    Args:
        state: Current ResearchState

    Returns:
        dict: Partial state update with current draft and log entry
    """
    iteration = state.get("iteration_count", 0)

    return {
        "current_draft": f"Draft for iteration {iteration}",
        "execution_log": [{
            "node": "synthesize",
            "iteration": iteration,
            "action": "Synthesized research findings"
        }]
    }


def critique_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critique Node: Evaluates draft quality (THE LOGIC DRIVER).

    This node controls the loop behavior:
    - Iteration 0: Returns quality_score = 0.5 (FORCE FAIL -> triggers loop)
    - Iteration 1+: Returns quality_score = 0.9 (FORCE PASS -> triggers end)

    This ensures we test both:
    1. The "refine" edge (loop back)
    2. The "end" edge (finish)

    Args:
        state: Current ResearchState

    Returns:
        dict: Partial state update with quality score, critique, and log entry
    """
    iteration = state.get("iteration_count", 0)

    # The scripted behavior that drives our test
    if iteration < 1:
        # First pass: Force failure to trigger loop
        quality_score = 0.5
        critique_message = "Too short - needs more detail"
    else:
        # Second pass: Force success to trigger end
        quality_score = 0.9
        critique_message = "Good job - meets quality standards"

    return {
        "quality_score": quality_score,
        "critique": {
            "score": quality_score,
            "message": critique_message,
            "passed": quality_score > 0.85
        },
        "execution_log": [{
            "node": "critique",
            "iteration": iteration,
            "action": f"Evaluated quality: {quality_score}",
            "message": critique_message
        }]
    }


def refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refinement Node: Increments iteration and prepares for next loop.

    This node is only called when quality is insufficient.
    It updates the iteration counter and logs the refinement action.

    Args:
        state: Current ResearchState

    Returns:
        dict: Partial state update with incremented iteration and log entry
    """
    current_iteration = state.get("iteration_count", 0)
    new_iteration = current_iteration + 1

    return {
        "iteration_count": new_iteration,
        "execution_log": [{
            "node": "refine",
            "iteration": current_iteration,
            "action": f"Refining plan... moving to iteration {new_iteration}"
        }]
    }
