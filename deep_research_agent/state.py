"""
ResearchState schema for the Deep Research Agent.

Fields marked with operator.add accumulate across iterations and parallel workers
(lists append). All other fields overwrite on each update.
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional


class ResearchState(TypedDict):
    """
    State schema for the Deep Research Agent.

    Accumulative fields (operator.add — values append, never replace):
        search_queries   All search queries executed across iterations
        visited_urls     All URLs visited during research
        failed_queries   Queries that failed to return results
        raw_search_results  Raw search result objects
        knowledge_base   Extracted knowledge chunks with metadata
        execution_log    Log entries for debugging and auditability
        knowledge_gaps   Knowledge gaps identified by refine_node (detective logic)

    Overwriting fields (standard replace semantics):
        task             The research question
        current_draft    Latest version of the synthesized answer
        quality_score    Most recent quality score from critique_node (0.0–1.0)
        current_plan     Active research plan (sub_questions, strategy)
        critique         Most recent critique result
        user_feedback    Human steering instruction; cleared after one use

    Control fields:
        next_action        Internal routing hint
        iteration_count    Current iteration (incremented by refine_node)
        max_iterations     Circuit breaker limit
        quality_threshold  Score required to accept the draft and exit the loop
    """

    # Accumulative — operator.add merges lists from parallel workers and iterations
    search_queries: Annotated[List[str], operator.add]
    visited_urls: Annotated[List[str], operator.add]
    failed_queries: Annotated[List[str], operator.add]
    raw_search_results: Annotated[List[Dict[str, Any]], operator.add]
    knowledge_base: Annotated[List[Dict[str, Any]], operator.add]
    execution_log: Annotated[List[Dict[str, Any]], operator.add]
    knowledge_gaps: Annotated[List[str], operator.add]

    # Overwriting
    task: str
    current_draft: str
    quality_score: float
    current_plan: Dict[str, Any]
    critique: Dict[str, Any]
    user_feedback: Optional[str]

    # Control
    next_action: str
    iteration_count: int
    max_iterations: int
    quality_threshold: float
