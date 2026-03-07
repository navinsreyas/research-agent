"""
Nodes package for Deep Research Agent

This package contains all node logic for the research agent.
Phase 3: Real API integrations (ChatAnthropic + Tavily)

The default imports now use real_nodes with live APIs.
Mock nodes are still available via: from nodes import mock_nodes
"""

# Import real nodes by default (Phase 3 Senior Upgrades)
from .real_nodes import (
    plan_node,
    search_node,  # Deprecated: kept for backward compatibility
    execute_search_query,  # NEW: Map-reduce worker for parallel execution
    synthesize_node,
    critique_node,
    refine_node
)

# Keep mock nodes available for testing
from . import mock_nodes

__all__ = [
    'plan_node',
    'search_node',  # Deprecated
    'execute_search_query',  # NEW
    'synthesize_node',
    'critique_node',
    'refine_node',
    'mock_nodes'  # Available as nodes.mock_nodes
]
