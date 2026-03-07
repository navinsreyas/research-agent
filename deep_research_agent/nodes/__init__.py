"""
Nodes package for the Deep Research Agent.

Default imports use real_nodes (live APIs).
Mock nodes are available for testing via: from nodes import mock_nodes
"""

from .real_nodes import (
    plan_node,
    execute_search_query,
    synthesize_node,
    critique_node,
    refine_node
)
from . import mock_nodes

__all__ = [
    'plan_node',
    'execute_search_query',
    'synthesize_node',
    'critique_node',
    'refine_node',
    'mock_nodes'
]
