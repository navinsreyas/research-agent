"""
Shared test setup.

The production package uses top-level imports (`from utils.scoring import ...`,
`import graph`, etc.), which assumes `deep_research_agent/` is on sys.path — that's
how it runs in production. We replicate that here so tests can import the real code
without any refactor.
"""

import os
import sys

AGENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "deep_research_agent")
)
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)
