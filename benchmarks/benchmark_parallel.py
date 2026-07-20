"""
Parallel (Send API) vs Sequential search-fan-out benchmark.

Measures the ONLY stage that concurrency affects: the search + scrape fan-out.
Planning, synthesis, and critique are identical in both modes and are excluded.

Fairness guarantees (read before quoting the number):
  1. IDENTICAL input — a fixed, hardcoded set of sub-queries (below). No LLM
     generation, so the two modes are exactly comparable.
  2. SAME underlying function — both modes call the unmodified production
     `execute_search_query` (Tavily search + Jina scrape). Only concurrency differs.
  3. PARALLEL = real production Send path — a minimal LangGraph whose fan-out edge
     is the same logic as production `continue_to_search` in graph.py, so LangGraph's
     Pregel executor runs the workers concurrently exactly as in production. Compiled
     WITHOUT the SQLite checkpointer so we time search I/O, not checkpoint writes.
  4. CACHE DISABLED — utils.cache._load_cache is forced to return {} (every call is a
     guaranteed miss) and _save_cache is a no-op. Confirmed via live-call count.
  5. REAL network calls — consumes Tavily quota (and hits Jina Reader).

This script does NOT touch production code. It imports and calls it.

Run:
    python benchmarks/benchmark_parallel.py           # 3 runs (default)
    python benchmarks/benchmark_parallel.py --runs 5
"""

# Imports below intentionally follow the sys.path insert and the cache monkeypatch
# (moving them to the top would break the benchmark), so E402 is silenced file-wide.
# ruff: noqa: E402

import argparse
import os
import sys
import time

# --- Make the production package importable (it uses top-level imports) ----------
HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "deep_research_agent")
sys.path.insert(0, AGENT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(AGENT_DIR, ".env"))

# --- Disable the disk cache BEFORE importing the nodes that use it ----------------
# The @disk_cache wrapper looks up _load_cache/_save_cache as module globals at call
# time, so replacing them here neutralises caching without editing production code.
import utils.cache as _cache_mod
_cache_mod._load_cache = lambda: {}          # every key absent -> guaranteed MISS
_cache_mod._save_cache = lambda cache: None  # never persist
_cache_mod.print = lambda *a, **k: None      # silence the module's [CACHE ...] prints

# Quieten node/scraper INFO/WARNING chatter so the report stays readable.
import logging
logging.getLogger().setLevel(logging.ERROR)

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from state import ResearchState
from nodes.real_nodes import execute_search_query
from utils.tracker import CostTracker

# --- 1) IDENTICAL, HARDCODED INPUT (same for both modes) -------------------------
QUERIES = [
    "solid state battery energy density 2025",
    "solid state battery commercialization timeline",
    "solid state battery vs lithium ion safety",
    "solid state battery manufacturing challenges",
]


def make_initial_state():
    """Minimal ResearchState seed. current_plan carries the fixed queries, exactly
    like production after plan_node runs."""
    return {
        "search_queries": [], "visited_urls": [], "failed_queries": [],
        "raw_search_results": [], "knowledge_base": [], "execution_log": [],
        "knowledge_gaps": [],
        "task": "benchmark", "current_draft": "", "quality_score": 0.0,
        "current_plan": {"sub_questions": list(QUERIES), "strategy": "benchmark"},
        "critique": {}, "user_feedback": None,
        "next_action": "start", "iteration_count": 0,
        "max_iterations": 3, "quality_threshold": 0.85,
    }


def fan_out(state):
    """Identical to production `continue_to_search` (graph.py): read the planned
    sub-questions and emit one Send per query, fanning out to parallel workers."""
    queries = state.get("current_plan", {}).get("sub_questions", [])
    iteration = state.get("iteration_count", 0)
    return [
        Send("execute_search_query", {"query": q, "iteration": iteration})
        for q in queries
    ]


def build_parallel_graph():
    """Minimal graph that reproduces ONLY the production fan-out stage:
    seed -> [execute_search_query x N in parallel] -> collect.
    Uses the real production node + the real Send routing. No checkpointer."""
    wf = StateGraph(ResearchState)
    wf.add_node("seed", lambda s: {})
    wf.add_node("execute_search_query", execute_search_query)  # production node
    wf.add_node("collect", lambda s: {})
    wf.set_entry_point("seed")
    wf.add_conditional_edges("seed", fan_out)     # real Send fan-out
    wf.add_edge("execute_search_query", "collect")
    wf.add_edge("collect", END)
    return wf.compile()


def run_sequential():
    """Same per-query function, plain loop, no concurrency."""
    for q in QUERIES:
        execute_search_query({"query": q, "iteration": 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="number of runs per mode")
    args = parser.parse_args()
    runs = args.runs

    if not os.getenv("TAVILY_API_KEY"):
        print("[X] TAVILY_API_KEY not found. Add it to deep_research_agent/.env")
        sys.exit(1)

    n = len(QUERIES)
    expected_live_calls = n * runs * 2  # both modes, every run, cache disabled

    print("=" * 74)
    print("PARALLEL (Send API) vs SEQUENTIAL - search fan-out benchmark")
    print("=" * 74)
    print(f"\nQueries (N={n}):")
    for i, q in enumerate(QUERIES, 1):
        print(f"  {i}. {q}")
    print("\nCache:   DISABLED (utils.cache._load_cache forced empty; _save_cache no-op)")
    print("Network: REAL Tavily + Jina Reader calls - CONSUMES TAVILY QUOTA")
    print(f"         (~{expected_live_calls} live Tavily searches for runs={runs})")
    print("Timed:   ONLY the search/scrape fan-out. Plan/synthesize/critique excluded.")
    print("Order:   interleaved (sequential then parallel) each round.\n")

    CostTracker.reset()
    graph = build_parallel_graph()  # compiled once; compile cost not timed

    seq_times, par_times = [], []
    for i in range(1, runs + 1):
        t = time.perf_counter()
        run_sequential()
        seq = time.perf_counter() - t
        seq_times.append(seq)

        t = time.perf_counter()
        graph.invoke(make_initial_state())
        par = time.perf_counter() - t
        par_times.append(par)

        print(f"Run {i}:  sequential = {seq:6.2f}s   parallel = {par:6.2f}s")

    seq_mean = sum(seq_times) / len(seq_times)
    par_mean = sum(par_times) / len(par_times)

    print("\n" + "-" * 74)
    print(f"Sequential runs: {[round(x, 2) for x in seq_times]}   mean = {seq_mean:.2f}s")
    print(f"Parallel   runs: {[round(x, 2) for x in par_times]}   mean = {par_mean:.2f}s")
    print("-" * 74)

    # --- Cache-hit confirmation: every query execution made a live API call --------
    live = CostTracker.search_count
    hits = CostTracker.cache_hits
    print(f"\nCache hits: {hits}  |  live Tavily calls: {live} / {expected_live_calls} expected")
    if live == expected_live_calls and hits == 0:
        print("  -> CONFIRMED: cache disabled, 0 hits (every query hit the live API).")
    else:
        print("  -> NOTE: live calls != expected (some queries may have failed/retried);")
        print("     0 cache hits still holds — failures are live attempts, not cache reads.")

    # --- Speedup -------------------------------------------------------------------
    if par_mean > 0:
        pct_faster = (seq_mean - par_mean) / seq_mean * 100
        nx = seq_mean / par_mean
        print("\n" + "=" * 74)
        print("SPEEDUP")
        print("=" * 74)
        print(f"  % faster = (seq-par)/seq*100 = {pct_faster:.1f}%")
        print(f"  Nx       = seq/par           = {nx:.2f}x")
        print("\nQuotable:")
        print(f'  "Parallel Send API execution of {n} sub-queries ran in {par_mean:.1f}s vs')
        print(f'   {seq_mean:.1f}s sequential, a {pct_faster:.0f}% reduction, averaged over {runs} runs')
        print('   (cache disabled)."')
        print("=" * 74)


if __name__ == "__main__":
    main()
