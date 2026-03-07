"""
Session cost tracker for the Deep Research Agent.

Aggregates Tavily search, Jina scrape, and Claude token usage across the
entire session using class-level variables (singleton pattern). Call
CostTracker.display_costs() at the end of a session to print a summary.

Pricing (as of 2026-02):
  Tavily:  $0.001 per search
  Claude Sonnet 4.5:  $3.00 / 1M input tokens, $15.00 / 1M output tokens
  Jina Reader: free
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Singleton cost tracker. All state is stored in class variables so any
    node can call CostTracker.track_*() without passing an instance around.

    Usage:
        CostTracker.track_search()
        CostTracker.track_scrape("https://example.com")
        CostTracker.track_llm(input_tokens=1500, output_tokens=300)
        CostTracker.display_costs()
    """

    # Pricing constants
    COST_PER_SEARCH = 0.001
    COST_PER_INPUT_TOKEN = 3.00 / 1_000_000
    COST_PER_OUTPUT_TOKEN = 15.00 / 1_000_000
    COST_PER_SCRAPE = 0.0

    # Counters
    search_count = 0
    cache_hits = 0
    scrape_count = 0
    scraped_urls: list = []
    input_tokens = 0
    output_tokens = 0
    llm_calls = 0

    @classmethod
    def reset(cls):
        """Reset all counters. Call at the start of a new session."""
        cls.search_count = 0
        cls.cache_hits = 0
        cls.scrape_count = 0
        cls.scraped_urls = []
        cls.input_tokens = 0
        cls.output_tokens = 0
        cls.llm_calls = 0
        logger.debug("[tracker] Reset")

    @classmethod
    def track_search(cls, from_cache: bool = False):
        """Record a search. Cache hits are counted separately (no API cost)."""
        if from_cache:
            cls.cache_hits += 1
        else:
            cls.search_count += 1

    @classmethod
    def track_scrape(cls, url: str):
        """Record a Jina Reader scrape."""
        cls.scrape_count += 1
        cls.scraped_urls.append(url)

    @classmethod
    def track_llm(cls, input_tokens: int, output_tokens: int):
        """Record token usage from a Claude API call."""
        cls.input_tokens += input_tokens
        cls.output_tokens += output_tokens
        cls.llm_calls += 1
        logger.debug(f"[tracker] LLM call #{cls.llm_calls}: {input_tokens} in, {output_tokens} out")

    @classmethod
    def get_costs(cls) -> Dict[str, float]:
        """Return estimated costs by category."""
        search_cost = cls.search_count * cls.COST_PER_SEARCH
        scrape_cost = cls.scrape_count * cls.COST_PER_SCRAPE
        llm_cost = (cls.input_tokens * cls.COST_PER_INPUT_TOKEN +
                    cls.output_tokens * cls.COST_PER_OUTPUT_TOKEN)
        return {
            "search_cost": round(search_cost, 4),
            "scrape_cost": round(scrape_cost, 4),
            "llm_cost": round(llm_cost, 4),
            "total_cost": round(search_cost + scrape_cost + llm_cost, 4)
        }

    @classmethod
    def display_costs(cls):
        """Print a formatted cost breakdown to the console."""
        costs = cls.get_costs()
        total_searches = cls.search_count + cls.cache_hits
        cache_savings = cls.cache_hits * cls.COST_PER_SEARCH
        cache_hit_rate = (cls.cache_hits / total_searches * 100) if total_searches > 0 else 0

        print("\n" + "=" * 70)
        print("SESSION COST BREAKDOWN")
        print("=" * 70)

        print("\nSEARCHES:")
        print(f"  Actual API calls:  {cls.search_count} × ${cls.COST_PER_SEARCH:.3f} = ${costs['search_cost']:.4f}")
        if cls.cache_hits > 0:
            print(f"  Cache hits:        {cls.cache_hits} (saved ${cache_savings:.4f} | {cache_hit_rate:.1f}% hit rate)")

        print("\nSCRAPING (Jina Reader):")
        print(f"  URLs scraped:      {cls.scrape_count} (free)")

        print("\nLLM (Claude Sonnet 4.5):")
        if cls.llm_calls > 0:
            print(f"  API calls:         {cls.llm_calls}")
            print(f"  Input tokens:      {cls.input_tokens:,} × ${cls.COST_PER_INPUT_TOKEN:.6f} = ${cls.input_tokens * cls.COST_PER_INPUT_TOKEN:.4f}")
            print(f"  Output tokens:     {cls.output_tokens:,} × ${cls.COST_PER_OUTPUT_TOKEN:.6f} = ${cls.output_tokens * cls.COST_PER_OUTPUT_TOKEN:.4f}")
            print(f"  LLM subtotal:      ${costs['llm_cost']:.4f}")
        else:
            print("  No LLM calls recorded")

        print("\n" + "=" * 70)
        print(f"TOTAL ESTIMATED COST:  ${costs['total_cost']:.4f}")
        print("=" * 70)
        print("\nNote: Estimates may vary slightly from actual billing.")

        logger.info(f"[tracker] Session total: ${costs['total_cost']:.4f} "
                    f"({cls.search_count} searches, {cls.scrape_count} scrapes, {cls.llm_calls} LLM calls)")

    @classmethod
    def get_summary(cls) -> Dict:
        """Return full session metrics as a dict (for logging or analytics)."""
        costs = cls.get_costs()
        total = cls.search_count + cls.cache_hits
        return {
            "searches": {
                "actual": cls.search_count,
                "cached": cls.cache_hits,
                "total": total,
                "cache_hit_rate": (cls.cache_hits / total * 100) if total > 0 else 0
            },
            "scrapes": {
                "count": cls.scrape_count,
                "urls": cls.scraped_urls
            },
            "llm": {
                "calls": cls.llm_calls,
                "input_tokens": cls.input_tokens,
                "output_tokens": cls.output_tokens,
                "total_tokens": cls.input_tokens + cls.output_tokens
            },
            "costs": costs
        }
