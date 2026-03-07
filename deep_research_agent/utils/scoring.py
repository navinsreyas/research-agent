"""
Source credibility scoring for the Deep Research Agent.

Scores each source on a 0.0-1.0 scale using three heuristics:
  Base:      0.5  (neutral prior for any source)
  Authority: +0.2 (known credible domain — .edu, .gov, Reuters, etc.)
  Freshness: +0.2 (content mentions the current or previous year)
  Depth:     +0.1 (content exceeds 5,000 characters)
  Max:       1.0
"""

import logging
import re
from typing import Dict, List


def calculate_source_score(url: str, content: str) -> Dict:
    """
    Calculate a credibility score for a source.

    Args:
        url:     Source URL used to extract domain for authority check.
        content: Full text content used for freshness and depth checks.

    Returns:
        dict with keys:
            score   float 0.0-1.0
            reasons list[str] human-readable explanation of score components
    """
    logger = logging.getLogger(__name__)

    score = 0.5
    reasons = []

    domain = ""
    try:
        match = re.search(r'https?://([^/]+)', url)
        if match:
            domain = match.group(1).lower()
    except Exception:
        pass

    logger.debug(f"[score_source] domain={domain}, content_len={len(content)}")

    # Authority: known credible domains
    authority_domains = [
        ".edu", ".gov", ".org",
        "reuters.com", "bloomberg.com", "nature.com",
        "science.org", "arxiv.org", "ieee.org",
        "nytimes.com", "washingtonpost.com", "wsj.com"
    ]
    if any(auth in domain for auth in authority_domains):
        score += 0.2
        reasons.append(f"Authority: {domain}")

    # Freshness: mentions current or previous year
    current_year = 2026
    if any(str(y) in content for y in [current_year, current_year - 1]):
        score += 0.2
        reasons.append(f"Fresh: mentions {current_year} or {current_year - 1}")

    # Depth: substantial content
    if len(content) > 5000:
        score += 0.1
        reasons.append(f"Deep: {len(content)} chars")

    score = max(0.0, min(1.0, score))
    logger.info(f"[score_source] {score:.2f} for {domain} — {reasons}")

    return {"score": score, "reasons": reasons}
