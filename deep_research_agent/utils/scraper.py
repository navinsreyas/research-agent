"""
Web scraper utility using the Jina Reader API.

Fetches full article content from a URL as clean Markdown, stripping ads,
navigation, and boilerplate. Free tier supports ~1,000 requests/day.
"""

import logging
import requests


def scrape_url(url: str, max_chars: int = 25000) -> str:
    """
    Fetch article content from a URL via Jina Reader (https://r.jina.ai/).

    Returns clean Markdown truncated to max_chars. Returns an empty string on
    any failure so callers can fall back to Tavily snippets gracefully.

    Args:
        url:       The target URL.
        max_chars: Maximum characters to return (default 25,000 ~ 6,250 tokens).

    Returns:
        Markdown content string, or "" on failure.
    """
    logger = logging.getLogger(__name__)

    if not url or not url.startswith(("http://", "https://")):
        logger.warning(f"[scrape_url] Invalid URL: {url}")
        return ""

    jina_url = f"https://r.jina.ai/{url}"
    logger.info(f"[scrape_url] Fetching: {url[:100]}...")

    try:
        response = requests.get(
            jina_url,
            timeout=10,
            headers={"User-Agent": "DeepResearchAgent/1.0 (Educational Research Tool)"}
        )

        if response.status_code != 200:
            logger.warning(f"[scrape_url] HTTP {response.status_code} for {url}")
            return ""

        content = response.text.strip()
        if len(content) > max_chars:
            content = content[:max_chars]

        logger.info(f"[scrape_url] {len(content)} chars from {url[:60]}...")
        return content

    except requests.Timeout:
        logger.warning(f"[scrape_url] Timeout after 10s: {url[:100]}...")
        return ""
    except requests.RequestException as e:
        logger.warning(f"[scrape_url] Request failed: {type(e).__name__}: {str(e)[:100]}")
        return ""
    except Exception as e:
        logger.error(f"[scrape_url] Unexpected error: {type(e).__name__}: {str(e)[:100]}")
        return ""
