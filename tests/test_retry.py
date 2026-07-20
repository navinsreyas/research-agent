"""
Retry / exponential backoff on the Tavily search (nodes/real_nodes.py).

The attempt count and delay bounds are read from the actual tenacity decorator
(`cached_tavily_search.retry`), not assumed. A mocked transient failure (429) is
retried and succeeds without raising. `time.sleep` is patched so the test is fast.
"""

import time
from unittest.mock import MagicMock

from tenacity import wait_exponential


def _get_retrying(func):
    # @disk_cache wraps @retry; functools.wraps copies the `.retry` attribute up,
    # but fall back to __wrapped__ just in case.
    return getattr(func, "retry", None) or func.__wrapped__.retry


def test_retry_decorator_config_matches_code():
    import nodes.real_nodes as real_nodes
    retrying = _get_retrying(real_nodes.cached_tavily_search)
    assert retrying.stop.max_attempt_number == 3        # stop_after_attempt(3)
    assert isinstance(retrying.wait, wait_exponential)
    assert retrying.wait.multiplier == 1                # wait_exponential(multiplier=1,
    assert retrying.wait.min == 2                       #   min=2,
    assert retrying.wait.max == 10                      #   max=10)


def test_transient_failure_is_retried_then_succeeds(tmp_path, monkeypatch):
    import utils.cache as cache
    import nodes.real_nodes as real_nodes

    monkeypatch.setattr(cache, "CACHE_FILE", str(tmp_path / "cache.json"))

    mock_client = MagicMock()
    mock_client.search.side_effect = [
        Exception("429 Too Many Requests"),   # transient failure 1
        Exception("429 Too Many Requests"),   # transient failure 2
        {"results": []},                       # success on 3rd attempt
    ]
    monkeypatch.setattr(real_nodes, "TavilyClient", lambda api_key=None: mock_client)

    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))   # record backoff, don't wait

    result = real_nodes.cached_tavily_search("q-retry-unique", max_results=3)

    assert result == {"results": []}            # recovered, no exception raised
    assert mock_client.search.call_count == 3   # 2 failures + 1 success == 3 attempts
    assert len(slept) == 2                        # a backoff pause between each attempt
    assert all(2 <= s <= 10 for s in slept)       # bounded by wait_exponential(min=2, max=10)
    assert slept == sorted(slept)                 # non-decreasing (exponential backoff)
