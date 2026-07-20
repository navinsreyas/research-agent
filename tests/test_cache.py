"""
Disk cache (utils/cache.py) via the real production `cached_tavily_search`.

Two identical queries -> the network layer is hit exactly ONCE; the second query is
served from disk. The network (TavilyClient) is mocked; no real web calls.
"""

from unittest.mock import MagicMock


def test_identical_queries_hit_disk_cache_once(tmp_path, monkeypatch):
    import utils.cache as cache
    import nodes.real_nodes as real_nodes

    # Isolate the cache file to a fresh temp path (no cross-test contamination,
    # and it starts empty so the first call is a guaranteed miss).
    monkeypatch.setattr(cache, "CACHE_FILE", str(tmp_path / "cache.json"))

    # Mock the network layer: TavilyClient(...).search(...) -> fixed serializable dict.
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": [{"url": "u", "title": "t", "content": "c"}]}
    monkeypatch.setattr(real_nodes, "TavilyClient", lambda api_key=None: mock_client)

    r1 = real_nodes.cached_tavily_search("solid state battery 2025", max_results=3)
    r2 = real_nodes.cached_tavily_search("solid state battery 2025", max_results=3)

    assert r1 == r2
    assert mock_client.search.call_count == 1   # 2nd identical query served from disk
