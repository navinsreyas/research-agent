"""
Disk-based caching for API calls.

Caches function results to cache.json using MD5-hashed keys derived from
the function name and arguments. Repeated searches are served instantly at
zero cost instead of hitting the Tavily API again.

Cache invalidation: manual only (delete cache.json to start fresh).
Thread safety: single-threaded use only (no concurrent writes).
"""

import json
import hashlib
import os
import logging
from functools import wraps
from typing import Any, Callable

CACHE_FILE = "cache.json"

logger = logging.getLogger(__name__)


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a deterministic MD5 cache key from function name + arguments."""
    key_string = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
    return hashlib.md5(key_string.encode()).hexdigest()


def _load_cache() -> dict:
    """Load cache from disk. Returns empty dict if file missing or corrupt."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            logger.debug(f"[cache] Loaded {len(cache)} entries")
            return cache
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"[cache] Corrupt cache, starting fresh: {e}")
        return {}


def _save_cache(cache: dict) -> None:
    """Persist cache to disk. Silently skips on I/O failure (cache is optional)."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
            logger.debug(f"[cache] Saved {len(cache)} entries")
    except IOError as e:
        logger.error(f"[cache] Failed to save cache: {e}")


def disk_cache(func: Callable) -> Callable:
    """
    Decorator: cache function results to disk (cache.json).

    Requirements:
    - All arguments must be JSON-serializable
    - Return value must be JSON-serializable
    - Function must be deterministic (same inputs → same output)

    Usage:
        @disk_cache
        def expensive_api_call(query: str) -> dict:
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = _generate_cache_key(func.__name__, args, kwargs)
        cache = _load_cache()

        if cache_key in cache:
            logger.info(f"[CACHE HIT] {func.__name__}")
            print(f"[CACHE HIT] {func.__name__}")
            return cache[cache_key]

        logger.info(f"[CACHE MISS] {func.__name__} — calling API")
        result = func(*args, **kwargs)

        try:
            cache[cache_key] = result
            _save_cache(cache)
        except (TypeError, ValueError) as e:
            logger.warning(f"[cache] Result not serializable, skipping cache: {e}")

        return result

    return wrapper


def clear_cache() -> None:
    """Delete the cache file to force fresh API calls on next run."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        logger.info(f"[cache] Cleared: {CACHE_FILE}")
        print("[CACHE] Cache cleared successfully")
    else:
        print("[CACHE] No cache file found")


def get_cache_stats() -> dict:
    """Return entry count and file size of the current cache."""
    if not os.path.exists(CACHE_FILE):
        return {"entries": 0, "size_bytes": 0, "size_mb": 0.0}

    cache = _load_cache()
    size_bytes = os.path.getsize(CACHE_FILE)

    return {
        "entries": len(cache),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2)
    }
