"""
Source credibility scorer (utils/scoring.py).

Actual rule (read from the code, not assumed):
    base 0.5 + authority +0.2 + freshness +0.2 (mentions 2025/2026) + depth +0.1 (>5000 chars)
So the ONLY achievable scores are {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}. 0.85 is impossible.
"""

import pytest

from utils.scoring import calculate_source_score

LONG = "x" * 5001          # > 5000 chars -> depth bonus
SHORT = "short content"    # < 5000 chars, no year -> no bonuses
YEAR = "In 2025 the field advanced. "   # freshness bonus


def test_high_authority_all_bonuses():
    # nature.com + mentions 2025 + >5000 chars => 0.5 +.2 +.2 +.1 = 1.0
    r = calculate_source_score("https://www.nature.com/articles/x", YEAR + LONG)
    assert r["score"] == pytest.approx(1.0)


def test_authority_only():
    # .edu domain, short, no year => 0.5 +.2 = 0.7
    r = calculate_source_score("https://web.mit.edu/paper", SHORT)
    assert r["score"] == pytest.approx(0.7)


def test_plain_com_base_only():
    # non-authority domain, short, no year => 0.5 (base only)
    r = calculate_source_score("https://techblog.example.com/post", SHORT)
    assert r["score"] == pytest.approx(0.5)


def test_achievable_score_set_has_no_impossible_values():
    auth = "https://www.nature.com/x"
    plain = "https://techblog.example.com/x"
    cases = {
        "none":             (plain, SHORT),
        "depth":            (plain, LONG),
        "fresh":            (plain, YEAR + SHORT),
        "fresh_depth":      (plain, YEAR + LONG),
        "auth":             (auth,  SHORT),
        "auth_depth":       (auth,  LONG),
        "auth_fresh":       (auth,  YEAR + SHORT),
        "auth_fresh_depth": (auth,  YEAR + LONG),
    }
    # round() neutralises float accumulation noise (e.g. 0.7+0.2+0.1)
    produced = {round(calculate_source_score(u, c)["score"], 10) for u, c in cases.values()}
    assert produced == {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    assert 0.85 not in produced    # the impossible combo must never appear
