"""
The career-development counterpart to historical_data.py / technology_data.py
— same idea again: a small curated dataset, this time of realistic entry-path
patterns for a handful of directions (ML engineering, AI research, quant,
startups, plus two adjacent baseline paths), so "next steps" can be grounded
in how people actually break into a field instead of generic advice.

This dataset is pattern-level (what kind of preparation and portfolio
signals matter, what the common entry roles and pitfalls are), not
labor-market statistics — treat it the same way as the other two datasets:
a small hand-curated starting point to verify and expand, not an
authoritative source. Specific job openings and course names should come
from web_search, not this file, since those change constantly and this
file doesn't.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "career_paths.json"
_PATHS = json.loads(_DATA_PATH.read_text())

VALID_DIRECTIONS = sorted({p["direction"] for p in _PATHS})


def _format(entries: list[dict]) -> str:
    parts = []
    for p in entries:
        parts.append(
            f"## {p['id'].replace('_', ' ').title()} (direction: {p['direction']})\n"
            f"Typical entry path: {p['typical_entry_path']}\n"
            f"Common prerequisites: {p['common_prerequisites']}\n"
            f"Common first roles: {p['common_first_roles']}\n"
            f"Common pitfalls: {p['common_pitfalls']}\n"
        )
    return "\n".join(parts)


def get_path_pattern(direction: str) -> str:
    """
    Return every dataset entry matching a stated direction (case-insensitive
    substring match against the 'direction' field and the entry name/tags)
    — a direction like "ML" or "quant" will match all relevant entries,
    including adjacent ones worth knowing about (e.g. "ML" also surfaces
    the Data Scientist entry, since it's a common confusion).
    """
    d = direction.lower().strip()
    matches = [
        p for p in _PATHS
        if d in p["direction"].lower() or d in p["id"] or any(d in t for t in p["tags"])
    ]
    if not matches:
        return (
            f"No dataset entry matches '{direction}'. Known directions: "
            f"{', '.join(VALID_DIRECTIONS)}. Say plainly this direction isn't "
            f"covered by the dataset rather than inventing a pattern for it."
        )
    return _format(matches)


def search_path_patterns(query: str, top_n: int = 3) -> str:
    """Flexible keyword search across all fields, for when the stated
    direction doesn't map cleanly onto one of the known categories."""
    query_words = {w for w in query.lower().split() if len(w) > 2}

    def score(p: dict) -> int:
        haystack = " ".join(
            [p["direction"], p["typical_entry_path"], p["common_prerequisites"], " ".join(p["tags"])]
        ).lower()
        return sum(1 for w in query_words if w in haystack)

    scored = sorted(((p, score(p)) for p in _PATHS), key=lambda ps: ps[1], reverse=True)
    top = [p for p, s in scored[:top_n] if s > 0]

    if not top:
        return "No dataset entry matched this query well. Say so rather than forcing a fit."
    return _format(top)
