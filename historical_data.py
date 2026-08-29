"""
A tiny "historical analogues" search tool.

This is intentionally simple — no embeddings, no vector DB — just keyword
overlap scoring against a curated JSON file of past cases. That's the point:
it's not trying to be a smart search engine, it's trying to be a fixed,
checkable source of ground truth the agent can cite instead of pattern-
matching purely from its own training data. Simple and inspectable beats
clever and opaque for something meant to keep the model honest.

The dataset (historical_cases.json) is a small hand-curated seed set of
~18 well-known cases — expand it as you find good analogues for the topics
you actually care about. Treat it as a starting point to verify and grow,
not an authoritative historical record.
"""

import json
import re
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "historical_cases.json"
_CASES = json.loads(_DATA_PATH.read_text())


def _end_year(case: dict) -> int | None:
    """
    The last year a case's period covers, or None if it's open-ended
    ("...-present"). Used for cutoff filtering (backtest.py) — an
    open-ended case is excluded from any cutoff-bound view because "this is
    still ongoing/unresolved" is itself a leak of information past the
    cutoff (you wouldn't have known it was still unresolved back then).
    """
    if "present" in case["period"].lower():
        return None
    years = [int(y) for y in re.findall(r"\d{4}", case["period"])]
    return max(years) if years else None


def _cases_as_of(cutoff_year: int | None) -> list[dict]:
    """All cases, or only those fully concluded at or before cutoff_year."""
    if cutoff_year is None:
        return _CASES
    return [c for c in _CASES if (ey := _end_year(c)) is not None and ey <= cutoff_year]


def _score(case: dict, query_words: set[str]) -> int:
    """Count how many query words appear in the case's searchable text."""
    haystack = " ".join(
        [
            case["name"],
            " ".join(case["parties"]),
            case["initial_dynamic"],
            case["outcome"],
            " ".join(case["tags"]),
        ]
    ).lower()
    return sum(1 for w in query_words if w in haystack)


def search_historical_analogues(query: str, top_n: int = 3, cutoff_year: int | None = None) -> str:
    """
    Search the curated historical-cases dataset for analogues to `query`.

    Returns the top matching cases formatted as text, including each case's
    actual outcome and the key lesson drawn from it. Returns a clear
    "no good match" message if nothing scores above zero, so the model
    doesn't force a bad analogy just because the tool was called.

    `cutoff_year`, if given, restricts the search to cases fully concluded
    at or before that year — see backtest.py. Default (None) is the normal,
    unrestricted behavior used by the live trend_direction_agent.
    """
    query_words = {w for w in query.lower().split() if len(w) > 2}
    scored = [(c, _score(c, query_words)) for c in _cases_as_of(cutoff_year)]
    scored = [(c, s) for c, s in scored if s > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[:top_n]

    if not top:
        return (
            "No historical case in the dataset matched this query well. "
            "Don't force an analogy — say plainly that no strong precedent "
            "was found in the available data, and rely more heavily on "
            "current signals instead."
        )

    parts = []
    for case, score in top:
        parts.append(
            f"## {case['name']} ({case['period']})\n"
            f"Parties: {', '.join(case['parties'])}\n"
            f"Initial dynamic: {case['initial_dynamic']}\n"
            f"Trajectory: {case['trajectory']}\n"
            f"Outcome: {case['outcome']}\n"
            f"Key lesson: {case['key_lesson']}\n"
        )
    return "\n".join(parts)


def get_country_case_history(country: str, cutoff_year: int | None = None) -> str:
    """
    Return every case in the dataset where `country` was a party, sorted by
    period. This exists so "structural tendency" claims about a country can
    be traced to specific documented cases instead of asserted from general
    impression — if a claim can't be tied to one of these cases, it doesn't
    belong in the fixed/structural section.

    Returns an explicit "too few cases" note (not silence) when the dataset
    has 0-1 matching cases, since a pattern claim needs more than one data
    point to be a pattern.

    `cutoff_year`, if given, restricts to cases fully concluded at or before
    that year — see backtest.py. Default (None) is the normal, unrestricted
    behavior used by the live trend_direction_agent.
    """
    country_lower = country.lower().strip()
    matches = [
        c for c in _cases_as_of(cutoff_year) if any(country_lower in p.lower() for p in c["parties"])
    ]
    matches.sort(key=lambda c: c["period"])

    cutoff_note = f" (as of cutoff year {cutoff_year})" if cutoff_year is not None else ""

    if len(matches) == 0:
        return (
            f"No cases in the dataset involve '{country}'{cutoff_note}. Do not "
            f"assert any structural/historical tendency for this country from "
            f"this tool — there is nothing here to back it. Say plainly that no "
            f"dataset-backed historical pattern is available."
        )

    if len(matches) == 1:
        header = (
            f"Only ONE case in the dataset involves '{country}' — that is not "
            f"enough to claim a consistent long-run pattern. Cite this single "
            f"case narrowly (\"in this one documented instance...\") rather than "
            f"generalizing it into a national tendency.\n\n"
        )
    else:
        header = (
            f"{len(matches)} cases in the dataset involve '{country}'. Look "
            f"across all of them for what's actually consistent — e.g. does "
            f"this country tend toward escalation, negotiated resolution, "
            f"long unresolved standoffs? Cite specific cases for any pattern "
            f"you claim; do not generalize beyond what these cases show.\n\n"
        )

    parts = [header]
    for case in matches:
        parts.append(
            f"## {case['name']} ({case['period']})\n"
            f"Parties: {', '.join(case['parties'])}\n"
            f"Initial dynamic: {case['initial_dynamic']}\n"
            f"Outcome: {case['outcome']}\n"
            f"Key lesson: {case['key_lesson']}\n"
        )
    return "\n".join(parts)
