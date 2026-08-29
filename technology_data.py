"""
The technology-forecasting counterpart to historical_data.py — same idea,
different domain: a small curated dataset of past technology trajectories
(what the early hype looked like, what actually happened), searchable by
keyword, so the agent can ground a "will this be huge" call in real
precedent instead of just pattern-matching from training data.

Two lookups, mirroring the geopolitical version's two tools:
  - search_technology_analogues(query): find the closest-matching past
    technology to THIS one's specific situation.
  - get_category_track_record(category): pull every case in a given
    category (consumer_hardware, infrastructure, energy, software_platform,
    biotech) to check the base rate for that TYPE of technology — e.g. "of
    the hyped consumer_hardware cases in this dataset, how many actually
    went mainstream vs. faded?"

Both take an optional cutoff_year, same as historical_data.py, so this is
backtest-ready even though backtest.py itself hasn't been adapted to this
domain yet.
"""

import json
import re
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "technology_cases.json"
_CASES = json.loads(_DATA_PATH.read_text())

VALID_CATEGORIES = sorted({c["category"] for c in _CASES})


def _end_year(case: dict) -> int | None:
    """Last year a case's period covers, or None if open-ended
    ('...-present') — excluded under a cutoff for the same reason as in
    historical_data.py: still being open/unresolved is itself information
    from after the cutoff."""
    if "present" in case["period"].lower():
        return None
    years = [int(y) for y in re.findall(r"\d{4}", case["period"])]
    return max(years) if years else None


def _cases_as_of(cutoff_year: int | None) -> list[dict]:
    if cutoff_year is None:
        return _CASES
    return [c for c in _CASES if (ey := _end_year(c)) is not None and ey <= cutoff_year]


def _score(case: dict, query_words: set[str]) -> int:
    haystack = " ".join(
        [
            case["name"],
            case["category"],
            case["early_signals"],
            case["actual_trajectory"],
            case["outcome"],
            " ".join(case["tags"]),
        ]
    ).lower()
    return sum(1 for w in query_words if w in haystack)


def search_technology_analogues(query: str, top_n: int = 3, cutoff_year: int | None = None) -> str:
    """
    Search the curated technology-cases dataset for analogues to `query`.
    Returns each matching case's actual outcome and key lesson — not just a
    similar-sounding name. Returns an explicit "no good match" message if
    nothing scores above zero, so the model doesn't force a bad analogy.
    """
    query_words = {w for w in query.lower().split() if len(w) > 2}
    scored = [(c, _score(c, query_words)) for c in _cases_as_of(cutoff_year)]
    scored = [(c, s) for c, s in scored if s > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[:top_n]

    if not top:
        return (
            "No technology case in the dataset matched this query well. "
            "Don't force an analogy — say plainly that no strong precedent "
            "was found in the available data."
        )

    parts = []
    for case, score in top:
        parts.append(
            f"## {case['name']} ({case['period']}, category: {case['category']})\n"
            f"Early signals: {case['early_signals']}\n"
            f"Actual trajectory: {case['actual_trajectory']}\n"
            f"Outcome: {case['outcome']}\n"
            f"Key lesson: {case['key_lesson']}\n"
        )
    return "\n".join(parts)


def get_category_track_record(category: str, cutoff_year: int | None = None) -> str:
    """
    Return every case in the dataset in a given category (e.g.
    'consumer_hardware', 'infrastructure', 'energy', 'software_platform',
    'biotech'), so the model can check the base rate for that TYPE of
    technology rather than asserting a general vibe about it.

    If the category string doesn't exactly match one in the dataset, returns
    the list of valid categories instead of silently returning nothing.
    """
    category_norm = category.lower().strip().replace(" ", "_")
    if category_norm not in VALID_CATEGORIES:
        return (
            f"'{category}' isn't a category in the dataset. Valid categories: "
            f"{', '.join(VALID_CATEGORIES)}. Pick the closest one, or say the "
            f"dataset doesn't cover this technology's category."
        )

    matches = [c for c in _cases_as_of(cutoff_year) if c["category"] == category_norm]
    matches.sort(key=lambda c: c["period"])

    cutoff_note = f" (as of cutoff year {cutoff_year})" if cutoff_year is not None else ""

    if len(matches) == 0:
        return f"No '{category_norm}' cases in the dataset{cutoff_note}. No base rate available for this category."

    outcomes = [c["outcome"] for c in matches]
    tally = {o: outcomes.count(o) for o in set(outcomes)}
    tally_str = ", ".join(f"{v} {k}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))

    header = (
        f"{len(matches)} '{category_norm}' cases in the dataset{cutoff_note}. "
        f"Outcome tally: {tally_str}. This is a small, hand-curated sample — "
        f"treat it as directional context, not a statistically reliable base "
        f"rate. Cite specific cases for any pattern you claim.\n\n"
    )

    parts = [header]
    for case in matches:
        parts.append(
            f"## {case['name']} ({case['period']})\n"
            f"Early signals: {case['early_signals']}\n"
            f"Outcome: {case['outcome']}\n"
            f"Key lesson: {case['key_lesson']}\n"
        )
    return "\n".join(parts)
