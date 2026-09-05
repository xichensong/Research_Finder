"""
A persistent, trackable to-do list plus the disk-writing helpers
startup_outreach_agent.py uses:

  - save_startup_outreach: writes a structured draft (company, backer, what
    they do, the role or a note that there's no posting, the fit point, and
    the drafted email) and logs a tracked "outreach" action item pointing
    at it. Records the company as contacted so future runs don't duplicate
    it. Never sends anything.
  - add_action_item / list_action_items / mark_item_status: the tracked
    list itself, usable standalone.

Everything here is local file I/O only — no network calls, no side effects
outside this project. That boundary is intentional: sending an email on
someone's behalf needs their explicit action each time, not a script that
does it for them.

Storage: action_items.json in the project root (not sandbox/, since this is
a persistent record meant to accumulate across many agent runs, unlike the
disposable per-run reports in sandbox/).

Standalone use (no agent run needed):
  python action_tools.py list
  python action_tools.py list open
  python action_tools.py done 3
  python action_tools.py add "Follow up with <company>" outreach
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_STORE_PATH = Path(__file__).parent / "action_items.json"
_STARTUPS_DIR = Path(__file__).parent / "sandbox" / "drafts" / "startups"
_CONTACTED_COMPANIES_PATH = Path(__file__).parent / "contacted_companies.json"

VALID_CATEGORIES = ["application", "opportunity", "skill_building", "outreach", "project", "other"]
VALID_STATUSES = ["open", "in_progress", "done"]


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:60] or "item"


def _load() -> list[dict]:
    if not _STORE_PATH.exists():
        return []
    return json.loads(_STORE_PATH.read_text())


def _save(items: list[dict]) -> None:
    _STORE_PATH.write_text(json.dumps(items, indent=2))


def _next_id(items: list[dict]) -> int:
    return max((it["id"] for it in items), default=0) + 1


def add_action_item(
    title: str, category: str, direction: str = "", notes: str = "", url: str = "", deadline_note: str = ""
) -> str:
    """Add a tracked action item. `category` should be one of
    VALID_CATEGORIES — an unrecognized value is kept as-is but flagged in
    the return message so a caller notices the typo rather than silently
    filing it under the wrong bucket."""
    items = _load()
    item = {
        "id": _next_id(items),
        "title": title,
        "category": category,
        "direction": direction,
        "notes": notes,
        "url": url,
        "deadline_note": deadline_note,
        "status": "open",
        "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    items.append(item)
    _save(items)

    warning = "" if category in VALID_CATEGORIES else f" (note: '{category}' isn't a standard category)"
    return f"Added item #{item['id']}: {title}{warning}"


def list_action_items(status: str = "all") -> str:
    """List tracked items, optionally filtered by status (open/in_progress/done)."""
    items = _load()
    if status != "all":
        items = [it for it in items if it["status"] == status]
    if not items:
        return f"No action items{'' if status == 'all' else f' with status {status!r}'}."

    lines = []
    for it in items:
        extra = []
        if it.get("url"):
            extra.append(f"url: {it['url']}")
        if it.get("deadline_note"):
            extra.append(f"deadline: {it['deadline_note']}")
        extra_str = f" [{', '.join(extra)}]" if extra else ""
        lines.append(f"#{it['id']} [{it['status']}] ({it['category']}) {it['title']}{extra_str}")
    return "\n".join(lines)


def mark_item_status(item_id: int, status: str) -> str:
    if status not in VALID_STATUSES:
        return f"'{status}' isn't a valid status. Use one of: {', '.join(VALID_STATUSES)}"
    items = _load()
    for it in items:
        if it["id"] == item_id:
            it["status"] = status
            _save(items)
            return f"Marked #{item_id} as {status}: {it['title']}"
    return f"No item with id {item_id}."


def get_contacted_companies() -> list[str]:
    """Names of every company a draft has ever been finalized for, across
    ALL past runs of startup_outreach_agent.py — not reset between runs.
    Used to keep successive batches non-overlapping."""
    if not _CONTACTED_COMPANIES_PATH.exists():
        return []
    return json.loads(_CONTACTED_COMPANIES_PATH.read_text())


def mark_company_contacted(company_name: str) -> None:
    """Record a company as contacted, persistently. Called automatically by
    save_startup_outreach — not meant to be called directly."""
    contacted = get_contacted_companies()
    if company_name not in contacted:
        contacted.append(company_name)
        _CONTACTED_COMPANIES_PATH.write_text(json.dumps(contacted, indent=2))


def save_startup_outreach(
    company_name: str,
    backer: str,
    what_they_do: str,
    posting_url: str,
    fit_point: str,
    email_draft: str,
    open_role_note: str,
) -> str:
    """
    Write a structured draft for ONE startup cold-email: the company and
    its backer, what they build, the open role (or a note that there isn't
    one), the single fit point identified, and the actual drafted email.
    Logs a tracked "outreach" action item pointing at the file, and records
    the company as contacted (see get_contacted_companies) so future runs
    don't duplicate it. Never sends anything — the person reviews, edits,
    and sends it themselves.
    """
    _STARTUPS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(company_name)
    path = _STARTUPS_DIR / f"{slug}.md"

    content = (
        f"# {company_name}\n\n"
        f"**Backer:** {backer}\n\n"
        f"**What they do:** {what_they_do}\n\n"
        f"**Role / posting:** {open_role_note}\n{posting_url}\n\n"
        f"## Fit point\n{fit_point}\n\n"
        f"## Drafted email\n{email_draft}\n"
    )
    path.write_text(content)
    mark_company_contacted(company_name)

    rel_path = path.relative_to(Path(__file__).parent)
    add_action_item(
        title=f"Send outreach email: {company_name}",
        category="outreach",
        notes=f"Draft at {rel_path} — review and send yourself.",
        url=posting_url,
    )
    return (
        f"Outreach draft saved to {rel_path}, and a tracked item was logged — "
        f"review and send it yourself; nothing was sent automatically."
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(
            "Usage:\n"
            "  python action_tools.py list [open|in_progress|done]\n"
            "  python action_tools.py done <id>\n"
            "  python action_tools.py in_progress <id>\n"
            "  python action_tools.py add \"<title>\" <category> [direction]"
        )
        sys.exit(1)

    cmd = args[0]
    if cmd == "list":
        status = args[1] if len(args) > 1 else "all"
        print(list_action_items(status))
    elif cmd in ("done", "in_progress", "open"):
        if len(args) < 2:
            print(f"Usage: python action_tools.py {cmd} <id>")
            sys.exit(1)
        print(mark_item_status(int(args[1]), cmd))
    elif cmd == "add":
        if len(args) < 3:
            print('Usage: python action_tools.py add "<title>" <category> [direction]')
            sys.exit(1)
        title, category = args[1], args[2]
        direction = args[3] if len(args) > 3 else ""
        print(add_action_item(title, category, direction))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
