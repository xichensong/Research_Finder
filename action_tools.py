"""
Turns next_steps_agent.py's one-shot report into a persistent, trackable
todo list, plus adjacent capabilities:

  - log_opportunity: capture a SPECIFIC real thing found via web_search
    (an actual posting, course, competition, deadline) as a tracked item
    with its URL — not just a category of advice.
  - save_draft: write actual drafted outreach text to disk for the person
    to review and send THEMSELVES. Never sends anything.
  - save_application_materials: for one specific logged opportunity, write
    a dedicated file with the actual application materials — a tailored
    cover letter, plus draft answers to the application's real questions
    IF web_search actually found them, or an honest note that the specific
    questions weren't found (never invented ones) if not. Also logs a
    tracked "submit application" item pointing at the file. Never submits
    anything.
  - save_professor_outreach: for professor_outreach_agent.py — writes a
    structured draft (professor, paper, connection point, email) and logs
    a tracked outreach item pointing at it. Never sends anything.

Everything here is local file I/O only — no network calls except what the
agent's own web_search already does, no side effects outside this project's
sandbox. That boundary is intentional: submitting an application or sending
a message on someone's behalf needs their explicit action each time, not a
script that does it for them.

Storage: action_items.json in the project root (not sandbox/, since this is
a persistent record meant to survive and accumulate across many agent runs,
unlike the disposable per-run reports in sandbox/).

Standalone use (no agent run needed):
  python action_tools.py list
  python action_tools.py list open
  python action_tools.py done 3
  python action_tools.py add "Register for Stat 134 next term" skill_building
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_STORE_PATH = Path(__file__).parent / "action_items.json"
_DRAFTS_DIR = Path(__file__).parent / "sandbox" / "drafts"
_APPLICATIONS_DIR = Path(__file__).parent / "sandbox" / "applications"
_PROFESSORS_DIR = Path(__file__).parent / "sandbox" / "drafts" / "professors"
_CONTACTED_PROFESSORS_PATH = Path(__file__).parent / "contacted_professors.json"

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


def log_opportunity(title: str, url: str, direction: str, deadline_note: str = "") -> str:
    """Log a SPECIFIC real opportunity found via web_search — not a general
    category of advice. Files it as an action item with category
    'opportunity'. Only call this for something concrete and real that was
    actually found this run (a real posting, program, or competition), with
    a real URL — never a plausible-sounding invented one."""
    return add_action_item(
        title=title, category="opportunity", direction=direction, url=url, deadline_note=deadline_note
    )


def save_draft(filename: str, content: str) -> str:
    """Write drafted outreach text to sandbox/drafts/ for the person to
    review and send themselves. This never sends anything — it only saves
    text to a local file."""
    _DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _DRAFTS_DIR / filename
    path.write_text(content)
    return f"Draft saved to {path.relative_to(Path(__file__).parent)} — review and send it yourself; nothing was sent automatically."


def save_application_materials(opportunity_title: str, content: str) -> str:
    """
    Write drafted application materials for ONE specific logged opportunity
    to its own file (filename derived from the opportunity title, so
    materials are easy to find alongside the tracked item), and log a
    "submit application" action item pointing at the file.

    `content` should be the actual materials — a tailored cover letter,
    and either real draft answers to application questions that were
    genuinely found via web_search, or an explicit honest note that the
    specific questions weren't found (never fabricated ones). This never
    submits anything — it's a draft for the person to review, tailor
    further, and submit themselves.
    """
    _APPLICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(opportunity_title)
    path = _APPLICATIONS_DIR / f"{slug}.md"
    path.write_text(content)

    rel_path = path.relative_to(Path(__file__).parent)
    add_action_item(
        title=f"Submit application: {opportunity_title}",
        category="application",
        notes=f"Draft materials at {rel_path} — review, tailor, and submit yourself.",
    )
    return (
        f"Application materials saved to {rel_path}, and a 'submit application' item was "
        f"logged pointing at it — review and submit yourself; nothing was submitted automatically."
    )


def get_contacted_professors() -> list[str]:
    """Names of every professor a draft has ever been finalized for, across
    ALL past runs of professor_outreach_agent.py — not reset between runs.
    Used to keep successive batches non-overlapping."""
    if not _CONTACTED_PROFESSORS_PATH.exists():
        return []
    return json.loads(_CONTACTED_PROFESSORS_PATH.read_text())


def mark_professor_contacted(professor_name: str) -> None:
    """Record a professor as contacted, persistently. Called automatically
    by save_professor_outreach — not meant to be called directly."""
    contacted = get_contacted_professors()
    if professor_name not in contacted:
        contacted.append(professor_name)
        _CONTACTED_PROFESSORS_PATH.write_text(json.dumps(contacted, indent=2))


def save_professor_outreach(
    professor_name: str, department: str, paper_title: str, paper_url: str, connection_point: str, email_draft: str
) -> str:
    """
    Write a structured draft for ONE professor cold-email: who they are,
    the specific paper the outreach is anchored on, the one connection
    point identified, and the actual drafted email text. Logs a tracked
    "outreach" action item pointing at the file, and records the professor
    as contacted (see get_contacted_professors) so future runs don't
    duplicate them. Never sends anything — the person reviews, edits, and
    sends it themselves.
    """
    _PROFESSORS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(professor_name)
    path = _PROFESSORS_DIR / f"{slug}.md"

    content = (
        f"# {professor_name} ({department})\n\n"
        f"## Paper this is anchored on\n{paper_title}\n{paper_url}\n\n"
        f"## Connection point\n{connection_point}\n\n"
        f"## Drafted email\n{email_draft}\n"
    )
    path.write_text(content)
    mark_professor_contacted(professor_name)

    rel_path = path.relative_to(Path(__file__).parent)
    add_action_item(
        title=f"Send outreach email: {professor_name}",
        category="outreach",
        notes=f"Draft at {rel_path} — review and send yourself.",
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
