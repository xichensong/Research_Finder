"""
A persistent, trackable to-do list plus the disk-writing helpers
startup_outreach_agent.py uses:

  - set_applicant_contact / get_applicant_contact: the applicant's own
    details (name, email, phone, LinkedIn, GitHub, portfolio), pulled from
    their profile once per run and used to pre-fill the review outbox.
  - save_startup_outreach: writes a structured draft (company, backer, what
    they do, the role or a note that there's no posting, the fit point, the
    drafted email + subject, and where to send it), logs a tracked
    "outreach" item, records the company as contacted so future runs don't
    duplicate it, and rebuilds OUTBOX.html.
  - OUTBOX.html: a local review page. For each company it shows either a
    prefilled mailto: link (subject + body baked in) or, when no public
    email was found, the application URL and the exact field values to
    paste. NOTHING IS SENT. The person opens the page, checks each one, and
    clicks the final button themselves.
  - add_action_item / list_action_items / mark_item_status: the tracked
    list itself, usable standalone.

Everything here is local file I/O only — no network calls, no send, no
submit. That boundary is deliberate: an email or a form submission goes to
a real person and can't be taken back, so the final click stays with the
human every time.

Storage: action_items.json / applicant.json in the project root (persistent
records); the per-company drafts, _leads.json, and OUTBOX.html live in
sandbox/drafts/startups/ with the disposable per-run output.

Standalone use (no agent run needed):
  python action_tools.py list
  python action_tools.py done 3
  python action_tools.py outbox          # rebuild OUTBOX.html from _leads.json
"""

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

_ROOT = Path(__file__).parent
_STORE_PATH = _ROOT / "action_items.json"
_STARTUPS_DIR = _ROOT / "sandbox" / "drafts" / "startups"
_CONTACTED_COMPANIES_PATH = _ROOT / "contacted_companies.json"
_APPLICANT_PATH = _ROOT / "applicant.json"
_LEADS_PATH = _STARTUPS_DIR / "_leads.json"
_OUTBOX_PATH = _STARTUPS_DIR / "OUTBOX.html"

VALID_CATEGORIES = ["application", "opportunity", "skill_building", "outreach", "project", "other"]
VALID_STATUSES = ["open", "in_progress", "done"]
APPLICANT_FIELDS = ["name", "email", "phone", "linkedin", "github", "portfolio"]


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


# --- applicant contact (for pre-filling the outbox) ------------------------

def set_applicant_contact(
    name: str = "", email: str = "", phone: str = "", linkedin: str = "", github: str = "", portfolio: str = ""
) -> str:
    """Store the applicant's own contact details, taken from their profile.
    Used only to pre-fill the review outbox: the mailto: sender identity and
    the name/email/link fields on web application forms. Never invent a
    value — pass '' for anything not in the profile."""
    data = dict(name=name, email=email, phone=phone, linkedin=linkedin, github=github, portfolio=portfolio)
    _APPLICANT_PATH.write_text(json.dumps(data, indent=2))
    have = ", ".join(k for k, v in data.items() if v) or "(none)"
    return f"Applicant contact stored for the outbox. Fields present: {have}."


def get_applicant_contact() -> dict:
    if not _APPLICANT_PATH.exists():
        return {k: "" for k in APPLICANT_FIELDS}
    d = json.loads(_APPLICANT_PATH.read_text())
    return {k: d.get(k, "") for k in APPLICANT_FIELDS}


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


# --- the review outbox ---------------------------------------------------

def _load_leads() -> list[dict]:
    return json.loads(_LEADS_PATH.read_text()) if _LEADS_PATH.exists() else []


def _append_lead(lead: dict) -> None:
    leads = [l for l in _load_leads() if l["company"].lower() != lead["company"].lower()]
    leads.append(lead)
    _STARTUPS_DIR.mkdir(parents=True, exist_ok=True)
    _LEADS_PATH.write_text(json.dumps(leads, indent=2))


def _project_note(body: str) -> str:
    """The substantive paragraph of the email (the longest one, per the
    draft spec where P2 carries the bulk), reused verbatim as the 'a project
    you're proud of' answer on web forms. Skips the greeting and sign-off."""
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    candidates = [p for p in paras if len(p.split()) > 6] or paras
    return max(candidates, key=len) if candidates else body.strip()


def _mailto(to: str, subject: str, body: str) -> str:
    return f"mailto:{quote(to)}?subject={quote(subject)}&body={quote(body)}"


def _gmail_compose(to: str, subject: str, body: str) -> str:
    """A Gmail web compose URL — opens a compose window, in whatever account
    the browser is signed into, with everything pre-filled. Still needs the
    person to press Send."""
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote(to)}&su={quote(subject)}&body={quote(body)}"
    )


def rebuild_outbox() -> str:
    """Regenerate OUTBOX.html from _leads.json. Safe to call standalone."""
    leads = _load_leads()
    app = get_applicant_contact()
    e = html.escape

    ready = len(leads)
    email_n = sum(1 for l in leads if l.get("send_to"))
    form_n = ready - email_n

    sig_bits = [app["name"], app["email"], app["phone"], app["linkedin"], app["portfolio"]]
    sig = "  |  ".join(b for b in sig_bits if b)

    cards = []
    for l in leads:
        head = (
            f"<h2>{e(l['company'])}</h2>"
            f"<p class='meta'>{e(l.get('backer',''))}<br>{e(l.get('open_role_note',''))}</p>"
            f"<details><summary>Fit point</summary><p>{e(l.get('fit_point',''))}</p></details>"
        )
        if l.get("send_to"):
            # The email body already carries the sign-off; append only a
            # contact line so the founder can reply or look you up.
            contact_line = "  |  ".join(b for b in [app["phone"], app["linkedin"], app["portfolio"]] if b)
            body_with_sig = l["body"] + (f"\n\n{contact_line}" if contact_line else "")
            gmail = _gmail_compose(l["send_to"], l.get("subject", ""), body_with_sig)
            mail = _mailto(l["send_to"], l.get("subject", ""), body_with_sig)
            action = (
                f"<p><b>To</b> {e(l['send_to'])}<br><b>Subject</b> {e(l.get('subject',''))}</p>"
                f"<pre>{e(body_with_sig)}</pre>"
                f"<a class='btn' href=\"{e(gmail)}\" target='_blank' rel='noopener'>Open in Gmail</a> "
                f"<a class='btn secondary' href=\"{e(mail)}\">Open in default mail app</a>"
                f"<p class='hint'>Opens a compose window with everything filled. Read it, then press Send yourself.</p>"
            )
        else:
            url = l.get("apply_url") or l.get("posting_url", "")
            rows = "".join(
                f"<tr><td>{e(k)}</td><td>{e(v) if v else '<span class=todo>fill in applicant.json</span>'}</td></tr>"
                for k, v in [
                    ("Name", app["name"]), ("Email", app["email"]),
                    ("LinkedIn", app["linkedin"]), ("GitHub", app["github"]),
                ]
            )
            rows += f"<tr><td>Project note</td><td>{e(_project_note(l['body']))}</td></tr>"
            action = (
                f"<p><b>No public email — apply via form:</b> <a href=\"{e(url)}\">{e(url)}</a></p>"
                f"<table>{rows}</table>"
                f"<a class='btn' href=\"{e(url)}\" target='_blank' rel='noopener'>Open application form</a>"
                f"<p class='hint'>Paste the values above, review, then submit.</p>"
            )
        cards.append(f"<section>{head}{action}</section>")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Outreach outbox</title>
<style>
 body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{font-size:1.4rem}} h2{{font-size:1.15rem;margin:.2rem 0}}
 .lead{{color:#555}} .meta{{color:#666;font-size:.9rem;margin:.2rem 0 .6rem}}
 section{{border:1px solid #ddd;border-radius:8px;padding:1rem 1.1rem;margin:1rem 0}}
 pre{{white-space:pre-wrap;background:#f6f6f6;border-radius:6px;padding:.8rem;font:13px/1.5 ui-monospace,monospace}}
 .btn{{display:inline-block;background:#1a1a1a;color:#fff;text-decoration:none;padding:.5rem .9rem;border-radius:6px;font-weight:600;margin:.2rem .3rem .2rem 0}}
 .btn.secondary{{background:#eee;color:#1a1a1a;border:1px solid #ccc}}
 .hint{{color:#777;font-size:.85rem}} table{{border-collapse:collapse;width:100%;margin:.5rem 0}}
 td{{border:1px solid #e2e2e2;padding:.4rem .6rem;vertical-align:top}} td:first-child{{width:110px;color:#555;font-weight:600}}
 .todo{{color:#b00}} details{{margin:.4rem 0}} summary{{cursor:pointer;color:#555}}
 @media(prefers-color-scheme:dark){{body{{background:#151515;color:#e8e8e8}}section{{border-color:#333}}pre{{background:#1f1f1f}}
  .btn{{background:#e8e8e8;color:#151515}}.btn.secondary{{background:#2a2a2a;color:#e8e8e8;border-color:#444}}td{{border-color:#333}}}}
</style></head><body>
<h1>Outreach outbox</h1>
<p class="lead">{ready} ready — {email_n} email, {form_n} form. Nothing is sent or submitted. Review each one and click the button yourself.</p>
{f'<p class="meta">Sending as: {e(sig)}</p>' if sig else '<p class="meta todo">applicant.json is empty — fill it in so emails are signed and form fields pre-fill.</p>'}
{''.join(cards)}
</body></html>"""
    _STARTUPS_DIR.mkdir(parents=True, exist_ok=True)
    _OUTBOX_PATH.write_text(doc)
    return str(_OUTBOX_PATH.relative_to(_ROOT))


def save_startup_outreach(
    company_name: str,
    backer: str,
    what_they_do: str,
    posting_url: str,
    fit_point: str,
    email_draft: str,
    open_role_note: str,
    email_subject: str = "",
    send_to: str = "",
    apply_url: str = "",
) -> str:
    """
    Write a structured draft for ONE startup cold-email and add it to the
    review outbox. `send_to` is a real outreach email if one was found;
    when it's empty the outbox points at `apply_url` (or the posting URL)
    and lays out the form fields instead. Logs a tracked "outreach" item,
    records the company as contacted, and rebuilds OUTBOX.html. Never sends
    or submits anything.
    """
    _STARTUPS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(company_name)
    path = _STARTUPS_DIR / f"{slug}.md"

    where = f"Email: {send_to}" if send_to else f"Apply via form: {apply_url or posting_url}"
    content = (
        f"# {company_name}\n\n"
        f"**Backer:** {backer}\n\n"
        f"**What they do:** {what_they_do}\n\n"
        f"**Role / posting:** {open_role_note}\n{posting_url}\n\n"
        f"**Send to:** {where}\n\n"
        f"## Fit point\n{fit_point}\n\n"
        f"## Drafted email\n"
        f"{('Subject: ' + email_subject + chr(10) + chr(10)) if email_subject else ''}{email_draft}\n"
    )
    path.write_text(content)
    mark_company_contacted(company_name)

    _append_lead(
        {
            "company": company_name,
            "backer": backer,
            "what_they_do": what_they_do,
            "open_role_note": open_role_note,
            "fit_point": fit_point,
            "subject": email_subject,
            "body": email_draft,
            "send_to": send_to,
            "apply_url": apply_url,
            "posting_url": posting_url,
        }
    )
    outbox_rel = rebuild_outbox()

    rel_path = path.relative_to(_ROOT)
    add_action_item(
        title=f"Send outreach: {company_name}",
        category="outreach",
        notes=f"Draft at {rel_path}; review + send from {outbox_rel}.",
        url=send_to and f"mailto:{send_to}" or (apply_url or posting_url),
    )
    return (
        f"Draft saved to {rel_path} and added to {outbox_rel} "
        f"({'prefilled email' if send_to else 'application form + field values'}). "
        f"Nothing was sent — the person reviews and clicks in the outbox."
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(
            "Usage:\n"
            "  python action_tools.py list [open|in_progress|done]\n"
            "  python action_tools.py done <id>\n"
            "  python action_tools.py in_progress <id>\n"
            "  python action_tools.py add \"<title>\" <category> [direction]\n"
            "  python action_tools.py outbox"
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
    elif cmd == "outbox":
        print(f"Rebuilt {rebuild_outbox()}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
