"""
Finds ML/AI/quant startups backed by a credible accelerator or tier-1 VC,
figures out ONE specific point where what they build genuinely connects to
something the person has actually done, and drafts a cold email around that
connection. It fills everything in — a prefilled email or, when no public
address exists, an application URL with the exact field values — but never
sends or submits. The final click stays with the person.

A startup does NOT need an open internship posting to be included. If a
real opening is found, the email references it; if not, the email just
makes the case for what the person could contribute. Either way the anchor
is a real, specific connection, not generic enthusiasm.

Honesty boundary: web_search returns company descriptions, batch pages,
funding announcements, and job-post summaries, not a deep read of a
codebase or a full JD. "What they do" and "the role" mean what search
actually surfaced, and the gates check the draft against that.

PIPELINE:

  0. set_applicant_contact — pull the person's own name/email/phone/links
     from the profile, once, so the outbox can pre-fill sender identity and
     web-form fields.

  1. log_startup_found — name, backer, what they do, a posting/careers URL,
     a search summary, whether a specific opening exists, and a real
     outreach email if one is published.
     -> GATE 1 (credibility + plausibility): real-looking company, backer
        in the credible set, real URL, specific content, a matching search
        query. Not a fact-check — a fabrication/weak-backer screen.

  2. propose_fit_point — the one connection between the company's work and
     the person's background.
     -> GATE 2 (grounding): the fit point's claims ABOUT THE COMPANY stay
        within what the logged summary supports.
     -> HUMANIZER PASS.

  3. propose_email_draft — subject + body.
     -> HUMANIZER PASS (subject and body).

  4. save_startup_outreach(company_name) — takes ONLY the name; the gated,
     humanized fit point / subject / body are pulled from code. Writes the
     draft, adds it to OUTBOX.html (prefilled mailto: if an email was
     found, else the form URL + field values), logs a tracked item.

Then one index report (write_file) — humanized, then grounding-checked.

CROSS-RUN NON-OVERLAP: every finished company is recorded permanently
(contacted_companies.json). Each run loads that list, tells the model to
skip them, and rejects log_startup_found in code if the name matches.

Run it:
  python startup_outreach_agent.py my_profile.md "ML,AI,Quant" 8
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

from action_tools import (
    get_contacted_companies,
    save_startup_outreach as _persist_startup_outreach,
    set_applicant_contact as _set_applicant_contact,
)
from humanizer import humanize_text
from tools import write_file
from verification import MAX_VERIFY_ATTEMPTS, verify_report

MODEL = "gpt-5.6"

# What counts as a "credible" backer. A company clears the bar if it went
# through one of these accelerators/programs OR has a disclosed funding
# round led by or including one of the tier-1 firms listed here. This is
# injected into both the system prompt and GATE 1.
CREDIBLE_BACKERS = (
    "Accelerators / programs: Y Combinator (YC), Techstars, a16z Speedrun, "
    "Neo, South Park Commons, Pear VC, AI Grant, Entrepreneur First. "
    "Tier-1 VCs (a disclosed round led by or including one of these counts): "
    "Sequoia, Andreessen Horowitz (a16z), Benchmark, Founders Fund, Greylock, "
    "Lightspeed, Index Ventures, Accel, Kleiner Perkins, Khosla Ventures, "
    "General Catalyst, Thrive Capital, Bessemer, NEA, Craft Ventures, "
    "Conviction, Radical Ventures."
)

TOOLS = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "set_applicant_contact",
        "description": (
            "Step 0. Call ONCE before the pipeline. Pull the person's own "
            "contact details straight from the profile below: name, email, "
            "phone, LinkedIn URL, GitHub URL, portfolio/website URL. These "
            "pre-fill the outbox (the mailto: sender line and the fields on "
            "web application forms). Pass '' for anything not in the profile "
            "— never invent an address or URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "linkedin": {"type": "string"},
                "github": {"type": "string"},
                "portfolio": {"type": "string"},
            },
            "required": ["name", "email", "phone", "linkedin", "github", "portfolio"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "log_startup_found",
        "description": (
            "Step 1 of 4. Log a startup you found via web_search. Only for a "
            "REAL company whose backer clears the credible-set bar in your "
            "instructions — never invent a name, URL, backer, or batch. "
            "`what_they_do` and `posting_summary` are what search actually "
            "surfaced, specific and honest. Set `has_open_role` true only if "
            "you found an actual early-career posting. `contact_email` is a "
            "real outreach address published on the company's own site "
            "(founders@, team@, careers@, or a named founder address) — pass "
            "'' if none is public, and the outbox will use the application "
            "form instead. Never guess an email."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "backer": {"type": "string", "description": "The specific accelerator/batch or the specific investor(s) and round."},
                "what_they_do": {"type": "string"},
                "posting_url": {
                    "type": "string",
                    "description": "The specific job posting if has_open_role, else the careers page or company/YC page.",
                },
                "posting_summary": {"type": "string"},
                "has_open_role": {"type": "boolean"},
                "contact_email": {"type": "string", "description": "Real outreach email published on their site, or '' if none."},
            },
            "required": ["company_name", "backer", "what_they_do", "posting_url", "posting_summary", "has_open_role", "contact_email"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "propose_fit_point",
        "description": (
            "Step 2 of 4. AFTER log_startup_found for this company. Propose "
            "the ONE specific point where what this company builds connects "
            "to the HIGHEST-PRIORITY piece of the person's background that "
            "genuinely applies — check the priority order in your "
            "instructions. It should show the person could contribute, not "
            "just that they're interested. Checked against the logged "
            "company summary; if it introduces unsupported claims about the "
            "company it's rejected for revision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "fit_point_draft": {"type": "string"},
            },
            "required": ["company_name", "fit_point_draft"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "propose_email_draft",
        "description": (
            "Step 3 of 4. AFTER propose_fit_point succeeded. Provide:\n"
            "`email_subject` — short and plain, under 9 words, no cliche "
            "colon-hooks, no hype. Just what this is about.\n"
            "`email_draft` — exactly THREE paragraphs, 120-180 words total:\n"
            "P1 (shortest): show you know specifically what they build; name "
            "the role here if one was found.\n"
            "P2 (LONGEST): connect their work to the person's own "
            "experience and say what the person could contribute, leading "
            "with the highest-priority background item that fits (see "
            "PRIORITY ORDER).\n"
            "P3: other relevant background, then a direct ask — be "
            "considered / whether they take interns, and a short call.\n"
            "P2 must clearly be the longest. Avoid cold-email cliches: no "
            "\"the part that caught my eye\", \"what I keep coming back to\", "
            "\"this resonates\", \"I'd love to\". State the specific thing "
            "plainly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "email_subject": {"type": "string"},
                "email_draft": {"type": "string"},
            },
            "required": ["company_name", "email_subject", "email_draft"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "save_startup_outreach",
        "description": (
            "Step 4 of 4. Finalizes one company — takes only the name; the "
            "approved, humanized fit point, subject, and email are pulled "
            "from what passed the earlier steps. AFTER propose_email_draft. "
            "Writes the draft, adds it to the review outbox, logs a tracked "
            "item. Never sends or submits."
        ),
        "parameters": {
            "type": "object",
            "properties": {"company_name": {"type": "string"}},
            "required": ["company_name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write the index report listing all startups found this run.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

_BASE_PROMPT = """\
You find credible early-stage startups, connect their real work to a \
specific person's real background, and draft cold emails. You fill \
everything in for review. You never send or submit anything — the person \
does the final check and click.

PURPOSE: get the person an internship or early-career role, or failing a \
formal opening, a short call with a founder open to bringing them on. The \
fit point exists to show the person could contribute, not that they find \
the work interesting. Every email ends with a direct, specific ask.

FOCUS: startups working in {areas}, where the technical work is close to \
what the person has actually done.

CREDIBLE BACKERS — a company is only eligible if it clears this bar. \
Confirm the backer via search before logging; skip any company you can't \
tie to one of these:
{backers}

OPEN ROLE IS OPTIONAL: a company with no visible internship posting is \
still eligible. If you find a real opening, reference it specifically. If \
not, make the case without pretending a posting exists.

CONTACT: for each company, also look for a real outreach email published \
on the company's own site (founders@, team@, careers@, or a named founder \
address). Pass it as contact_email. If there is no public email, pass '' \
and the outbox will point at the application form instead. Never guess an \
address.

PRIORITY ORDER for the connection — check the profile for the person's own \
stated order; if given, follow it. Otherwise: (1) a company or product the \
person founded or shipped themselves, especially with real users — for a \
startup this is the strongest signal, lead with it whenever it genuinely \
connects; (2) independent or first-author research papers; (3) internships \
or research-assistant positions. Only fall back to a lower-priority item \
when a higher one doesn't genuinely fit this company's work.

ALREADY CONTACTED — do not pick any of these, covered in a previous run: \
{already_contacted}

WORKFLOW:

0. First, call set_applicant_contact with the person's own name, email, \
phone, and LinkedIn/GitHub/portfolio URLs exactly as they appear in the \
profile below (use '' for any that aren't there).

Then, for a target of about {count} companies, run the four-step pipeline \
(log_startup_found -> propose_fit_point -> propose_email_draft -> \
save_startup_outreach) FOR EACH:

1. Search for startups in {areas} tied to a credible backer \
(ycombinator.com/companies filtered by industry, a16z Speedrun / AI Grant \
cohort pages, Techstars portfolios, recent funding announcements naming a \
tier-1 firm). Confirm the backer, find what they build, any early-career \
role, and a public contact email. 2-4 web_search calls per company. Skip a \
company you can't tie to a credible backer.
2. log_startup_found with what you actually found.
3. propose_fit_point with the ONE specific point from the highest-priority \
applicable background item. If rejected, revise within the logged summary \
and retry.
4. propose_email_draft with a plain subject and the three-paragraph email \
(P2 longest, leads with the highest-priority background item).
5. save_startup_outreach with just the company name.

Do not fabricate a company, backer, opening, email, or a connection that \
isn't in the profile. If you can't find enough real material, skip the \
company — fewer real drafts beat more fabricated ones.

After the pipeline for each company, write an index report using \
write_file, in this exact structure:

  # Startup outreach — {areas}

  ## Summary
  2-3 sentences: how many companies, how many with an open role vs cold \
outreach, how many had a public email vs form-only, any notable gap.

  ## <One "###" subsection per company, name as heading>
  1 sentence on what they do and their backer, 1 on the role situation, \
1-2 on the fit point, 1 on how to reach them (email or form).

  ## Confidence & what would change this
  1-2 sentences on how solid the fits and backer confirmations are.

  ## Sources
  Flat list of every URL you used.

After write_file, an independent reliability check reviews the report \
against what your tools returned. Address any gaps and call write_file \
again. Limited attempts. Keep every section tight.

PROFILE (use only real facts from this):
{profile_text}
"""


def build_system_prompt(areas: list[str], count: int, profile_text: str, already_contacted: list[str]) -> str:
    return _BASE_PROMPT.format(
        count=count,
        areas=", ".join(areas),
        backers=CREDIBLE_BACKERS,
        profile_text=profile_text,
        already_contacted=", ".join(already_contacted) if already_contacted else "(none yet — this is the first run)",
    )


# --- Gate helpers ----------------------------------------------------------

def _run_gate(client: OpenAI, system_instructions: str, user_content: str) -> tuple[bool, list[str]]:
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ],
        reasoning={"effort": "low"},
    )
    text = response.output_text.strip()
    if text.upper().startswith("VERDICT: FAIL"):
        gaps = [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]
        return False, gaps or ["Gate failed but returned no specific gaps."]
    return True, []


_PLAUSIBILITY_GATE_PROMPT = """\
You are a plausibility and credibility auditor. You cannot browse the web, \
so you cannot confirm a company or its funding exists — catch OBVIOUS signs \
of fabrication or a backer that plainly doesn't clear the bar.

The credible-backer bar is:
{backers}

Given a company's logged details and the web_search queries this run, FAIL if:
- No real-looking URL (empty or an obvious placeholder)
- The description/summary reads as generic ("an AI startup" with no product)
- The stated backer is clearly not in the credible set, or too vague to check
- No web_search query plausibly relates to finding this company or its backer

Otherwise PASS. Do not fail just because you don't recognize the company.

Respond in EXACTLY this format:
VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap>
"""

_FIT_GATE_PROMPT = """\
You are a grounding auditor for one claim. You'll see a company summary \
(what search found) and a proposed "fit point" connecting that company's \
work to someone's background.

Check ONLY whether the fit point's claims ABOUT THE COMPANY stay consistent \
with the summary, or introduce specific technical details (stack, methods, \
roadmap) the summary doesn't support. Claims about the PERSON are not your \
concern here. If the summary is thin and the fit point makes an oddly \
specific claim about the company, FAIL it as likely fabricated elaboration.

Respond in EXACTLY this format:
VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap>
"""

_INDEX_VERIFIER_PROMPT = """\
You are a methodology auditor for a startup-outreach index report. You did \
NOT write it. Check ONLY, against the evidence log:

1. Every company named as a "###" subsection has a matching \
log_startup_found entry (not invented after the fact).
2. The backer named for each company matches what was logged, not an \
upgraded one.
3. Any company the report says has an open role was logged with \
has_open_role true.
4. Any company the report says has a contact email was logged with a \
non-empty contact_email (form-only companies must not be described as \
emailable).
5. Nothing claims a company was drafted if its pipeline never completed.

Do NOT critique writing style or fit quality, and do NOT invent gaps \
beyond these five categories.

Respond in EXACTLY this format, nothing else:
VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap, naming the company>
"""


class _TrackedClient:
    """
    Wraps the real OpenAI client so every .responses.create() call — main
    loop, gates, humanizer, verifier — is counted. `token_budget`, if set,
    raises _BudgetExceeded once the running total crosses it. Reports
    TOKENS, not dollars; convert at your billing-dashboard rate.
    """

    def __init__(self, real_client: OpenAI, token_budget: int | None = None):
        self._real = real_client
        self.token_budget = token_budget
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    @property
    def responses(self):
        return self

    def create(self, **kwargs):
        response = self._real.responses.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.total_output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.call_count += 1
        if self.token_budget is not None and self.total_tokens > self.token_budget:
            raise _BudgetExceeded(self.total_tokens, self.token_budget)
        return response

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> str:
        return (
            f"{self.call_count} API calls this run — {self.total_input_tokens:,} input tokens, "
            f"{self.total_output_tokens:,} output tokens, {self.total_tokens:,} total. "
            f"Check platform.openai.com/usage for the actual dollar cost at your account's rate."
        )


class _BudgetExceeded(Exception):
    def __init__(self, used: int, budget: int):
        super().__init__(f"Token budget exceeded: used {used:,}, budget was {budget:,}.")
        self.used = used
        self.budget = budget


def run_agent(
    areas: list[str],
    profile_text: str,
    count: int = 8,
    max_turns: int = 120,
    token_budget: int | None = None,
) -> str:
    real_client = OpenAI()
    client = _TrackedClient(real_client, token_budget=token_budget)

    already_contacted = get_contacted_companies()
    already_contacted_lower = {name.lower() for name in already_contacted}
    system_prompt = build_system_prompt(areas, count, profile_text, already_contacted)
    plausibility_prompt = _PLAUSIBILITY_GATE_PROMPT.format(backers=CREDIBLE_BACKERS)

    evidence_log: list[dict] = []
    companies_drafted: list[str] = []
    company_state: dict[str, dict] = {}
    verify_attempts = {"count": 0}
    last_written_content = {"text": None}

    def set_applicant_contact(name, email, phone, linkedin, github, portfolio) -> str:
        return _set_applicant_contact(
            name=name, email=email, phone=phone, linkedin=linkedin, github=github, portfolio=portfolio
        )

    def log_startup_found(company_name, backer, what_they_do, posting_url, posting_summary, has_open_role, contact_email) -> str:
        if company_name.lower() in already_contacted_lower:
            return (
                f"REJECTED — '{company_name}' was already contacted in a previous run "
                f"(enforced, not just a reminder). Pick a different company."
            )
        evidence_log.append(
            {
                "tool": "log_startup_found",
                "args": {
                    "company_name": company_name, "backer": backer, "posting_url": posting_url,
                    "has_open_role": has_open_role, "contact_email": contact_email,
                },
                "result": f"backer={backer}, url={posting_url}, has_open_role={has_open_role}, "
                f"contact_email={contact_email or '(none)'}, what_they_do={what_they_do}, summary={posting_summary}",
            }
        )
        passed, gaps = _run_gate(
            client,
            plausibility_prompt,
            f"Company: {company_name}\nBacker: {backer}\nWhat they do: {what_they_do}\nURL: {posting_url}\n"
            f"Has open role: {has_open_role}\nSummary: {posting_summary}\n\nSearch queries this run:\n"
            + "\n".join(e["query"] for e in evidence_log if e["tool"] == "web_search"),
        )
        if not passed:
            return (
                "GATE 1 (credibility + plausibility) FAILED — do not proceed for this company yet:\n"
                + "\n".join(f"- {g}" for g in gaps)
                + "\nSearch again for a confirmable backer, or skip this company."
            )
        company_state[company_name] = {
            "backer": backer,
            "what_they_do": what_they_do,
            "posting_url": posting_url,
            "posting_summary": posting_summary,
            "has_open_role": bool(has_open_role),
            "contact_email": contact_email or "",
        }
        return f"Logged and passed the credibility check. Proceed to propose_fit_point for {company_name}."

    def propose_fit_point(company_name, fit_point_draft) -> str:
        if company_name not in company_state:
            return f"No company logged for '{company_name}' yet — call log_startup_found first."
        passed, gaps = _run_gate(
            client,
            _FIT_GATE_PROMPT,
            f"Company summary: {company_state[company_name]['what_they_do']}\n"
            f"{company_state[company_name]['posting_summary']}\n\nProposed fit point: {fit_point_draft}",
        )
        if not passed:
            return (
                "GATE 2 (grounding) FAILED — fit point not saved:\n"
                + "\n".join(f"- {g}" for g in gaps)
                + "\nRevise within what the logged summary supports and call propose_fit_point again."
            )
        humanized = humanize_text(
            client, fit_point_draft, context="a specific technical fit point in a cold email to a startup founder"
        )
        company_state[company_name]["fit_point"] = humanized
        evidence_log.append({"tool": "propose_fit_point", "args": {"company_name": company_name}, "result": humanized})
        return f"Fit point passed grounding check and was humanized. Proceed to propose_email_draft for {company_name}."

    def propose_email_draft(company_name, email_subject, email_draft) -> str:
        if company_name not in company_state or "fit_point" not in company_state[company_name]:
            return f"No approved fit point for '{company_name}' yet — call propose_fit_point first."
        humanized_body = humanize_text(
            client, email_draft, context="a cold email from a student to a startup founder, 120-180 words, three paragraphs"
        )
        humanized_subject = humanize_text(
            client, email_subject, context="a short email subject line, keep it under 9 words, plain, no hype or cliche hooks"
        )
        company_state[company_name]["email_subject"] = humanized_subject
        company_state[company_name]["email_draft"] = humanized_body
        return f"Subject and email drafted and humanized. Proceed to save_startup_outreach for {company_name}."

    def save_startup_outreach_step(company_name) -> str:
        state = company_state.get(company_name, {})
        if "fit_point" not in state or "email_draft" not in state:
            return (
                f"Cannot finalize '{company_name}' — pipeline incomplete "
                f"(need propose_fit_point and propose_email_draft first)."
            )
        send_to = state["contact_email"]
        apply_url = "" if send_to else state["posting_url"]
        open_role_note = (
            "Open early-career role found (see posting URL)" if state["has_open_role"]
            else "No specific opening found — cold outreach"
        )
        result = _persist_startup_outreach(
            company_name,
            state["backer"],
            state["what_they_do"],
            state["posting_url"],
            state["fit_point"],
            state["email_draft"],
            open_role_note,
            state.get("email_subject", ""),
            send_to,
            apply_url,
        )
        companies_drafted.append(company_name)
        return result

    def verified_write_file(filename: str, content: str) -> str:
        # The index report is model-generated prose like the fit points and
        # emails, so it gets the same de-slop pass. The context note keeps
        # headings and URLs intact so the section check still finds them.
        content = humanize_text(
            client,
            content,
            context=(
                "a markdown index report. Preserve every heading line (starting with #, "
                "##, or ###) verbatim, including company names in them, and preserve every "
                "URL exactly. Only rewrite the prose sentences under the headings."
            ),
        )
        write_file(filename, content)
        last_written_content["text"] = content

        required_headers = ["## Summary", "## Confidence & what would change this", "## Sources"] + [
            f"### {name}" for name in companies_drafted
        ]
        passed, gaps = verify_report(client, content, evidence_log, required_headers, system_prompt=_INDEX_VERIFIER_PROMPT)
        if passed:
            return (
                "Index report written and passed the reliability check. Done — no need to call write_file again."
            )
        verify_attempts["count"] += 1
        if verify_attempts["count"] > MAX_VERIFY_ATTEMPTS:
            caveat_block = (
                f"\n\n## Reliability caveats (unresolved after {MAX_VERIFY_ATTEMPTS} correction attempts)\n"
                + "\n".join(f"- {g}" for g in gaps)
            )
            write_file(filename, content + caveat_block)
            last_written_content["text"] = content + caveat_block
            return (
                f"Correction attempts exhausted ({MAX_VERIFY_ATTEMPTS} max). Finalized as-is with "
                f"remaining gaps appended. Do not call write_file again — give your closing summary now."
            )
        gap_list = "\n".join(f"- {g}" for g in gaps)
        return (
            f"Index report written, but the reliability check (attempt "
            f"{verify_attempts['count']}/{MAX_VERIFY_ATTEMPTS}) found gaps:\n{gap_list}\n\nAddress these, then call write_file again."
        )

    tool_functions = {
        "set_applicant_contact": lambda **kw: set_applicant_contact(**kw),
        "log_startup_found": lambda **kw: log_startup_found(**kw),
        "propose_fit_point": lambda **kw: propose_fit_point(**kw),
        "propose_email_draft": lambda **kw: propose_email_draft(**kw),
        "save_startup_outreach": lambda **kw: save_startup_outreach_step(**kw),
        "write_file": lambda **kw: verified_write_file(kw["filename"], kw["content"]),
    }

    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Set applicant contact, then find and draft outreach for about {count} startups."},
    ]

    def finish(reason: str) -> str:
        print(f"\n{client.summary()}")
        print(f"({reason})")
        return last_written_content["text"] or "Stopped before writing a final report — check sandbox/drafts/startups/ for completed drafts and OUTBOX.html."

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ({client.total_tokens:,} tokens used so far) ---")
        try:
            response = client.responses.create(model=MODEL, tools=TOOLS, input=input_items)
        except _BudgetExceeded as e:
            print(f"\nTOKEN BUDGET EXCEEDED: {e}")
            return finish("stopped mid-turn on the main model call — any companies already saved are safe on disk")
        input_items += response.output

        function_calls = []
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if getattr(content, "text", "").strip():
                        print(f"Model: {content.text.strip()}")
            elif item.type == "function_call":
                print(f"Model wants to call: {item.name}({item.arguments[:150]})")
                function_calls.append(item)
            elif item.type == "web_search_call":
                query = getattr(item.action, "query", "")
                print(f"Model is web-searching: {query}")
                evidence_log.append({"tool": "web_search", "query": query})

        if not function_calls:
            print(f"\n{client.summary()}")
            return last_written_content["text"] or response.output_text

        for item in function_calls:
            args = json.loads(item.arguments)
            func = tool_functions.get(item.name)
            try:
                result = func(**args)
            except _BudgetExceeded as e:
                print(f"\nTOKEN BUDGET EXCEEDED: {e}")
                return finish(f"stopped mid-turn during {item.name} — any companies already saved are safe on disk")
            except Exception as e:
                result = f"Error running {item.name}: {e}"
            print(f"  -> {result[:200]}")
            input_items.append({"type": "function_call_output", "call_id": item.call_id, "output": result})

    return finish(f"hit max_turns ({max_turns}) without finishing")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            'Usage: python startup_outreach_agent.py <profile_file> "<areas_comma_separated>" [count] [token_budget]\n'
            'Example: python startup_outreach_agent.py my_profile.md "ML,AI,Quant" 8 500000\n\n'
            "Default count is 8. Each run excludes companies drafted in a previous run "
            "(see contacted_companies.json). Output lands in sandbox/drafts/startups/, including "
            "OUTBOX.html — a review page with a prefilled email or form fields per company. "
            "Nothing is sent; you open the outbox and click.\n\n"
            "token_budget is optional — if set, the run stops itself (keeping saved drafts) once "
            "total tokens cross that number."
        )
        sys.exit(1)

    profile_path = sys.argv[1]
    if not Path(profile_path).exists():
        print(f"Profile file not found: {profile_path}")
        sys.exit(1)
    profile_text = Path(profile_path).read_text()

    areas = [a.strip() for a in sys.argv[2].split(",") if a.strip()]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    token_budget = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"Areas: {areas}\nTarget count: {count}\nToken budget: {token_budget or '(none set)'}\n")

    answer = run_agent(areas, profile_text, count, token_budget=token_budget)
    print("\n=== Final answer ===")
    print(answer)
