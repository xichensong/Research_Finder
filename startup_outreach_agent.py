"""
Finds ML/AI/quant startups backed by a credible accelerator or tier-1 VC,
figures out ONE specific point where what they build genuinely connects to
something the person has actually done, and drafts a cold email around that
connection. Never sends anything.

A startup does NOT need an open internship posting to be included. If a real
opening is found, the email references it; if not, the email just makes the
case for what the person could contribute. Either way the anchor is a real,
specific connection between the company's work and the person's work, not
generic enthusiasm.

Honesty boundary: web_search returns company descriptions, batch pages,
funding announcements, and job-post summaries, not a deep read of a
codebase or a full JD. "What they do" and "the role" mean what search
actually surfaced, and the gates below check the draft against that, not
against some fuller truth nobody retrieved.

PIPELINE (four sequential, code-enforced steps, so there's something real
to check between them instead of just asking the model to be careful):

  1. log_startup_found — captures what was actually found for one company
     (name, backer, what they do, a posting or careers URL, a summary of
     what search returned, whether a specific opening exists).
     -> GATE 1 (credibility + plausibility): independent check that this
        looks like a real company with a backer in the credible set, a
        real-looking URL, specific not generic content, and a matching
        search query in the evidence log. Not a fact-check (there is no
        fetch tool to confirm a URL) — a sanity check for obvious
        fabrication and for backers that don't clear the bar.

  2. propose_fit_point — the model's draft of the one connection between
     the company's work and the person's background.
     -> GATE 2 (grounding): independent check that the fit point's claims
        ABOUT THE COMPANY stay within what the logged summary supports —
        catches specific technical claims bolted onto a thin description.
     -> HUMANIZER PASS (humanizer.py): only after the gate passes.

  3. propose_email_draft — the model's draft of the actual email.
     -> HUMANIZER PASS.

  4. save_startup_outreach(company_name) — takes ONLY the name. The fit
     point and email text are pulled from the gated/humanized versions
     tracked in code, not whatever the model passes here — so a gate can't
     be bypassed by retyping different text at the last step.

Then one index report (write_file) listing every company that made it all
the way through — humanized (same de-slop pass as the fit points and
emails), then checked by the grounding verifier.

CROSS-RUN NON-OVERLAP: every company that finishes the pipeline is recorded
permanently (action_tools.mark_company_contacted, called inside
save_startup_outreach) — this persists across separate runs. Each new run
loads that list and (a) tells the model who to skip, and (b) rejects
log_startup_found in code if the name matches — so a later run can't
overlap with an earlier one even if the model ignores the instruction.

Run it:
  python startup_outreach_agent.py my_profile.md "ML,AI,Quant" 8
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

from action_tools import get_contacted_companies, save_startup_outreach as _persist_startup_outreach
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
        "name": "log_startup_found",
        "description": (
            "Step 1 of 4. Log a startup you found via web_search. Only call "
            "this for a REAL company whose backer clears the credible-set "
            "bar in your instructions — never invent a name, URL, backer, or "
            "batch. `what_they_do` and `posting_summary` should be what "
            "web_search actually surfaced, specific and honest — this "
            "becomes the evidence everything downstream is checked against. "
            "Set `has_open_role` true only if you found an actual open "
            "internship / new-grad / early-career posting; otherwise false, "
            "and put the careers or company page in `posting_url`. Skip a "
            "company entirely if you can't confirm a credible backer rather "
            "than logging a guess."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "backer": {
                    "type": "string",
                    "description": "The specific accelerator/batch or the specific investor(s) and round, as found via search.",
                },
                "what_they_do": {"type": "string", "description": "What search actually said the company builds, specifically."},
                "posting_url": {
                    "type": "string",
                    "description": "Real URL: the specific job posting if has_open_role, else the careers page or company/YC page.",
                },
                "posting_summary": {
                    "type": "string",
                    "description": "What you actually found — the role description if there's an opening, plus company/funding context.",
                },
                "has_open_role": {"type": "boolean"},
            },
            "required": ["company_name", "backer", "what_they_do", "posting_url", "posting_summary", "has_open_role"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "propose_fit_point",
        "description": (
            "Step 2 of 4. Must be called AFTER log_startup_found for this "
            "company. Propose the ONE specific point where what this company "
            "builds connects to the HIGHEST-PRIORITY piece of the person's "
            "background that genuinely applies — check the priority order in "
            "your instructions before falling back to a lower-priority item. "
            "The point should show the person could contribute to this "
            "company's work, not just that they find it interesting. This is "
            "checked against the logged company summary — if it introduces "
            "claims about the company that summary doesn't support, it will "
            "be rejected and you'll need to revise it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Must match a company already logged via log_startup_found."},
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
            "Step 3 of 4. Must be called AFTER propose_fit_point succeeded "
            "for this company. Draft the actual cold email as exactly THREE "
            "paragraphs, 120-180 words total:\n"
            "Paragraph 1 (shortest): show you know specifically what they're "
            "building, not generic praise. If a specific role was found, "
            "name it here and say you're writing about it.\n"
            "Paragraph 2 (LONGEST — the bulk of the email): connect their "
            "work to the person's own experience and explain specifically "
            "what the person could contribute. This is the approved fit "
            "point, developed in full.\n"
            "Paragraph 3: add other relevant background, then a direct ask "
            "— if there's an open role, ask to be considered and to talk; "
            "if not, ask whether they take interns / would consider bringing "
            "someone on, and ask for a short call.\n"
            "Paragraph 2 must clearly be the longest of the three."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "email_draft": {"type": "string"},
            },
            "required": ["company_name", "email_draft"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "save_startup_outreach",
        "description": (
            "Step 4 of 4. Finalizes and saves one company's outreach — "
            "takes only the name; the approved, humanized fit point and "
            "email are pulled from what already passed the earlier steps. "
            "Must be called after propose_email_draft succeeded for this "
            "company. Never sends anything — saves a draft and logs a "
            "tracked item."
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
specific person's real background, and draft cold emails — never send \
anything.

PURPOSE — read this carefully, it changes what a good email looks like: \
the goal is to get the person an internship or early-career role at the \
company, or failing a formal opening, to get a founder or hiring lead on a \
short call open to bringing them on. The fit point exists to show the \
person could contribute to what the company is building, not to show they \
find it interesting. Every email ends with a direct, specific ask — not \
"I'd love to hear your thoughts," an actual request to be considered and \
to talk.

FOCUS: startups working in {areas}. Prefer companies where the technical \
work is close to what the person has actually done.

CREDIBLE BACKERS — a company is only eligible if it clears this bar. \
Confirm the backer via search before logging the company; skip any company \
you can't tie to one of these:
{backers}

OPEN ROLE IS OPTIONAL: a company with no visible internship posting is \
still eligible. If you find a real opening (internship, new-grad, \
early-career), the email references it specifically. If not, the email \
makes the case for what the person could contribute without pretending a \
posting exists.

PRIORITY ORDER for what background material to draw the connection from — \
check the profile below for the person's own stated priority order; if \
given, follow it. Otherwise use this default, picking the item that \
genuinely and specifically fits THIS company's work: (1) a company or \
product the person founded or shipped themselves, especially one with real \
users — for a startup this is the strongest signal, so lead with it \
whenever it genuinely connects; (2) independent or first-author research \
papers; (3) internships or research-assistant positions. Only fall back to \
a lower-priority item when a higher one doesn't genuinely fit this \
company's work — not just because a lower one is topically closer.

ALREADY CONTACTED — do not pick any of these companies, they were covered \
in a previous run: {already_contacted}

WORKFLOW, for a target of about {count} companies, using the four-step \
pipeline (log_startup_found -> propose_fit_point -> propose_email_draft -> \
save_startup_outreach) FOR EACH company:

1. Search for startups in {areas} tied to a credible backer. Good starting \
points: ycombinator.com/companies (filter by industry), a16z Speedrun and \
AI Grant cohort pages, Techstars portfolio pages, and recent funding \
announcements naming a tier-1 firm. Do not rely on a guessed URL — find \
the real one via search.
2. For EACH company, confirm the backer and find what they build, plus any \
open early-career role, then call log_startup_found with what you actually \
found. Budget 2-4 web_search calls per company. Skip a company if you \
can't confirm a credible backer rather than logging a guess.
3. Call propose_fit_point with the ONE specific point, drawn from the \
highest-priority applicable background item, that shows the person could \
contribute. Checked against what you logged — if rejected, revise to stay \
within what the logged summary supports and call it again.
4. Call propose_email_draft with the email as exactly three paragraphs, \
120-180 words: (1) short, show you know what they build, name the role if \
there is one; (2) LONGEST, connect it to the person's experience and what \
they could contribute, leading with the highest-priority background item \
that fits (see PRIORITY ORDER above); (3) other relevant background plus a \
direct ask. Paragraph 2 must clearly be the longest. Avoid cold-email \
cliches: no "the part that caught my eye," "what I keep coming back to," \
"this resonates with," "I'd love to." Say the specific thing plainly.
5. Call save_startup_outreach with just the company name to finalize.

Do not fabricate a company, a backer, an opening, or a connection to the \
person's background that isn't in their profile below. If you can't find \
enough real material for a company, skip it — fewer real, well-grounded \
drafts beat more fabricated ones.

After completing the pipeline for each company, write an index report using \
write_file, in this exact structure:

  # Startup outreach — {areas}

  ## Summary
  2-3 sentences: how many companies you found and drafted outreach for, \
how many had an open role vs cold outreach, and any notable gap.

  ## <One "###" subsection per company you completed the pipeline for, \
using the company name as the heading>
  For each: 1 sentence on what they do and their backer, 1 sentence on \
whether there's an open role, 1-2 sentences on the fit point. A summary of \
the dedicated draft file, not a copy of the full email.

  ## Confidence & what would change this
  1-2 sentences: how confident you are in the QUALITY of the fits found and \
in the backer confirmations, and what would improve them.

  ## Sources
  Flat list of every URL you used.

After you call write_file, an independent reliability check reviews the \
index report against what your tools actually returned this run. If it \
finds gaps, address them and call write_file again. Limited correction \
attempts.

Keep every section tight — stick to the stated length.

PROFILE (use only real facts from this — do not invent anything about this \
person):
{profile_text}
"""


def build_system_prompt(
    areas: list[str], count: int, profile_text: str, already_contacted: list[str]
) -> str:
    return _BASE_PROMPT.format(
        count=count,
        areas=", ".join(areas),
        backers=CREDIBLE_BACKERS,
        profile_text=profile_text,
        already_contacted=", ".join(already_contacted) if already_contacted else "(none yet — this is the first run)",
    )


# --- Gate helpers ----------------------------------------------------------
#
# Same VERDICT: PASS/FAIL + GAPS: pattern as verification.py. Neither gate
# can browse the web, so neither can CONFIRM a company or backer is real —
# they catch obvious signs of fabrication (missing URL, generic content,
# a backer nowhere near the credible set, claims that outrun the logged
# evidence), not certify accuracy.

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
You are a plausibility and credibility auditor. You cannot browse the web \
yourself, so you cannot confirm a company or its funding actually exists — \
your job is to catch OBVIOUS signs of fabrication or a backer that plainly \
doesn't clear the bar, not to certify accuracy.

The credible-backer bar is:
{backers}

Given a company's logged details and the web_search queries performed this \
run, FAIL if:
- No real-looking URL was given (empty, or an obvious placeholder)
- The company description or posting summary reads as generic/vague rather \
than a specific real company (e.g. "an AI startup" with no actual product)
- The stated backer is clearly not in the credible set above, or is so \
vague it can't be checked ("some VCs", "well funded")
- No web_search query in the evidence log plausibly relates to finding \
this company or confirming its backer

Otherwise PASS. Do not fail something just because you personally don't \
recognize the company — that's not evidence of fabrication.

Respond in EXACTLY this format:
VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap>
"""

_FIT_GATE_PROMPT = """\
You are a grounding auditor for one specific claim. You'll see a company \
summary (what was actually found via search) and a proposed "fit point" \
describing how that company's work connects to someone's background.

Check ONLY whether the fit point's claims ABOUT THE COMPANY stay consistent \
with the summary, or introduce specific technical details the summary \
doesn't support. Claims about the PERSON's background are not what you're \
checking here.

If the summary is thin and the fit point makes an oddly specific technical \
claim about the company's stack, methods, or roadmap that the summary \
doesn't support, FAIL it as likely fabricated elaboration.

Respond in EXACTLY this format:
VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap>
"""


_INDEX_VERIFIER_PROMPT = """\
You are a methodology auditor for a startup-outreach index report. You did \
NOT write it and have no stake in it being good — your only job is to catch \
grounding problems.

You'll see the index report and a log of everything the tools actually \
returned this run (log_startup_found entries, propose_fit_point results, \
web_search queries). Check ONLY:

1. Every company named as a "###" subsection actually has a matching \
log_startup_found entry in the evidence log (i.e. wasn't invented for the \
report after the fact).
2. The backer named for each company in the report matches what was \
actually logged for them, not a different or upgraded backer.
3. Any company the report says has an open role was actually logged with \
has_open_role true.
4. Nothing in the report claims a company was drafted if its pipeline \
never completed in the evidence log.

Do NOT critique writing style or judge whether the fits are strong enough \
(that's a quality judgment, not a grounding one), and do NOT invent gaps \
beyond these four categories.

Respond in EXACTLY this format, nothing else:

VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap, naming the company and what's wrong>
"""


class _TrackedClient:
    """
    Wraps the real OpenAI client so every call to .responses.create() —
    from the main loop, AND from _run_gate / humanize_text / verify_report,
    since they all receive this same object — gets its token usage counted.
    This is the only way to get a true total: those side-calls are real,
    separate API requests, easy to forget when estimating cost by eyeballing
    the main loop alone.

    Reports TOKENS, not dollars — convert using the exact rate on your
    OpenAI billing dashboard.

    `token_budget`, if set, raises _BudgetExceeded once the running total
    crosses it — checked after every call, so a run can't blow past the cap
    by more than one call's worth of tokens.
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
    real_client = OpenAI()  # constructed here, not at import time
    client = _TrackedClient(real_client, token_budget=token_budget)

    already_contacted = get_contacted_companies()  # persists across runs — see module docstring
    already_contacted_lower = {name.lower() for name in already_contacted}
    system_prompt = build_system_prompt(areas, count, profile_text, already_contacted)
    plausibility_prompt = _PLAUSIBILITY_GATE_PROMPT.format(backers=CREDIBLE_BACKERS)

    evidence_log: list[dict] = []
    companies_drafted: list[str] = []  # only companies that finish the FULL pipeline
    company_state: dict[str, dict] = {}
    verify_attempts = {"count": 0}
    last_written_content = {"text": None}

    def log_startup_found(company_name, backer, what_they_do, posting_url, posting_summary, has_open_role) -> str:
        if company_name.lower() in already_contacted_lower:
            return (
                f"REJECTED — '{company_name}' was already contacted in a previous run "
                f"(this is enforced, not just a prompt reminder). Pick a different company."
            )
        evidence_log.append(
            {
                "tool": "log_startup_found",
                "args": {"company_name": company_name, "backer": backer, "posting_url": posting_url, "has_open_role": has_open_role},
                "result": f"backer={backer}, url={posting_url}, has_open_role={has_open_role}, "
                f"what_they_do={what_they_do}, summary={posting_summary}",
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
                "GATE 1 (credibility + plausibility) FAILED — do not proceed to propose_fit_point for this "
                "company yet:\n" + "\n".join(f"- {g}" for g in gaps)
                + "\nEither search again for stronger evidence (especially a confirmable backer), or skip this company."
            )
        company_state[company_name] = {
            "backer": backer,
            "what_they_do": what_they_do,
            "posting_url": posting_url,
            "posting_summary": posting_summary,
            "has_open_role": bool(has_open_role),
        }
        return f"Logged and passed the credibility check. Proceed to propose_fit_point for {company_name}."

    def propose_fit_point(company_name, fit_point_draft) -> str:
        if company_name not in company_state:
            return f"No company logged for '{company_name}' yet — call log_startup_found first."

        passed, gaps = _run_gate(
            client,
            _FIT_GATE_PROMPT,
            f"Company summary: {company_state[company_name]['what_they_do']}\n"
            f"{company_state[company_name]['posting_summary']}\n\n"
            f"Proposed fit point: {fit_point_draft}",
        )
        if not passed:
            return (
                "GATE 2 (grounding) FAILED — this fit point was not saved:\n"
                + "\n".join(f"- {g}" for g in gaps)
                + "\nRevise it to stay within what the logged company summary actually supports, and call "
                "propose_fit_point again."
            )

        humanized = humanize_text(
            client, fit_point_draft, context="a specific technical fit point in a cold email to a startup founder"
        )
        company_state[company_name]["fit_point"] = humanized
        evidence_log.append(
            {"tool": "propose_fit_point", "args": {"company_name": company_name}, "result": humanized}
        )
        return f"Fit point passed grounding check and was humanized. Proceed to propose_email_draft for {company_name}."

    def propose_email_draft(company_name, email_draft) -> str:
        if company_name not in company_state or "fit_point" not in company_state[company_name]:
            return f"No approved fit point for '{company_name}' yet — call propose_fit_point first."

        humanized = humanize_text(client, email_draft, context="a cold email from a student to a startup founder, 120-180 words")
        company_state[company_name]["email_draft"] = humanized
        return f"Email drafted and humanized. Proceed to save_startup_outreach for {company_name}."

    def save_startup_outreach_step(company_name) -> str:
        state = company_state.get(company_name, {})
        if "fit_point" not in state or "email_draft" not in state:
            return (
                f"Cannot finalize '{company_name}' — the pipeline isn't complete "
                f"(need propose_fit_point and propose_email_draft to succeed first)."
            )
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
        )
        companies_drafted.append(company_name)
        return result

    def verified_write_file(filename: str, content: str) -> str:
        # Humanize before writing — the index report is model-generated prose
        # like the fit points and emails, so it gets the same de-slop pass.
        # The context note keeps headings and URLs intact so the section
        # check below still finds them.
        content = humanize_text(
            client,
            content,
            context=(
                "a markdown index report. Preserve every heading line (starting with "
                "#, ##, or ###) verbatim, including company names in them, and preserve "
                "every URL exactly. Only rewrite the prose sentences under the headings."
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
                "Index report written and passed the reliability check: every company that completed "
                "the pipeline is represented. Done — no need to call write_file again."
            )

        verify_attempts["count"] += 1
        if verify_attempts["count"] > MAX_VERIFY_ATTEMPTS:
            caveat_block = (
                f"\n\n## Reliability caveats (unresolved after {MAX_VERIFY_ATTEMPTS} "
                f"correction attempts)\n" + "\n".join(f"- {g}" for g in gaps)
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
            f"{verify_attempts['count']}/{MAX_VERIFY_ATTEMPTS}) found gaps:\n{gap_list}\n\n"
            f"Address these, then call write_file again."
        )

    tool_functions = {
        "log_startup_found": lambda **kw: log_startup_found(**kw),
        "propose_fit_point": lambda **kw: propose_fit_point(**kw),
        "propose_email_draft": lambda **kw: propose_email_draft(**kw),
        "save_startup_outreach": lambda **kw: save_startup_outreach_step(**kw),
        "write_file": lambda **kw: verified_write_file(kw["filename"], kw["content"]),
    }

    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Find and draft outreach for about {count} startups, as instructed."},
    ]

    def finish(reason: str) -> str:
        print(f"\n{client.summary()}")
        print(f"({reason})")
        return last_written_content["text"] or "Stopped before writing a final report — check sandbox/drafts/startups/ for any completed drafts."

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ({client.total_tokens:,} tokens used so far) ---")
        try:
            response = client.responses.create(model=MODEL, tools=TOOLS, input=input_items)
        except _BudgetExceeded as e:
            print(f"\nTOKEN BUDGET EXCEEDED: {e}")
            return finish("stopped mid-turn on the main model call — any companies already saved before this are safe on disk")
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
                return finish(f"stopped mid-turn during {item.name} — any companies already saved before this are safe on disk")
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
            "Default count is 8. Each run automatically excludes companies already drafted in a "
            "previous run (see contacted_companies.json) — non-overlapping batches, no need to track it yourself.\n\n"
            "token_budget is optional — if set, the run stops itself (keeping any drafts already saved) "
            "once total input+output tokens for the run cross that number. Leave it off for your first run "
            "so you can see real usage in the printed summary, then set an informed budget for later runs."
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
