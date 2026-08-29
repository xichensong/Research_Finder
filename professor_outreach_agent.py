"""
Finds professors at a given university working in given research areas,
finds each one's most recent paper, identifies ONE specific point from it
that genuinely connects to your background, and drafts a cold email around
that connection — never sends it.

Honesty boundary worth being explicit about: web_search gives the model
paper abstracts/summaries/citations, not necessarily full-text access to a
PDF. So "read through it and pick out a point" realistically means "read
whatever web_search actually surfaced (usually an abstract or summary) and
find a genuine point in THAT" — not a deep technical read of the full
paper.

PIPELINE (this is the part that changed from a single do-everything tool
call into four sequential, code-enforced steps, so there's something real
to check between them instead of just prompting the model to be careful):

  1. log_paper_found — captures what was actually found for one professor
     (name, department, paper, and a summary of what search returned).
     This becomes real evidence, not a claim.
     -> GATE 1 (plausibility): independent check that this looks like a
        real find (real-looking URL, specific not generic, a matching
        search query in the evidence log) — not a fact-check (no fetch
        tool exists to confirm the URL is real), a sanity check for
        obvious fabrication.

  2. propose_connection_point — the model's draft of the one connection.
     -> GATE 2 (grounding): independent check that the connection point's
        claims ABOUT THE PAPER don't go beyond what the logged summary
        actually supports — catches "hallucinated elaboration," where a
        thin abstract gets an oddly specific technical claim bolted onto it.
     -> HUMANIZER PASS (humanizer.py): only runs after the gate passes —
        never humanize something that might still get rejected.

  3. propose_email_draft — the model's draft of the actual email.
     -> HUMANIZER PASS.

  4. save_professor_outreach(professor_name) — takes ONLY the name. The
     connection point and email text are pulled from the gated/humanized
     versions tracked in code, not whatever the model might pass here —
     so there's no path where a gate gets bypassed by just retyping
     different text at the last step.

Then one index report (write_file, verified — same grounding checkpoint
pattern as the other agents) listing every professor who made it all the
way through the pipeline.

CROSS-RUN NON-OVERLAP: every professor who finishes the pipeline gets
recorded permanently (action_tools.mark_professor_contacted, called
automatically inside save_professor_outreach) — this persists across
separate runs of this script, not just within one run. Each new run loads
that list and (a) tells the model explicitly who to skip, and (b) rejects
log_paper_found in code if the name matches — so a later run can't overlap
with an earlier one even if the model ignores the prompt instruction.

Run it:
  python professor_outreach_agent.py my_profile.md "UC Berkeley" "ML,AI,Quant,Probability,RL,Deep Learning" 10
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

from action_tools import get_contacted_professors, save_professor_outreach as _persist_professor_outreach
from humanizer import humanize_text
from tools import write_file
from verification import MAX_VERIFY_ATTEMPTS, verify_report

MODEL = "gpt-5.6"

TOOLS = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "log_paper_found",
        "description": (
            "Step 1 of 4. Log a professor and the specific paper you found "
            "for them via web_search. `paper_summary` should be what "
            "web_search actually surfaced (usually an abstract or a short "
            "summary) — be honest and specific, this becomes the evidence "
            "everything downstream is checked against. Only call this for "
            "a REAL professor and REAL paper — never invent a name, paper, "
            "or URL. Skip a professor entirely if you can't find a real "
            "recent paper rather than filling this in with something vague."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "professor_name": {"type": "string"},
                "department": {"type": "string"},
                "paper_title": {"type": "string"},
                "paper_url": {"type": "string", "description": "The real URL where you found it."},
                "paper_summary": {
                    "type": "string",
                    "description": "What you actually found — the abstract or summary content, as specifically as you have it.",
                },
            },
            "required": ["professor_name", "department", "paper_title", "paper_url", "paper_summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "propose_connection_point",
        "description": (
            "Step 2 of 4. Must be called AFTER log_paper_found for this "
            "professor. Propose the ONE specific technical point from the "
            "logged paper that connects to the HIGHEST-PRIORITY piece of "
            "the person's background that genuinely applies — check the "
            "priority order given in your instructions (papers first, then "
            "the shipped app, then internships) before falling back to a "
            "lower-priority item. The point should demonstrate the person "
            "could contribute to this research, not just that they find it "
            "interesting — this is groundwork for an ask to join the work, "
            "not small talk about the paper. State explicitly whether your "
            "read of the paper was abstract/summary-level or deeper. This "
            "gets checked against the logged paper summary — if it "
            "introduces claims about the paper that summary doesn't "
            "support, it will be rejected and you'll need to revise it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "professor_name": {"type": "string", "description": "Must match a professor already logged via log_paper_found."},
                "connection_point_draft": {"type": "string"},
            },
            "required": ["professor_name", "connection_point_draft"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "propose_email_draft",
        "description": (
            "Step 3 of 4. Must be called AFTER propose_connection_point "
            "succeeded for this professor. Draft the actual cold email as "
            "exactly THREE paragraphs, 120-180 words total:\n"
            "Paragraph 1 (shortest): state that you read this specific "
            "recent paper and found it genuinely interesting — reference "
            "something specific about it, not generic praise.\n"
            "Paragraph 2 (LONGEST — the bulk of the email): connect the "
            "paper to your own experience, explaining specifically why "
            "that connection matters. This is the approved connection "
            "point, developed in full — the substantive core of the email.\n"
            "Paragraph 3: add other relevant background beyond the "
            "paragraph 2 connection, then explicitly express interest in "
            "participating in the professor's research and ask to meet or "
            "talk.\n"
            "Paragraph 2 must clearly be the longest of the three."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "professor_name": {"type": "string"},
                "email_draft": {"type": "string"},
            },
            "required": ["professor_name", "email_draft"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "save_professor_outreach",
        "description": (
            "Step 4 of 4. Finalizes and saves one professor's outreach — "
            "takes only the name; the approved, humanized connection point "
            "and email are pulled from what already passed the earlier "
            "steps. Must be called after propose_email_draft succeeded for "
            "this professor. Never sends anything — saves a draft and logs "
            "a tracked item."
        ),
        "parameters": {
            "type": "object",
            "properties": {"professor_name": {"type": "string"}},
            "required": ["professor_name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write the index report listing all professors found this run.",
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
You find professors, connect their real recent work to a specific person's \
real background, and draft cold emails — never send anything.

PURPOSE — read this carefully, it changes what a good email looks like: \
the goal is NOT to strike up a research discussion. The goal is to get \
involved in the professor's research — join their lab, work with them as \
an RA, contribute to a specific project — and to get a face-to-face \
conversation (a meeting or call) to make that happen. The connection point \
exists to demonstrate the person could contribute to this research, not to \
show they find it interesting. Every email must end with a direct, \
specific ask along those lines — not "I'd love to hear your thoughts" or \
"happy to discuss further," an actual request to get involved and to talk.

PRIORITY ORDER for what background material to draw the connection from — \
check the profile below for the person's own stated priority order (they \
may have ranked what to lead with); if given, follow it. Otherwise use \
this default: (1) any self-initiated or first-author research work — this \
is the strongest signal of independent research capability, especially if \
the profile notes it was built without a formal research position (that's \
worth mentioning briefly in the email — it shows initiative, not just \
skill); (2) a shipped, real product with real users; (3) internships or \
research-assistant positions. Find the connection from the HIGHEST-priority \
item that genuinely and specifically applies to this particular \
professor's paper — only fall back to a lower-priority item when the \
higher one truly doesn't fit this paper. Do not default to whatever seems \
topically closest if a higher-priority item also works.

ALREADY CONTACTED — do not pick any of these professors, they were covered \
in a previous run: {already_contacted}

WORKFLOW, for a target of about {count} professors, using the four-step \
pipeline (log_paper_found -> propose_connection_point -> \
propose_email_draft -> save_professor_outreach) FOR EACH professor:

1. Search for faculty at {university} working in: {areas}. Use web_search — \
department faculty listing pages are usually the most reliable starting \
point. Find the university's actual domain yourself via search (don't \
guess it — e.g. "UC Berkeley" is berkeley.edu, not the naive guess you'd \
construct from the name) before using a site: filter.
2. For EACH professor found, search for their most recent publication, \
then call log_paper_found with what you actually found. Only proceed with \
a professor if you find a REAL, specific, recent paper — skip a professor \
if you can't find one rather than inventing one. Budget your searches: 2-3 \
web_search calls per professor is enough (one for the faculty listing, one \
or two for their most recent paper) — don't keep searching once you have a \
specific real paper, and don't re-search a professor you've already logged.
3. Call propose_connection_point with the ONE specific technical point, \
drawn from the highest-priority applicable background item (see PRIORITY \
ORDER above), that shows the person could contribute to this research — \
not just that they find it interesting. This is checked against what you \
logged — if it's rejected, revise it to stay within what the logged \
summary actually supports and call it again.
4. Call propose_email_draft with the actual email as exactly three \
paragraphs, 120-180 words total: (1) short — read this specific paper, \
found it genuinely interesting, reference something specific about it; \
(2) LONGEST, the substantive core — connect it to the person's own \
experience and explain why that connection matters, developing the \
approved connection point in full; (3) other relevant background plus an \
explicit ask to participate in the research and to meet or talk. \
Paragraph 2 must clearly be the longest of the three.
5. Call save_professor_outreach with just the professor's name to finalize.

Do not fabricate a professor, a paper, or a connection to the person's \
background that isn't actually in their profile below. If you can't find \
enough real material for a professor, skip them entirely rather than \
filling gaps with plausible-sounding invention — fewer, real, \
well-grounded drafts beat more fabricated ones.

After you have completed the pipeline for each professor, write an index \
report using write_file, in this exact structure:

  # Professor outreach — {university}, {areas}

  ## Summary
  2-3 sentences: how many professors you found and drafted outreach for, \
and any notable gap (e.g. an area with no strong match found).

  ## <One "###" subsection per professor you completed the pipeline for, \
using their name as the heading>
  For each: 1-2 sentences on who they are and their department, 1 sentence \
naming the paper, 1-2 sentences on the connection point. This is a summary \
of what's in their dedicated draft file, not a duplicate of the full email.

  ## Confidence & what would change this
  1-2 sentences: how confident you are in the QUALITY of the connections \
found (not whether emailing will work — that's not knowable), and what \
would improve them (e.g. finding more than an abstract for a given paper).

  ## Sources
  Flat list of every URL you used.

After you call write_file, an independent reliability check reviews the \
index report against what your tools actually returned this run — checking \
that every professor named in it actually completed the full pipeline. If \
it finds gaps, address them and call write_file again. Limited correction \
attempts.

Keep every section tight — stick to the stated length.

PROFILE (use only real facts from this — do not invent anything about this \
person):
{profile_text}
"""


def build_system_prompt(
    university: str, areas: list[str], count: int, profile_text: str, already_contacted: list[str]
) -> str:
    return _BASE_PROMPT.format(
        count=count,
        university=university,
        areas=", ".join(areas),
        profile_text=profile_text,
        already_contacted=", ".join(already_contacted) if already_contacted else "(none yet — this is the first run)",
    )


# --- Gate helpers ------------------------------------------------------------
#
# Same VERDICT: PASS/FAIL + GAPS: pattern as verification.py, reused here
# for two lightweight, honest checks: neither can independently browse the
# web, so neither can CONFIRM a paper is real — they catch obvious signs of
# fabrication (missing URL, generic content, claims that outrun the logged
# evidence), not certify accuracy.

def _run_gate(client: OpenAI, system_instructions: str, user_content: str) -> tuple[bool, list[str]]:
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ],
        reasoning={"effort": "low"},  # a PASS/FAIL sanity check, not a task needing deep reasoning
    )
    text = response.output_text.strip()
    if text.upper().startswith("VERDICT: FAIL"):
        gaps = [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]
        return False, gaps or ["Gate failed but returned no specific gaps."]
    return True, []


_PLAUSIBILITY_GATE_PROMPT = """\
You are a plausibility auditor. You cannot browse the web yourself, so you \
cannot confirm a paper actually exists — your job is to catch OBVIOUS signs \
of fabrication, not certify accuracy.

Given a professor's logged details and the web_search queries performed \
this run, FAIL if:
- No real-looking URL was given (empty, or an obvious placeholder)
- The paper title or summary reads as generic/vague rather than a specific \
real paper (e.g. "a paper about machine learning" instead of an actual \
topic/method)
- No web_search query in the evidence log plausibly relates to finding \
this professor or paper

Otherwise PASS. Do not fail something just because you personally don't \
recognize the professor or paper — that's not evidence of fabrication.

Respond in EXACTLY this format:
VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap>
"""

_CONNECTION_GATE_PROMPT = """\
You are a grounding auditor for one specific claim. You'll see a paper \
summary (what was actually found via search) and a proposed "connection \
point" describing how that paper connects to someone's background.

Check ONLY whether the connection point's claims ABOUT THE PAPER stay \
consistent with the summary, or introduce specific technical details the \
summary doesn't support. Claims about the PERSON's background are not what \
you're checking here.

If the summary is thin (abstract-only) and the connection point makes an \
oddly specific technical claim the summary doesn't support, FAIL it as \
likely fabricated elaboration.

Respond in EXACTLY this format:
VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap>
"""


_INDEX_VERIFIER_PROMPT = """\
You are a methodology auditor for a professor-outreach index report. You did \
NOT write it and have no stake in it being good — your only job is to catch \
grounding problems.

You'll see the index report and a log of everything the tools actually \
returned this run (log_paper_found entries, propose_connection_point \
results, web_search queries). Check ONLY:

1. Every professor named as a "###" subsection actually has a matching \
log_paper_found entry in the evidence log (i.e. wasn't invented for the \
report after the fact).
2. The paper named for each professor in the report matches what was \
actually logged for them, not a different or embellished paper.
3. Nothing in the report claims a professor was contacted/drafted if their \
pipeline never completed in the evidence log.

Do NOT critique writing style or judge whether the connections are strong \
enough (that's a quality judgment, not a grounding one), and do NOT invent \
gaps beyond these three categories.

Respond in EXACTLY this format, nothing else:

VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap, naming the professor and what's wrong>
"""


class _TrackedClient:
    """
    Wraps the real OpenAI client so every call to .responses.create() —
    from the main loop, AND from _run_gate / humanize_text / verify_report,
    since they all receive this same object — gets its token usage counted.
    This is the only way to get a true total: those side-calls are real,
    separate API requests, easy to forget when estimating cost by eyeballing
    the main loop alone.

    Reports TOKENS, not dollars — third-party pricing sources for this model
    were inconsistent and OpenAI's own pricing page couldn't be fetched to
    verify, so a fabricated cost estimate would be worse than none. Convert
    using the exact rate on your OpenAI billing dashboard.

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
    university: str,
    areas: list[str],
    profile_text: str,
    count: int = 10,
    max_turns: int = 120,
    token_budget: int | None = None,
) -> str:
    real_client = OpenAI()  # constructed here, not at import time
    client = _TrackedClient(real_client, token_budget=token_budget)

    already_contacted = get_contacted_professors()  # persists across runs — see module docstring
    already_contacted_lower = {name.lower() for name in already_contacted}
    system_prompt = build_system_prompt(university, areas, count, profile_text, already_contacted)

    evidence_log: list[dict] = []
    professors_drafted: list[str] = []  # only professors that finish the FULL pipeline
    professor_state: dict[str, dict] = {}  # per-professor: department, paper_*, connection_point, email_draft
    verify_attempts = {"count": 0}
    last_written_content = {"text": None}

    def log_paper_found(professor_name, department, paper_title, paper_url, paper_summary) -> str:
        if professor_name.lower() in already_contacted_lower:
            return (
                f"REJECTED — '{professor_name}' was already contacted in a previous run "
                f"(this is enforced, not just a prompt reminder). Pick a different professor."
            )
        evidence_log.append(
            {
                "tool": "log_paper_found",
                "args": {"professor_name": professor_name, "paper_title": paper_title, "paper_url": paper_url},
                "result": f"department={department}, url={paper_url}, summary={paper_summary}",
            }
        )
        passed, gaps = _run_gate(
            client,
            _PLAUSIBILITY_GATE_PROMPT,
            f"Professor: {professor_name}\nDepartment: {department}\nPaper: {paper_title}\nURL: {paper_url}\n"
            f"Summary: {paper_summary}\n\nSearch queries this run:\n"
            + "\n".join(e["query"] for e in evidence_log if e["tool"] == "web_search"),
        )
        if not passed:
            return (
                "GATE 1 (plausibility) FAILED — do not proceed to propose_connection_point for this "
                "professor yet:\n" + "\n".join(f"- {g}" for g in gaps)
                + "\nEither search again for stronger evidence, or skip this professor entirely."
            )
        professor_state[professor_name] = {
            "department": department,
            "paper_title": paper_title,
            "paper_url": paper_url,
            "paper_summary": paper_summary,
        }
        return f"Logged and passed plausibility check. Proceed to propose_connection_point for {professor_name}."

    def propose_connection_point(professor_name, connection_point_draft) -> str:
        if professor_name not in professor_state:
            return f"No paper logged for '{professor_name}' yet — call log_paper_found first."

        passed, gaps = _run_gate(
            client,
            _CONNECTION_GATE_PROMPT,
            f"Paper summary: {professor_state[professor_name]['paper_summary']}\n\n"
            f"Proposed connection point: {connection_point_draft}",
        )
        if not passed:
            return (
                "GATE 2 (grounding) FAILED — this connection point was not saved:\n"
                + "\n".join(f"- {g}" for g in gaps)
                + "\nRevise it to stay within what the logged paper summary actually supports, and call "
                "propose_connection_point again."
            )

        humanized = humanize_text(
            client, connection_point_draft, context="a specific technical connection point in a cold email to a professor"
        )
        professor_state[professor_name]["connection_point"] = humanized
        evidence_log.append(
            {"tool": "propose_connection_point", "args": {"professor_name": professor_name}, "result": humanized}
        )
        return (
            f"Connection point passed grounding check and was humanized. Proceed to propose_email_draft "
            f"for {professor_name}."
        )

    def propose_email_draft(professor_name, email_draft) -> str:
        if professor_name not in professor_state or "connection_point" not in professor_state[professor_name]:
            return f"No approved connection point for '{professor_name}' yet — call propose_connection_point first."

        humanized = humanize_text(client, email_draft, context="a cold email from a student to a professor, 120-180 words")
        professor_state[professor_name]["email_draft"] = humanized
        return f"Email drafted and humanized. Proceed to save_professor_outreach for {professor_name}."

    def save_professor_outreach_step(professor_name) -> str:
        state = professor_state.get(professor_name, {})
        if "connection_point" not in state or "email_draft" not in state:
            return (
                f"Cannot finalize '{professor_name}' — the pipeline isn't complete "
                f"(need propose_connection_point and propose_email_draft to succeed first)."
            )
        result = _persist_professor_outreach(
            professor_name,
            state["department"],
            state["paper_title"],
            state["paper_url"],
            state["connection_point"],
            state["email_draft"],
        )
        professors_drafted.append(professor_name)
        return result

    def verified_write_file(filename: str, content: str) -> str:
        write_file(filename, content)
        last_written_content["text"] = content

        required_headers = ["## Summary", "## Confidence & what would change this", "## Sources"] + [
            f"### {name}" for name in professors_drafted
        ]
        passed, gaps = verify_report(client, content, evidence_log, required_headers, system_prompt=_INDEX_VERIFIER_PROMPT)
        if passed:
            return (
                "Index report written and passed the reliability check: every professor who completed "
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
        "log_paper_found": lambda **kw: log_paper_found(**kw),
        "propose_connection_point": lambda **kw: propose_connection_point(**kw),
        "propose_email_draft": lambda **kw: propose_email_draft(**kw),
        "save_professor_outreach": lambda **kw: save_professor_outreach_step(**kw),
        "write_file": lambda **kw: verified_write_file(kw["filename"], kw["content"]),
    }

    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Find and draft outreach for about {count} professors, as instructed."},
    ]

    def finish(reason: str) -> str:
        print(f"\n{client.summary()}")
        print(f"({reason})")
        return last_written_content["text"] or "Stopped before writing a final report — check sandbox/drafts/professors/ for any completed drafts."

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ({client.total_tokens:,} tokens used so far) ---")
        try:
            response = client.responses.create(model=MODEL, tools=TOOLS, input=input_items)
        except _BudgetExceeded as e:
            print(f"\nTOKEN BUDGET EXCEEDED: {e}")
            return finish("stopped mid-turn on the main model call — any professors already saved before this are safe on disk")
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
                return finish(f"stopped mid-turn during {item.name} — any professors already saved before this are safe on disk")
            except Exception as e:
                result = f"Error running {item.name}: {e}"
            print(f"  -> {result[:200]}")
            input_items.append({"type": "function_call_output", "call_id": item.call_id, "output": result})

    return finish(f"hit max_turns ({max_turns}) without finishing")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            'Usage: python professor_outreach_agent.py <profile_file> "<university>" "<areas_comma_separated>" [count] [token_budget]\n'
            'Example: python professor_outreach_agent.py my_profile.md "UC Berkeley" "ML,AI,Quant,Probability,RL,Deep Learning" 10 500000\n\n'
            "Default count is 10. Each run automatically excludes professors already drafted in a "
            "previous run (see contacted_professors.json) — non-overlapping batches, no need to track it yourself.\n\n"
            "token_budget is optional — if set, the run stops itself (keeping any drafts already saved) "
            "once total input+output tokens for the run cross that number, instead of running unchecked. "
            "Leave it off for your first run so you can see real usage in the printed summary, then set an "
            "informed budget for later runs based on that."
        )
        sys.exit(1)

    profile_path = sys.argv[1]
    if not Path(profile_path).exists():
        print(f"Profile file not found: {profile_path}")
        sys.exit(1)
    profile_text = Path(profile_path).read_text()

    university = sys.argv[2]
    areas = [a.strip() for a in sys.argv[3].split(",") if a.strip()]
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    token_budget = int(sys.argv[5]) if len(sys.argv) > 5 else None

    print(f"University: {university}\nAreas: {areas}\nTarget count: {count}\nToken budget: {token_budget or '(none set)'}\n")

    answer = run_agent(university, areas, profile_text, count, token_budget=token_budget)
    print("\n=== Final answer ===")
    print(answer)
