"""
The verification checkpoint: after the agent writes a report, this checks
whether the *method* behind it was reliable before letting the run finish.

Important honesty boundary: this cannot check whether a forecast will turn
out to be TRUE — nobody can verify the future today. What it checks is
GROUNDING — did every claim actually trace back to something the tools
returned, or did the model just assert something plausible with nothing
behind it. That's the accuracy lever that's actually available at generation
time.

Two checks, run every time write_file is called:

  1. Mechanical section check (no LLM, 100% reliable): are the required
     section headers actually present in what got written?

  2. Independent grounding check: a SEPARATE model call, with a fresh
     conversation that has no memory of writing the report, given only the
     report text and a log of what the tools actually returned this run.
     It checks whether structural claims cite real cases and whether
     research-dependent layers actually had research behind them. This has
     to be a separate call, not the same agent re-reading its own work —
     the same reasoning that produced a mistake is generally not the
     reasoning that catches it (see: why code review isn't done by the
     original author reading their own diff a second time).

Both check ONLY the current run's evidence — a fresh evidence_log per call
to run_agent, not some accumulated global history.
"""

from openai import OpenAI

VERIFIER_MODEL = "gpt-5.6"

MAX_VERIFY_ATTEMPTS = 2  # correction cycles allowed before finalizing with caveats


def format_evidence_log(evidence_log: list[dict]) -> str:
    """Render the tool-call evidence log as text for the verifier to read."""
    if not evidence_log:
        return "(no tool calls were logged this run)"

    lines = []
    for i, entry in enumerate(evidence_log, 1):
        if entry["tool"] == "web_search":
            lines.append(f"{i}. web_search query: {entry['query']}")
        else:
            lines.append(
                f"{i}. {entry['tool']}({entry['args']}) returned:\n{entry['result']}\n"
            )
    return "\n".join(lines)


def check_sections_present(report_text: str, required_headers: list[str]) -> list[str]:
    """Mechanical check — no LLM. Returns a gap message per missing header."""
    lowered = report_text.lower()
    gaps = []
    for header in required_headers:
        if header.lower() not in lowered:
            gaps.append(f"Missing required section: '{header}'")
    return gaps


_VERIFIER_SYSTEM_PROMPT = """\
You are a methodology auditor. You did NOT write the report below and have \
no stake in it being good — your only job is to find grounding problems.

You will be shown a report and an evidence log of everything the research \
tools actually returned while producing it. Check ONLY for these things:

1. Any claim in a "Structural tendencies" section that names a historical \
case which does NOT appear in the evidence log (i.e. the case was not \
actually returned by get_country_case_history this run) — this is a \
fabricated citation.
2. Any structural/character claim about a country made despite the evidence \
log showing that country had zero or one case returned for it (the report \
should have said the dataset doesn't support a pattern claim instead).
3. Any individual layer (Economic, Technological, Military, Demographic, \
Diplomatic/political) whose content is NOT backed by at least one web_search \
query in the evidence log that's clearly about that topic — i.e. the layer \
reads like it was written from general knowledge with no research behind it \
at all.
4. Any place where the verdict/synthesis directly contradicts what an \
individual or comparative layer actually says.

Do NOT critique writing style, do NOT judge whether the forecast is likely \
correct (that's unknowable), and do NOT invent gaps beyond these four \
categories — this must stay a grounding check, not a general critique.

Respond in EXACTLY this format, nothing else:

VERDICT: PASS
or
VERDICT: FAIL
GAPS:
- <specific gap, naming the section and what's wrong>
- <specific gap>
"""


def verify_report(
    client: OpenAI,
    report_text: str,
    evidence_log: list[dict],
    required_headers: list[str],
    system_prompt: str = _VERIFIER_SYSTEM_PROMPT,
) -> tuple[bool, list[str]]:
    """
    Run both checks. Returns (passed, gaps). `passed` is True only if BOTH
    the mechanical section check and the independent grounding check pass.

    `system_prompt` defaults to the historical-case-grounding checklist below
    (what trend_direction_agent.py / technology_direction_agent.py /
    next_steps_agent.py need). A caller checking a different kind of report
    — e.g. professor_outreach_agent.py, which has nothing to do with country
    cases or Economic/Technological/Military layers — should pass its own
    prompt in the same VERDICT: PASS / VERDICT: FAIL + GAPS: format instead
    of silently being checked against criteria that don't apply to it.
    """
    gaps = check_sections_present(report_text, required_headers)

    verifier_input = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"REPORT:\n{report_text}\n\n"
                f"EVIDENCE LOG (everything the tools actually returned this run):\n"
                f"{format_evidence_log(evidence_log)}"
            ),
        },
    ]
    response = client.responses.create(
        model=VERIFIER_MODEL, input=verifier_input, reasoning={"effort": "low"}
    )
    verdict_text = response.output_text.strip()

    if verdict_text.upper().startswith("VERDICT: FAIL"):
        for line in verdict_text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                gaps.append(line[2:].strip())

    return (len(gaps) == 0), gaps
