"""
A career next-steps agent: given your profile (resume/transcript-shaped —
see profile_template.md) and a list of candidate directions you're weighing,
produces concrete next steps — grounded in real current opportunities via
web_search and in realistic entry-path patterns via career_data.py, not
generic "learn to code" advice.

Structurally different from trend_direction_agent.py / technology_direction_
agent.py in one real way: the "individual" layers aren't a fixed list
(Economic/Technological/...) — they're one section PER DIRECTION YOU GIVE IT,
so the tool list and prompt are built dynamically at runtime instead of from
a static registry. Everything else carries over: dataset-grounded claims,
the verification checkpoint (verification.py — still zero changes needed),
length caps built in from the start.

The structure, adapted to this domain:

  Per-direction sections (one per direction you list): fit today, the gap,
  concrete next steps — grounded in career_data.py's entry-path patterns
  and web_search for real current opportunities.

  Cross-direction sections: overlap & shared actions (what serves multiple
  directions at once — most of ML/AI/Quant prep genuinely overlaps),
  prioritization (given your stated constraints, what to lead with now vs.
  defer), reality check (explicit "is this actually realistic given where
  you are today" — the direct analog of "hype vs. substance" in the tech
  agent), and the recommended plan (the synthesized, time-sequenced verdict).

It also connects to action_tools.py so the report isn't just prose: real
opportunities it finds get logged as tracked, checkable items (with URLs),
concrete application materials get drafted per opportunity, and recommended
outreach gets an actual drafted message saved to disk — never sent. See
action_tools.py's own docstring for why sending/submitting isn't automated
here; that boundary is intentional, not a missing feature.

Run it:
  python next_steps_agent.py my_profile.md "next 6 months" "ML,AI,Quant,Startup"
  python next_steps_agent.py my_profile.md "next 6 months" "ML,AI,Quant,Startup" Quant

Then, any time, independent of the agent:
  python action_tools.py list
  python action_tools.py done 3
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

from action_tools import add_action_item, list_action_items, log_opportunity, save_application_materials, save_draft
from career_data import VALID_DIRECTIONS, get_path_pattern, search_path_patterns
from tools import write_file
from verification import MAX_VERIFY_ATTEMPTS, verify_report

MODEL = "gpt-5.6"

TOOLS = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "get_path_pattern",
        "description": (
            "Get the dataset's realistic entry-path pattern(s) for a stated "
            "direction (e.g. 'ML', 'quant', 'startup') — typical "
            "prerequisites, common first roles, and common pitfalls. Use "
            "this once per direction before writing that direction's "
            "section. Never assert a claim about 'how people typically "
            "break into X' that isn't traceable to what this tool or "
            "search_path_patterns returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {"direction": {"type": "string"}},
            "required": ["direction"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_path_patterns",
        "description": (
            "Flexible keyword search over the same dataset, for when a "
            "direction doesn't map cleanly onto get_path_pattern's exact "
            f"categories ({', '.join(VALID_DIRECTIONS)})."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write the finished plan to a file in the local sandbox directory.",
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
    {
        "type": "function",
        "name": "log_opportunity",
        "description": (
            "Log a SPECIFIC real opportunity you found via web_search this "
            "run — an actual internship posting, program, competition, or "
            "course, with its real URL — as a tracked, checkable item. Only "
            "call this for something concrete you actually found, never for "
            "a general category ('apply to quant internships' is not an "
            "opportunity — a specific real program with a real URL is). "
            "Skip this if you can't find a real URL rather than inventing one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The specific opportunity's name."},
                "url": {"type": "string", "description": "The real URL you found it at."},
                "direction": {"type": "string", "description": "Which direction this relates to."},
                "deadline_note": {"type": "string", "description": "Deadline if known, else empty string."},
            },
            "required": ["title", "url", "direction", "deadline_note"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "add_action_item",
        "description": (
            "Log a concrete next step as a tracked, checkable item — for "
            "things that aren't a specific found opportunity (use "
            "log_opportunity for those) but are still a real, specific "
            "action: a course to register for, a skill to build, a project "
            "to start. One item per concrete action, not one item per "
            "direction — 'take Stat 134' is an item; 'get better at quant' "
            "is not specific enough to be one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "category": {
                    "type": "string",
                    "description": "One of: application, skill_building, outreach, project, other.",
                },
                "direction": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["title", "category", "direction", "notes"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "save_draft",
        "description": (
            "Save actual drafted outreach text (a cold email, a message to "
            "a program contact) to disk for the person to review and send "
            "THEMSELVES. This never sends anything — write the real draft "
            "content, not a description of what a draft would contain. Only "
            "use this when the plan recommends outreach to a specific real "
            "person, program, or organization; skip it for generic "
            "'network more' advice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "e.g. 'outreach_professor_x.txt'"},
                "content": {"type": "string", "description": "The actual drafted message text."},
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "save_application_materials",
        "description": (
            "For ONE opportunity you already logged with log_opportunity, "
            "write the actual application materials it would need — a "
            "tailored cover letter using REAL facts from the profile, plus "
            "draft answers to the application's specific questions IF "
            "web_search actually found them. If you could not find the "
            "specific questions, say so explicitly in the content ('specific "
            "application questions not found via search — check the "
            "official application page') instead of inventing plausible- "
            "sounding ones — fabricated application questions are actively "
            "misleading, not just unhelpful. Always call this once per "
            "logged opportunity that has concrete next-step material to "
            "draft (skip for opportunities where you have nothing more to "
            "add beyond the listing itself). Never submits anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "opportunity_title": {
                    "type": "string",
                    "description": "Must match the title used in the corresponding log_opportunity call.",
                },
                "content": {
                    "type": "string",
                    "description": "The actual materials: a 'what this requires' checklist, a cover letter draft, and question answers (or an honest note if questions weren't found).",
                },
            },
            "required": ["opportunity_title", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

TOOL_FUNCTIONS = {
    "get_path_pattern": lambda **kw: get_path_pattern(kw["direction"]),
    "search_path_patterns": lambda **kw: search_path_patterns(kw["query"]),
    "write_file": lambda **kw: write_file(kw["filename"], kw["content"]),
    "log_opportunity": lambda **kw: log_opportunity(kw["title"], kw["url"], kw["direction"], kw["deadline_note"]),
    "add_action_item": lambda **kw: add_action_item(kw["title"], kw["category"], kw["direction"], kw["notes"]),
    "save_draft": lambda **kw: save_draft(kw["filename"], kw["content"]),
    "save_application_materials": lambda **kw: save_application_materials(kw["opportunity_title"], kw["content"]),
}

_BASE_PROMPT = """\
You are a career next-steps analyst. You are given someone's profile \
(resume/transcript-shaped) and a list of directions they're weighing. Your \
job is to produce CONCRETE next steps, not generic advice — every \
recommendation should be something specific enough to act on this week, not \
"build your skills" or "network more."

Ground every claim: use web_search to find REAL current things (actual \
course names, actual certification programs, actual types of roles/programs \
companies in this space are hiring for) rather than inventing plausible- \
sounding ones. Use get_path_pattern / search_path_patterns to check your \
"how people typically break into X" claims against the dataset rather than \
asserting them from general impression.

Be honest, not encouraging-by-default. If the profile doesn't yet support a \
stated direction, say so plainly and say what would need to change — a \
falsely encouraging plan wastes the person's time more than a blunt one. \
Do not invent facts about the person's background beyond what's in their \
profile.

As you find things, don't just describe them in prose — log them:
- Found a specific real opportunity (an actual posting, program, \
competition) via web_search? Call log_opportunity with its real URL. Skip \
it rather than inventing a URL if you can't find a real one. This applies \
to ANY opportunity type — internships, research programs, competitions, \
fellowships — not internships only.
- For every opportunity you log, also call save_application_materials \
with the same title: a "what this requires" checklist, a cover letter \
draft using real facts from the profile, and draft answers to the \
application's actual questions IF web_search found them — otherwise an \
explicit honest note that the specific questions weren't found, never \
invented ones. Same treatment for every opportunity type, not just \
internships — a competition's submission requirements or a program's \
essay prompts get the same materials-drafting treatment.
- Recommending a concrete action that isn't a found opportunity (register \
for a specific course, build a specific project)? Call add_action_item.
- Recommending outreach to a specific real person, program, or \
organization? Call save_draft with the ACTUAL drafted message text, not a \
description of what to write. This is never sent automatically — it's \
saved for the person to review and send themselves.
Do this alongside writing the report, not instead of it — the report is \
still the full narrative; these calls make the concrete items in it \
trackable and actionable outside the report too. None of this submits \
anything on the person's behalf — everything stops at a saved draft they \
review and act on themselves.

After you call write_file, an independent reliability check reviews what \
you wrote against what your tools actually returned this run — not whether \
the advice is right (nobody can fully verify that), only whether claims are \
grounded in evidence. If it finds gaps, the write_file result will list \
them — address them and call write_file again. Limited correction attempts.

Keep every section tight — stick to the stated length. Specificity over \
volume: one real, well-chosen recommendation beats five vague ones.
"""

_CONFIDENCE_AND_SOURCES = """\
  ## Confidence & what would change this
  1-2 sentences: confidence in this plan, and the biggest single thing that \
would change it (e.g. a specific skill gap closing, a constraint changing).

  ## Sources
  A flat list of web sources and dataset entries used — no commentary per \
source.
"""


def _direction_section_key(direction: str) -> str:
    return f"### {direction.strip()}"


def build_system_prompt(profile_text: str, timeframe: str, directions: list[str], focus: str | None) -> str:
    if focus is not None:
        return f"""{_BASE_PROMPT}
You are producing ONLY the "{focus}" section this run, for the timeframe \
"{timeframe}" — not the full multi-direction report. Do not write sections \
for other directions or the cross-direction synthesis.

Write your answer using write_file, in this exact structure:

  # Next steps: {focus} — {timeframe}

  ## {focus}
  Fit today (2-3 sentences), the gap (2-3 sentences), then 3-5 concrete, \
specific next steps as a bulleted list — real things (actual courses, \
actual types of programs/roles), not generic advice.

{_CONFIDENCE_AND_SOURCES}

PROFILE:
{profile_text}
"""

    direction_headers = "\n".join(f"  {_direction_section_key(d)}" for d in directions)
    return f"""{_BASE_PROMPT}
Directions to assess: {', '.join(directions)}. Timeframe: {timeframe}.

For EACH direction, call get_path_pattern once, and use web_search for \
current specifics. Then write the full plan using write_file, in this \
exact structure:

  # Next steps — {timeframe}

  ## Recommended plan
  One tight paragraph (4-6 sentences): the sequenced, prioritized \
recommendation across all directions given — what to do first, what to \
defer, and why. This must synthesize everything below, not restate it.

  ## Per-direction assessment
{direction_headers}
  For each: fit today (2-3 sentences), the gap (2-3 sentences), then 3-5 \
concrete next steps as a bulleted list — real things, not generic advice.

  ## Overlap & shared actions
  3-5 sentences: what preparation serves multiple directions at once, so \
effort isn't duplicated across near-identical prep for different tracks.

  ## Prioritization
  3-5 sentences: given the stated constraints (time, timeline, financial), \
which direction(s) to lead with now vs. defer, and why.

  ## Reality check
  3-5 sentences: is this plan actually realistic given where the profile \
shows this person is today? Say plainly if a direction is a stretch and \
what would need to be true for it to work, rather than hedging.

{_CONFIDENCE_AND_SOURCES}

PROFILE:
{profile_text}
"""


def required_headers(directions: list[str], focus: str | None) -> list[str]:
    if focus is not None:
        return [f"## {focus}"]
    return (
        ["## Recommended plan", "## Per-direction assessment"]
        + [_direction_section_key(d) for d in directions]
        + ["## Overlap & shared actions", "## Prioritization", "## Reality check"]
    )


def run_agent(
    profile_text: str, timeframe: str, directions: list[str], focus: str | None = None, max_turns: int = 30
) -> str:
    client = OpenAI()  # constructed here, not at import time

    system_prompt = build_system_prompt(profile_text, timeframe, directions, focus)
    headers = required_headers(directions, focus)

    evidence_log: list[dict] = []
    verify_attempts = {"count": 0}
    last_written_content = {"text": None}

    def verified_write_file(filename: str, content: str) -> str:
        write_file(filename, content)
        last_written_content["text"] = content

        passed, gaps = verify_report(client, content, evidence_log, headers)
        if passed:
            return (
                "Plan written and passed the reliability check: all required "
                "sections present, claims grounded in this run's evidence. "
                "Done — no need to call write_file again."
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
                f"Correction attempts exhausted ({MAX_VERIFY_ATTEMPTS} max). "
                f"Finalized as-is with remaining gaps appended. Do not call "
                f"write_file again — give your closing summary now."
            )

        gap_list = "\n".join(f"- {g}" for g in gaps)
        return (
            f"Plan written, but the reliability check (attempt "
            f"{verify_attempts['count']}/{MAX_VERIFY_ATTEMPTS}) found gaps:\n"
            f"{gap_list}\n\nAddress these, then call write_file again."
        )

    tool_functions = dict(TOOL_FUNCTIONS)
    tool_functions["write_file"] = lambda **kw: verified_write_file(kw["filename"], kw["content"])

    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Produce the plan as instructed."},
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ---")
        response = client.responses.create(model=MODEL, tools=TOOLS, input=input_items)
        input_items += response.output

        function_calls = []
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if getattr(content, "text", "").strip():
                        print(f"Model: {content.text.strip()}")
            elif item.type == "function_call":
                print(f"Model wants to call: {item.name}({item.arguments})")
                function_calls.append(item)
            elif item.type == "web_search_call":
                query = getattr(item.action, "query", "")
                print(f"Model is web-searching: {query}")
                evidence_log.append({"tool": "web_search", "query": query})

        if not function_calls:
            return last_written_content["text"] or response.output_text

        for item in function_calls:
            args = json.loads(item.arguments)
            func = tool_functions.get(item.name)
            try:
                result = func(**args)
            except Exception as e:
                result = f"Error running {item.name}: {e}"
            print(f"  -> {result[:200]}")

            if item.name in ("get_path_pattern", "search_path_patterns"):
                evidence_log.append({"tool": item.name, "args": args, "result": result})

            input_items.append({"type": "function_call_output", "call_id": item.call_id, "output": result})

    return last_written_content["text"] or "Stopped: hit max_turns without finishing."


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            'Usage: python next_steps_agent.py <profile_file> "<timeframe>" "<directions_comma_separated>" [focus]\n'
            'Example (full report): python next_steps_agent.py my_profile.md "next 6 months" "ML,AI,Quant,Startup"\n'
            'Example (one direction): python next_steps_agent.py my_profile.md "next 6 months" "ML,AI,Quant,Startup" Quant\n\n'
            f"Directions don't have to match a known category exactly (get_path_pattern/search_path_patterns handle "
            f"fuzzy matches), but the dataset currently covers: {', '.join(VALID_DIRECTIONS)}"
        )
        sys.exit(1)

    profile_path = sys.argv[1]
    if not Path(profile_path).exists():
        print(f"Profile file not found: {profile_path}\nSee profile_template.md to create one.")
        sys.exit(1)
    profile_text = Path(profile_path).read_text()

    timeframe = sys.argv[2]
    directions = [d.strip() for d in sys.argv[3].split(",") if d.strip()]
    focus = sys.argv[4] if len(sys.argv) > 4 else None

    if focus and focus not in directions and focus.lower() not in [d.lower() for d in directions]:
        print(f"Focus '{focus}' doesn't match any of the given directions: {', '.join(directions)}")
        sys.exit(1)

    print(f"Directions: {directions}\nTimeframe: {timeframe}")
    print(f"Focus: {focus}" if focus else "Focus: (full report)")
    print()

    answer = run_agent(profile_text, timeframe, directions, focus)
    print("\n=== Final answer ===")
    print(answer)
