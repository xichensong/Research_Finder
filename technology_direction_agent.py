"""
A "technology direction" agent: given ONE technology/trend and a timeframe,
assesses whether it's headed toward mainstream/huge, a real-but-narrow
niche, or fading — the "would I have correctly called AI/ML a few years
before it blew up" question.

Same architecture as trend_direction_agent.py, repointed at a different
domain and a different input shape (one subject, not a pair):

  Dimension layers (assessed for the one technology, via web_search):
    adoption, investment, cost trajectory, talent/research signals,
    regulatory/social signals

  Synthesis layers (cross-cutting, dataset-grounded where possible):
    category track record (technology_data.py, base rate for this TYPE of
    tech), historical analogue (closest-matching past technology's actual
    trajectory), hype vs. substance (explicit signal-vs-noise check — this
    is the single most common failure mode in tech forecasting, so it gets
    its own required section rather than being left implicit), and the
    verdict (the synthesized call).

Same verification checkpoint as before (verification.py is domain-agnostic
— no changes needed there), and length caps are built into every section
from the start this time, rather than bolted on after the fact once output
turned out too long (see: the actual history of trend_direction_agent.py).

Run it:
  python technology_direction_agent.py "quantum computing" "next 15 years"
  python technology_direction_agent.py "quantum computing" "next 15 years" "hype vs substance"
"""

import json
import sys

from openai import OpenAI

from technology_data import VALID_CATEGORIES, get_category_track_record, search_technology_analogues
from tools import write_file
from verification import MAX_VERIFY_ATTEMPTS, verify_report

MODEL = "gpt-5.6"

ALL_TOOL_DEFS = {
    "web_search": {"type": "web_search"},
    "search_technology_analogues": {
        "type": "function",
        "name": "search_technology_analogues",
        "description": (
            "Search a curated dataset of ~16 past technology trajectories for "
            "analogues to this situation. Returns each matching case's actual "
            "outcome (mainstream / niche / faded / delayed_but_mainstream / "
            "too_early_to_call), not just a similar-sounding name. Use this to "
            "find the closest-matching past technology — gives a verified "
            "outcome to reason from instead of relying on vague memory of "
            "'similar things.'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the situation, e.g. 'expensive hardware with high pre-launch hype and unclear use case' or 'infrastructure shift with falling cost curve'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    "get_category_track_record": {
        "type": "function",
        "name": "get_category_track_record",
        "description": (
            "Get every dataset case in a given category, with an outcome "
            f"tally, to check the base rate for that TYPE of technology. "
            f"Valid categories: {', '.join(VALID_CATEGORIES)}. Pick the "
            "closest one for the technology being assessed. Use this once, "
            "before writing the 'Category track record' section — never "
            "assert a base-rate claim ('technologies like this usually...') "
            "that isn't traceable to what this tool returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": f"One of: {', '.join(VALID_CATEGORIES)}",
                }
            },
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    "write_file": {
        "type": "function",
        "name": "write_file",
        "description": "Write the finished analysis to a file in the local sandbox directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the file to write."},
                "content": {"type": "string", "description": "The text content to write."},
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

ALL_TOOL_FUNCTIONS = {
    "search_technology_analogues": lambda **kw: search_technology_analogues(kw["query"]),
    "get_category_track_record": lambda **kw: get_category_track_record(kw["category"]),
    "write_file": lambda **kw: write_file(kw["filename"], kw["content"]),
}

# --- Layer registry ----------------------------------------------------------

LAYER_SPECS = {
    "adoption": {
        "category": "dimension",
        "display": "Adoption",
        "aliases": ["adoption", "adoption curve", "user growth"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Adoption: current position on the adoption curve (early "
            "adopter / growth / mainstream), user or customer growth rate, "
            "and whether growth is accelerating or plateauing. Research via "
            "web_search."
        ),
    },
    "investment": {
        "category": "dimension",
        "display": "Investment & funding",
        "aliases": ["investment", "funding", "vc funding", "investment and funding"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Investment & funding: VC funding trends, corporate R&D "
            "investment, M&A activity in the space. Research via web_search."
        ),
    },
    "cost_trajectory": {
        "category": "dimension",
        "display": "Cost trajectory",
        "aliases": ["cost trajectory", "cost curve", "price trend"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Cost trajectory: is the underlying cost/performance of this "
            "technology improving on a steady, measurable curve (like "
            "solar or batteries), or has it plateaued? This is one of the "
            "strongest signals in the dataset (see: solar, EVs, nuclear "
            "power as a counterexample) — don't skip it even if hard to "
            "quantify precisely. Research via web_search."
        ),
    },
    "talent_research": {
        "category": "dimension",
        "display": "Talent & research signals",
        "aliases": ["talent", "research signals", "talent and research", "papers", "patents"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Talent & research signals: research paper/patent volume "
            "trend, whether top researchers or top companies are entering "
            "the field, and whether there's been a specific measurable "
            "step-change result (not just steady incremental progress). "
            "Research via web_search."
        ),
    },
    "regulatory_social": {
        "category": "dimension",
        "display": "Regulatory & social signals",
        "aliases": ["regulatory", "social signals", "regulatory and social", "policy", "public sentiment"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Regulatory & social signals: policy/regulatory attention "
            "(supportive, neutral, or restrictive), and public sentiment. "
            "This is where technically-sound technologies have historically "
            "failed (nuclear power, supersonic flight) — don't assume good "
            "engineering is sufficient. Research via web_search."
        ),
    },
    "category_track_record": {
        "category": "synthesis",
        "display": "Category track record",
        "aliases": ["category track record", "base rate", "track record"],
        "tools": ["get_category_track_record", "write_file"],
        "instructions": (
            "Category track record: call get_category_track_record ONCE "
            "with the closest matching category for this technology. Every "
            "base-rate claim here must trace to a case that tool returned. "
            "Do not use web_search for this layer — it isn't offered, on "
            "purpose."
        ),
    },
    "historical_analogue": {
        "category": "synthesis",
        "display": "Historical analogue",
        "aliases": ["historical analogue", "historical analogy", "closest analogue"],
        "tools": ["search_technology_analogues", "write_file"],
        "instructions": (
            "Historical analogue: call search_technology_analogues to find "
            "the closest-matching past technology's actual trajectory. Try "
            "a few different queries if the first doesn't return a strong "
            "match; if nothing matches well, say so rather than forcing an "
            "analogy. Do not use web_search for this layer."
        ),
    },
    "hype_vs_substance": {
        "category": "synthesis",
        "display": "Hype vs. substance",
        "aliases": ["hype vs substance", "hype vs. substance", "signal vs noise", "hype check"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Hype vs. substance: explicitly separate media/buzz signal "
            "(coverage volume, prominent people talking about it) from "
            "substance signal (measurable adoption, cost curve, revenue, "
            "usage). Media coverage volume is the weakest signal in this "
            "whole assessment — see Segway, Google Glass, and enterprise "
            "blockchain pilots in the dataset for cases where hype "
            "substantially outran substance. Research via web_search."
        ),
    },
    "verdict": {
        "category": "synthesis",
        "display": "Verdict",
        "aliases": ["verdict", "direction", "call"],
        # Needs the full picture — no fast path, same reasoning as
        # power_trajectory in trend_direction_agent.py.
        "tools": ["web_search", "search_technology_analogues", "get_category_track_record", "write_file"],
        "instructions": (
            "Verdict: this is a synthesized call, so first quickly check "
            "adoption, investment, cost trajectory, and regulatory signals "
            "via web_search, plus the category track record and closest "
            "historical analogue — then commit to: mainstream / niche / "
            "faded / too_early_to_call, over the given timeframe."
        ),
    },
}

_ALL_LAYER_KEYS_IN_ORDER = [
    "adoption",
    "investment",
    "cost_trajectory",
    "talent_research",
    "regulatory_social",
    "category_track_record",
    "historical_analogue",
    "hype_vs_substance",
    "verdict",
]


def resolve_layer(query: str) -> str | None:
    normalized = " ".join(query.lower().replace(",", " ").split())
    best_key, best_len = None, 0
    for key, spec in LAYER_SPECS.items():
        for alias in spec["aliases"]:
            if alias in normalized and len(alias) > best_len:
                best_key, best_len = key, len(alias)
    return best_key


def required_headers(layer: str | None) -> list[str]:
    if layer is not None:
        return [f"## {LAYER_SPECS[layer]['display']}"]
    return ["## Verdict"] + [
        f"### {LAYER_SPECS[k]['display']}"
        for k in _ALL_LAYER_KEYS_IN_ORDER
        if k != "verdict"
    ]


def build_tools_and_functions(layer: str | None):
    if layer is None:
        tool_names = list(ALL_TOOL_DEFS.keys())
    else:
        tool_names = LAYER_SPECS[layer]["tools"]
    tools = [ALL_TOOL_DEFS[name] for name in tool_names]
    tool_functions = {name: ALL_TOOL_FUNCTIONS[name] for name in tool_names if name in ALL_TOOL_FUNCTIONS}
    return tools, tool_functions


_BASE_PROMPT = """\
You are a technology-direction analyst. Given a technology/trend and a \
timeframe, assess whether it's headed toward mainstream adoption, a real \
but narrow niche, fading out, or is too early to call — you must not hedge \
into "could go either way" with no lean, unless the honest answer really is \
too-early-to-call (that's a valid, distinct verdict, not a cop-out — use it \
when the evidence genuinely doesn't support a stronger claim yet).

This is speculative forecasting, not fact — the structure exists so a \
reader can see your reasoning and disagree with a specific layer, not so \
you can sound certain.

After you call write_file, an independent reliability check reviews what \
you wrote against what your tools actually returned this run — not whether \
your forecast is right (nobody can check that), only whether every claim is \
grounded in evidence. If it finds gaps, the write_file result will list \
them specifically — address them and call write_file again. Limited \
correction attempts, so prioritize the most important gap first.

Keep every section tight — stick to the stated length. Pick the strongest \
one or two supporting points where the format asks for that, not an \
exhaustive list.
"""

_VERDICT_AND_SOURCES = """\
  ## Confidence & what would change this call
  1-2 sentences: confidence (low/medium/high), and the 1-3 specific events \
that would flip this call.

  ## Sources
  A flat list of web sources and dataset cases used — no commentary per \
source.
"""


def build_system_prompt(layer: str | None) -> str:
    if layer is None:
        dimension = "\n".join(
            f"  - {LAYER_SPECS[k]['instructions']}"
            for k in _ALL_LAYER_KEYS_IN_ORDER
            if LAYER_SPECS[k]["category"] == "dimension"
        )
        synthesis = "\n".join(
            f"  - {LAYER_SPECS[k]['instructions']}"
            for k in _ALL_LAYER_KEYS_IN_ORDER
            if LAYER_SPECS[k]["category"] == "synthesis" and k != "verdict"
        )
        return f"""{_BASE_PROMPT}
DIMENSION LAYERS (each assessed for this one technology):
{dimension}

SYNTHESIS LAYERS (cross-cutting, dataset-grounded where possible):
{synthesis}
  - {LAYER_SPECS['verdict']['instructions']}

Once you have enough to reason from, write your analysis to a file using \
write_file, in this exact structure (stick to the stated lengths):

  # <technology> — <timeframe>

  ## Verdict
  One tight paragraph (4-5 sentences max): mainstream / niche / faded / \
too_early_to_call, over the given timeframe. State it plainly. This must \
synthesize every layer below, not just restate current news.

  ## Dimension layers
  ### Adoption
  ### Investment & funding
  ### Cost trajectory
  ### Talent & research signals
  ### Regulatory & social signals
  For each: 3-4 sentences, not a full essay.

  ## Synthesis layers
  ### Category track record
  ### Historical analogue
  ### Hype vs. substance
  For each: 3-5 sentences.

{_VERDICT_AND_SOURCES}"""

    spec = LAYER_SPECS[layer]
    return f"""{_BASE_PROMPT}
You are producing ONLY the "{spec['display']}" layer this run — not the \
full 9-layer report. Do not research or write any other layer. The tools \
offered to you are limited to exactly what this layer needs.

{spec['instructions']}

Once you have enough to reason from, write your analysis to a file using \
write_file, in this exact structure:

  # <technology> — <timeframe> — {spec['display']}

  ## {spec['display']}
  4-6 sentences. This layer only — not an essay.

{_VERDICT_AND_SOURCES}"""


def run_agent(technology: str, timeframe: str, layer: str | None = None, max_turns: int = 25) -> str:
    client = OpenAI()  # constructed here, not at import time — see trend_direction_agent.py for why

    tools, tool_functions = build_tools_and_functions(layer)
    system_prompt = build_system_prompt(layer)
    headers = required_headers(layer)

    evidence_log: list[dict] = []
    verify_attempts = {"count": 0}
    last_written_content = {"text": None}

    def verified_write_file(filename: str, content: str) -> str:
        write_file(filename, content)
        last_written_content["text"] = content

        passed, gaps = verify_report(client, content, evidence_log, headers)
        if passed:
            return (
                "Report written and passed the reliability check: all required "
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
                f"Finalized as-is with remaining gaps appended as an explicit "
                f"caveats section. Do not call write_file again — give your "
                f"closing summary now."
            )

        gap_list = "\n".join(f"- {g}" for g in gaps)
        return (
            f"Report written, but the reliability check (attempt "
            f"{verify_attempts['count']}/{MAX_VERIFY_ATTEMPTS}) found gaps:\n"
            f"{gap_list}\n\nAddress these, then call write_file again with the "
            f"corrected content."
        )

    tool_functions["write_file"] = lambda **kw: verified_write_file(kw["filename"], kw["content"])

    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Technology: {technology}\nTimeframe: {timeframe}"},
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ---")
        response = client.responses.create(model=MODEL, tools=tools, input=input_items)
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

            if item.name in ("search_technology_analogues", "get_category_track_record"):
                evidence_log.append({"tool": item.name, "args": args, "result": result})

            input_items.append({"type": "function_call_output", "call_id": item.call_id, "output": result})

    return last_written_content["text"] or "Stopped: hit max_turns without finishing."


if __name__ == "__main__":
    if len(sys.argv) < 3:
        available = ", ".join(spec["display"] for spec in LAYER_SPECS.values())
        print(
            'Usage: python technology_direction_agent.py "<technology>" "<timeframe>" [layer]\n'
            'Example (full report):  python technology_direction_agent.py "quantum computing" "next 15 years"\n'
            'Example (one layer):    python technology_direction_agent.py "quantum computing" "next 15 years" "hype vs substance"\n\n'
            f"Available layers: {available}"
        )
        sys.exit(1)

    technology, timeframe = sys.argv[1], sys.argv[2]
    layer_query = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else None

    layer = None
    if layer_query:
        layer = resolve_layer(layer_query)
        if layer is None:
            available = ", ".join(spec["display"] for spec in LAYER_SPECS.values())
            print(f"Couldn't match '{layer_query}' to a known layer.\nAvailable layers: {available}")
            sys.exit(1)

    print(f"Technology: {technology}\nTimeframe: {timeframe}")
    print(f"Layer: {LAYER_SPECS[layer]['display']}" if layer else "Layer: (full report)")
    print()

    answer = run_agent(technology, timeframe, layer)
    print("\n=== Final answer ===")
    print(answer)
