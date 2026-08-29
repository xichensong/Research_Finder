"""
A "trend direction" agent: given two countries and a timeframe, it produces
a layered comparison and a directional call for how their relationship is
headed — improving, worsening, or stable.

This is the same loop as agent_manual_openai.py — the interesting part still
isn't the tools, it's the task prompt, which has to tell the model *what a
good comparison looks like*. There's a fixed set of "layers":

  Individual layers (scored for each country, then compared):
    economic, technological, military, demographic, diplomatic/political

  Comparative layers (about the pair itself, not either country alone):
    bilateral relationship history, structural tendencies (dataset-grounded,
    see historical_data.py), relative power trajectory (the synthesized
    verdict), third-party alliance overlap

Forcing this structure is what turns "write me an essay about X and Y" into
something checkable — a reader can look at the Economic layer and disagree
with just that part, instead of the whole thing being one undifferentiated
paragraph of vibes.

You can also ask for just ONE layer instead of the full 9-section report.
This isn't just a shorter prompt — the tool list itself shrinks (e.g. the
Structural tendencies layer only offers get_country_case_history, not
web_search at all), so a narrow run makes genuinely fewer tool calls, not
just a genuinely shorter final write-up.

There's also a verification checkpoint (see verification.py): every time the
model calls write_file, a mechanical check confirms all required sections
are present, and a SEPARATE, independent model call checks whether every
claim actually traces back to something the tools returned this run. This
checks GROUNDING, not truth — nobody can verify a forecast is correct before
the future happens. If gaps are found, they come back as the result of the
write_file call itself, so the model naturally loops back to fix them using
the same "read the tool result, decide what to do next" mechanism as
everything else — capped at a couple of correction attempts so it can't
loop forever.

Run it:
  python trend_direction_agent.py "United States" "China" "next 20 years"
  python trend_direction_agent.py "United States" "China" "next 20 years" technological
  python trend_direction_agent.py "United States" "China" "next 20 years" "structural tendencies"
"""

import json
import sys

from openai import OpenAI

from historical_data import get_country_case_history, search_historical_analogues
from tools import write_file
from verification import MAX_VERIFY_ATTEMPTS, verify_report

MODEL = "gpt-5.6"

# --- Full tool definitions (a layer's "tools" list below is a subset of the
# names here — see LAYER_SPECS) --------------------------------------------

ALL_TOOL_DEFS = {
    "web_search": {"type": "web_search"},
    "search_historical_analogues": {
        "type": "function",
        "name": "search_historical_analogues",
        "description": (
            "Search a curated dataset of ~18 past geopolitical/rivalry cases "
            "for historical analogues to a situation. Returns each matching "
            "case's actual trajectory and outcome, not just a similar-sounding "
            "headline. Use this to find the closest-matching past SITUATION — "
            "this gives verified outcomes to reason from instead of relying "
            "on memory of history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the dynamic, e.g. 'trade war between rival powers' or 'territorial dispute frozen conflict'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    "get_country_case_history": {
        "type": "function",
        "name": "get_country_case_history",
        "description": (
            "Get every case in the dataset involving a specific country/party "
            "(not a matched-pair search like search_historical_analogues — this "
            "pulls ALL cases that party appears in, across different opponents "
            "and eras). Use this once per country to find dataset-backed "
            "patterns in how that country has historically behaved (escalates? "
            "settles? drags on unresolved?). Never assert a structural/"
            "character claim about a country that isn't traceable to a case "
            "this tool returned — if it returns zero or one case, say the "
            "dataset doesn't support a pattern claim rather than filling the "
            "gap from general impression."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Country/party name, e.g. 'China' or 'United States'.",
                }
            },
            "required": ["country"],
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
    "search_historical_analogues": lambda **kw: search_historical_analogues(kw["query"]),
    "get_country_case_history": lambda **kw: get_country_case_history(kw["country"]),
    "write_file": lambda **kw: write_file(kw["filename"], kw["content"]),
}

# --- Layer registry ---------------------------------------------------------
#
# Each layer knows: which category it's in, what to call it in output, which
# aliases resolve to it from the command line, which tools it actually needs
# (this is what makes a single-layer run faster — fewer tools offered means
# the model isn't tempted to go research things this layer doesn't need),
# and the specific research/writing instructions for it.

LAYER_SPECS = {
    "economic": {
        "category": "individual",
        "display": "Economic",
        "aliases": ["economic", "economy", "economics"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Economic: GDP trajectory, trade position, growth rate, key "
            "industries. Research via web_search."
        ),
    },
    "technological": {
        "category": "individual",
        "display": "Technological",
        "aliases": ["technological", "technology", "tech", "technological advancements"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Technological: R&D output, key sectors (semiconductors, AI, "
            "etc.), innovation trajectory. Research via web_search."
        ),
    },
    "military": {
        "category": "individual",
        "display": "Military",
        "aliases": ["military", "defense", "defence"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Military: spending, capability trajectory, force posture. "
            "Research via web_search."
        ),
    },
    "demographic": {
        "category": "individual",
        "display": "Demographic",
        "aliases": ["demographic", "demographics", "population"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Demographic: population trend, age structure, workforce "
            "trajectory. These move slowly — do not let a demographic claim "
            "be based on something that could change in under a decade. "
            "Research via web_search."
        ),
    },
    "diplomatic": {
        "category": "individual",
        "display": "Diplomatic/political",
        "aliases": ["diplomatic", "political", "diplomacy", "diplomatic/political"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Diplomatic/political: domestic political stability, alliance "
            "network, international standing. Research via web_search."
        ),
    },
    "bilateral": {
        "category": "comparative",
        "display": "Bilateral relationship history",
        "aliases": ["bilateral", "bilateral relationship history", "relationship history"],
        "tools": ["web_search", "search_historical_analogues", "write_file"],
        "instructions": (
            "Bilateral relationship history: trade volume between them "
            "specifically, and past diplomatic/conflict history specifically "
            "between these two countries. Use web_search for current "
            "bilateral data and search_historical_analogues to find the "
            "closest-matching past SITUATION to this specific dynamic (a "
            "different question from either country's general tendencies). "
            "Call search_historical_analogues with a few different queries "
            "if the first doesn't return a strong match; if nothing matches, "
            "say so rather than forcing an analogy."
        ),
    },
    "structural": {
        "category": "comparative",
        "display": "Structural tendencies",
        "aliases": ["structural", "structural tendencies", "tendencies"],
        "tools": ["get_country_case_history", "write_file"],
        "instructions": (
            "Structural tendencies: call get_country_case_history ONCE for "
            "EACH of the two countries, then compare their dataset-backed "
            "patterns side by side. Every claim here must trace to a "
            "specific case one of those calls returned. If a country had "
            "zero or one case in the dataset, say plainly that the dataset "
            "doesn't support a structural claim for that country — do not "
            "fill the gap with general impression or stereotype. This is "
            "the layer most likely to sound authoritative while being "
            "wrong, so treat the case-count caveats from "
            "get_country_case_history as binding, not optional. Do not use "
            "web_search for this layer — it isn't offered, on purpose."
        ),
    },
    "alliance": {
        "category": "comparative",
        "display": "Third-party alliance overlap",
        "aliases": ["alliance", "alliances", "third-party", "third party", "third-party alliance overlap"],
        "tools": ["web_search", "write_file"],
        "instructions": (
            "Third-party alliance overlap: shared or conflicting "
            "allies/partners — who else has a stake in this relationship "
            "and which side they lean toward. From web_search."
        ),
    },
    "power_trajectory": {
        "category": "comparative",
        "display": "Relative power trajectory",
        "aliases": ["power trajectory", "relative power", "verdict", "trajectory", "relative power trajectory"],
        # Needs the full picture to synthesize from — no fast path here,
        # unlike every other layer, since a verdict without the underlying
        # research is just a guess.
        "tools": ["web_search", "search_historical_analogues", "get_country_case_history", "write_file"],
        "instructions": (
            "Relative power trajectory: this is a synthesized verdict, so "
            "first quickly research the economic, technological, military, "
            "demographic, and diplomatic pictures for both countries via "
            "web_search, and check get_country_case_history for both "
            "countries for structural pattern context — then commit to who "
            "is gaining relative ground, over the given timeframe."
        ),
    },
}

_ALL_LAYER_KEYS_IN_ORDER = [
    "economic",
    "technological",
    "military",
    "demographic",
    "diplomatic",
    "bilateral",
    "structural",
    "alliance",
    "power_trajectory",
]


def resolve_layer(query: str) -> str | None:
    """
    Match free-text like "individual, technological advancements" or just
    "structural tendencies" to a layer key. Matches the LONGEST alias found
    as a substring of the (comma-stripped, lowercased) query, so a more
    specific alias wins over a shorter, looser one.
    """
    normalized = query.lower().replace(",", " ")
    normalized = " ".join(normalized.split())  # collapse whitespace

    best_key, best_len = None, 0
    for key, spec in LAYER_SPECS.items():
        for alias in spec["aliases"]:
            if alias in normalized and len(alias) > best_len:
                best_key, best_len = key, len(alias)
    return best_key


def required_headers(layer: str | None) -> list[str]:
    """The exact section headers the verifier's mechanical check looks for
    — must match what build_system_prompt() actually instructs the model to
    write."""
    if layer is not None:
        return [f"## {LAYER_SPECS[layer]['display']}"]
    return (
        ["## Verdict: relative power trajectory"]
        + [f"### {LAYER_SPECS[k]['display']}" for k in _ALL_LAYER_KEYS_IN_ORDER if LAYER_SPECS[k]["category"] == "individual"]
        + [f"### {LAYER_SPECS[k]['display']}" for k in _ALL_LAYER_KEYS_IN_ORDER if LAYER_SPECS[k]["category"] == "comparative" and k != "power_trajectory"]
    )


def build_tools_and_functions(layer: str | None):
    """Return the (tools, tool_functions) subset relevant to `layer`, or the
    full set if layer is None (whole-report mode)."""
    if layer is None:
        tool_names = list(ALL_TOOL_DEFS.keys())
    else:
        tool_names = LAYER_SPECS[layer]["tools"]

    tools = [ALL_TOOL_DEFS[name] for name in tool_names]
    tool_functions = {
        name: ALL_TOOL_FUNCTIONS[name] for name in tool_names if name in ALL_TOOL_FUNCTIONS
    }
    return tools, tool_functions


_BASE_PROMPT = """\
You are a comparative trend-direction analyst. Given two countries and a \
timeframe, produce a layered comparison and commit to a directional call for \
how their relationship is headed over that timeframe — you must not hedge \
into "it could go either way" with no lean.

This is speculative forecasting, not fact — the structure exists so a reader \
can see your reasoning and disagree with a specific layer, not so you can \
sound certain. Do not present this as a factual prediction; present it as an \
evidence-based judgment call.

After you call write_file, an independent reliability check reviews what \
you wrote against what your tools actually returned this run — not whether \
your forecast is right (nobody can check that), only whether every claim is \
actually grounded in evidence rather than asserted from general impression. \
If it finds gaps, the result of your write_file call will list them \
specifically — go address them (gather more evidence with your tools if \
needed) and call write_file again with the corrected content. You get a \
limited number of correction attempts, so prioritize the most important gap \
first.

Keep every section tight — stick to the stated length. Pick the strongest \
one or two supporting points where the format asks for that, not an \
exhaustive list.
"""

_CONFIDENCE_AND_SOURCES = """\
  ## Confidence & what would change this call
  1-2 sentences: your confidence (low/medium/high) and the 1-3 specific \
events that would flip this call if they happened. A list of events, not \
paragraphs about each.

  ## Sources
  A flat list of the web sources you used and the dataset cases you drew \
on, if any — no commentary per source.
"""


def build_system_prompt(layer: str | None) -> str:
    if layer is None:
        individual = "\n".join(
            f"  - {LAYER_SPECS[k]['instructions']}"
            for k in _ALL_LAYER_KEYS_IN_ORDER
            if LAYER_SPECS[k]["category"] == "individual"
        )
        comparative = "\n".join(
            f"  - {LAYER_SPECS[k]['instructions']}"
            for k in _ALL_LAYER_KEYS_IN_ORDER
            if LAYER_SPECS[k]["category"] == "comparative"
        )
        return f"""{_BASE_PROMPT}
The output has a fixed set of layers. Do not merge them into one narrative \
— a reader should be able to look at any single layer and evaluate it on \
its own.

INDIVIDUAL LAYERS (assess each country separately, then compare):
{individual}

COMPARATIVE LAYERS (about the pair itself, not either country alone):
{comparative}

Once you have enough to reason from, write your analysis to a file using \
write_file, in this exact structure:

  # <country A> vs <country B> — <timeframe>

  ## Verdict: relative power trajectory
  One tight paragraph (4-5 sentences max): improving / worsening / stable \
for the relationship, and which country is gaining relative ground, over \
the given timeframe. State it plainly, don't hedge. This must be a \
synthesis of every layer below, not just a restatement of current news.

  ## Individual layers
  ### Economic
  ### Technological
  ### Military
  ### Demographic
  ### Diplomatic/political
  For each: 2-3 sentences on country A, 2-3 sentences on country B, then one \
explicit comparison sentence ("A leads on X; B is closing the gap because \
Y").

  ## Comparative layers
  ### Bilateral relationship history
  ### Structural tendencies
  ### Third-party alliance overlap

{_CONFIDENCE_AND_SOURCES}"""

    spec = LAYER_SPECS[layer]
    return f"""{_BASE_PROMPT}
You are producing ONLY the "{spec['display']}" layer this run — not the full \
9-layer report. Do not research or write any other layer. This is a \
deliberately narrow, fast run: the tools offered to you are limited to \
exactly what this layer needs.

{spec['instructions']}

Once you have enough to reason from, write your analysis to a file using \
write_file, in this exact structure:

  # <country A> vs <country B> — <timeframe> — {spec['display']}

  ## {spec['display']}
  2-3 sentences on country A, 2-3 sentences on country B, then one explicit \
comparison sentence. This layer only — 6-8 sentences total, not an essay.

{_CONFIDENCE_AND_SOURCES}"""


def run_agent(
    country_a: str, country_b: str, timeframe: str, layer: str | None = None, max_turns: int = 25
) -> str:
    # Constructed here, not at module import time, so a missing OPENAI_API_KEY
    # doesn't crash before the usage/error messages below get a chance to run.
    client = OpenAI()

    tools, tool_functions = build_tools_and_functions(layer)
    system_prompt = build_system_prompt(layer)
    headers = required_headers(layer)

    # Everything the tools actually return this run, so the verifier can
    # check claims against real evidence instead of trusting the report.
    evidence_log: list[dict] = []
    verify_attempts = {"count": 0}  # mutable so the closure below can update it

    # The model's final chat message is often just a short acknowledgement
    # ("Report written.") rather than the report itself — trusting that as
    # "the result" silently discards the actual content. Track the real
    # written text here and prefer it at both return points below.
    last_written_content = {"text": None}

    def verified_write_file(filename: str, content: str) -> str:
        """Wraps the plain write_file tool with a reliability checkpoint.
        The model only ever sees this version — see the write_file override
        right after build_tools_and_functions() below."""
        write_file(filename, content)  # persist to disk first, always
        last_written_content["text"] = content

        passed, gaps = verify_report(client, content, evidence_log, headers)
        if passed:
            return (
                "Report written and passed the reliability check: all required "
                "sections present, and claims are grounded in this run's "
                "evidence. Done — no need to call write_file again."
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
                f"Finalized as-is with the remaining gaps appended to the file "
                f"as an explicit caveats section. Do not call write_file again "
                f"— give your closing summary now."
            )

        gap_list = "\n".join(f"- {g}" for g in gaps)
        return (
            f"Report written, but the reliability check (attempt "
            f"{verify_attempts['count']}/{MAX_VERIFY_ATTEMPTS}) found gaps:\n"
            f"{gap_list}\n\n"
            f"Address these — gather more evidence via your tools if needed — "
            f"then call write_file again with the corrected content."
        )

    tool_functions["write_file"] = lambda **kw: verified_write_file(kw["filename"], kw["content"])

    input_items = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Country A: {country_a}\nCountry B: {country_b}\nTimeframe: {timeframe}",
        },
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ---")

        response = client.responses.create(
            model=MODEL,
            tools=tools,
            input=input_items,
        )
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

            # write_file's own result already IS the verification feedback —
            # nothing extra to log there. The two research tools get logged
            # so the verifier can check claims against them.
            if item.name in ("search_historical_analogues", "get_country_case_history"):
                evidence_log.append({"tool": item.name, "args": args, "result": result})

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                }
            )

    return last_written_content["text"] or "Stopped: hit max_turns without finishing."


if __name__ == "__main__":
    if len(sys.argv) < 4:
        available = ", ".join(spec["display"] for spec in LAYER_SPECS.values())
        print(
            'Usage: python trend_direction_agent.py "<country A>" "<country B>" "<timeframe>" [layer]\n'
            'Example (full report):  python trend_direction_agent.py "United States" "China" "next 20 years"\n'
            'Example (one layer):    python trend_direction_agent.py "United States" "China" "next 20 years" technological\n\n'
            f"Available layers: {available}"
        )
        sys.exit(1)

    country_a, country_b, timeframe = sys.argv[1], sys.argv[2], sys.argv[3]
    layer_query = " ".join(sys.argv[4:]) if len(sys.argv) > 4 else None

    layer = None
    if layer_query:
        layer = resolve_layer(layer_query)
        if layer is None:
            available = ", ".join(spec["display"] for spec in LAYER_SPECS.values())
            print(f"Couldn't match '{layer_query}' to a known layer.\nAvailable layers: {available}")
            sys.exit(1)

    print(f"Country A: {country_a}\nCountry B: {country_b}\nTimeframe: {timeframe}")
    print(f"Layer: {LAYER_SPECS[layer]['display']}" if layer else "Layer: (full report)")
    print()

    answer = run_agent(country_a, country_b, timeframe, layer)
    print("\n=== Final answer ===")
    print(answer)
