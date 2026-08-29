"""
A backtest harness for trend_direction_agent's method — see the conversation
that led here for the full reasoning. Short version:

  Real backtesting = run the method blind on old data, check it against
  what actually happened. That's the only way to actually validate a
  forecasting method.

  The problem: an LLM already knows what happened after any cutoff you
  pick, baked into its training. Restricting which TOOLS it can see doesn't
  erase what it already knows. So this cannot be a clean, leak-free
  validation of predictive accuracy — that would require a model that was
  actually trained only on pre-cutoff data, which isn't available here.

  What this DOES give you: a controlled PROCESS check. Two separate passes,
  each honest about what it knows:

    1. BLIND prediction — given only a manually-written snapshot of what
       was known as of the cutoff, plus the historical-cases dataset
       filtered to exclude anything that concluded after the cutoff, it
       produces a directional call for the years following the cutoff.
       No web_search (there's no reliable way to date-filter it), so this
       step's tool access is genuinely restricted — even though the
       model's own latent knowledge isn't.

    2. INFORMED grading — a SEPARATE pass, with full web_search access and
       no restrictions, checks the blind prediction against what actually
       happened. This one is *supposed* to have hindsight — that's the
       point of grading.

Treat the result as "did the method reach a defensible call given
period-appropriate evidence," not "did it prove it can predict the future."
It can't prove that, and no LLM-based method honestly can, yet.

Run it:
  python backtest.py "United States" "China" 2005 20 snapshots/us_china_2005.txt
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

from historical_data import get_country_case_history, search_historical_analogues
from tools import write_file

MODEL = "gpt-5.6"


def _agent_loop(
    client: OpenAI, system_prompt: str, user_content: str, tools: list, tool_functions: dict, max_turns: int = 10
) -> str:
    """The same loop shape as everywhere else in this project — kept
    standalone here rather than imported, so this file works as a
    self-contained example of the backtest mechanics.

    Returns the actual content passed to write_file, if it was ever called
    — NOT the model's final chat message. Those are often just a short
    acknowledgement ("Report written.") rather than the report itself, and
    trusting that as "the result" silently throws away the real content —
    exactly the bug this comment is here to prevent you from reintroducing.
    """
    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    last_written_content: str | None = None

    for turn in range(1, max_turns + 1):
        print(f"  --- turn {turn} ---")
        response = client.responses.create(model=MODEL, tools=tools, input=input_items)
        input_items += response.output

        function_calls = []
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if getattr(content, "text", "").strip():
                        print(f"  Model: {content.text.strip()[:200]}")
            elif item.type == "function_call":
                print(f"  Model wants to call: {item.name}({item.arguments})")
                function_calls.append(item)
            elif item.type == "web_search_call":
                print(f"  Model is web-searching: {getattr(item.action, 'query', '')}")

        if not function_calls:
            return last_written_content or response.output_text

        for item in function_calls:
            args = json.loads(item.arguments)
            if item.name == "write_file" and "content" in args:
                last_written_content = args["content"]
            func = tool_functions.get(item.name)
            try:
                result = func(**args)
            except Exception as e:
                result = f"Error running {item.name}: {e}"
            print(f"    -> {result[:150]}")
            input_items.append({"type": "function_call_output", "call_id": item.call_id, "output": result})

    return last_written_content or "Stopped: hit max_turns without finishing."


# --- Step 1: blind prediction ----------------------------------------------

_BLIND_TOOLS = [
    {
        "type": "function",
        "name": "search_historical_analogues",
        "description": (
            "Search the historical-cases dataset for analogues to a situation, "
            "restricted to cases fully concluded before the cutoff year — "
            "nothing from after the cutoff is visible."
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
        "name": "get_country_case_history",
        "description": (
            "Get every dataset case involving a country, restricted to cases "
            "fully concluded before the cutoff year."
        ),
        "parameters": {
            "type": "object",
            "properties": {"country": {"type": "string"}},
            "required": ["country"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write the finished blind prediction to a file in the sandbox.",
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


def run_blind_prediction(
    client: OpenAI, country_a: str, country_b: str, cutoff_year: int, years_forward: int, period_snapshot: str
) -> str:
    tool_functions = {
        "search_historical_analogues": lambda **kw: search_historical_analogues(kw["query"], cutoff_year=cutoff_year),
        "get_country_case_history": lambda **kw: get_country_case_history(kw["country"], cutoff_year=cutoff_year),
        "write_file": lambda **kw: write_file(kw["filename"], kw["content"]),
    }

    system_prompt = f"""\
You are making a BLIND forecast as if it were the year {cutoff_year}. You \
must reason ONLY from:
  1. The "known context as of {cutoff_year}" snapshot given below — this is \
everything you're allowed to treat as known fact.
  2. search_historical_analogues and get_country_case_history — both are \
restricted to cases that concluded before {cutoff_year}; nothing later is \
visible to them.

You do NOT have web_search. Do not use any knowledge you have of what \
actually happened after {cutoff_year} — reason as if you genuinely don't \
know yet. If the given snapshot doesn't cover something you'd want to know, \
say the prediction is limited by that gap rather than filling it in.

Produce a directional call — improving / worsening / stable — for the \
{country_a}-{country_b} relationship over the {years_forward} years \
following {cutoff_year}. State it plainly, don't hedge into "could go \
either way."

Write your answer using write_file, in this structure:

  # Blind prediction: {country_a} vs {country_b}, made as of {cutoff_year}

  ## Prediction
  One tight paragraph (4-5 sentences max): the directional call for the \
next {years_forward} years, and why, drawn only from the snapshot and the \
cutoff-restricted dataset tools.

  ## Historical grounding used
  2-3 sentences. Name the case(s) this leans on and the one-line reason \
each applies — not a paragraph per case.

  ## Confidence & known gaps
  1-2 sentences: confidence level, and the single biggest gap in the \
{cutoff_year} snapshot that limits this call.
"""

    user_content = f"Known context as of {cutoff_year}:\n\n{period_snapshot}"
    return _agent_loop(client, system_prompt, user_content, _BLIND_TOOLS, tool_functions)


# --- Step 2: informed grading ------------------------------------------------

_GRADING_TOOLS = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "write_file",
        "description": "Write the finished grading report to a file in the sandbox.",
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

_GRADING_TOOL_FUNCTIONS = {
    "write_file": lambda **kw: write_file(kw["filename"], kw["content"]),
}


def grade_backtest(
    client: OpenAI, country_a: str, country_b: str, cutoff_year: int, years_forward: int, prediction_text: str
) -> str:
    system_prompt = f"""\
You are grading a forecast that was made BLIND, as if it were {cutoff_year}, \
for the {country_a}-{country_b} relationship over the following \
{years_forward} years. You have full, current knowledge and web_search \
access — this pass is SUPPOSED to use hindsight, that's the point of \
grading.

Research via web_search what actually happened to this relationship between \
{cutoff_year} and {cutoff_year + years_forward} (or up to now, if that's \
sooner). 2-3 searches is enough — you're checking the direction of the \
relationship, not writing its complete history. Then compare the actual \
trajectory to the blind prediction's directional call.

Keep the whole thing tight. This is a grade, not a research report — every \
sentence should be doing work toward "did the direction match," not adding \
one more supporting fact. If you're tempted to list a fourth or fifth \
example of the same point, don't — pick the strongest one or two and move \
on. Skip citation markers/footnotes; name sources in plain text only if it's \
actually load-bearing for the reader.

Write your grading using write_file, in this structure (stick to the stated \
lengths):

  # Backtest grade: {country_a} vs {country_b}, predicted from {cutoff_year}

  ## What was predicted
  One sentence: quote the blind prediction's directional call.

  ## What actually happened
  3-5 sentences. The events that most bear on the direction call, not a \
timeline of everything you found.

  ## Match assessment
  One tight paragraph. Did the direction match? Name the single biggest \
thing it got right and the single biggest thing it missed — not an \
exhaustive list of both.

  ## What this does and doesn't tell you
  1-2 sentences: this is one data point from one method run, and the blind \
step can't fully rule out the model's own knowledge of real history leaking \
into its reasoning — a good grade here is suggestive, not proof the method \
works.
"""
    user_content = f"Blind prediction to grade:\n\n{prediction_text}"
    return _agent_loop(client, system_prompt, user_content, _GRADING_TOOLS, _GRADING_TOOL_FUNCTIONS)


def run_full_backtest(country_a: str, country_b: str, cutoff_year: int, years_forward: int, snapshot_path: str):
    client = OpenAI()
    period_snapshot = Path(snapshot_path).read_text()

    print(f"\n=== Step 1: blind prediction as of {cutoff_year} ===")
    prediction = run_blind_prediction(client, country_a, country_b, cutoff_year, years_forward, period_snapshot)
    print("\n--- blind prediction ---")
    print(prediction)

    print(f"\n=== Step 2: informed grading against real history ===")
    grade = grade_backtest(client, country_a, country_b, cutoff_year, years_forward, prediction)
    print("\n--- grade ---")
    print(grade)

    return prediction, grade


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            'Usage: python backtest.py "<country A>" "<country B>" <cutoff_year> <years_forward> <snapshot_file>\n'
            'Example: python backtest.py "United States" "China" 2005 20 snapshots/us_china_2005.txt'
        )
        sys.exit(1)

    country_a, country_b = sys.argv[1], sys.argv[2]
    cutoff_year = int(sys.argv[3])
    years_forward = int(sys.argv[4])
    snapshot_path = sys.argv[5]

    if not Path(snapshot_path).exists():
        print(f"Snapshot file not found: {snapshot_path}")
        sys.exit(1)

    run_full_backtest(country_a, country_b, cutoff_year, years_forward, snapshot_path)
