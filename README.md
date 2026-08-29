# Research Finder

A small collection of LLM agents for career / research direction work. They
share one idea: a plain while-loop around a single model call, where each
pass lets the model call tools (`web_search`, local datasets, file writes),
read the results, and decide what to do next — with a verification
checkpoint that rejects any report whose claims aren't grounded in what the
tools actually returned this run.

Nothing is ever sent on your behalf. Outreach emails and application
materials are drafted to disk for you to review and send yourself.

## Setup

```bash
pip install -r requirements.txt
```

Set whichever API key the agent you're running needs (see table below):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

No key is stored in the repo — each agent constructs its client from the
environment at runtime.

## Your profile

Most agents take a profile file as their first argument. Copy the template
and fill it in:

```bash
cp profile_template.md my_profile.md
```

`my_profile.md` stays local — it's git-ignored and only ever read from disk
and passed to the model as context.

## The agents

| Script | Provider | What it does |
| --- | --- | --- |
| `professor_outreach_agent.py` | OpenAI | Finds professors at a university whose recent work matches your interests, drafts a tailored outreach email per professor to `sandbox/drafts/professors/`, and writes an index report. Remembers who it drafted across runs so repeat runs don't overlap. |
| `next_steps_agent.py` | OpenAI | Given your profile and directions you're weighing, produces concrete next steps grounded in real current opportunities and realistic entry paths. Logs opportunities and drafts application materials via `action_tools.py`. |
| `trend_direction_agent.py` | OpenAI | Layered comparison of two countries and a directional call on where their relationship is headed. |
| `technology_direction_agent.py` | OpenAI | Assesses whether one technology/trend is headed toward mainstream, a narrow niche, or fading. |
| `backtest.py` | OpenAI | Process-check harness for the trend agent's method: blind prediction on a period snapshot, then a separate informed grading pass. |
| `agent_manual.py` / `agent_runner.py` | Anthropic | Minimal reference agents showing the raw loop (hand-written vs. SDK Tool Runner). |
| `agent_manual_openai.py` | OpenAI | The same minimal agent on OpenAI's Responses API. |

### Examples

```bash
python professor_outreach_agent.py my_profile.md "UC Berkeley" "ML,AI,Probability,RL" 10
```

```bash
python next_steps_agent.py my_profile.md "next 6 months" "ML,AI,Quant,Startup"
```

```bash
python trend_direction_agent.py "United States" "China" "next 20 years"
python trend_direction_agent.py "United States" "China" "next 20 years" technological
```

```bash
python technology_direction_agent.py "quantum computing" "next 15 years"
```

```bash
python backtest.py "United States" "China" 2005 20 snapshots/us_china_2005.txt
```

Most agents accept an optional trailing argument to run a single section
instead of the full report, which also shrinks the tool list. The
professor agent accepts an optional `count` and `token_budget`.

## Action items

`next_steps_agent.py` and `professor_outreach_agent.py` log tracked items to
`action_items.json` (git-ignored). Manage them any time:

```bash
python action_tools.py list
python action_tools.py list open
python action_tools.py done 3
python action_tools.py in_progress 5
```

## Outputs

- `sandbox/drafts/` — drafted emails and materials (git-ignored)
- `action_items.json`, `contacted_professors.json` — local run state, created
  automatically, git-ignored

## Layout

```
*_agent.py          the agents
action_tools.py     action-item + draft tracking (CLI + tools)
tools.py            shared file tools (sandboxed read/write/list)
verification.py     grounding checkpoint used by the report agents
humanizer.py        light pass to de-robotify drafted prose
*_data.py           local datasets the agents ground claims in
*_cases.json         "
career_paths.json    "
snapshots/          period snapshots for backtest.py
```
