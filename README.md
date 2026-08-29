# Research Finder

An LLM agent that finds professors at a university whose recent work matches
your research interests, then drafts a tailored outreach email for each one.

How it works: a plain while-loop around a single model call. Each pass lets
the model call tools (`web_search`, file writes), read the results, and
decide what to do next. A verification checkpoint rejects any draft or index
report whose claims aren't grounded in what the tools actually returned that
run.

Nothing is ever sent on your behalf. Emails are drafted to disk for you to
review and send yourself. The agent also remembers who it drafted across
runs, so repeat runs give you non-overlapping batches.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

No key is stored in the repo — the client is built from the environment at
runtime.

## Your profile

The agent takes a profile file (Markdown) as its first argument. Copy the
template and fill it in:

```bash
cp profile_template.md my_profile.md
```

`my_profile.md` stays local — it's git-ignored, and only ever read from disk
and passed to the model as context. Keep it out of any public repo; it's
personal data in plain text.

### Turning a resume into the profile

The agent reads Markdown, not PDFs. To build `my_profile.md` from a
resume/transcript:

1. Open `profile_template.md` — the `EXAMPLE` blocks show the level of
   detail that's actually useful. Vague entries ("good at coding") give the
   agent nothing to work with; specific ones ("built X with Y, took
   graduate-level Z") do.
2. Fill in each section from your resume: education (degree, school,
   expected graduation, relevant coursework, GPA if you want it weighed),
   research/work experience, projects, technical skills, and — most
   important — the research areas and problems you actually want to work on.
3. If a fact isn't on your resume, either leave it out or mark it
   `unknown` rather than guessing. Delete any section that doesn't apply.
4. Shortcut: paste your resume text (and transcript, if relevant) into any
   LLM along with `profile_template.md` and ask it to fill the template in
   that exact structure — then read the result and correct anything it
   inferred wrong before saving as `my_profile.md`.

## Run it

```bash
python professor_outreach_agent.py my_profile.md "UC Berkeley" "ML,AI,Probability,RL" 10
```

Arguments:

| Position | Argument | Notes |
| --- | --- | --- |
| 1 | profile file | e.g. `my_profile.md` |
| 2 | university | quoted, e.g. `"UC Berkeley"` |
| 3 | research areas | comma-separated, quoted, e.g. `"ML,AI,Probability,RL"` |
| 4 | count | optional, default 10 — how many professors to draft |
| 5 | token budget | optional — stop the run (keeping drafts already saved) once total tokens cross this number. Leave off the first run to see real usage in the printed summary, then set an informed budget. |

Each run automatically excludes professors drafted in a previous run (see
`contacted_professors.json`), so you can just run it again for the next
batch.

## Outputs

- `sandbox/drafts/professors/*.md` — one drafted email per professor, plus
  an index report. Git-ignored.
- `action_items.json` — a tracked "send this email" item per draft, created
  automatically. Git-ignored.
- `contacted_professors.json` — the cross-run list of who's been drafted.
  Git-ignored.

Manage the tracked items any time:

```bash
python action_tools.py list
python action_tools.py list open
python action_tools.py done 3
python action_tools.py in_progress 5
```

## Layout

```
professor_outreach_agent.py   the agent
action_tools.py               action-item + draft tracking (CLI + tools)
tools.py                      sandboxed file read/write/list
verification.py               grounding checkpoint for drafts and reports
humanizer.py                  light pass to de-robotify drafted prose
profile_template.md           copy to my_profile.md and fill in
```
