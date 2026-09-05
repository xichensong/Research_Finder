# Startup Internship Finder

An LLM agent that finds ML / AI / quant startups backed by a credible
accelerator or tier-1 VC, works out one specific point where what they build
connects to something you have actually done, and drafts a cold email around
that connection. It never sends anything.

A company does not need an open internship posting to be included. If the
agent finds a real opening, the email references it; if not, the email just
makes the case for what you could contribute. Either way the anchor is a
real, specific connection between the company's work and yours, not generic
enthusiasm.

How it works: a plain while-loop around a single model call. Each pass lets
the model call tools (`web_search`, file writes), read the results, and
decide what to do next. Gated checkpoints reject a draft whose claims about
a company aren't supported by what search actually returned, and a final
check rejects an index report that names a company its pipeline never
finished. The agent also remembers which companies it drafted across runs,
so repeat runs give you non-overlapping batches.

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

The agent reads Markdown, not PDFs. To build `my_profile.md` from a resume:

1. Open `profile_template.md` — the `EXAMPLE` blocks show the level of
   detail that's useful. Vague entries ("good at ML") give the agent
   nothing; specific ones ("built and shipped X with Y users using Z") do.
2. Fill in each section from your resume: education, work and research
   experience, projects, publications, technical skills, and the areas you
   want to work in.
3. Optionally state a priority order for what the agent should lead with in
   an email. The default is: (1) a shipped product with real users, (2) a
   directly relevant internship or research role, (3) self-initiated or
   first-author research. The agent picks the highest-priority item that
   genuinely fits each company.
4. If a fact isn't on your resume, leave it out or mark it `unknown` rather
   than guessing.
5. Shortcut: paste your resume text into any LLM with `profile_template.md`
   and ask it to fill the template in that structure, then review and
   correct anything it inferred wrong before saving as `my_profile.md`.

## Run it

```bash
python startup_outreach_agent.py my_profile.md "ML,AI,Quant" 8
```

| Position | Argument | Notes |
| --- | --- | --- |
| 1 | profile file | e.g. `my_profile.md` |
| 2 | focus areas | comma-separated, quoted, e.g. `"ML,AI,Quant"` |
| 3 | count | optional, default 8 — how many companies to draft |
| 4 | token budget | optional — stop the run (keeping drafts already saved) once total tokens cross this number. Leave it off the first run to see real usage in the printed summary, then set an informed budget. |

Each run automatically excludes companies drafted in a previous run (see
`contacted_companies.json`), so you can just run it again for the next batch.

### What counts as a "credible" backer

Defined in `CREDIBLE_BACKERS` at the top of `startup_outreach_agent.py`.
Currently: YC, Techstars, a16z Speedrun, Neo, South Park Commons, Pear VC,
AI Grant, Entrepreneur First — or a disclosed funding round led by or
including a tier-1 firm (Sequoia, a16z, Benchmark, Founders Fund, Greylock,
Lightspeed, Index, Accel, and similar). Edit that constant to change the
bar. The credibility gate can't truly verify funding — it catches obvious
misses and vague backers, and the model is told to confirm the backer via
search before logging a company.

## Outputs

- `sandbox/drafts/startups/*.md` — one file per company: backer, what they
  do, the role (or a note that there's no posting), the fit point, and the
  drafted email. Plus an index report. Git-ignored.
- `action_items.json` — a tracked "send this email" item per draft, created
  automatically. Git-ignored.
- `contacted_companies.json` — the cross-run list of who's been drafted.
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
startup_outreach_agent.py   the agent
action_tools.py             action-item + draft tracking (CLI + tools)
tools.py                    sandboxed file read/write/list
verification.py             grounding checkpoint for the index report
humanizer.py                light pass to de-robotify drafted prose
profile_template.md         copy to my_profile.md and fill in
```
