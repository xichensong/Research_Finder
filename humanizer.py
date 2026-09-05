"""
Ports the anthropic-skills:humanizer skill's actual rules into a standalone
rewriting pass, for use inside startup_outreach_agent.py (and reusable
elsewhere). This exists because that skill only runs inside a Claude Code
session — it can't be invoked directly from a standalone script hitting the
OpenAI API. So instead of guessing at "sound more human," this module
carries the skill's real rule set (Wikipedia's "Signs of AI writing"
categories) as the system prompt for an independent rewriting call.

Not a verification gate — a rewriting gate. It always returns text, it
doesn't pass/fail. Use it AFTER a grounding gate has approved the content
(see startup_outreach_agent.py's propose_fit_point / propose_email_draft),
so you're not humanizing something that might still get rejected.
"""

from openai import OpenAI

HUMANIZER_MODEL = "gpt-5.6"

_HUMANIZER_SYSTEM_PROMPT = """\
You are a writing editor. Rewrite the given text to remove AI-writing \
patterns and sound like a real person wrote it, without changing its \
meaning or the facts in it.

NEVER use an em dash (—) anywhere in your output, for any reason. Replace \
every one with a period, a comma, or a semicolon — whichever reads most \
naturally at that spot — never leave one in and never substitute a hyphen \
surrounded by spaces either. This rule overrides any instinct to use one \
for a "punchy" aside.

Cut these patterns wherever they appear:
- Inflated significance ("stands as a testament to", "marks a pivotal \
moment", "underscores the importance of")
- Superficial "-ing" tack-ons ("...ensuring users can...", "...reflecting \
its commitment to...")
- Promotional language ("vibrant", "cutting-edge", "seamless", "robust")
- Vague attributions ("industry experts believe", "studies show" with no \
actual source)
- Rule-of-three padding (forcing every list into exactly three items to \
sound comprehensive)
- Negative parallelism ("it's not just X, it's Y")
- Overused AI vocabulary: additionally, crucial, delve, align with, \
foster, garner, highlight (as a verb), intricate, key (as filler adjective), \
landscape (abstract sense), pivotal, showcase, testament, underscore (as a \
verb), tapestry, robust, leverage (as a verb), utilize (use "use" instead)
- Copula avoidance ("serves as", "stands as", "represents" in place of a \
plain "is")
- Excessive hedging ("could potentially possibly")
- Generic upbeat closers ("the future looks bright", "exciting times \
ahead")
- Filler phrases ("in order to" → "to", "due to the fact that" → "because")
- Collaborative-artifact leftovers if any slipped in ("I hope this helps!", \
"let me know if...")

Then add actual voice: vary sentence length and rhythm, be specific instead \
of vague, and let it read like one particular person wrote it — not the \
statistically average sentence for the topic. Keep the tone appropriate to \
the context you're given (e.g. a cold email to a startup founder should \
stay professional and concise — "voice" here means specific and genuine, not \
casual or jokey).

Do not change any factual claim, name, paper title, or URL — only rewrite \
the prose around them. If the input is already clean, return it close to \
unchanged rather than rewriting for the sake of it.

Return ONLY the rewritten text — no preamble, no explanation, no "Here is \
the rewritten version:".
"""


def _strip_em_dashes(text: str) -> str:
    """
    Mechanical backstop. The system prompt above already forbids em dashes
    explicitly and forcefully — but a model can still drop one in anyway,
    and "prompt harder" isn't a reliable fix for something this checkable.
    Same philosophy as the rest of this project: enforce mechanically
    wherever you can, don't just hope the prompt was followed.

    " — " (spaced, the common case) becomes ". " if it looks like it's
    joining two independent clauses (both sides start with a capital
    letter after trimming), else ", " — a reasonable default for the
    appositive/aside case. A bare "—" with no surrounding spaces just
    becomes ", ".
    """
    while " — " in text:
        idx = text.index(" — ")
        before, after = text[:idx], text[idx + 3:]
        # If what follows already starts with a capital, it reads as an
        # independent clause — split it into its own sentence. Otherwise
        # treat it as an aside and just comma it in. `after[0]` is already
        # uppercase in the first branch, so no extra capitalization needed.
        sep = ". " if after[:1].isupper() else ", "
        text = before + sep + after
    text = text.replace("—", ", ").rstrip()
    # A dash at the very end of a sentence (rare, malformed input) leaves a
    # dangling trailing comma — trim it rather than ship one.
    if text.endswith(","):
        text = text[:-1].rstrip()
    return text


def humanize_text(client: OpenAI, text: str, context: str = "") -> str:
    """
    Rewrite `text` to remove AI-writing patterns per the rules above.
    `context` is optional free text telling the rewriter what the piece is
    (e.g. "a cold email to a startup founder" or "a one-paragraph technical
    connection point") so tone stays appropriate.
    """
    user_content = f"CONTEXT: {context}\n\nTEXT TO HUMANIZE:\n{text}" if context else text
    response = client.responses.create(
        model=HUMANIZER_MODEL,
        input=[
            {"role": "system", "content": _HUMANIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        reasoning={"effort": "low"},
    )
    return _strip_em_dashes(response.output_text.strip())
