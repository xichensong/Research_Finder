"""
A minimal agent, built by hand, so you can see exactly how the loop works.

The core idea of "an agent" is just this:

    1. Send Claude the conversation so far, plus a list of tools it's allowed
       to call.
    2. Claude replies with either a final answer, or a request to call one
       or more tools ("tool_use" blocks).
    3. If it asked for tools: we run them ourselves, append the results to
       the conversation as "tool_result" blocks, and go back to step 1.
    4. If it gave a final answer: we're done.

That's it. There's no magic — it's a while-loop around one API call, where
each pass through the loop is Claude "checking its progress" and deciding
what to do next based on what the tools returned.

Run it:  python agent_manual.py
"""

import json

import anthropic

from tools import list_files, read_file, write_file

client = anthropic.Anthropic()

MODEL = "claude-opus-5"

# --- 1. Describe our tools to Claude -----------------------------------
#
# Each tool needs a name, a description (Claude decides *when* to use a
# tool almost entirely from this text, so be specific), and a JSON Schema
# describing its arguments.

TOOLS = [
    {
        "type": "web_search_20260209",  # built-in server-side tool — Anthropic runs the actual search
        "name": "web_search",
    },
    {
        "name": "read_file",
        "description": "Read the contents of a text file from the local sandbox directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the file to read."}
            },
            "required": ["filename"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text content to a file in the local sandbox directory, creating or overwriting it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the file to write."},
                "content": {"type": "string", "description": "The text content to write."},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files currently in the local sandbox directory.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# Map tool names to the actual Python functions that execute them.
# (web_search isn't here because it runs on Anthropic's servers, not ours.)
TOOL_FUNCTIONS = {
    "read_file": lambda **kw: read_file(kw["filename"]),
    "write_file": lambda **kw: write_file(kw["filename"], kw["content"]),
    "list_files": lambda **kw: list_files(),
}


def run_agent(task: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": task}]

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        # Always append Claude's full response to history before doing
        # anything else with it — the tool_use blocks it contains need to
        # stay paired with the tool_result blocks we're about to add.
        messages.append({"role": "assistant", "content": response.content})

        # Show what Claude said / decided to do this turn, so you can watch
        # it "think" — this is the progress-checking in action.
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"Claude: {block.text.strip()}")
            elif block.type == "tool_use":
                print(f"Claude wants to call: {block.name}({json.dumps(block.input)})")
            elif block.type == "server_tool_use":
                print(f"Claude is searching: {block.input.get('query')}")

        # stop_reason tells us why the model stopped generating this turn.
        if response.stop_reason == "end_turn":
            # No more tools requested — Claude is done.
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return final_text

        if response.stop_reason != "tool_use":
            # Anything else (max_tokens, refusal, ...) — bail out rather
            # than looping forever.
            return f"Stopped early: {response.stop_reason}"

        # Execute every tool_use block from this turn and collect results.
        # (Server-side tools like web_search are handled by Anthropic and
        # already show up as *_tool_result blocks — we only execute the
        # client-side ones ourselves.)
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            func = TOOL_FUNCTIONS.get(block.name)
            if func is None:
                result = f"Error: no such tool '{block.name}'"
                is_error = True
            else:
                try:
                    result = func(**block.input)
                    is_error = False
                except Exception as e:
                    result = f"Error running {block.name}: {e}"
                    is_error = True
            print(f"  -> {result[:200]}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                }
            )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return "Stopped: hit max_turns without finishing."


if __name__ == "__main__":
    task = (
        "Search the web for the current version number of Python, "
        "then write a file called python_version.txt in the sandbox "
        "containing that version number and one sentence about what's new in it. "
        "Afterwards, list the files in the sandbox to confirm it's there."
    )
    print(f"Task: {task}\n")
    answer = run_agent(task)
    print("\n=== Final answer ===")
    print(answer)
