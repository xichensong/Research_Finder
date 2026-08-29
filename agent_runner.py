"""
The same agent as agent_manual.py, but using the Anthropic SDK's built-in
Tool Runner instead of a hand-written loop.

Tool Runner does exactly what agent_manual.py's while-loop does — call the
model, run any requested tools, feed results back, repeat until Claude stops
asking for tools — but you don't write the loop yourself. You just define
your tools as plain Python functions with the @beta_tool decorator (which
reads the type hints and docstring to build the schema for you) and hand
them to the runner.

Use this version once you understand the manual one. It's what you'd
actually use in real code.

Run it:  python agent_runner.py
"""

import anthropic
from anthropic import beta_tool

from tools import list_files as _list_files
from tools import read_file as _read_file
from tools import write_file as _write_file

client = anthropic.Anthropic()

MODEL = "claude-opus-5"


# The @beta_tool decorator turns a typed function + docstring into a tool
# schema automatically — no hand-written JSON Schema needed.
@beta_tool
def read_file(filename: str) -> str:
    """Read the contents of a text file from the local sandbox directory.

    Args:
        filename: Name of the file to read.
    """
    return _read_file(filename)


@beta_tool
def write_file(filename: str, content: str) -> str:
    """Write text content to a file in the local sandbox directory, creating or overwriting it.

    Args:
        filename: Name of the file to write.
        content: The text content to write.
    """
    return _write_file(filename, content)


@beta_tool
def list_files() -> str:
    """List all files currently in the local sandbox directory."""
    return _list_files()


def run_agent(task: str) -> str:
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=4096,
        tools=[
            {"type": "web_search_20260209", "name": "web_search"},  # server-side tool
            read_file,
            write_file,
            list_files,
        ],
        messages=[{"role": "user", "content": task}],
    )

    final_text = ""
    for message in runner:
        # Each iteration is one full turn. Print what happened, same as the
        # manual version, so you can watch its progress.
        for block in message.content:
            if block.type == "text" and block.text.strip():
                print(f"Claude: {block.text.strip()}")
            elif block.type == "tool_use":
                print(f"Claude called: {block.name}({block.input})")
            elif block.type == "server_tool_use":
                print(f"Claude searched: {block.input.get('query')}")
        if message.stop_reason == "end_turn":
            final_text = "".join(b.text for b in message.content if b.type == "text")

    return final_text


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
