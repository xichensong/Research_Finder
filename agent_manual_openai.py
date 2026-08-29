"""
Same agent as agent_manual.py, rebuilt on OpenAI's Responses API instead of
Anthropic's Messages API. The *shape* of an agent is identical between
providers — only the field names differ:

    Anthropic                          OpenAI (Responses API)
    ----------------------------------------------------------------
    messages=[...]                     input=[...]
    tool_use block                     function_call output item
    block.input (dict, pre-parsed)     item.arguments (JSON string - must
                                        json.loads() it yourself)
    tool_result block                  function_call_output item
    tool_use_id                        call_id
    stop_reason == "tool_use"          response.output contains any
                                        function_call items
    stop_reason == "end_turn"          response.output contains no
                                        function_call items

The loop logic is the same as agent_manual.py:
    1. Send the conversation + tool definitions.
    2. If the model asked to call a function: run it, append the result,
       go back to step 1.
    3. If it didn't: we're done, print response.output_text.

Run it:  python agent_manual_openai.py
"""

import json

from openai import OpenAI

from tools import list_files, read_file, write_file

MODEL = "gpt-5.6"

TOOLS = [
    {"type": "web_search"},  # built-in server-side tool
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a text file from the local sandbox directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the file to read."}
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write text content to a file in the local sandbox directory, creating or overwriting it.",
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
    {
        "type": "function",
        "name": "list_files",
        "description": "List all files currently in the local sandbox directory.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
]

TOOL_FUNCTIONS = {
    "read_file": lambda **kw: read_file(kw["filename"]),
    "write_file": lambda **kw: write_file(kw["filename"], kw["content"]),
    "list_files": lambda **kw: list_files(),
}


def run_agent(task: str, max_turns: int = 10) -> str:
    client = OpenAI()
    input_items = [{"role": "user", "content": task}]

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ---")

        response = client.responses.create(
            model=MODEL,
            tools=TOOLS,
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
                print(f"Model is searching: {getattr(item.action, 'query', '')}")

        if not function_calls:
            return response.output_text

        for item in function_calls:
            args = json.loads(item.arguments)
            func = TOOL_FUNCTIONS.get(item.name)
            try:
                result = func(**args)
            except Exception as e:
                result = f"Error running {item.name}: {e}"
            print(f"  -> {result[:200]}")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                }
            )

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
