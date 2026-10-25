"""Tiny stand-in for the vLLM chat endpoint, for testing the runner offline.

    PYTHONPATH=. external/collusion_hackathon/colosseum/.venv/bin/python tests/mock_server.py
"""

import json
import random
import re

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
rng = random.Random(0)


def tool_call(name, args):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": f"call_{rng.randint(0, 10**9)}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    tools = {t["function"]["name"] for t in body.get("tools") or []}
    messages = body["messages"]
    last = messages[-1]
    prompt = next(m["content"] for m in messages if m["role"] == "user")
    if last["role"] == "tool":
        message = {"role": "assistant", "content": "Done."}
    elif "post_message" in tools:
        boards = re.findall(r"^Blackboard (\d+) \(members", prompt, re.M)
        message = tool_call("post_message", {"message": "I plan to take the cheapest option.",
                                             "blackboard_id": int(rng.choice(boards))})
    elif "assign_task" in tools:
        ids = re.findall(r"^- ([A-Z]+-\d+(?:::\w+)?) \|", prompt, re.M)
        args = {"task_id": rng.choice(ids + ["skip"])}
        if "pace" in json.dumps(body["tools"]):
            args["pace"] = rng.choice(["standard", "rushed"])
        message = tool_call("assign_task", args)
    elif "submit_sharing_plan" in tools:
        others = [h for h in ("MRMC", "SBRH", "LGH", "HPMC") if h not in prompt.split("\n")[1]]
        loans = [{"to_hospital": rng.choice(others), "resource": rng.choice(["AMB", "CATH", "MRI", "ECMO"]), "count": 1}]
        message = tool_call("submit_sharing_plan", {"loans": loans if rng.random() < 0.6 else []})
    else:
        message = {"role": "assistant", "content": "ok"}
    return {"id": "mock", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 20, "total_tokens": len(prompt) // 4 + 20}}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
