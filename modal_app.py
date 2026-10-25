"""Run the collusion study on Modal.

One H100 container serves Qwen3.8-27B-FP8 with vLLM on localhost and runs the
agents against it, so the model endpoint is never exposed.

    modal run modal_app.py --action smoke
    modal run --detach modal_app.py --action sweep --config jira.yaml,hospital.yaml --seeds 1-5 --tag main
    modal volume get collusion-results main ./results/main
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.8-27B-FP8"
VLLM_PORT = 8000
REPO = Path(__file__).resolve().parent
REMOTE = "/work"
COLOSSEUM = f"{REMOTE}/external/collusion_hackathon/colosseum"
PYTHON = f"{COLOSSEUM}/.venv/bin/python"

image = (
    modal.Image.from_registry("vllm/vllm-openai:qwen38")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python", "python --version", "pip install uv")
    .add_local_dir(
        REPO / "external" / "collusion_hackathon",
        f"{REMOTE}/external/collusion_hackathon",
        copy=True,
        ignore=["**/.git", ".git", "**/.venv", "**/__pycache__", "**/outputs", "compute.env"],
    )
    .run_commands(f"cd {COLOSSEUM} && uv sync --frozen")
    .add_local_dir(REPO / "study", f"{REMOTE}/study", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "configs", f"{REMOTE}/configs")
)

app = modal.App("collusion-study")
results = modal.Volume.from_name("collusion-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
VOLUMES = {"/results": results, "/root/.cache/huggingface": hf_cache}


def start_vllm(timeout_s: int = 1500) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", MODEL,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--max-model-len", "32768",
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.90",
        "--language-model-only",
        "--enable-prefix-caching",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    proc = subprocess.Popen(cmd)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=5):
                return proc
        except Exception:
            time.sleep(5)
    proc.terminate()
    raise TimeoutError("vLLM did not become ready")


def chat(payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


@app.function(image=image, gpu="H100", timeout=3600, volumes=VOLUMES)
def smoke():
    proc = start_vllm()
    tool = {
        "type": "function",
        "function": {
            "name": "assign_task",
            "description": "Claim a ticket by id, or 'skip'.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You assign yourself one ticket. Always use the tool."},
            {"role": "user", "content": "Tickets: PLAT-101 (cost 3), PLAT-102 (cost 9). Claim the cheaper one."},
        ],
        "tools": [tool],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    out = chat(payload)
    msg = out["choices"][0]["message"]
    print("seconds:", round(time.time() - t0, 2))
    print("content:", msg.get("content"))
    print("reasoning:", (msg.get("reasoning_content") or msg.get("reasoning") or "")[:300])
    print("tool_calls:", json.dumps(msg.get("tool_calls"), indent=1))
    print("usage:", out.get("usage"))
    hf_cache.commit()
    proc.terminate()


@app.function(image=image, gpu="H100", timeout=14 * 3600, volumes=VOLUMES)
def sweep(configs: str, seeds: str, cells: str, tag: str, concurrency: int):
    """Run one sweep per config in parallel against the same vLLM server.
    Results go to /results/<tag>/<config name>."""
    proc = start_vllm()
    hf_cache.commit()
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": REMOTE,
        "LLM_BASE_URL": f"http://127.0.0.1:{VLLM_PORT}/v1",
        "LLM_API_KEY": "local",
    })
    runs = []
    for config in [c.strip() for c in configs.split(",") if c.strip()]:
        cmd = [
            PYTHON, "-m", "study.runner",
            "--config", f"{REMOTE}/configs/{config}",
            "--out", f"/results/{tag}/{Path(config).stem}",
            "--seeds", seeds,
            "--cells", cells,
            "--concurrency", str(concurrency),
        ]
        runs.append(subprocess.Popen(cmd, cwd=REMOTE, env=env))
    while any(r.poll() is None for r in runs):
        time.sleep(300)
        results.commit()
    results.commit()
    proc.terminate()
    codes = [r.returncode for r in runs]
    if any(codes):
        raise SystemExit(f"runner exit codes: {codes}")


@app.local_entrypoint()
def main(
    action: str = "smoke",
    config: str = "jira.yaml",
    seeds: str = "1",
    cells: str = "all",
    tag: str = "pilot",
    concurrency: int = 12,
):
    if action == "smoke":
        smoke.remote()
    elif action == "sweep":
        sweep.remote(config, seeds, cells, tag, concurrency)
    else:
        raise SystemExit(f"unknown action {action!r}")
