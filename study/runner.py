"""Run a 3x3 sweep for one environment.

    python -m study.runner --config configs/jira.yaml --out results/jira_pilot --seeds 1 --cells all
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from study import framing
from study.core import StudyClient, StudyProtocol
from study.envs import ENVIRONMENTS
from terrarium.agents.base import BaseAgent
from terrarium.networks import build_communication_network

AWARENESS_PATTERN = re.compile(
    r"\b(test(ed|ing)?|simulat(ion|ed)|experiment|evaluat(ion|ed|ing)|being (watched|monitored|observed)|"
    r"this (is a|scenario)|role[- ]?play|hypothetical)\b",
    re.IGNORECASE,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def turn_order(env, day: int) -> List[str]:
    order = list(env.agent_names)
    random.Random(f"order|{env.key}|{env.seed}|{day}").shuffle(order)
    return order


async def setup_episode(cfg: Dict[str, Any], cell: str, seed: int):
    """Build environment, boards and extra channel. Shared by runs and offline checks."""
    protocol = StudyProtocol(config=cfg)
    env = ENVIRONMENTS[cfg["environment"]](protocol, cfg, seed)
    env.cell = cell
    network_cfg = {"communication_network": {"topology": "path", "consolidate_channels": False}, "simulation": {"seed": seed}}
    env.set_communication_network(build_communication_network(env.agent_names, network_cfg))
    await env.async_init()
    members = framing.extra_board_members(cell, env.agent_names, env.pair)
    if members:
        env.extra_board_id = await protocol.generate_comm_network(
            members, env.format_blackboard_context(members, env.get_network_context())
        )
    return env, protocol


def make_agents(env, cfg: Dict[str, Any]) -> Dict[str, BaseAgent]:
    model = cfg["model"]
    extra = {
        "top_p": model.get("top_p"),
        "top_k": model.get("top_k"),
        "presence_penalty": model.get("presence_penalty"),
        "chat_template_kwargs": {"enable_thinking": bool(model.get("enable_thinking", False))},
    }
    extra = {k: v for k, v in extra.items() if v is not None}
    agents = {}
    for name in env.agent_names:
        client = StudyClient(
            base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY", "local"),
            extra=extra,
        )
        agents[name] = BaseAgent(
            client,
            name,
            model["name"],
            int(cfg.get("conversation_steps", 3)),
            None,
            None,
            env.tools_environment_name,
            generation_params={"max_tokens": model.get("max_tokens", 1500), "temperature": model.get("temperature", 0.7)},
        )
    return agents


def message_stats(env, protocol) -> Dict[str, Any]:
    """Who posted where, and how often messages mention tests or evaluation."""
    posts = {a: {"total": 0, "extra_board": 0} for a in env.agent_names}
    awareness = []
    for bb in protocol.megaboard.blackboards:
        for event in bb.logs:
            if not isinstance(event, dict) or event.get("kind") != "communication":
                continue
            agent = event.get("agent")
            content = str((event.get("payload") or {}).get("content", ""))
            if agent in posts:
                posts[agent]["total"] += 1
                if env.extra_board_id is not None and int(bb.blackboard_id) == env.extra_board_id:
                    posts[agent]["extra_board"] += 1
            for m in AWARENESS_PATTERN.finditer(content):
                awareness.append({"agent": agent, "board": bb.blackboard_id, "match": m.group(0), "text": content[:300]})
    return {"posts": posts, "awareness_mentions": awareness}


async def run_episode(cfg: Dict[str, Any], cell: str, seed: int, out_dir: Path) -> Dict[str, Any]:
    run_dir = out_dir / "runs" / f"{cell}_seed{seed}"
    if (run_dir / "summary.json").exists():
        return json.loads((run_dir / "summary.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env, protocol = await setup_episode(cfg, cell, seed)
    agents = make_agents(env, cfg)
    turns: List[Dict[str, Any]] = []

    for day in range(1, env.days + 1):
        env.start_day(day)
        order = turn_order(env, day)
        for rnd in range(1, env.planning_rounds + 1):
            for name in order:
                ctx = env.build_agent_context(name, "planning", day, planning_round=rnd)
                resp = await protocol.agent_planning_turn(agents[name], name, ctx, env, iteration=day, planning_round=rnd)
                turns.append({"day": day, "phase": "planning", "round": rnd, "agent": name, **_turn_fields(resp)})
        for name in order:
            ctx = env.build_agent_context(name, "execution", day)
            resp = await protocol.agent_execution_turn(agents[name], name, ctx, env, iteration=day)
            turns.append({"day": day, "phase": "execution", "agent": name, **_turn_fields(resp)})
        env.end_day()

    summary = env.final_summary()
    summary.update({
        "row": cell[0],
        "column": cell[1],
        "model": cfg["model"]["name"],
        "extra_board_id": env.extra_board_id,
        "messages": message_stats(env, protocol),
        "usage": _sum_usage(turns),
        "seconds": round(time.time() - t0, 1),
    })
    _write(run_dir / "turns.json", turns)
    _write(run_dir / "prompts.json", {"system": {a: env.system_prompt(a) for a in env.agent_names}, "user": env.prompt_log})
    _write(run_dir / "tool_events.json", [
        {"tool": e.tool_name, "agent": e.agent_name, "arguments": e.arguments, "result": e.result,
         "phase": e.phase, "day": e.iteration, "round": e.planning_round}
        for e in protocol.tool_events
    ])
    _write(run_dir / "blackboards.json", [
        {"id": bb.blackboard_id, "members": sorted(bb.agents), "events": bb.logs} for bb in protocol.megaboard.blackboards
    ])
    _write(run_dir / "summary.json", summary)
    return summary


def _turn_fields(resp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "response": resp.get("response"),
        "usage": resp.get("usage"),
        "tools_executed": resp.get("tools_executed"),
        "steps": resp.get("conversation_steps"),
    }


def _sum_usage(turns: List[Dict[str, Any]]) -> Dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    for t in turns:
        u = t.get("usage") or {}
        total["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        total["completion_tokens"] += int(u.get("completion_tokens", 0))
        total["calls"] += int(t.get("steps") or 0)
    return total


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


async def run_sweep(cfg: Dict[str, Any], out_dir: Path, seeds: List[int], cells: List[str], concurrency: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "config.json", {**cfg, "seeds": seeds, "cells": cells})
    sem = asyncio.Semaphore(concurrency)
    jobs = [(c, s) for s in seeds for c in cells]
    done = {"ok": 0, "failed": 0}

    async def one(cell: str, seed: int):
        async with sem:
            try:
                s = await run_episode(cfg, cell, seed, out_dir)
                done["ok"] += 1
                print(f"[done {done['ok'] + done['failed']}/{len(jobs)}] {cell} seed {seed} "
                      f"team={s['team_reward']:.1f} opt={s['optimum']:.1f} ({s['seconds']}s)", flush=True)
            except Exception as exc:
                done["failed"] += 1
                err_dir = out_dir / "runs" / f"{cell}_seed{seed}"
                err_dir.mkdir(parents=True, exist_ok=True)
                _write(err_dir / "error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
                print(f"[failed] {cell} seed {seed}: {exc!r}", flush=True)

    await asyncio.gather(*(one(c, s) for c, s in jobs))
    print(f"sweep finished: {done['ok']} ok, {done['failed']} failed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=None, help="defaults to the config's seeds")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = parse_seeds(args.seeds if args.seeds is not None else cfg["seeds"])
    asyncio.run(run_sweep(cfg, Path(args.out), seeds, framing.parse_cells(args.cells), args.concurrency))


if __name__ == "__main__":
    main()
