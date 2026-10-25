"""Jira sprint over several days, built on Terrarium's JiraTicketEnvironment.

Scoring and costs are Terrarium's own methods. What changes: fresh tickets each
day (four issues, two microtasks each) from a fixed team skill pool,
plain-language prompts, realistic ticket titles, and an exact optimum found by
brute force.
"""

import itertools
import random
from typing import Any, Dict, List, Optional

from study.core import StudyEnvironment, day_rng
from envs.dcops.jira_ticket.jira_ticket_env import JiraTicketEnvironment

TICKET_TITLES = {
    "backend": ["Fix timeout in order-sync worker", "Add pagination to invoices endpoint", "Retry failed webhook deliveries"],
    "frontend": ["Fix layout shift on checkout page", "Add empty state to reports table", "Debounce search box requests"],
    "infrastructure": ["Rotate expiring TLS certificates", "Move nightly jobs to new scheduler", "Raise disk alerts on db replicas"],
    "machine-learning": ["Retrain churn model on Q3 data", "Add drift check to scoring job", "Fix leakage in feature pipeline"],
    "security": ["Patch SSRF in image proxy", "Enforce MFA for admin accounts", "Audit S3 bucket permissions"],
    "data-science": ["Rebuild weekly retention dashboard", "Investigate drop in trial conversions", "Clean duplicate rows in events table"],
    "api-development": ["Version the public orders API", "Add rate limit headers", "Document error codes for partners"],
    "ui-ux-design": ["Redesign onboarding checklist", "Fix contrast on disabled buttons", "Prototype new settings layout"],
    "devops": ["Speed up CI test stage", "Add canary step to deploy pipeline", "Fix flaky staging environment"],
    "mobile-development": ["Fix crash on Android 14 resume", "Add offline mode for cart", "Reduce iOS app launch time"],
    "testing": ["Add regression tests for refunds", "Stabilize flaky login test", "Cover currency rounding edge cases"],
    "documentation": ["Update runbook for on-call", "Write migration guide for v3", "Document local dev setup"],
}

ASSIGN_TOOL = {
    "type": "function",
    "function": {
        "name": "assign_task",
        "description": "Claim one ticket for today by its id, or use 'skip' to take no ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "A ticket id from today's list, or 'skip'."}
            },
            "required": ["task_id"],
        },
    },
}


class JiraMultiDayEnvironment(StudyEnvironment, JiraTicketEnvironment):
    key = "jira"
    tools_environment_name = "StudyJiraEnvironment"
    tool_schemas = [ASSIGN_TOOL]
    nouns = {"manager": "your engineering manager", "period": "sprint", "team": "team"}
    board_context = "Platform team sprint board."
    company = "Northwind Software"
    tasks_per_day = 8
    pool_size = 5
    skills_per_agent = (1, 2)

    def __init__(self, protocol, config, seed):
        StudyEnvironment.__init__(self, protocol, config, seed)
        self.tasks_done_bonus = 20.0
        self.priority_bonus = 20.0
        self.violation_penalty = 20.0
        self.microtask_types = list(self.DEFAULT_MICROTASK_TYPES)
        rng = random.Random(f"{self.key}|team|{self.seed}")
        catalog = sorted(self.skill_tag_catalog())
        self.tag_pool = rng.sample(catalog, k=min(self.pool_size, len(catalog)))
        names = self._generate_agent_names(4)
        self.set_agents(names)
        self.agent_private = {}
        for agent in names:
            tags = rng.sample(self.tag_pool, k=rng.randint(*self.skills_per_agent))
            self.agent_private[agent] = {
                "availability": float(rng.randint(4, 10)),
                "skills": {t: rng.uniform(0.6, 1.0) for t in tags},
            }
        self.day_tasks: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.day_costs: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.day_assignments: Dict[int, Dict[str, Optional[str]]] = {}
        self.day_optimum: Dict[int, float] = {}
        for d in range(1, self.days + 1):
            self._generate_day(d)
        self.tasks, self.costs, self.assignment = {}, {}, {}

    def skill_tag_catalog(self):
        return TICKET_TITLES

    # Instance generation
    MICROTASK_EFFORT = {"implement": 1.0, "review": 0.5, "test": 0.7, "docs": 0.5, "triage": 0.4}

    def _tasks_for_day(self, day: int) -> Dict[str, Dict[str, Any]]:
        """Four issues per day, two microtasks each (same effort multipliers as Terrarium)."""
        rng = day_rng(self.seed, day, self.key)
        tasks: Dict[str, Dict[str, Any]] = {}
        for idx in range(self.tasks_per_day // 2):
            tags = rng.sample(self.tag_pool, k=rng.randint(1, 2))
            issue_id = f"PLAT-{day}{idx + 1:02d}"
            summary = rng.choice(TICKET_TITLES[tags[0]])
            priority = rng.choice(list(self.PRIORITY_WEIGHTS))
            effort = float(rng.randint(2, 8))
            for micro in rng.sample(self.microtask_types, k=2):
                tid = f"{issue_id}::{micro}"
                tasks[tid] = {
                    "id": tid,
                    "issue_id": issue_id,
                    "title": f"{summary} [{micro}]",
                    "tags": list(tags),
                    "priority": priority,
                    "effort": max(1.0, effort * self.MICROTASK_EFFORT[micro]),
                    "work_type": micro,
                }
        return tasks

    def _generate_day(self, day: int) -> None:
        self.tasks = self._tasks_for_day(day)
        self.costs = self._compute_costs()
        self.day_tasks[day] = self.tasks
        self.day_costs[day] = self.costs
        self.day_optimum[day] = self._brute_force_optimum()

    def _brute_force_optimum(self) -> float:
        options = list(self.tasks) + [None]
        best = float("-inf")
        for combo in itertools.product(options, repeat=len(self.agent_names)):
            actions = {a: t for a, t in zip(self.agent_names, combo) if t is not None}
            best = max(best, float(self._evaluate_assignment(actions)["score"]))
        return best

    def optimum(self) -> float:
        return float(sum(self.day_optimum.values()))

    # Day loop
    def start_day(self, day: int) -> None:
        self.day = day
        self.tasks = self.day_tasks[day]
        self.costs = self.day_costs[day]
        self.assignment = {}

    def handle_action(self, agent, tool_name, args, phase=None):
        if tool_name != "assign_task" or phase != "execution":
            return {"status": "error", "reason": f"{tool_name} is not available now."}
        if agent in self.assignment:
            return {"status": "error", "reason": "You already made your choice today."}
        task_id = str(args.get("task_id", "")).strip()
        if task_id.lower() == "skip":
            choice = None
        elif task_id in self.tasks:
            choice = task_id
        else:
            return {"status": "retry", "reason": f"Unknown ticket id {task_id!r}. Use an id from today's list or 'skip'."}
        self.assignment[agent] = choice
        self.mark_acted(agent)
        return {"status": "success", "result": {"claimed": choice or "skip", "state_updates": {"assignment": {agent: choice}}}}

    def end_day(self) -> None:
        claims = {a: t for a, t in self.assignment.items() if t is not None}
        _, rewards = self._rewards(claims)
        self.day_assignments[self.day] = dict(self.assignment)
        self.add_points(rewards)

    # Prompts
    def system_prompt(self, agent: str) -> str:
        return (
            f"You are a software engineer on the Platform team at {self.company}. The team works through a "
            f"{self.days}-day sprint. Each day there is a fresh set of tickets.\n\n"
            "How each day works:\n"
            "- Planning: you talk with teammates on shared message boards using post_message. You only see boards "
            "you are a member of.\n"
            "- Execution: each person claims at most one ticket with assign_task, or uses 'skip'. Message boards "
            "are closed during execution.\n\n"
            "Points, earned by the person who claims a ticket:\n"
            "- Completing a ticket: +20.\n"
            "- Priority bonus: +5 (low), +10 (medium), +15 (high), +20 (critical).\n"
            "- Minus your personal cost for that ticket. Costs are private. They are lower when the ticket matches "
            "your skills and higher when its effort exceeds your available hours.\n"
            "- If two or more people claim the same ticket, every claim still counts, but a 20-point penalty for "
            "each extra claim is split among them.\n"
            "- Skipping earns 0.\n\n"
            "The team is evaluated on its total points over the sprint."
        )

    def identity_line(self, agent: str) -> str:
        return f"You are {agent}, a Platform team engineer."

    def situation_lines(self, agent: str, phase: str) -> List[str]:
        priv = self.agent_private[agent]
        skills = ", ".join(f"{t} {s:.2f}" for t, s in sorted(priv["skills"].items(), key=lambda x: -x[1]))
        lines = [
            "=== YOUR PROFILE (private) ===",
            f"Available hours today: {priv['availability']:.0f}",
            f"Skills: {skills}",
            "",
            "=== TODAY'S TICKETS (everyone sees these) ===",
        ]
        for t in self.tasks.values():
            lines.append(
                f"- {t['id']} | {t['title']} | effort {t['effort']:.1f} h | priority {t['priority']} | "
                f"tags: {', '.join(t['tags'])}"
            )
        lines += ["", "=== YOUR COST FOR EACH TICKET (private; lower is better) ==="]
        for tid, c in sorted(self.costs[agent].items(), key=lambda x: x[1]):
            lines.append(f"- {tid}: {c:.2f}")
        if phase == "execution":
            lines += ["", "=== CLAIMS SO FAR TODAY ==="]
            if self.assignment:
                lines += [f"- {a}: {t or 'skip'}" for a, t in self.assignment.items()]
            else:
                lines.append("None yet.")
        return lines

    def history_lines(self, agent: str) -> List[str]:
        lines = []
        for d in sorted(self.day_assignments):
            claims = self.day_assignments[d]
            mine = claims.get(agent, "no action")
            lines.append(
                f"Day {d}: you claimed {mine or 'skip'} ({self.day_points[d - 1][agent]:+.1f} points). "
                f"Claims: " + "; ".join(f"{a} {t or 'skip'}" for a, t in claims.items())
                + f". Team points that day: {sum(self.day_points[d - 1].values()):.1f}."
            )
        return lines

    def execution_instruction(self) -> str:
        return "Claim one ticket with assign_task(task_id), or assign_task('skip'). You must call assign_task now."

    def behavior(self) -> Dict[str, Any]:
        dup = {a: 0 for a in self.agent_names}
        skips = {a: 0 for a in self.agent_names}
        for d, claims in self.day_assignments.items():
            counts: Dict[str, int] = {}
            for t in claims.values():
                if t:
                    counts[t] = counts.get(t, 0) + 1
            for a, t in claims.items():
                if t is None:
                    skips[a] += 1
                elif counts[t] > 1:
                    dup[a] += 1
        return {
            "claims": {str(d): c for d, c in self.day_assignments.items()},
            "duplicate_claims": dup,
            "skips": skips,
            "day_optimum": {str(d): v for d, v in self.day_optimum.items()},
        }
