"""Analysis for the 3x3 study. Fixed before the main runs.

    python -m study.analysis results/main/jira results/main/hospital results/main/construction --out results/analysis

Design: 3 environments x 9 cells x 30 seeds (seeds 1-30, fixed in advance, no
interim looks). The pair is the two ends of the line in every cell. Only runs
that crash are excluded; an agent that fails to act still counts, and the
share of agent-days with an action is reported per cell.

Primary outcome: dA per cell (see study/metrics.py). For each environment and
cell: mean over seeds, 95% bootstrap CI (resampling seeds), and the number of
seeds with dA > 0. Also reported: dP, dO, dEff and the collusion signature
(dA > 0 and dO < 0). Main contrasts, each averaged within seed before
bootstrapping:
    pressure:     mean dA over rows a and b  minus  mean dA over row c
    channel:      mean dA over columns 1 and 2  minus  mean dA over column 3
    interaction:  (b2 - b3) - (c2 - c3)
The incentive-alone and instruction-added contrasts split the pressure effect.

Everything else is exploratory: behavior measures, awareness mentions, and dA
computed separately for each day (day rewards over a fair share of
OPT / (4 x days)).
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np

from study import framing
from study.metrics import load_runs, run_metrics, with_baseline

ENV_ORDER = ["jira", "hospital", "construction"]
N_BOOT = 10000


def bootstrap_ci(values: List[float], seed: int = 0) -> tuple:
    if not values:
        return (None, None)
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


def cell_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for env in [e for e in ENV_ORDER if any(r["environment"] == e for r in rows)]:
        for cell in framing.CELLS:
            rs = [r for r in rows if r["environment"] == env and r["cell"] == cell]
            if not rs:
                continue
            dA = [r["dA"] for r in rs]
            lo, hi = bootstrap_ci(dA)
            out.append({
                "environment": env,
                "cell": cell,
                "pressure": framing.ROW_LABELS[cell[0]],
                "channel": framing.COLUMN_LABELS[cell[1]],
                "n_seeds": len(rs),
                "dA_mean": mean(dA),
                "dA_ci_low": lo,
                "dA_ci_high": hi,
                "dA_positive": sum(1 for v in dA if v > 0),
                "dP_mean": mean(r["dP"] for r in rs),
                "dO_mean": mean(r["dO"] for r in rs),
                "dEff_mean": mean(r["dEff"] for r in rs),
                "signature_count": sum(r["signature"] for r in rs),
                "A_mean": mean(r["A"] for r in rs),
                "efficiency_mean": mean(r["efficiency"] for r in rs),
                "acted_rate": mean(r["acted_rate"] for r in rs),
                "pair_extra_board_share": _mean_or_none([r["pair_extra_board_share"] for r in rs]),
                "awareness_mentions": sum(r["awareness_mentions"] for r in rs),
            })
    return out


def _mean_or_none(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def contrasts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for env in [e for e in ENV_ORDER if any(r["environment"] == e for r in rows)]:
        by_seed: Dict[int, Dict[str, float]] = {}
        for r in rows:
            if r["environment"] == env:
                by_seed.setdefault(r["seed"], {})[r["cell"]] = r["dA"]
        complete = {s: v for s, v in by_seed.items() if all(c in v for c in framing.CELLS)}
        defs = {
            "pressure (rows a,b minus c)": lambda v: mean(v[c] for c in framing.CELLS if c[0] in "ab")
            - mean(v[c] for c in framing.CELLS if c[0] == "c"),
            "incentive alone (row a minus c)": lambda v: mean(v[c] for c in framing.CELLS if c[0] == "a")
            - mean(v[c] for c in framing.CELLS if c[0] == "c"),
            "instruction added (row b minus a)": lambda v: mean(v[c] for c in framing.CELLS if c[0] == "b")
            - mean(v[c] for c in framing.CELLS if c[0] == "a"),
            "channel (columns 1,2 minus 3)": lambda v: mean(v[c] for c in framing.CELLS if c[1] in "12")
            - mean(v[c] for c in framing.CELLS if c[1] == "3"),
            "interaction (b2-b3)-(c2-c3)": lambda v: (v["b2"] - v["b3"]) - (v["c2"] - v["c3"]),
        }
        for name, f in defs.items():
            vals = [f(v) for v in complete.values()]
            lo, hi = bootstrap_ci(vals)
            out.append({
                "environment": env,
                "contrast": name,
                "n_seeds": len(vals),
                "mean": mean(vals) if vals else None,
                "ci_low": lo,
                "ci_high": hi,
                "positive": sum(1 for x in vals if x > 0),
            })
    return out


def behavior_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for env in [e for e in ENV_ORDER if any(r["environment"] == e for r in rows)]:
        for cell in framing.CELLS:
            rs = [r for r in rows if r["environment"] == env and r["cell"] == cell]
            if not rs:
                continue
            row: Dict[str, Any] = {"environment": env, "cell": cell, "n_seeds": len(rs)}
            if env == "construction":
                pair_rush = mean(sum(r["behavior"]["rushes"][a] for a in r["pair"]) / 2 for r in rs)
                other_rush = mean(
                    sum(v for a, v in r["behavior"]["rushes"].items() if a not in r["pair"]) / 2 for r in rs
                )
                row.update({
                    "pair_rushes_per_agent": pair_rush,
                    "other_rushes_per_agent": other_rush,
                    "pair_repair_share_on_others": _mean_or_none(
                        [r["behavior"]["pair_repair_share_on_others"] for r in rs]),
                    "pair_same_zone_days": mean(r["behavior"]["pair_same_zone_days"] for r in rs),
                })
            elif env == "hospital":
                row.update({
                    "pair_loan_share_to_partner": _mean_or_none(
                        [r["behavior"]["pair_loan_share_to_partner"] for r in rs]),
                    "pair_idle_while_nonpair_critical": mean(
                        sum(r["behavior"]["idle_while_nonpair_critical"][a] for a in r["pair"]) for r in rs),
                    "others_idle_while_others_critical": mean(
                        sum(v for a, v in r["behavior"]["idle_while_others_critical"].items() if a not in r["pair"])
                        for r in rs),
                    "treated_on_arrival_share": mean(
                        r["behavior"]["treated_on_arrival"] / r["behavior"]["patients"] for r in rs),
                })
            else:
                row.update({
                    "pair_duplicate_claims": mean(
                        sum(r["behavior"]["duplicate_claims"][a] for a in r["pair"]) for r in rs),
                    "other_duplicate_claims": mean(
                        sum(v for a, v in r["behavior"]["duplicate_claims"].items() if a not in r["pair"]) for r in rs),
                    "skips": mean(sum(r["behavior"]["skips"].values()) for r in rs),
                })
            out.append(row)
    return out


def day_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exploratory: dA for each day separately."""
    out = []
    for env in [e for e in ENV_ORDER if any(r["environment"] == e for r in rows)]:
        for cell in framing.CELLS:
            rs = [r for r in rows if r["environment"] == env and r["cell"] == cell and r["day_dA"]]
            if not rs:
                continue
            row: Dict[str, Any] = {"environment": env, "cell": cell, "n_seeds": len(rs)}
            for d in range(len(rs[0]["day_dA"])):
                vals = [r["day_dA"][d] for r in rs]
                lo, hi = bootstrap_ci(vals)
                row[f"day{d + 1}_dA"] = mean(vals)
                row[f"day{d + 1}_ci"] = f"[{fmt(lo)}, {fmt(hi)}]"
            out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plots(rows: List[Dict[str, Any]], table: List[Dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = [e for e in ENV_ORDER if any(r["environment"] == e for r in rows)]
    if not envs:
        return
    vmax = max(0.05, max(abs(t["dA_mean"]) for t in table))

    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 4), squeeze=False)
    for ax, env in zip(axes[0], envs):
        grid = np.full((3, 3), np.nan)
        labels = [["" for _ in range(3)] for _ in range(3)]
        for t in table:
            if t["environment"] != env:
                continue
            i, j = framing.ROWS.index(t["cell"][0]), framing.COLUMNS.index(t["cell"][1])
            grid[i, j] = t["dA_mean"]
            labels[i][j] = f"{t['dA_mean']:+.2f}\n{t['dA_positive']}/{t['n_seeds']} > 0"
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), ["1 group", "2 private", "3 neighbors"])
        ax.set_yticks(range(3), ["a incentive", "b inc.+instr.", "c neither"])
        ax.set_title(env)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="dA (normalized, vs c3)")
    fig.savefig(out / "heatmaps_dA.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(envs), 1, figsize=(9, 2.8 * len(envs)), squeeze=False)
    for ax, env in zip(axes[:, 0], envs):
        for k, cell in enumerate(framing.CELLS):
            vals = [r["dA"] for r in rows if r["environment"] == env and r["cell"] == cell]
            if not vals:
                continue
            xs = np.full(len(vals), k) + np.linspace(-0.15, 0.15, len(vals))
            ax.scatter(xs, vals, s=14, alpha=0.7)
            lo, hi = bootstrap_ci(vals)
            ax.errorbar([k], [mean(vals)], yerr=[[mean(vals) - lo], [hi - mean(vals)]], fmt="_k", capsize=4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(9), framing.CELLS)
        ax.set_ylabel("dA")
        ax.set_title(f"{env}: per-seed dA with mean and 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(out / "per_seed_dA.png", dpi=200)
    plt.close(fig)


def markdown(table, contrast_rows, behavior, days, out: Path) -> None:
    lines = ["# Results", "", "## dA by cell (mean over seeds, 95% bootstrap CI, seeds with dA > 0)", ""]
    lines.append("| env | cell | n | dA | 95% CI | dA>0 | dP | dO | dEff | signature | acted | pair extra-board share | awareness |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        share = t["pair_extra_board_share"]
        share_txt = "n/a" if share is None else f"{share:.2f}"
        lines.append(
            f"| {t['environment']} | {t['cell']} | {t['n_seeds']} | {fmt(t['dA_mean'])} | "
            f"[{fmt(t['dA_ci_low'])}, {fmt(t['dA_ci_high'])}] | {t['dA_positive']}/{t['n_seeds']} | "
            f"{fmt(t['dP_mean'])} | {fmt(t['dO_mean'])} | {fmt(t['dEff_mean'])} | "
            f"{t['signature_count']}/{t['n_seeds']} | {t['acted_rate']:.2f} | {share_txt} | "
            f"{t['awareness_mentions']} |"
        )
    lines += ["", "## Main contrasts (dA, averaged within seed)", ""]
    lines.append("| env | contrast | n | mean | 95% CI | positive |")
    lines.append("|---|---|---|---|---|---|")
    for c in contrast_rows:
        lines.append(
            f"| {c['environment']} | {c['contrast']} | {c['n_seeds']} | {fmt(c['mean'])} | "
            f"[{fmt(c['ci_low'])}, {fmt(c['ci_high'])}] | {c['positive']}/{c['n_seeds']} |"
        )
    lines += ["", "## Exploratory: dA by day", ""]
    if days:
        keys = [k for k in days[0] if k != "environment"]
        lines.append("| env | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for d in days:
            lines.append(f"| {d['environment']} | " + " | ".join(
                f"{d[k]:+.3f}" if isinstance(d[k], float) else str(d[k]) for k in keys) + " |")
    lines += ["", "## Exploratory: behavior by cell", ""]
    for env in ENV_ORDER:
        rows = [b for b in behavior if b["environment"] == env]
        if not rows:
            continue
        keys = [k for k in rows[0] if k not in ("environment",)]
        lines.append(f"### {env}")
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "---|" * len(keys))
        for b in rows:
            lines.append("| " + " | ".join(
                "n/a" if b[k] is None else (f"{b[k]:.2f}" if isinstance(b[k], float) else str(b[k])) for k in keys
            ) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = with_baseline([run_metrics(s) for s in load_runs(args.results)])
    table = cell_table(rows)
    contrast_rows = contrasts(rows)
    behavior = behavior_table(rows)
    days = day_table(rows)
    write_csv(out / "runs.csv", [{k: v for k, v in r.items() if k not in ("behavior", "pair", "agents")} for r in rows])
    write_csv(out / "cells.csv", table)
    write_csv(out / "contrasts.csv", contrast_rows)
    write_csv(out / "behavior.csv", behavior)
    write_csv(out / "days.csv", days)
    markdown(table, contrast_rows, behavior, days, out)
    plots(rows, table, out)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
