#!/usr/bin/env python3
"""
Combined baldintza pass rates (same metrics and graph as correlation/3_judge_baldintza.py,
02_cumulative_pass.png) for every judged results file, next to a hardcoded Latxa 70B
baseline (human gold).

  * Incorrect sentences: B1 · B1 & B5 · all five (B1–B5)
  * Correct sentences:   B1 · B1 & B2
  * All sentences:       B1 · B1 & B5 / B2 (no nonexistent errors)

Partial credit (0.5) counts as not met; an item with an unannotated (null)
baldintza in a set is left out of that set.

Writes <out>/cumulative_pass.png and prints the same values as a table.

Usage:
    python3 04_compute_metrics.py [-i ../output/results] [-o ../output/metrics]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

BASELINE_LABEL = "Latxa 70B (human gold)"
# Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8, computed from
# results/emaitzak/human_anotazioak_2026-10-18_1635.json (item 41 has B5 null)
BASELINE = {
    BASELINE_LABEL: {
        "incorrect": {"n": 59, "B1": 39 / 59, "B1 & B5": 18 / 59, "B1-B5": 2 / 59},
        "correct": {"n": 60, "B1": 42 / 60, "B1 & B2": 30 / 60},
        "all": {"n": 119, "B1": 81 / 119, "B1 & B5/B2": 48 / 119},
    },
}

SET_TITLE = {"incorrect": "Incorrect sentences (5 baldintzak)",
             "correct": "Correct sentences (2 baldintzak)",
             "all": "All sentences (B1, B5 / B2)"}
NESTED = ["#9ec5f4", "#3987e5", "#104281"]  # light -> dark: B1 ⊇ B1 & B5 ⊇ all
# metric key -> (legend label, color), per set
METRICS = {
    "incorrect": {"B1": ("B1 · flags it as incorrect", NESTED[0]),
                  "B1 & B5": ("B1 & B5 · + flags no nonexistent errors", NESTED[1]),
                  "B1-B5": ("All (B1–B5) · + names the rule (B2), explains why (B3),\n"
                            "   gives a valid correction (B4)", NESTED[2])},
    "correct": {"B1": ("B1 · flags it as correct", NESTED[0]),
                "B1 & B2": ("All (B1 & B2) · + flags no nonexistent errors", NESTED[2])},
    "all": {"B1": ("B1 · right judgment", NESTED[0]),
            "B1 & B5/B2": ("B1 & B5 / B2 · + flags no nonexistent errors", NESTED[1])},
}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.edgecolor": "#c3c2b7",
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def met(v):
    return float(v) >= 1


def rate(items, cond):
    return sum(map(cond, items)) / len(items) if items else float("nan")


def compute(records):
    """records: list of baldintzak dicts (5 keys = incorrect sentence, 2 = correct)."""
    inc = [b for b in records if len(b) == 5]
    cor = [b for b in records if len(b) == 2]
    # "all" pools B1 and the no-nonexistent-errors baldintza (B5 / B2)
    pooled = [(b["baldintza_1"], b["baldintza_5" if len(b) == 5 else "baldintza_2"])
              for b in inc + cor]

    inc = [b for b in inc if None not in b.values()]
    cor = [b for b in cor if None not in b.values()]
    pooled = [p for p in pooled if None not in p]
    return {
        "incorrect": {"n": len(inc),
                      "B1": rate(inc, lambda b: met(b["baldintza_1"])),
                      "B1 & B5": rate(inc, lambda b: met(b["baldintza_1"]) and met(b["baldintza_5"])),
                      "B1-B5": rate(inc, lambda b: all(map(met, b.values())))},
        "correct": {"n": len(cor),
                    "B1": rate(cor, lambda b: met(b["baldintza_1"])),
                    "B1 & B2": rate(cor, lambda b: all(map(met, b.values())))},
        "all": {"n": len(pooled),
                "B1": rate(pooled, lambda p: met(p[0])),
                "B1 & B5/B2": rate(pooled, lambda p: met(p[0]) and met(p[1]))},
    }


def load(path):
    """Results file -> (infer model, judge model, baldintzak list).
    Records without a judge result are skipped."""
    recs = json.loads(path.read_text(encoding="utf-8"))
    infer = recs[0].get("infer", {}).get("infer_model", path.stem)
    judge = recs[0].get("judge", {}).get("judge_model", "?")
    baldintzak = [(r.get("result") or {}).get("baldintzak") for r in recs]
    missing = sum(b is None for b in baldintzak)
    if missing:
        print(f"{path.name}: {missing} records without baldintzak skipped")
    return infer, judge, [b for b in baldintzak if b]


def print_table(metrics):
    cols = [(s, k) for s in METRICS for k in METRICS[s]]
    w = max(map(len, metrics)) + 2
    print("".ljust(w) + "".join(f"{s[:3]}:{k}".rjust(15) for s, k in cols))
    for label, m in metrics.items():
        print(label.ljust(w) + "".join(f"{m[s][k]:15.1%}" for s, k in cols))


def plot_cumulative(metrics, title, out):
    """One panel per set; one group of nested-metric bars per model, values on the bars.
    The baseline row is hatched and separated from the models by a dashed line."""
    labels = list(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.9 * len(labels) + 1.8), sharey=True)
    y = np.arange(len(labels))
    for ax, subset in zip(axes, SET_TITLE):
        h = 0.8 / len(METRICS[subset])
        for k, (key, (name, color)) in enumerate(METRICS[subset].items()):
            vals = [metrics[l][subset][key] for l in labels]
            bars = ax.barh(y + (k - (len(METRICS[subset]) - 1) / 2) * h, vals, h, color=color,
                           edgecolor=SURFACE, linewidth=1.2, label=name)
            for bar, l in zip(bars, labels):
                if l == BASELINE_LABEL:
                    bar.set_hatch("///")
                    bar.set_edgecolor(INK2)
            ax.bar_label(bars, [f"{v:.1%}" for v in vals], padding=3, fontsize=8, color=INK2)
        if BASELINE_LABEL in labels:
            ax.axhline(labels.index(BASELINE_LABEL) + 0.5, color=INK2, linewidth=0.8, linestyle="--")
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1.1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x")
        ax.legend(handles=[Patch(color=c, label=n) for n, c in METRICS[subset].values()],
                  loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
        ax.set_title(SET_TITLE[subset])
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    path = out / "cumulative_pass.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input-dir", type=Path, default=here.parent / "output" / "results")
    ap.add_argument("-o", "--out", type=Path, default=here.parent / "output" / "metrics")
    args = ap.parse_args()

    runs = [load(p) for p in sorted(args.input_dir.glob("*.json"))]
    judges = sorted({judge for _, judge, _ in runs})
    metrics = dict(BASELINE)
    for infer, judge, records in runs:
        # Judge name only needed in the label when several judges are mixed
        metrics[infer if len(judges) == 1 else f"{infer} [{judge}]"] = compute(records)

    title = "Combined pass rate"
    if len(judges) == 1:
        title += f" — judge: {judges[0]} (baseline: human gold)"

    print_table(metrics)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_cumulative(metrics, title, args.out)


if __name__ == "__main__":
    main()
