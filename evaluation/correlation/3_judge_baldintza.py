#!/usr/bin/env python3
"""
Baldintza (criteria) pass rates per generator model.

gold.json maps generator model -> list of items {"id", "baldintzak"}.

  * Incorrect sentences (5 baldintzak):
      B1  flags the sentence as incorrect
      B2  names the grammar rule that is violated
      B3  explains why the sentence breaks that rule
      B4  gives a valid corrected sentence
      B5  flags no other, nonexistent errors
  * Correct sentences (2 baldintzak):
      B1  flags the sentence as correct
      B2  flags no nonexistent errors
  * All sentences: B1 (right judgment) and "no nonexistent errors"
    (B5 for incorrect, B2 for correct) pooled over both sets.

Writes to <out>/ (default: 3_baldintza_results/):
    01_pass_rate_per_baldintza.png   share of responses meeting each baldintza
    02_cumulative_pass.png           incorrect: B1, B1 & B5, all five · correct: B1, both
                                     · all: B1, B1 & B5/B2
Each graph has one panel per set: incorrect, correct, all.

Usage:
    python 3_judge_baldintza.py [--gold FILE] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

CRITERIA = {
    "incorrect": {
        "baldintza_1": "B1 · flags it\nas incorrect",
        "baldintza_2": "B2 · names the\nviolated rule",
        "baldintza_3": "B3 · explains why\nit breaks the rule",
        "baldintza_4": "B4 · gives a valid\ncorrection",
        "baldintza_5": "B5 · flags no\nnonexistent errors",
    },
    "correct": {
        "baldintza_1": "B1 · flags it\nas correct",
        "baldintza_2": "B2 · flags no\nnonexistent errors",
    },
    "all": {
        "baldintza_1": "B1 · right\njudgment",
        "no_spurious": "B5 / B2 · flags no\nnonexistent errors",
    },
}
SET_TITLE = {"incorrect": "Incorrect sentences (5 baldintzak)",
             "correct": "Correct sentences (2 baldintzak)",
             "all": "All sentences (B1, B5 / B2)"}

GEN_SHORT = {
    "Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8": "Latxa 70B",
    "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
NESTED = ["#9ec5f4", "#3987e5", "#104281"]  # light -> dark: B1 ⊇ B1 & B5 ⊇ all

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.edgecolor": "#c3c2b7",
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def load_gold(path: Path) -> pd.DataFrame:
    """One row per (generator, subset, id, criterion); partial credit (0.5) counts as
    not met. Subset "all" repeats B1 and B5 (incorrect) / B2 (correct) of every item."""
    rows = []
    for gen, items in json.loads(path.read_text()).items():
        for it in items:
            b = it["baldintzak"]
            subset = "incorrect" if len(b) == 5 else "correct"
            spurious = "baldintza_5" if subset == "incorrect" else "baldintza_2"
            for crit, v in b.items():
                row = {"generator": GEN_SHORT.get(gen, gen), "id": f"{subset}-{it['id']}",
                       "value": np.nan if v is None else float(float(v) >= 1)}
                rows.append({**row, "subset": subset, "criterion": crit})
                if crit in ("baldintza_1", spurious):
                    rows.append({**row, "subset": "all", "criterion":
                                 "baldintza_1" if crit == "baldintza_1" else "no_spurious"})
    return pd.DataFrame(rows)


def save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_pass_rates(df, gens, colors, out):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [5, 2.3, 2.3]})
    width = 0.8 / len(gens)
    for ax, subset in zip(axes, SET_TITLE):
        t = (df[df.subset == subset].groupby(["criterion", "generator"]).value.mean()
             .unstack()[gens])
        x = np.arange(len(t))
        for k, g in enumerate(gens):
            bars = ax.bar(x + (k - (len(gens) - 1) / 2) * width, t[g], width,
                          color=colors[g], edgecolor=SURFACE, linewidth=1.2, label=g)
            ax.bar_label(bars, [f"{v:.0%}" for v in t[g]], padding=2, fontsize=7, color=INK2)
        ax.set_xticks(x, [CRITERIA[subset][c] for c in t.index])
        ax.set_title(SET_TITLE[subset])
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y")
    axes[0].set_ylabel("Share of responses meeting the baldintza")
    axes[0].legend(ncol=len(gens), loc="upper left", bbox_to_anchor=(0, -0.15))
    fig.suptitle("Gold — pass rate per baldintza", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    save(fig, out / "01_pass_rate_per_baldintza.png")


def plot_cumulative(df, gens, out):
    """Incorrect: B1, B1 & B5, all five met. Correct: B1, both met. All: B1, B1 & B5/B2."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.9 * len(gens) + 1.8), sharey=True)
    y = np.arange(len(gens))
    for ax, subset in zip(axes, SET_TITLE):
        w = (df[df.subset == subset]
             .pivot_table(index=["generator", "id"], columns="criterion", values="value")
             .dropna() == 1)
        b1 = w["baldintza_1"]
        if subset == "incorrect":
            metrics = {"B1 · flags it as incorrect": (b1, NESTED[0]),
                       "B1 & B5 · + flags no nonexistent errors":
                           (b1 & w["baldintza_5"], NESTED[1]),
                       "All (B1–B5) · + names the rule (B2), explains why (B3),\n"
                       "   gives a valid correction (B4)": (w.all(axis=1), NESTED[2])}
        elif subset == "correct":
            metrics = {"B1 · flags it as correct": (b1, NESTED[0]),
                       "All (B1 & B2) · + flags no nonexistent errors":
                           (w.all(axis=1), NESTED[2])}
        else:
            metrics = {"B1 · right judgment": (b1, NESTED[0]),
                       "B1 & B5 / B2 · + flags no nonexistent errors":
                           (b1 & w["no_spurious"], NESTED[1])}
        h = 0.8 / len(metrics)
        for k, (m, (met, color)) in enumerate(metrics.items()):
            vals = met.groupby(level="generator").mean().reindex(gens)
            bars = ax.barh(y + (k - (len(metrics) - 1) / 2) * h, vals, h, color=color,
                           edgecolor=SURFACE, linewidth=1.2, label=m)
            ax.bar_label(bars, [f"{v:.0%}" for v in vals], padding=3, fontsize=8, color=INK2)
        ax.set_yticks(y, gens)
        ax.set_xlim(0, 1.08)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
        ax.set_title(SET_TITLE[subset])
    axes[0].invert_yaxis()
    fig.suptitle("Gold — combined pass rate",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    save(fig, out / "02_cumulative_pass.png")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=here / "gold.json")
    ap.add_argument("--out", type=Path, default=here / "3_baldintza_results")
    args = ap.parse_args()

    df = load_gold(args.gold)
    gens = [g for g in GEN_SHORT.values() if g in set(df.generator)]
    gens += sorted(set(df.generator) - set(gens))
    colors = dict(zip(gens, SERIES))
    args.out.mkdir(parents=True, exist_ok=True)

    plot_pass_rates(df, gens, colors, args.out)
    plot_cumulative(df, gens, args.out)


if __name__ == "__main__":
    main()
