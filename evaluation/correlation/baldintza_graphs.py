#!/usr/bin/env python3
"""
Baldintza (criteria) analysis graphs.

gold.json maps generator model -> list of items {"id", "baldintzak"}.

  * Incorrect sentences (5 baldintzak):
      B1 says the sentence is incorrect     B2 identifies the broken rule
      B3 explains the rule/sentence conflict B4 gives a valid correction
      B5 does not flag any other (spurious) error
  * Correct sentences (2 baldintzak):
      B1 says the sentence is correct        B2 does not flag any (spurious) error

Graphs are written to <out>/ (default: 3_baldintza_results/).

Usage:
    python baldintza_graphs.py [--gold FILE] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Labels & style
# ----------------------------------------------------------------------------
CRITERIA = {
    "incorrect": {
        "baldintza_1": "B1 · says incorrect",
        "baldintza_2": "B2 · identifies rule",
        "baldintza_3": "B3 · explains conflict",
        "baldintza_4": "B4 · gives correction",
        "baldintza_5": "B5 · no spurious errors",
    },
    "correct": {
        "baldintza_1": "B1 · says correct",
        "baldintza_2": "B2 · no spurious errors",
    },
}
SET_TITLE = {"incorrect": "Incorrect sentences (5 baldintzak)",
             "correct": "Correct sentences (2 baldintzak)"}

GEN_SHORT = {
    "Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8": "Latxa 70B",
    "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
}

# Categorical slots in fixed order (one per generator, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])
ORDINAL = ["#e1e0d9", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def clean_value(v, where):
    """Binary 0/1; strings are cast, partial credit (0.5) counts as not met
    (the judging rubric says partially met = 0); None stays missing."""
    if v is None:
        print(f"  [warn] missing value at {where} -> excluded")
        return np.nan
    v = float(v)
    if v not in (0.0, 1.0):
        print(f"  [warn] non-binary value {v} at {where} -> {int(v >= 1)}")
        v = float(v >= 1)
    return v


def load_gold(path):
    data = json.loads(path.read_text())
    rows = []
    for gen, items in data.items():
        for it in items:
            b = it["baldintzak"]
            subset = "incorrect" if len(b) == 5 else "correct"
            for crit, v in b.items():
                rows.append({
                    "generator": GEN_SHORT.get(gen, gen), "id": int(it["id"]),
                    "subset": subset, "criterion": crit,
                    "value": clean_value(v, f"{path.name}:{gen}:{it['id']}:{crit}"),
                })
    return pd.DataFrame(rows)


def gen_order(df):
    known = [g for g in GEN_SHORT.values() if g in set(df.generator)]
    return known + sorted(set(df.generator) - set(known))


def wide(df, subset):
    """item x criterion matrix per generator (index: generator, id)."""
    d = df[df.subset == subset]
    return d.pivot_table(index=["generator", "id"], columns="criterion",
                         values="value", aggfunc="first", dropna=False)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pct_axis(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def annotate_heatmap(ax, mat, fmt="{:.0%}", vmin=0, vmax=1):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            dark = (v - vmin) / (vmax - vmin) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK)


def grouped_bars(ax, table, gens, colors, label_vals=True):
    """table: index = categories (x), columns = generators."""
    n = len(gens)
    width = 0.8 / n
    x = np.arange(len(table.index))
    for k, g in enumerate(gens):
        vals = table[g].values
        bars = ax.bar(x + (k - (n - 1) / 2) * width, vals, width,
                      color=colors[g], edgecolor=SURFACE, linewidth=1.2, label=g)
        if label_vals:
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}",
                            ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=0)
    ax.grid(axis="x", visible=False)


# ----------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------
def plot_pass_rates(df, gens, colors, title, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [5, 2.3]}, sharey=True)
    for ax, subset in zip(axes, ["incorrect", "correct"]):
        d = df[df.subset == subset]
        t = d.groupby(["criterion", "generator"]).value.mean().unstack()[gens]
        t.index = [CRITERIA[subset][c].replace(" · ", "\n") for c in t.index]
        grouped_bars(ax, t, gens, colors)
        ax.set_title(SET_TITLE[subset])
        ax.set_ylim(0, 1.1)
        pct_axis(ax)
    axes[0].set_ylabel("Share of responses meeting the criterion")
    axes[0].legend(ncol=len(gens), loc="upper left", bbox_to_anchor=(0, -0.13))
    fig.suptitle(f"{title} — pass rate per baldintza", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    save(fig, out / "01_pass_rate_per_baldintza.png")


def plot_heatmap(df, gens, title, out):
    cols, labels = [], []
    for subset in ["incorrect", "correct"]:
        for c, lab in CRITERIA[subset].items():
            cols.append((subset, c))
            labels.append(("INC " if subset == "incorrect" else "COR ") + lab)
    t = df.groupby(["generator", "subset", "criterion"]).value.mean()
    mat = np.array([[t.get((g, s, c), np.nan) for s, c in cols] for g in gens])
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(gens) + 2.4))
    im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    annotate_heatmap(ax, mat)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" · ", "\n") for l in labels], fontsize=8.5)
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels(gens)
    ax.axvline(len(CRITERIA["incorrect"]) - 0.5, color=SURFACE, lw=4)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, format="{x:.0%}")
    cb.outline.set_visible(False)
    ax.set_title(f"{title} — pass rate heatmap (INC = incorrect sentences, COR = correct)")
    save(fig, out / "02_pass_rate_heatmap.png")


def plot_score_distribution(df, gens, title, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(gens) + 2.6),
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, subset in zip(axes, ["incorrect", "correct"]):
        w = wide(df, subset)
        n_crit = len(CRITERIA[subset])
        score = w.sum(axis=1, min_count=1).dropna().round().astype(int)
        dist = (score.groupby(level="generator").value_counts(normalize=True)
                .unstack(fill_value=0).reindex(columns=range(n_crit + 1), fill_value=0)
                .reindex(gens))
        palette = ORDINAL if n_crit == 5 else [ORDINAL[0], ORDINAL[3], ORDINAL[6]]
        left = np.zeros(len(gens))
        for k in range(n_crit + 1):
            vals = dist[k].values
            ax.barh(range(len(gens)), vals, left=left, color=palette[k],
                    edgecolor=SURFACE, linewidth=1.5, height=0.62, label=f"{k}/{n_crit}")
            for i, (l, v) in enumerate(zip(left, vals)):
                if v >= 0.06:
                    ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=7.5, color="#ffffff" if k >= (n_crit + 1) / 2 else INK)
            left += vals
        ax.set_yticks(range(len(gens)))
        ax.set_yticklabels(gens)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        pct_axis(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{SET_TITLE[subset]}\nbaldintzak met per response")
        ax.legend(ncol=n_crit + 1, loc="upper left", bbox_to_anchor=(0, -0.08),
                  title="criteria met", fontsize=8, title_fontsize=8)
    fig.suptitle(f"{title} — distribution of response scores", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, out / "03_score_distribution.png")


def plot_funnel(df, gens, colors, title, out):
    """Cumulative strict pass: B1, B1&B2, ..., all five (incorrect sentences)."""
    w = wide(df, "incorrect")
    crits = list(CRITERIA["incorrect"])
    steps = ["B1", "+B2", "+B3", "+B4", "+B5\n(all met)"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for g in gens:
        m = w.loc[g][crits].dropna()
        cum = [(m[crits[:k + 1]] == 1).all(axis=1).mean() for k in range(len(crits))]
        ax.plot(steps, cum, color=colors[g], lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=g)
        ax.annotate(f"{g}  {cum[-1]:.0%}", (len(steps) - 1, cum[-1]),
                    xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.2, len(steps) - 0.3)
    pct_axis(ax)
    ax.set_ylabel("Responses meeting all criteria so far")
    ax.set_title(f"{title} — from detecting the error to a full explanation\n"
                 "(incorrect sentences: cumulative share meeting B1…Bk)")
    ax.legend(loc="lower left")
    save(fig, out / "04_explanation_funnel.png")


def plot_conditional(df, gens, colors, title, out):
    """P(Bk=1 | B1=1): once the error is detected, how good is the explanation?"""
    w = wide(df, "incorrect")
    crits = list(CRITERIA["incorrect"])[1:]
    rows = {}
    for g in gens:
        m = w.loc[g]
        hit = m[m["baldintza_1"] == 1]
        rows[g] = {CRITERIA["incorrect"][c].replace(" · ", "\n"): hit[c].mean() for c in crits}
        rows[g]["_n"] = len(hit)
    t = pd.DataFrame(rows)
    n = t.loc["_n"]
    t = t.drop("_n").astype(float)
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bars(ax, t, gens, colors)
    ax.set_ylim(0, 1.1)
    pct_axis(ax)
    ax.set_ylabel("Pass rate, given B1 = 1")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [f"{g} (n={int(n[g])})" for g in gens], ncol=len(gens),
              loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.set_title(f"{title} — explanation quality when the error WAS detected\n"
                 "(incorrect sentences, conditioned on B1 = 1)")
    save(fig, out / "05_conditional_on_detection.png")


def plot_criteria_correlation(df, gens, title, out):
    """Phi correlation between the 5 criteria (incorrect sentences), per generator + pooled."""
    w = wide(df, "incorrect")
    crits = list(CRITERIA["incorrect"])
    labels = [c.replace("baldintza_", "B") for c in crits]
    panels = [("All generators", w)] + [(g, w.loc[[g]]) for g in gens]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6))
    for ax, (name, m) in zip(axes, panels):
        c = m[crits].astype(float).corr().values  # Pearson on binaries = phi
        im = ax.imshow(c, cmap=DIV, vmin=-1, vmax=1)
        mask = np.triu(np.ones_like(c, dtype=bool), 1)
        c_show = np.where(mask, np.nan, c)
        np.fill_diagonal(c_show, np.nan)
        for i in range(len(crits)):
            for j in range(len(crits)):
                if not np.isnan(c_show[i, j]):
                    ax.text(j, i, f"{c_show[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.5, color="#ffffff" if abs(c_show[i, j]) > 0.6 else INK)
        ax.imshow(np.where(mask | np.eye(len(crits), dtype=bool), 1, np.nan),
                  cmap=LinearSegmentedColormap.from_list("s", [SURFACE, SURFACE]))
        ax.set_xticks(range(len(crits)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(crits)))
        ax.set_yticklabels(labels)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(name, fontsize=9.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.outline.set_visible(False)
    fig.suptitle(f"{title} — how criteria co-occur (phi correlation, incorrect sentences;"
                 " NaN = constant criterion)", x=0.01, ha="left", fontweight="bold",
                 fontsize=12, y=1.04)
    save(fig, out / "06_criteria_correlation.png")


def plot_item_difficulty(df, gens, title, out):
    """Per-item score (share of criteria met) for every generator, items sorted by difficulty."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 15),
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, subset in zip(axes, ["incorrect", "correct"]):
        w = wide(df, subset)
        n_crit = len(CRITERIA[subset])
        frac = (w.mean(axis=1)).unstack(level="generator")[gens]
        frac = frac.loc[frac.mean(axis=1).sort_values().index]
        mat = frac.values
        im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(gens)))
        ax.set_xticklabels(gens, rotation=30, ha="right")
        ax.set_yticks(range(len(frac.index)))
        ax.set_yticklabels([f"#{i}" for i in frac.index], fontsize=6.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{SET_TITLE[subset]}\nhardest items on top "
                     f"(cell = share of {n_crit} criteria met)", fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, format="{x:.0%}")
    cb.outline.set_visible(False)
    fig.suptitle(f"{title} — item difficulty", x=0.01, ha="left",
                 fontweight="bold", fontsize=13, y=0.93)
    save(fig, out / "07_item_difficulty.png")


def summary_table(df, gens):
    rows = []
    for g in gens:
        r = {"generator": g}
        for subset in ["incorrect", "correct"]:
            w = wide(df, subset).loc[g]
            for c, lab in CRITERIA[subset].items():
                r[f"{subset}:{lab}"] = w[c].mean()
            r[f"{subset}:mean_score"] = w.mean(axis=1).mean()
            r[f"{subset}:all_met"] = (w == 1).all(axis=1).mean()
        inc_b1 = wide(df, "incorrect").loc[g]["baldintza_1"]
        cor_b1 = wide(df, "correct").loc[g]["baldintza_1"]
        r["judgment_accuracy (B1 both sets)"] = pd.concat([inc_b1, cor_b1]).mean()
        r["overall_all_met"] = (r["incorrect:all_met"] + r["correct:all_met"]) / 2
        rows.append(r)
    return pd.DataFrame(rows).set_index("generator")


def plot_overview(summary, gens, colors, title, out):
    metrics = {
        "judgment_accuracy (B1 both sets)": "Grammaticality\njudgment (B1)",
        "incorrect:mean_score": "Incorrect:\nmean criteria met",
        "incorrect:all_met": "Incorrect:\nall 5 met",
        "correct:mean_score": "Correct:\nmean criteria met",
        "correct:all_met": "Correct:\nboth met",
        "overall_all_met": "Overall strict\n(all met, avg sets)",
    }
    t = summary[list(metrics)].T
    t.index = list(metrics.values())
    fig, ax = plt.subplots(figsize=(13, 5))
    grouped_bars(ax, t[gens], gens, colors)
    ax.set_ylim(0, 1.1)
    pct_axis(ax)
    ax.legend(ncol=len(gens), loc="upper left", bbox_to_anchor=(0, -0.15))
    ax.set_title(f"{title} — overview: judging vs. explaining")
    save(fig, out / "08_overview.png")


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=here / "gold.json")
    ap.add_argument("--out", type=Path, default=here / "3_baldintza_results")
    args = ap.parse_args()

    print(f"== {args.gold}")
    df = load_gold(args.gold)
    gens = gen_order(df)
    colors = dict(zip(gens, SERIES))
    title = "Gold"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    plot_pass_rates(df, gens, colors, title, out)
    plot_heatmap(df, gens, title, out)
    plot_score_distribution(df, gens, title, out)
    plot_funnel(df, gens, colors, title, out)
    plot_conditional(df, gens, colors, title, out)
    plot_criteria_correlation(df, gens, title, out)
    plot_item_difficulty(df, gens, title, out)
    summary = summary_table(df, gens)
    plot_overview(summary, gens, colors, title, out)
    print(f"\nDone. Graphs written to {out}")


if __name__ == "__main__":
    main()
