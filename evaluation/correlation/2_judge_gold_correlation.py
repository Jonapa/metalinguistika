#!/usr/bin/env python3
"""
Compare LLM-judge annotations of the *baldintzak* (binary conditions) against
the human-annotated gold standard.

Expected JSON format (same for gold and every judge):

    {
      "<evaluated-model>": [
        {"id": "1", "baldintzak": {"baldintza_1": 1, ..., "baldintza_5": 0}},
        ...
      ],
      ...
    }

Usage:
    python 2_judge_gold_correlation.py                      # gold.json + every other *.json in this folder
    python 2_judge_gold_correlation.py --gold gold.json --judges a.json b.json --out plots/

Items are split by the number of baldintzak in gold: 5 -> incorrect sentences,
2 -> correct sentences. These are different baldintzak, so the groups are never
pooled: the full analysis runs separately for each group.

Outputs (in --out, default ./2_correlation_results/):
    incorrect/ correct/          each one containing:
    01_leaderboard.png          Cohen's kappa (95% bootstrap CI), raw agreement, F1, score-level Spearman
    02_kappa_by_baldintza.png   kappa per judge x baldintza
    03_kappa_by_model.png       kappa per judge x evaluated model
    04_confusion_matrices.png   2x2 confusion matrices (row-normalised) per judge
    05_score_correlation.png    per-item total score: gold vs judge bubble plots
    06_leniency.png             positive-rate bias (judge - gold) per baldintza
    07_pairwise_agreement.png   kappa between every pair of annotators (gold + judges)
    08_system_level.png         mean score per evaluated model, gold vs judges
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
GOLD_COLOR = INK

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec",
                     "#f4a5a4", "#e66767", "#b02c2c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
})

N_BOOT = 2000
RNG = np.random.default_rng(42)

# Items are split by how many baldintzak they have (taken from gold):
# 5 -> incorrect sentences, 2 -> correct sentences.
GROUP_NAMES = {5: "incorrect", 2: "correct"}
GROUP_TAG = ""  # set per analysed subset, appended to every figure title


def tagged(title: str) -> str:
    return f"{title} — {GROUP_TAG}" if GROUP_TAG else title


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def short_name(name: str) -> str:
    """Make model / judge names readable on plot axes."""
    name = Path(name).stem if name.endswith(".json") else name
    for prefix in ("anthropic_", "google_", "openai_", "anthropic/", "google/", "openai/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("-Instruct-exp_2_101_v2-FP8", "")
    return name


def to_label(value):
    """Normalise one annotation to 0/1; anything else becomes NaN."""
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(value)
    try:
        v = float(str(value).strip())
    except ValueError:
        return np.nan
    return v if v in (0.0, 1.0) else np.nan


def load_annotations(path: Path, source: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows, bad = [], []
    for model, items in data.items():
        for item in items:
            for cond, raw in item.get("baldintzak", {}).items():
                val = to_label(raw)
                if np.isnan(val):
                    bad.append(f"{model} id={item['id']} {cond}={raw!r}")
                rows.append({"model": short_name(model), "id": str(item["id"]),
                             "baldintza": cond, source: val})
    if bad:
        print(f"  [!] {path.name}: {len(bad)} non-binary value(s) ignored -> "
              + "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))
    return pd.DataFrame(rows).set_index(["model", "id", "baldintza"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return np.nan
    po = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def binary_metrics(g: np.ndarray, j: np.ndarray) -> dict:
    tp = np.sum((g == 1) & (j == 1))
    tn = np.sum((g == 0) & (j == 0))
    fp = np.sum((g == 0) & (j == 1))
    fn = np.sum((g == 1) & (j == 0))
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    phi = stats.pearsonr(g, j)[0] if g.std() and j.std() else np.nan
    return {"n": len(g), "kappa": cohen_kappa(g, j), "accuracy": (tp + tn) / len(g),
            "precision": prec, "recall": rec, "f1": f1, "phi": phi,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def bootstrap_kappa_ci(df: pd.DataFrame, judge: str, n_boot=N_BOOT):
    """Cluster bootstrap over items (model,id) so the 5 baldintzak stay together."""
    sub = df[["gold", judge]].dropna()
    items = sub.index.droplevel("baldintza")
    codes, uniq = pd.factorize(items)
    g = sub["gold"].to_numpy()
    j = sub[judge].to_numpy()
    # per-item sufficient statistics -> fast vectorised resampling
    k = len(uniq)
    n_i = np.bincount(codes, minlength=k)
    agree_i = np.bincount(codes, weights=(g == j), minlength=k)
    g1_i = np.bincount(codes, weights=g, minlength=k)
    j1_i = np.bincount(codes, weights=j, minlength=k)
    idx = RNG.integers(0, k, size=(n_boot, k))
    n = n_i[idx].sum(1)
    po = agree_i[idx].sum(1) / n
    pa = g1_i[idx].sum(1) / n
    pb = j1_i[idx].sum(1) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = (po - pe) / (1 - pe)
    return np.nanpercentile(kap, [2.5, 97.5])


def kappa_band(k: float) -> str:
    """Landis & Koch (1977)."""
    if np.isnan(k):
        return "n/a"
    for thr, lab in [(0, "poor"), (0.2, "slight"), (0.4, "fair"),
                     (0.6, "moderate"), (0.8, "substantial")]:
        if k <= thr:
            return lab
    return "almost perfect"


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def save(fig, out: Path, name: str):
    path = out / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  saved {path}")


def subtitle(ax, text):
    ax.text(0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=MUTED)


def annotated_heatmap(ax, mat: pd.DataFrame, cmap, norm, fmt="{:.2f}", cbar_label=""):
    im = ax.imshow(mat.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-.5, mat.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for i in range(mat.shape[0]):
        for jj in range(mat.shape[1]):
            v = mat.iat[i, jj]
            if pd.isna(v):
                continue
            r, g, b, _ = im.cmap(im.norm(v))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(jj, i, fmt.format(v), ha="center", va="center", fontsize=9.5,
                    color="white" if lum < 0.5 else INK, fontweight="normal")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, labelcolor=MUTED)
    cb.set_label(cbar_label, color=MUTED, fontsize=9)
    return im


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_leaderboard(summary: pd.DataFrame, colors: dict, out: Path):
    s = summary.sort_values("kappa")
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 4, figsize=(15, 1.0 + 0.62 * len(s)), sharey=True,
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1], "wspace": 0.12})
    fig.suptitle(tagged("Judge vs. gold — agreement leaderboard"), x=0.01, ha="left",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.text(0.01, 0.955, f"Binary baldintzak, n = {int(s['n'].iloc[0])} pairs per judge · "
             f"bootstrap over items ({N_BOOT}×)", fontsize=9.5, color=MUTED, ha="left")

    # (a) kappa forest plot
    ax = axes[0]
    lo_x = max(0, math.floor((s.kappa_lo.min() - 0.08) * 10) / 10)
    # Landis & Koch bands, alternating shade, labelled only where visible
    for i, (lo, hi, lab) in enumerate([(0, .2, "slight"), (.2, .4, "fair"),
                                       (.4, .6, "moderate"), (.6, .8, "substantial"),
                                       (.8, 1, "almost perfect")]):
        if hi <= lo_x:
            continue
        if i % 2:
            ax.axvspan(max(lo, lo_x), hi, color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.text((max(lo, lo_x) + hi) / 2, len(s) - 0.35, lab, ha="center",
                va="bottom", fontsize=7.5, color=MUTED)
    for yi, (judge, r) in zip(y, s.iterrows()):
        c = colors[judge]
        ax.plot([r.kappa_lo, r.kappa_hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.scatter(r.kappa, yi, s=90, color=c, edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(r.kappa_hi + 0.015, yi, f"{r.kappa:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(lo_x, 1.0)
    ax.set_ylim(-0.6, len(s) - 0.1)
    ax.set_title("Cohen's κ (95% CI)")
    ax.set_yticks(y, s.index)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=INK)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")

    def bar_panel(ax, col, title, fmt):
        vals = s[col].to_numpy()
        ax.barh(y, vals, height=0.55, color=[colors[j] for j in s.index],
                edgecolor=SURFACE, lw=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, fmt.format(v), va="center", fontsize=9, color=INK)
        ax.set_xlim(max(0, np.nanmin(vals) - 0.15), 1.05)
        ax.set_title(title)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    bar_panel(axes[1], "accuracy", "Raw agreement", "{:.1%}")
    bar_panel(axes[2], "f1", "F1 (positive = 1)", "{:.3f}")
    bar_panel(axes[3], "score_spearman", "Spearman ρ (item score)", "{:.3f}")
    save(fig, out, "01_leaderboard.png")


def fig_heatmap(mat: pd.DataFrame, title: str, sub: str, out: Path, name: str):
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * mat.shape[1], 1.4 + 0.6 * mat.shape[0]))
    vmin = min(0, np.nanmin(mat.to_numpy(dtype=float)))
    annotated_heatmap(ax, mat, SEQ_BLUE, plt.Normalize(vmin, 1), cbar_label="Cohen's κ")
    ax.set_title(tagged(title), pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, out, name)


def fig_confusions(df, judges, colors, out: Path):
    ncol = min(len(judges), 3)
    nrow = math.ceil(len(judges) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.9 * nrow), squeeze=False)
    fig.suptitle(tagged("Confusion matrices (row-normalised by gold label)"), x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    for ax, judge in zip(axes.flat, judges):
        sub = df[["gold", judge]].dropna()
        cm = pd.crosstab(sub["gold"], sub[judge]).reindex(index=[0, 1], columns=[0, 1],
                                                          fill_value=0)
        norm = cm.div(cm.sum(axis=1), axis=0)
        im = ax.imshow(norm.to_numpy(), cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(2):
            for k in range(2):
                v = norm.iat[i, k]
                ax.text(k, i, f"{v:.0%}\n", ha="center", va="center", fontsize=14,
                        fontweight="bold", color="white" if v > 0.55 else INK)
                ax.text(k, i + 0.17, f"n = {cm.iat[i, k]}", ha="center", va="center",
                        fontsize=8.5, color="white" if v > 0.55 else INK_2)
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel("judge")
        ax.set_ylabel("gold")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=SURFACE, lw=4)
        ax.set_title(judge, color=INK)
        ax.plot([-0.5, 1.5], [-0.62, -0.62], color=colors[judge], lw=4,
                clip_on=False, solid_capstyle="round")
    for ax in axes.flat[len(judges):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, out, "04_confusion_matrices.png")


def fig_score_scatter(scores: pd.DataFrame, judges, colors, out: Path):
    ncol = min(len(judges), 3)
    nrow = math.ceil(len(judges) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.3 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    maxv = int(np.nanmax(scores.to_numpy()))
    fig.suptitle(tagged(f"Per-item score (sum of baldintzak, 0–{maxv}): gold vs. judge"),
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.945, "Bubble area = number of items · dashed line = perfect agreement",
             fontsize=9.5, color=MUTED)
    for ax, judge in zip(axes.flat, judges):
        sub = scores[["gold", judge]].dropna()
        counts = sub.groupby(["gold", judge]).size().reset_index(name="c")
        ax.plot([0, maxv], [0, maxv], ls="--", color=AXIS, lw=1.2, zorder=1)
        ax.scatter(counts["gold"], counts[judge], s=counts["c"] * 22, color=colors[judge],
                   alpha=0.85, edgecolor=SURFACE, lw=2, zorder=2)
        for _, r in counts.iterrows():
            if r.c >= 5:
                ax.text(r["gold"], r[judge], int(r.c), ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold", zorder=3)
        rho = stats.spearmanr(sub["gold"], sub[judge])[0]
        r_p = stats.pearsonr(sub["gold"], sub[judge])[0]
        mae = np.mean(np.abs(sub["gold"] - sub[judge]))
        ax.text(0.03, 0.97, f"ρ = {rho:.3f}\nr = {r_p:.3f}\nMAE = {mae:.2f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
        ax.set_title(judge)
        ax.set_xlim(-0.6, maxv + 0.6)
        ax.set_ylim(-0.6, maxv + 0.6)
        ax.set_xticks(range(maxv + 1))
        ax.set_yticks(range(maxv + 1))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel("gold score")
    for ax in axes[:, 0]:
        ax.set_ylabel("judge score")
    for ax in axes.flat[len(judges):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "05_score_correlation.png")


def fig_leniency(df, judges, out: Path):
    rates = df.groupby(level="baldintza").mean()
    bias = rates[judges].sub(rates["gold"], axis=0).T
    bias["ALL"] = (df[judges].mean() - df["gold"].mean())
    lim = max(0.05, np.nanmax(np.abs(bias.to_numpy())))
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * bias.shape[1], 1.6 + 0.6 * bias.shape[0]))
    annotated_heatmap(ax, bias, DIVERGING, TwoSlopeNorm(0, -lim, lim), fmt="{:+.1%}",
                      cbar_label="judge − gold positive rate")
    ax.set_title(tagged("Leniency / strictness bias per baldintza"), pad=36)
    gold_line = " · ".join(f"{c.replace('baldintza_', 'B')}: {v:.0%}"
                           for c, v in rates["gold"].items())
    ax.text(0, 1.02, "Red = judge says “1” more often than humans (lenient) · "
            f"blue = stricter\nGold positive rate — {gold_line}",
            transform=ax.transAxes, fontsize=9, color=MUTED, linespacing=1.5)
    save(fig, out, "06_leniency.png")


def fig_pairwise(df, annotators, out: Path):
    mat = pd.DataFrame(index=annotators, columns=annotators, dtype=float)
    for a in annotators:
        for b in annotators:
            sub = df[[a, b]].dropna() if a != b else df[[a]].dropna()
            mat.loc[a, b] = 1.0 if a == b else cohen_kappa(sub[a].to_numpy(),
                                                           sub[b].to_numpy())
    mask = np.triu(np.ones_like(mat, dtype=bool), 1)
    shown = mat.mask(mask)
    fig, ax = plt.subplots(figsize=(2 + 1.15 * len(annotators), 1.5 + 0.95 * len(annotators)))
    annotated_heatmap(ax, shown, SEQ_BLUE,
                      plt.Normalize(min(0, np.nanmin(mat.to_numpy())), 1),
                      cbar_label="Cohen's κ")
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_title(tagged("Pairwise agreement between all annotators"), pad=22)
    ax.text(0, 1.02, "Do judges agree with each other more than with humans?",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, out, "07_pairwise_agreement.png")
    return mat


def fig_system_level(scores, judges, colors, out: Path):
    sys_mean = scores.groupby(level="model").mean()
    sys_mean = sys_mean.sort_values("gold")
    models = sys_mean.index.tolist()
    y = np.arange(len(models))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 1.6 + 0.85 * len(models)),
                                  gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.35})
    fig.suptitle(tagged("System-level view: mean item score per evaluated model"),
                 x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.02)
    lo = np.nanmin(sys_mean.to_numpy())
    hi = np.nanmax(sys_mean.to_numpy())
    for yi, m in zip(y, models):
        ax.plot([lo - 0.05, hi + 0.05], [yi, yi], color=GRID, lw=1, zorder=0)
    offs = np.linspace(-0.22, 0.22, len(judges))
    for off, judge in zip(offs, judges):
        ax.scatter(sys_mean[judge], y + off, s=60, color=colors[judge], edgecolor=SURFACE,
                   lw=1.5, zorder=2, label=judge)
    ax.scatter(sys_mean["gold"], y, s=260, marker="|", color=GOLD_COLOR, lw=3,
               zorder=3, label="gold (human)")
    ax.set_yticks(y, models)
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=10.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("mean item score")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(judges) + 1, 6),
              fontsize=9, handletextpad=0.3, columnspacing=1.2)

    # ranking correlation
    taus = {}
    for judge in judges:
        taus[judge] = (stats.kendalltau(sys_mean["gold"], sys_mean[judge])[0]
                       if len(models) > 2 else np.nan)
    order = sorted(judges, key=lambda j: np.nan_to_num(taus[j], nan=-2))
    vals = [taus[j] for j in order]
    ax2.barh(range(len(order)), vals, height=0.55, color=[colors[j] for j in order],
             edgecolor=SURFACE, lw=2)
    for i, v in enumerate(vals):
        ax2.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax2.axvline(0, color=AXIS, lw=1)
    ax2.set_xlim(-1.15, 1.25)
    ax2.set_yticks(range(len(order)), order)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    ax2.set_title(f"Kendall τ of model ranking (n = {len(models)})")
    ax2.grid(axis="x")
    ax2.set_axisbelow(True)
    save(fig, out, "08_system_level.png")
    return taus


def analyse(df: pd.DataFrame, judges, colors, out: Path, tag: str) -> dict:
    """Run every metric + figure on one subset of items; return the tables."""
    global GROUP_TAG
    GROUP_TAG = tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\n  {tag.upper()}  ->  {out}\n{'=' * 78}")

    # per-item score: only items where every baldintza of that item is valid
    item_df = df.groupby(level=["model", "id"])
    n_cond_item = df["gold"].groupby(level=["model", "id"]).size()
    scores = item_df.sum(min_count=1).where(item_df.count().eq(n_cond_item, axis=0))

    rows = []
    for judge in judges:
        sub = df[["gold", judge]].dropna()
        m = binary_metrics(sub["gold"].to_numpy(), sub[judge].to_numpy())
        m["kappa_lo"], m["kappa_hi"] = bootstrap_kappa_ci(df, judge)
        sc = scores[["gold", judge]].dropna()
        m["score_pearson"] = stats.pearsonr(sc["gold"], sc[judge])[0]
        m["score_spearman"] = stats.spearmanr(sc["gold"], sc[judge])[0]
        m["score_kendall"] = stats.kendalltau(sc["gold"], sc[judge])[0]
        m["score_mae"] = np.mean(np.abs(sc["gold"] - sc[judge]))
        m["exact_item_match"] = np.mean(sc["gold"] == sc[judge])
        m["pos_rate_judge"] = sub[judge].mean()
        m["pos_rate_gold"] = sub["gold"].mean()
        m["kappa_band"] = kappa_band(m["kappa"])
        rows.append(pd.Series(m, name=judge))
    summary = pd.DataFrame(rows)

    per_cond = pd.DataFrame({
        j: {c: cohen_kappa(*(g[["gold", j]].dropna().to_numpy().T))
            for c, g in df.groupby(level="baldintza")} for j in judges}).T
    per_model = pd.DataFrame({
        j: {mdl: cohen_kappa(*(g[["gold", j]].dropna().to_numpy().T))
            for mdl, g in df.groupby(level="model")} for j in judges}).T
    per_cond.columns = [c.replace("baldintza_", "baldintza ") for c in per_cond.columns]

    order = summary.sort_values("kappa", ascending=False).index.tolist()
    per_cond, per_model = per_cond.loc[order], per_model.loc[order]

    # ---- figures ------------------------------------------------------------
    print("Rendering figures ...")
    fig_leaderboard(summary, colors, out)
    fig_heatmap(per_cond, "Cohen's κ per baldintza",
                "Which conditions are hard for judges? (rows sorted by overall κ)",
                out, "02_kappa_by_baldintza.png")
    fig_heatmap(per_model, "Cohen's κ per evaluated model",
                "Does judge reliability depend on whose output is being judged?",
                out, "03_kappa_by_model.png")
    fig_confusions(df, order, colors, out)
    fig_score_scatter(scores, order, colors, out)
    fig_leniency(df, order, out)
    fig_pairwise(df, ["gold"] + order, out)
    taus = fig_system_level(scores, order, colors, out)
    summary["system_kendall_tau"] = pd.Series(taus)

    # ---- console tables ------------------------------------------------------
    cols = ["n", "kappa", "kappa_lo", "kappa_hi", "kappa_band", "accuracy", "f1",
            "score_spearman", "score_mae", "pos_rate_judge", "pos_rate_gold"]
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(f"\n--- Judge vs gold, {tag} (sorted by Cohen's κ) ---")
        print(summary.loc[order, cols].to_string())
        print(f"\n--- Cohen's κ per baldintza, {tag} ---")
        print(per_cond.to_string())
        print(f"\n--- Cohen's κ per evaluated model, {tag} ---")
        print(per_model.to_string())

    GROUP_TAG = ""
    return {"summary": summary, "per_cond": per_cond, "n_items": len(n_cond_item),
            "n_cond": int(n_cond_item.max()), "gold_pos": df["gold"].mean()}


def print_conclusions(results: dict):
    print(f"\n{'=' * 78}\n  CONCLUSIONS PER GROUP\n{'=' * 78}")
    for g, r in results.items():
        s = r["summary"].sort_values("kappa", ascending=False)
        best, worst = s.index[0], s.index[-1]
        hard = r["per_cond"].mean().idxmin()
        easy = r["per_cond"].mean().idxmax()
        print(f"\n[{g}]  {r['n_items']} items · gold positive rate {r['gold_pos']:.0%}")
        print(f"  best judge : {best:<24} κ = {s.at[best, 'kappa']:.3f} "
              f"[{s.at[best, 'kappa_lo']:.3f}, {s.at[best, 'kappa_hi']:.3f}] "
              f"({s.at[best, 'kappa_band']}), agreement {s.at[best, 'accuracy']:.1%}")
        print(f"  worst judge: {worst:<24} κ = {s.at[worst, 'kappa']:.3f} "
              f"[{s.at[worst, 'kappa_lo']:.3f}, {s.at[worst, 'kappa_hi']:.3f}] "
              f"({s.at[worst, 'kappa_band']}), agreement {s.at[worst, 'accuracy']:.1%}")
        if r["per_cond"].shape[1] > 1:
            print(f"  hardest baldintza: {hard} (mean κ {r['per_cond'][hard].mean():.3f}) · "
                  f"easiest: {easy} (mean κ {r['per_cond'][easy].mean():.3f})")
        lenient = (s["pos_rate_judge"] - s["pos_rate_gold"])
        print("  bias (judge − gold positive rate): "
              + ", ".join(f"{j} {v:+.1%}" for j, v in lenient.items()))
    groups = list(results)
    if len(groups) == 2:
        a, b = groups
        d = (results[b]["summary"]["kappa"] - results[a]["summary"]["kappa"]).sort_values()
        print(f"\n[{b} − {a}]  Δκ per judge: "
              + ", ".join(f"{j} {v:+.3f}" for j, v in d.items()))
        print("  Note: κ is sensitive to prevalence — when gold is mostly 1, a few "
              "disagreements\n  cost a lot of κ even if raw agreement stays high. "
              "Read κ together with agreement.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=here / "gold.json")
    ap.add_argument("--judges", type=Path, nargs="*",
                    help="judge JSON files (default: every *.json next to gold except gold)")
    ap.add_argument("--out", type=Path, default=here / "2_correlation_results")
    args = ap.parse_args()

    if not args.gold.exists():
        sys.exit(f"Gold file not found: {args.gold}")
    judge_files = args.judges or sorted(p for p in args.gold.parent.glob("*.json")
                                        if p.resolve() != args.gold.resolve())
    if not judge_files:
        sys.exit("No judge files found.")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Gold   : {args.gold.name}")
    gold = load_annotations(args.gold, "gold")
    frames = [gold]
    judges = []
    for p in judge_files:
        name = short_name(p.name)
        print(f"Judge  : {p.name}  ->  {name}")
        frames.append(load_annotations(p, name))
        judges.append(name)
    colors = {j: SERIES[i % len(SERIES)] for i, j in enumerate(judges)}
    if len(judges) > len(SERIES):
        warnings.warn("More judges than palette slots; colours repeat.")

    # keep exactly the (model, id, baldintza) cells defined by gold
    df = pd.concat(frames, axis=1, join="outer").reindex(gold.index).sort_index()
    missing = df[judges].isna().sum()
    if missing.any():
        print("  [i] missing/invalid labels per judge (pairs dropped):",
              missing[missing > 0].to_dict())

    # ---- split items by number of baldintzak in gold ------------------------
    n_cond = df["gold"].groupby(level=["model", "id"]).size().rename("n_cond")
    item_group = n_cond.map(lambda n: GROUP_NAMES.get(n, f"{n}_baldintzak"))
    group = item_group.reindex(df.index.droplevel("baldintza")).to_numpy()
    print("\nItem groups (per evaluated model × id):",
          ", ".join(f"{g}: {c}" for g, c in item_group.value_counts().items()))

    results = {}
    for g in sorted(set(group), key=lambda g: -n_cond[item_group == g].iloc[0]):
        k = int(n_cond[item_group == g].iloc[0])
        results[g] = analyse(df[group == g], judges, colors, args.out / g,
                             f"{g} sentences ({k} baldintzak)")

    print_conclusions(results)
    print(f"\nAll outputs in: {args.out}")


if __name__ == "__main__":
    main()
