#!/usr/bin/env python3
"""
Judge vs. gold agreement leaderboard on the *baldintzak* (binary conditions).

Expected JSON format (same for gold and every judge):

    {"<evaluated-model>": [{"id": "1", "baldintzak": {"baldintza_1": 1, ...}}, ...], ...}

Items are split by the number of baldintzak in gold (5 -> incorrect, 2 -> correct)
and also pooled ("both", per-item score = fraction of baldintzak satisfied).

Writes a single <out>/01_leaderboard.png: one row per group (incorrect, correct,
both) with Cohen's kappa (95% bootstrap CI over items), raw agreement, F1 and
Spearman of item scores.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
GROUP_NAMES = {5: "incorrect", 2: "correct"}
N_BOOT = 2000
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "grid.color": GRID, "xtick.color": MUTED, "ytick.major.size": 0, "savefig.dpi": 200,
})


def short_name(name: str) -> str:
    for prefix in ("anthropic_", "google_", "openai_", "anthropic/", "google/", "openai/"):
        name = name.removeprefix(prefix)
    return name.replace("-Instruct-exp_2_101_v2-FP8", "")


def load(path: Path, source: str) -> pd.DataFrame:
    """One row per (model, id, baldintza); anything other than 0/1 becomes NaN."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [(short_name(model), str(item["id"]), cond, value)
            for model, items in data.items() for item in items
            for cond, value in item.get("baldintzak", {}).items()]
    df = pd.DataFrame(rows, columns=["model", "id", "baldintza", source])
    df[source] = pd.to_numeric(df[source], errors="coerce").where(lambda v: v.isin([0, 1]))
    return df.set_index(["model", "id", "baldintza"])


def kappa(po, pa, pb):
    pe = pa * pb + (1 - pa) * (1 - pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (po - pe) / (1 - pe)


def judge_metrics(df: pd.DataFrame, scores: pd.DataFrame, judge: str) -> dict:
    sub = df[["gold", judge]].dropna()
    g, j = sub["gold"].to_numpy(), sub[judge].to_numpy()

    # cluster bootstrap over items (model, id) so an item's baldintzak stay together
    codes, uniq = pd.factorize(sub.index.droplevel("baldintza"))
    per_item = [np.bincount(codes, weights=w, minlength=len(uniq))
                for w in (np.ones_like(g), g == j, g, j)]
    idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    n, agree, g1, j1 = (w[idx].sum(1) for w in per_item)
    lo, hi = np.nanpercentile(kappa(agree / n, g1 / n, j1 / n), [2.5, 97.5])

    tp = np.sum((g == 1) & (j == 1))
    sc = scores[["gold", judge]].dropna()
    return {"n": len(g), "kappa": kappa(np.mean(g == j), g.mean(), j.mean()),
            "kappa_lo": lo, "kappa_hi": hi, "accuracy": np.mean(g == j),
            "f1": 2 * tp / (g.sum() + j.sum()),
            "spearman": stats.spearmanr(sc["gold"], sc[judge])[0]}


def group_metrics(df: pd.DataFrame, judges) -> pd.DataFrame:
    # per-item score: only items where every baldintza is valid; pooled -> fraction
    items = df.groupby(level=["model", "id"])
    n_cond = items["gold"].size()
    scores = items.sum().where(items.count().eq(n_cond, axis=0)).div(n_cond, axis=0)
    return pd.DataFrame({j: judge_metrics(df, scores, j) for j in judges}).T.astype(float)


def pair_breakdown(df: pd.DataFrame, n_pairs: pd.Series) -> str:
    """How the number of (gold, judge) pairs is made up, e.g.
    '3 models × 20 sentences × 5 baldintzak = 300 pairs'."""
    n_cond = df.groupby(level=["model", "id"])["gold"].size()
    parts = []
    for k, n_items in n_cond.value_counts().sort_index(ascending=False).items():
        n_models = n_cond[n_cond == k].index.get_level_values("model").nunique()
        parts.append(f"{n_models} models × {n_items // n_models} sentences × {k} baldintzak"
                     if n_items % n_models == 0 else f"{n_items} items × {k} baldintzak")
    text = " + ".join(parts) + f" = {len(df)} pairs per judge"
    if n_pairs.min() < len(df):
        text += f" (min {int(n_pairs.min())} after dropping missing judge labels)"
    return text


def leaderboard(group_dfs: dict, judges, colors, out: Path):
    """One row per group (title -> its slice of df), one column per metric."""
    groups = {tag: group_metrics(d, judges) for tag, d in group_dfs.items()}
    order = list(groups.values())[-1].sort_values("kappa").index  # by pooled kappa
    y = np.arange(len(order))
    cols = [("kappa", "Cohen's κ (95% CI)", "{:.3f}"), ("accuracy", "Raw agreement", "{:.1%}"),
            ("f1", "F1 (positive = 1)", "{:.3f}"), ("spearman", "Spearman ρ (item score)", "{:.3f}")]
    xmin = {c: max(0, np.nanmin([s[c].min() for s in groups.values()]) - 0.15) for c, _, _ in cols}
    xmin["kappa"] = max(0, np.floor((min(s["kappa_lo"].min() for s in groups.values()) - 0.08) * 10) / 10)

    fig, axes = plt.subplots(len(groups), 4, figsize=(15, len(groups) * (1.2 + 0.62 * len(order))),
                             sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1], "wspace": 0.12, "hspace": 0.75})
    fig.suptitle("Judge vs. gold — agreement leaderboard", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.text(0.01, 1.015, f"Binary baldintzak · κ CI from bootstrap over items ({N_BOOT}×)",
             fontsize=9.5, color=MUTED)

    for row, (tag, s) in zip(axes, groups.items()):
        s = s.loc[order]
        # Cohen's kappa forest plot over Landis & Koch bands
        ax, lo_x = row[0], xmin["kappa"]
        for i, lab in enumerate(["slight", "fair", "moderate", "substantial", "almost perfect"]):
            lo, hi = max(i / 5, lo_x), (i + 1) / 5
            if hi <= lo_x:
                continue
            if i % 2:
                ax.axvspan(lo, hi, color=GRID, alpha=0.35, lw=0, zorder=0)
            ax.text((lo + hi) / 2, len(s) - 0.35, lab, ha="center", fontsize=7.5, color=MUTED)
        for yi, (judge, r) in zip(y, s.iterrows()):
            ax.plot([r.kappa_lo, r.kappa_hi], [yi, yi], color=colors[judge], lw=2)
            ax.scatter(r.kappa, yi, s=90, color=colors[judge], edgecolor=SURFACE, lw=2, zorder=3)
            ax.text(r.kappa_hi + 0.015, yi, f"{r.kappa:.3f}", va="center", fontsize=9)
        ax.set(xlim=(lo_x, 1.0), ylim=(-0.6, len(s) - 0.1))
        ax.set_yticks(y, order, fontsize=10.5)
        ax.text(0, 1.32, tag, transform=ax.transAxes, fontsize=12.5, fontweight="bold")
        ax.text(0, 1.2, pair_breakdown(group_dfs[tag], s["n"]), transform=ax.transAxes,
                fontsize=9.5, color=MUTED)

        for ax, (col, _, fmt) in zip(row[1:], cols[1:]):
            vals = s[col].to_numpy()
            ax.barh(y, vals, height=0.55, color=[colors[j] for j in order])
            for yi, v in zip(y, vals):
                ax.text(v + 0.01, yi, fmt.format(v), va="center", fontsize=9)
            ax.set_xlim(xmin[col], 1.05)
        for ax, (_, title, _) in zip(row, cols):
            ax.set_title(title, fontsize=10.5)
            ax.grid(axis="x")
            ax.set_axisbelow(True)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "01_leaderboard.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"saved {out / '01_leaderboard.png'}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=here / "gold.json")
    ap.add_argument("--judges", type=Path, nargs="*",
                    help="judge JSON files (default: every other *.json next to gold)")
    ap.add_argument("--out", type=Path, default=here / "2_correlation_results")
    args = ap.parse_args()

    judge_files = args.judges or sorted(p for p in args.gold.parent.glob("*.json")
                                        if p.resolve() != args.gold.resolve())
    judges = [short_name(p.stem) for p in judge_files]
    colors = {j: SERIES[i % len(SERIES)] for i, j in enumerate(judges)}

    # keep exactly the (model, id, baldintza) cells defined by gold
    gold = load(args.gold, "gold")
    df = pd.concat([gold] + [load(p, j) for p, j in zip(judge_files, judges)],
                   axis=1).reindex(gold.index).sort_index()

    n_cond = df.groupby(level=["model", "id"])["gold"].transform("size")
    groups = {f"{name} sentences ({k} baldintzak)": df[n_cond == k]
              for k, name in GROUP_NAMES.items() if (n_cond == k).any()}
    groups["both groups (pooled)"] = df
    leaderboard(groups, judges, colors, args.out)

if __name__ == "__main__":
    main()
