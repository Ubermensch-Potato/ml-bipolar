"""Feature-importance figures, one PNG per (model, strategy) plus a cross-model heatmap.

Reads the *_importance.csv files written by eval/run_importance.py and
eval/run_importance_tabpfn.py, aggregates the 25 folds per (model, strategy), and writes
PNGs to outputs/figures/.

Importance units differ per model family (LR: coef, XGBoost: gain, BalancedRF: Gini,
TabPFN: permutation dAUC), so absolute values are NOT comparable across models. Each bar
chart is read on its own; the heatmap compares RANKS via per-combo max-normalization.

Usage:
  python -m eval.plot_importance --imp outputs/models_spec_at_sens_importance.csv \
                                       outputs/tabpfn_spec_at_sens_importance.csv
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ml_code/ on path
from utils.paths import OUTPUTS as OUT

METHOD_LABEL = {"coef": "logistic-regression coefficient (signed)",
                "gain": "XGBoost gain (normalized, sum=1)",
                "gini": "BalancedRF Gini importance (sum=1)",
                "permutation_dauc": "permutation importance (drop in test AUC)"}
POS, NEG, NEU = "#c0392b", "#2874a6", "#7f8c8d"      # risk / protective / undetermined


def aggregate(df):
    """Per (model, strategy, feature): mean importance over folds + 95% CI of the mean."""
    g = df.groupby(["model", "strategy", "method", "feature"])
    out = g.agg(mean=("importance", "mean"), sd=("importance", "std"), n=("importance", "size"),
                direction=("direction", "mean")).reset_index()
    out["sem"] = out["sd"] / np.sqrt(out["n"])            # bracket access: .sem/.sd shadow DataFrame methods
    out["lo"] = out["mean"] - 1.96 * out["sem"]
    out["hi"] = out["mean"] + 1.96 * out["sem"]
    # L1 sparsity: how often the feature survived (non-zero coefficient)
    nz = df.assign(nz=(df.importance.abs() > 1e-12)).groupby(
        ["model", "strategy", "feature"]).nz.mean().rename("selected_frac").reset_index()
    return out.merge(nz, on=["model", "strategy", "feature"], how="left")


def rank_and_orient(sub):
    """Add `plotted` (the signed bar value) and sort most-important first.

    coef  : already signed -> plot the coefficient, rank by |coef|.
    others: magnitude-only (>=0) -> plot magnitude on the side given by the feature's
            correlation with the outcome, rank by the magnitude itself. Ranking unsigned
            methods by |mean| would promote negative permutation dAUC, which is just noise.
    """
    signed = (sub.method.iloc[0] == "coef")
    sub = sub.copy()
    if signed:
        sub["plotted"] = sub["mean"]
        order = sub["mean"].abs()
    else:
        side = np.where(sub["direction"] >= 0, 1.0, -1.0)
        sub["plotted"] = sub["mean"] * side
        order = sub["mean"]
    return sub.assign(_o=order).sort_values("_o", ascending=False).drop(columns="_o"), signed


def bar_png(sub, model, strat, topk, outdir):
    method = sub.method.iloc[0]
    sub, signed = rank_and_orient(sub)
    sub = sub.head(topk).iloc[::-1]                         # biggest at the top of the bar chart

    colors = [POS if v > 0 else (NEG if v < 0 else NEU) for v in sub["plotted"]]

    labels = list(sub.feature)
    if model == "LR_L1":                                   # L1 is sparse: show selection stability
        labels = [f"{f}  ({s:.0%})" for f, s in zip(sub.feature, sub.selected_frac)]

    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(sub) + 2.4))
    ypos = np.arange(len(sub))
    ax.barh(ypos, sub["plotted"], color=colors, alpha=.85, height=.7)
    # CI half-widths are magnitudes; mirror them onto whichever side the bar was drawn
    err = np.vstack([np.maximum(sub["mean"] - sub["lo"], 0), np.maximum(sub["hi"] - sub["mean"], 0)])
    ax.errorbar(sub["plotted"], ypos, xerr=err, fmt="none", ecolor="#333", elinewidth=.9, capsize=2)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#333", lw=.8)

    lim = float(np.max(np.abs(sub["plotted"]) + err.max(axis=0))) * 1.12
    ax.set_xlim(-lim, lim)                                 # symmetric axis: left = protective
    ax.set_xlabel(METHOD_LABEL.get(method, method) + ("" if signed else "   (bar length = magnitude)"),
                  fontsize=9)
    note = ("bar side = coefficient sign" if signed else
            "bar side = sign of the feature's correlation with conversion (magnitude is unsigned)")
    extra = "; (%) = folds where L1 kept the feature" if model == "LR_L1" else ""
    ax.set_title(f"{model}:{strat}   —   top {len(sub)} features\n"
                 f"mean over 25 folds, error bars = 95% CI of the mean\n{note}{extra}", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=POS), plt.Rectangle((0, 0), 1, 1, color=NEG)]
    lab = (["raises predicted risk", "lowers predicted risk"] if signed
           else ["higher value -> more conversion", "higher value -> less conversion"])
    ax.legend(handles, lab, fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    p = os.path.join(outdir, f"imp_{model}_{strat}.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    return p


def heatmap_png(agg, topn, outdir):
    """Rank-comparable view: per (model,strategy) normalize |mean| to its own max."""
    agg = agg.copy()
    agg["abs"] = agg["mean"].abs()
    agg["norm"] = agg.groupby(["model", "strategy"])["abs"].transform(lambda s: s / (s.max() or 1))
    agg["combo"] = agg.model + ":" + agg.strategy

    top = (agg.sort_values("norm", ascending=False).groupby("combo").head(topn).feature.unique())
    piv = agg[agg.feature.isin(top)].pivot_table(index="feature", columns="combo", values="norm")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(0.55 * piv.shape[1] + 4.5, 0.3 * piv.shape[0] + 2.4))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_title("Feature importance, normalized within each model:strategy\n"
                 "(units differ across model families — compare ranks, not magnitudes)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.02, pad=.02, label="importance / max within combo")
    fig.tight_layout()
    p = os.path.join(outdir, "imp_heatmap.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    return p


def top_features_json(agg, df, topk, sources, path):
    """Machine-readable top-K per (model, strategy) + a cross-combo consensus ranking."""
    combos, appear = {}, {}
    for (m, s), sub in agg.groupby(["model", "strategy"]):
        ranked, signed = rank_and_orient(sub)
        ranked = ranked.head(topk)
        base = df[(df.model == m) & (df.strategy == s)].groupby(["repeat", "fold"]).baseline_auc.first()
        items = []
        for rank, (_, r) in enumerate(ranked.iterrows(), 1):
            appear.setdefault(r.feature, []).append(rank)
            item = {"rank": rank, "feature": r.feature, "mean": round(float(r["mean"]), 6),
                    "ci95": [round(float(r["lo"]), 6), round(float(r["hi"]), 6)],
                    "n_folds": int(r["n"]), "plotted": round(float(r["plotted"]), 6),
                    "direction": ("positive" if r["direction"] > 0 else
                                  "negative" if r["direction"] < 0 else "undetermined")}
            if m == "LR_L1":
                item["selected_frac"] = round(float(r["selected_frac"]), 4)
            items.append(item)
        combos[f"{m}:{s}"] = {"model": m, "strategy": s, "method": sub.method.iloc[0],
                              "signed": bool(signed), "mean_test_auc": round(float(base.mean()), 6),
                              "top": items}

    n_combos = len(combos)
    consensus = sorted(
        ({"feature": f, "n_combos_in_top": len(rs), "share": round(len(rs) / n_combos, 3),
          "mean_rank": round(float(np.mean(rs)), 2)} for f, rs in appear.items()),
        key=lambda d: (-d["n_combos_in_top"], d["mean_rank"]))

    doc = {"generated_from": [os.path.basename(s) for s in sources],
           "topk": topk, "n_combos": n_combos,
           "note": ("Importance units differ across model families (coef / gain / gini / "
                    "permutation dAUC); compare ranks, not magnitudes. `direction` is the sign of "
                    "the feature's correlation with conversion on the training folds — descriptive, "
                    "not causal."),
           "combos": combos, "consensus": consensus}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return consensus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imp", nargs="+", required=True, help="one or more *_importance.csv")
    ap.add_argument("--topk", type=int, default=20, help="features per bar chart / JSON")
    ap.add_argument("--topn", type=int, default=10, help="features per combo entering the heatmap")
    ap.add_argument("--outdir", default=os.path.join(OUT, "figures"))
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in a.imp], ignore_index=True)
    agg = aggregate(df)

    made = []
    for (m, s), sub in agg.groupby(["model", "strategy"]):
        made.append(bar_png(sub, m, s, a.topk, a.outdir))
    made.append(heatmap_png(agg, a.topn, a.outdir))

    agg.sort_values(["model", "strategy"]).to_csv(os.path.join(a.outdir, "importance_summary.csv"), index=False)
    jpath = os.path.join(a.outdir, "top_features.json")
    consensus = top_features_json(agg, df, a.topk, a.imp, jpath)

    print(f"{len(made)} figures -> {a.outdir}")
    for p in made:
        print("  " + os.path.basename(p))
    print(f"  top_features.json  importance_summary.csv")
    print(f"\nconsensus (feature appears in top-{a.topk} of how many of the {agg.groupby(['model','strategy']).ngroups} combos):")
    for d in consensus[:10]:
        print(f"  {d['feature']:30s} {d['n_combos_in_top']:>3} combos  mean rank {d['mean_rank']:.1f}")


if __name__ == "__main__":
    main()
