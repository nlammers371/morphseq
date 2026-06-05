"""
run_coverage_diagnostics.py
============================
Two things in one pass:

1. CovGap diagnostic on the pooled conformal benchmark output.
   Detects which classes are undercovered (subsidized) vs overcovering
   (subsidizing), with Wilson CIs so we don't react to sampling noise.

2. Mondrian (class-conditional) conformal re-run.
   For each LOEO fold and each method, recompute per-class qhat_c from
   the calibration q-scores, build Mondrian sets, and compare singleton
   rate, mean set size, and per-class coverage against the pooled baseline.

Output:
    coverage_diagnostics_covgap.csv          -- per-class CovGap table (pooled)
    coverage_diagnostics_mondrian_sets.csv   -- image-level Mondrian predictions
    coverage_diagnostics_mondrian_summary.csv -- fold-level comparison table
    plots/coverage_diagnostics_covgap.png
    plots/coverage_diagnostics_mondrian_vs_pooled.png

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260601_label_transfer_method/run_coverage_diagnostics.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "new_files"))

from conformal_sets import aps_scores, aps_quantile, build_sets  # noqa: E402
from coverage_diagnostics import (  # noqa: E402
    covgap_report,
    print_covgap_report,
    mondrian_qhat,
    build_sets_mondrian,
)

PREDICTIONS = HERE / "q_conformal_benchmark_full_time_image_predictions.csv"
PLOT_DIR = HERE / "plots"

LABEL_ORDER = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
LABEL_DISPLAY = {
    "Low_to_High": "Low→High",
    "High_to_Low": "High→Low",
    "Intermediate": "Intermediate",
    "Not Penetrant": "Not Penetrant",
}
ALPHA = 0.10
Q_COLS = [f"q_{l}" for l in LABEL_ORDER]
IN_SET_COLS = [f"in_set_{l}" for l in LABEL_ORDER]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABEL_ORDER)}


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: CovGap on pooled benchmark output
# ─────────────────────────────────────────────────────────────────────────────

def run_covgap(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in pred.groupby("method"):
        membership = grp[IN_SET_COLS].to_numpy(dtype=bool)
        y_true = grp["true_label"].map(LABEL_TO_IDX).to_numpy(dtype=int)
        report = covgap_report(membership, y_true, ALPHA, label_names=LABEL_ORDER)

        print(f"\n{'='*60}")
        print(f"CovGap report — {method} (pooled)")
        print(f"{'='*60}")
        print_covgap_report(report)

        for c in report["classes"]:
            rows.append({
                "method": method,
                "label": c["label"],
                "n": c["n"],
                "covered": c.get("covered"),
                "coverage": c.get("coverage"),
                "wilson_lo": c.get("wilson_lo"),
                "wilson_hi": c.get("wilson_hi"),
                "gap": c.get("gap"),
                "undercovered": c.get("undercovered"),
                "subsidizes": c.get("subsidizes"),
                "target": report["target"],
                "marginal_coverage": report["marginal_coverage"],
                "cov_gap": report["cov_gap"],
            })
    return pd.DataFrame(rows)


def plot_covgap(covgap_df: pd.DataFrame) -> None:
    methods = covgap_df["method"].unique().tolist()
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             sharey=True, constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    target = ALPHA

    for ax, method in zip(axes, methods):
        sub = covgap_df[covgap_df["method"] == method].copy()
        sub = sub[sub["n"].fillna(0) > 0]
        labels = sub["label"].tolist()
        cov = sub["coverage"].to_numpy(dtype=float)
        lo = sub["wilson_lo"].to_numpy(dtype=float)
        hi = sub["wilson_hi"].to_numpy(dtype=float)
        x = np.arange(len(labels))

        colors = []
        for _, row in sub.iterrows():
            if row["undercovered"]:
                colors.append("#B2182B")
            elif row["subsidizes"]:
                colors.append("#4DAC26")
            else:
                colors.append("#4C78A8")

        ax.barh(x, cov, color=colors, alpha=0.7, height=0.55)
        for i in range(len(labels)):
            ax.plot([lo[i], hi[i]], [x[i], x[i]], color="black", linewidth=1.5)
            ax.plot([lo[i], hi[i]], [x[i], x[i]], "k|", markersize=6)

        ax.axvline(1 - ALPHA, color="#333333", linewidth=1.5, linestyle="--",
                   label=f"target {1-ALPHA:.0%}")
        ax.set_yticks(x)
        ax.set_yticklabels([LABEL_DISPLAY.get(l, l) for l in labels])
        ax.set_xlabel("Coverage")
        ax.set_title(method)
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#B2182B", alpha=0.7, label="undercovered (subsidized)"),
        Patch(facecolor="#4DAC26", alpha=0.7, label="overcovering (subsidizes)"),
        Patch(facecolor="#4C78A8", alpha=0.7, label="within tolerance"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Per-class coverage with 95% Wilson CIs (pooled LOEO)", fontsize=12)
    PLOT_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOT_DIR / "coverage_diagnostics_covgap.png", dpi=180, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "coverage_diagnostics_covgap.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {PLOT_DIR / 'coverage_diagnostics_covgap.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Mondrian re-calibration per fold
# ─────────────────────────────────────────────────────────────────────────────

def normalize_q(q: np.ndarray, smoothing: float = 1e-6) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    q[~np.isfinite(q)] = 0.0
    q = np.clip(q, 0, None)
    if smoothing > 0:
        q += smoothing
    denom = q.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return q / denom


def run_mondrian(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run conformal set building with per-class qhats, LOEO fold by fold."""

    # The prediction table has calibration q-scores embedded: for each
    # heldout_experiment_id fold, the non-heldout rows are the calibration pool.
    # We need to reconstruct the calibration APS scores per fold.
    #
    # What we have: q_{label} columns are the MODEL q-scores for every row.
    # The calibration rows for fold F = rows where experiment_id != F and
    # heldout_experiment_id == F (they appear once per fold in the full table
    # because the benchmark stores ALL predictions including calibration points).
    #
    # Actually the table stores only query (heldout) predictions, not cal rows.
    # We need to re-derive qhat_c. We do this by treating the OTHER folds'
    # heldout predictions as a proxy calibration: for fold F, everything with
    # heldout_experiment_id != F is used as calibration.

    all_set_rows = []
    summary_rows = []

    for method, method_grp in pred.groupby("method"):
        folds = sorted(method_grp["heldout_experiment_id"].unique().tolist())

        for fold in folds:
            query = method_grp[method_grp["heldout_experiment_id"] == fold].copy()
            cal_pool = method_grp[method_grp["heldout_experiment_id"] != fold].copy()

            if len(cal_pool) == 0:
                continue

            # Build calibration APS scores from the other folds' q-scores
            q_cal = normalize_q(cal_pool[Q_COLS].to_numpy(dtype=float))
            y_cal = cal_pool["true_label"].map(LABEL_TO_IDX).to_numpy(dtype=int)
            s_cal = aps_scores(q_cal)

            # Pooled qhat (baseline, should match stored qhat approximately)
            q_query = normalize_q(query[Q_COLS].to_numpy(dtype=float))
            s_query = aps_scores(q_query)

            # Mondrian qhat per class
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                qhats_m = mondrian_qhat(s_cal, y_cal, ALPHA, n_classes=len(LABEL_ORDER))
            for w in caught:
                print(f"  [warning] {w.message}")

            # Pooled baseline: use median stored qhat for this fold
            pooled_qhat = float(query["qhat"].median())

            # Build sets
            sets_pooled = build_sets(s_query, pooled_qhat)
            sets_mondrian = build_sets_mondrian(s_query, qhats_m)

            y_query = query["true_label"].map(LABEL_TO_IDX).to_numpy(dtype=int)

            # Coverage reports
            report_pooled = covgap_report(sets_pooled, y_query, ALPHA, LABEL_ORDER)
            report_mondrian = covgap_report(sets_mondrian, y_query, ALPHA, LABEL_ORDER)

            def set_to_str(row_mask):
                return "|".join([LABEL_ORDER[j] for j in np.where(row_mask)[0]])

            # Per-image Mondrian predictions
            for i, (idx, orig_row) in enumerate(query.iterrows()):
                set_mask = sets_mondrian[i]
                all_set_rows.append({
                    "method": method,
                    "heldout_experiment_id": fold,
                    "snip_id": orig_row["snip_id"],
                    "true_label": orig_row["true_label"],
                    "argmax_label": orig_row["argmax_label"],
                    "mondrian_set": set_to_str(set_mask),
                    "mondrian_set_size": int(set_mask.sum()),
                    "mondrian_singleton": bool(set_mask.sum() == 1),
                    "mondrian_correct": bool(set_mask[y_query[i]]),
                    "pooled_set_size": int(sets_pooled[i].sum()),
                    "pooled_correct": bool(sets_pooled[i][y_query[i]]),
                })

            # Summary row per fold
            def _summary(tag, report, sets):
                return {
                    "method": method,
                    "heldout_experiment_id": fold,
                    "conformal_type": tag,
                    "n_query": len(query),
                    "marginal_coverage": report["marginal_coverage"],
                    "cov_gap": report["cov_gap"],
                    "mean_set_size": float(sets.sum(axis=1).mean()),
                    "singleton_rate": float((sets.sum(axis=1) == 1).mean()),
                    **{
                        f"coverage_{c['label']}": c["coverage"]
                        for c in report["classes"]
                        if c["n"] > 0
                    },
                    **{
                        f"undercovered_{c['label']}": c["undercovered"]
                        for c in report["classes"]
                        if c["n"] > 0
                    },
                    **{
                        f"qhat_{c['label']}": qhats_m[LABEL_TO_IDX[c['label']]]
                        for c in report["classes"]
                    },
                }

            summary_rows.append(_summary("pooled", report_pooled, sets_pooled))
            summary_rows.append(_summary("mondrian", report_mondrian, sets_mondrian))

        # Print pooled summary across all folds
        method_rows = [r for r in summary_rows if r["method"] == method]
        for ctype in ["pooled", "mondrian"]:
            type_rows = [r for r in method_rows if r["conformal_type"] == ctype]
            if not type_rows:
                continue
            sing = np.mean([r["singleton_rate"] for r in type_rows])
            mss = np.mean([r["mean_set_size"] for r in type_rows])
            cov = np.mean([r["marginal_coverage"] for r in type_rows])
            cgap = np.mean([r["cov_gap"] for r in type_rows if r["cov_gap"] is not None])
            print(f"  {method} / {ctype}: singleton={sing:.3f}  mean_set_size={mss:.3f}"
                  f"  coverage={cov:.3f}  cov_gap={cgap:.4f}")

    return pd.DataFrame(all_set_rows), pd.DataFrame(summary_rows)


def plot_mondrian_comparison(summary_df: pd.DataFrame) -> None:
    methods = summary_df["method"].unique().tolist()
    metrics = [
        ("singleton_rate", "Singleton rate"),
        ("mean_set_size", "Mean set size"),
        ("marginal_coverage", "Marginal coverage"),
        ("cov_gap", "CovGap"),
    ]
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, len(methods),
                             figsize=(5 * len(methods), 3.5 * n_metrics),
                             constrained_layout=True)
    if len(methods) == 1:
        axes = axes[:, None]

    colors = {"pooled": "#4C78A8", "mondrian": "#E45756"}

    for col, method in enumerate(methods):
        sub = summary_df[summary_df["method"] == method]
        folds = sorted(sub["heldout_experiment_id"].unique())

        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for ctype in ["pooled", "mondrian"]:
                vals = sub[sub["conformal_type"] == ctype].set_index(
                    "heldout_experiment_id")[metric].reindex(folds)
                x = np.arange(len(folds))
                ax.plot(x, vals.values, marker="o", color=colors[ctype],
                        linewidth=1.8, markersize=5, label=ctype)

            if row == 0:
                ax.set_title(method, fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(np.arange(len(folds)))
            ax.set_xticklabels([f.replace("20", "") for f in folds],
                               rotation=45, ha="right", fontsize=7)
            if metric == "marginal_coverage":
                ax.axhline(1 - ALPHA, color="#888", linewidth=1, linestyle="--")
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles = [plt.Line2D([0], [0], color=c, marker="o", linewidth=1.8, markersize=5,
                           label=t)
               for t, c in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Pooled vs Mondrian conformal: per-fold metrics", fontsize=12, y=1.04)
    PLOT_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOT_DIR / "coverage_diagnostics_mondrian_vs_pooled.png",
                dpi=180, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "coverage_diagnostics_mondrian_vs_pooled.svg",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PLOT_DIR / 'coverage_diagnostics_mondrian_vs_pooled.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading {PREDICTIONS}")
    pred = pd.read_csv(PREDICTIONS, low_memory=False)
    print(f"  {len(pred)} rows, {pred['method'].nunique()} methods, "
          f"{pred['heldout_experiment_id'].nunique()} folds")

    print("\n" + "="*60)
    print("PART 1: CovGap diagnostic (pooled)")
    print("="*60)
    covgap_df = run_covgap(pred)
    out_covgap = HERE / "coverage_diagnostics_covgap.csv"
    covgap_df.to_csv(out_covgap, index=False)
    print(f"\nSaved {out_covgap}")
    plot_covgap(covgap_df)

    print("\n" + "="*60)
    print("PART 2: Mondrian re-calibration (per fold)")
    print("="*60)
    set_df, summary_df = run_mondrian(pred)
    out_sets = HERE / "coverage_diagnostics_mondrian_sets.csv"
    out_summary = HERE / "coverage_diagnostics_mondrian_summary.csv"
    set_df.to_csv(out_sets, index=False)
    summary_df.to_csv(out_summary, index=False)
    print(f"\nSaved {out_sets}")
    print(f"Saved {out_summary}")
    plot_mondrian_comparison(summary_df)

    print("\n" + "="*60)
    print("SUMMARY: Does Mondrian help?")
    print("="*60)
    for method in summary_df["method"].unique():
        sub = summary_df[summary_df["method"] == method]
        for metric, label in [("singleton_rate", "singleton rate"),
                               ("cov_gap", "CovGap"),
                               ("mean_set_size", "mean set size")]:
            p = sub[sub["conformal_type"] == "pooled"][metric].mean()
            m = sub[sub["conformal_type"] == "mondrian"][metric].mean()
            arrow = "↑" if m > p else "↓"
            print(f"  {method} {label}: pooled={p:.4f}  mondrian={m:.4f}  {arrow} {abs(m-p):.4f}")


if __name__ == "__main__":
    main()
