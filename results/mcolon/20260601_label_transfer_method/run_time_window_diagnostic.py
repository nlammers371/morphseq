"""
Time-window diagnostic for CEP290 label transfer.

Core question: is the Low_to_High / Not Penetrant confusion a METHOD failure or a
TIME-WINDOW failure? Labels are trajectory-level; images are static. A broad 30-48 hpf
window may blur images where the phenotype has not yet diverged with images where it has.

For each HPF window we rebuild the reference leave-one-out null FROM IMAGES IN THAT
WINDOW ONLY, recompute raw + calibrated predictions, and track how separability and
calibration behavior change across developmental time. The headline diagnostic is whether
the LtH/NP neighbor profile shifts over time (separable later) or stays flat (a feature /
label / biology problem, not time).

Two calibration configs are compared per window:
    conservative : raw_q_knn  + uniform prior   (best macro_f1 globally)
    sensitive    : balanced_q_knn + uniform     (best rare-class recovery globally)

Outputs:
    time_window_diagnostic_metrics.csv      (one row per window per method)
    time_window_neighbor_profile.csv        (P(neighbor label | true label, window))
    plots/time_window_*.png

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python run_time_window_diagnostic.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from calibration_core import (
    MAIN_LABELS,
    build_reference_loo_table,
    likelihood_raw_q_knn,
    likelihood_balanced_q_knn,
    prior_uniform,
    combine_posterior,
    calibrated_predictions,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors

DATA_PATH = ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
OUT_DIR = HERE
PLOT_DIR = HERE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

LABEL_COL = "cluster_categories"
TIME_COL = "predicted_stage_hpf"
K_SMALL = 15
K_PRIOR = 100

# Sliding windows (2 hpf step, 6 hpf width) for finer resolution of the transition.
SLIDING_WINDOWS = [(lo, lo + 6) for lo in range(30, 43, 2)]  # 30-36 ... 42-48
# Non-overlapping windows for a cleaner per-bin readout.
BLOCK_WINDOWS = [(30, 34), (34, 38), (38, 42), (42, 48)]

COLORS = {"Low_to_High": "#E6194B", "High_to_Low": "#3CB44B",
          "Intermediate": "#4363D8", "Not Penetrant": "#F58231"}


def get_feature_cols(df):
    return sorted([c for c in df.columns if c.startswith("z_mu_b_")])


def _vector_block(loo_df, prefix, labels):
    return loo_df[[f"{prefix}{lbl}" for lbl in labels]].values.astype(float)


def evaluate(true_labels, pred_labels, labels):
    out = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, pred_labels),
        "macro_f1": f1_score(true_labels, pred_labels, labels=labels,
                             average="macro", zero_division=0),
    }
    prec, rec, _, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, zero_division=0)
    for i, lbl in enumerate(labels):
        out[f"recall[{lbl}]"] = rec[i]
        out[f"precision[{lbl}]"] = prec[i]
    t = np.asarray(true_labels); p = np.asarray(pred_labels)
    lth = t == "Low_to_High"; npm = t == "Not Penetrant"
    out["LtH->NP_collapse"] = (p[lth] == "Not Penetrant").mean() if lth.any() else np.nan
    out["NP->LtH_falsecall"] = (p[npm] == "Low_to_High").mean() if npm.any() else np.nan
    return out


def neighbor_profile(ref_X, ref_labels, k, labels):
    """P(neighbor label | true label) within this window (self excluded)."""
    if len(ref_X) <= k + 1:
        return None
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(ref_X)
    _, idxs = nn.kneighbors(ref_X)
    neigh = ref_labels[idxs[:, 1:]]
    rows = {}
    for fl in labels:
        mask = ref_labels == fl
        if mask.sum() == 0:
            rows[fl] = {nl: np.nan for nl in labels}
            continue
        nl = neigh[mask].ravel()
        rows[fl] = {nl2: (nl == nl2).mean() for nl2 in labels}
    return rows


def run_window(df_win, feature_cols, labels, k_small, k_prior):
    """Build LOO null + raw/conservative/sensitive predictions for one window."""
    ref_X = df_win[feature_cols].values.astype(float)
    ref_labels = df_win[LABEL_COL].values
    n = len(df_win)

    # k_prior must fit the window; shrink if needed.
    kp = min(k_prior, n - 1)
    ks = min(k_small, max(kp - 1, 1))
    if kp <= ks or n < ks + 2:
        return None  # too few images to calibrate

    loo = build_reference_loo_table(ref_X, ref_labels, labels=labels,
                                    k_small=ks, k_prior=kp)
    true = loo["true_label"].values
    q_ref = _vector_block(loo, "q_", labels)

    results = {}
    results["raw"] = evaluate(true, loo["raw_pred_label"].values, labels)

    # conservative: raw_q_knn + uniform
    lik_c = likelihood_raw_q_knn(q_ref, q_ref, true, labels)
    post_c = combine_posterior(lik_c, prior_uniform(q_ref, true, labels))
    pred_c, _ = calibrated_predictions(post_c, labels)
    results["conservative"] = evaluate(true, pred_c, labels)

    # sensitive: balanced_q_knn + uniform
    lik_s = likelihood_balanced_q_knn(q_ref, q_ref, true, labels)
    post_s = combine_posterior(lik_s, prior_uniform(q_ref, true, labels))
    pred_s, _ = calibrated_predictions(post_s, labels)
    results["sensitive"] = evaluate(true, pred_s, labels)

    profile = neighbor_profile(ref_X, ref_labels, ks, labels)
    label_counts = {lbl: int((ref_labels == lbl).sum()) for lbl in labels}
    return {"results": results, "profile": profile, "label_counts": label_counts,
            "n": n, "k_small": ks, "k_prior": kp}


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df[LABEL_COL].notna()].copy()
    df = df[df[LABEL_COL].isin(MAIN_LABELS)].copy()
    feature_cols = get_feature_cols(df)
    print(f"  {len(df)} labeled images, {len(feature_cols)} features")

    metric_rows = []
    profile_rows = []

    for window_set, set_name in [(SLIDING_WINDOWS, "sliding"),
                                 (BLOCK_WINDOWS, "block")]:
        print(f"\n{'='*70}\n{set_name.upper()} WINDOWS\n{'='*70}")
        for lo, hi in window_set:
            df_win = df[(df[TIME_COL] >= lo) & (df[TIME_COL] < hi)].copy()
            out = run_window(df_win, feature_cols, MAIN_LABELS, K_SMALL, K_PRIOR)
            wlabel = f"{lo}-{hi}"
            if out is None:
                print(f"  [{wlabel}] SKIP (n={len(df_win)} too small)")
                continue

            counts = out["label_counts"]
            print(f"\n  [{wlabel}] n={out['n']}  "
                  f"(LtH {counts['Low_to_High']}, NP {counts['Not Penetrant']}, "
                  f"HtL {counts['High_to_Low']}, Int {counts['Intermediate']})  "
                  f"k_small={out['k_small']}")

            for method, m in out["results"].items():
                metric_rows.append({
                    "window_set": set_name, "window": wlabel,
                    "lo": lo, "hi": hi, "n": out["n"], "method": method,
                    **{k: v for k, v in counts.items()}, **m,
                })

            # headline print
            for method in ["raw", "conservative", "sensitive"]:
                m = out["results"][method]
                print(f"    {method:<13s} macroF1={m['macro_f1']:.3f} "
                      f"balAcc={m['balanced_accuracy']:.3f} "
                      f"LtH_rec={m['recall[Low_to_High]']:.3f} "
                      f"Int_rec={m['recall[Intermediate]']:.3f} "
                      f"LtH->NP={m['LtH->NP_collapse']:.3f}")

            # neighbor profile
            if out["profile"] is not None:
                for fl, row in out["profile"].items():
                    profile_rows.append({
                        "window_set": set_name, "window": wlabel,
                        "lo": lo, "hi": hi, "true_label": fl,
                        **{f"neigh_{nl}": v for nl, v in row.items()},
                    })

    metrics_df = pd.DataFrame(metric_rows)
    profile_df = pd.DataFrame(profile_rows)
    metrics_df.to_csv(OUT_DIR / "time_window_diagnostic_metrics.csv", index=False)
    profile_df.to_csv(OUT_DIR / "time_window_neighbor_profile.csv", index=False)
    print(f"\nSaved time_window_diagnostic_metrics.csv ({len(metrics_df)} rows)")
    print(f"Saved time_window_neighbor_profile.csv ({len(profile_df)} rows)")

    _plot_lth_np_profile_over_time(profile_df)
    _plot_metric_over_time(metrics_df, "recall[Low_to_High]", "LtH recall")
    _plot_metric_over_time(metrics_df, "recall[Intermediate]", "Intermediate recall")
    _plot_metric_over_time(metrics_df, "LtH->NP_collapse", "LtH->NP collapse")
    _plot_metric_over_time(metrics_df, "macro_f1", "macro F1")
    print(f"Plots saved to {PLOT_DIR}/")


def _plot_lth_np_profile_over_time(profile_df):
    """Headline diagnostic: do true-LtH neighbors shift away from NP over time?"""
    sub = profile_df[(profile_df["window_set"] == "sliding") &
                     (profile_df["true_label"] == "Low_to_High")].copy()
    if sub.empty:
        return
    sub["mid"] = (sub["lo"] + sub["hi"]) / 2
    sub = sub.sort_values("mid")
    fig, ax = plt.subplots(figsize=(8, 5))
    for nl in MAIN_LABELS:
        col = f"neigh_{nl}"
        if col in sub.columns:
            ax.plot(sub["mid"], sub[col], marker="o", label=nl, color=COLORS[nl])
    ax.set_xlabel("window midpoint (hpf)")
    ax.set_ylabel("P(neighbor label | true = Low_to_High)")
    ax.set_title("Headline diagnostic: LtH neighbor composition over developmental time\n"
                 "(does LtH separate from NP later?)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "time_window_lth_neighbor_profile.png", dpi=150)
    plt.close(fig)


def _plot_metric_over_time(metrics_df, metric, title):
    sub = metrics_df[metrics_df["window_set"] == "sliding"].copy()
    if sub.empty or metric not in sub.columns:
        return
    sub["mid"] = (sub["lo"] + sub["hi"]) / 2
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {"raw": ("gray", "--"), "conservative": ("#1f77b4", "-"),
             "sensitive": ("#d62728", "-")}
    for method in ["raw", "conservative", "sensitive"]:
        m = sub[sub["method"] == method].sort_values("mid")
        if m.empty:
            continue
        c, ls = style[method]
        ax.plot(m["mid"], m[metric], marker="o", label=method, color=c, linestyle=ls)
    ax.set_xlabel("window midpoint (hpf)")
    ax.set_ylabel(metric)
    ax.set_title(f"{title} over developmental time (sliding windows)")
    ax.legend()
    fig.tight_layout()
    safe = metric.replace("[", "_").replace("]", "").replace("->", "_to_").replace(" ", "_")
    fig.savefig(PLOT_DIR / f"time_window_{safe}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
