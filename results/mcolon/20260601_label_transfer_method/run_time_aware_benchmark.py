"""
Time-aware reference-null calibration bake-off (image-LOO arena).

Compares, ALL under the W_q × W_t time kernel with a UNIFORM prior (local prior
abandoned — see findings Section 0):

    1. raw_q_knn      + time kernel
    2. balanced_q_knn + time kernel        (prior bake-off's rare-class winner; kept)
    3. kernel q-distance, jensenshannon
    4. kernel q-distance, cityblock (L1)
    5. kernel q-distance, euclidean

swept over σ_t ∈ {2, 3, 4, 6} hpf (and σ_t = ∞ as a time-agnostic control).

Plus a SEPARATE supervised baseline, run first as the bar to clear:
    multinomial logistic regression on [q-vector, hpf] (+ optional q×hpf interactions).

Decision metrics (rare-class focused, NOT raw accuracy): macro_f1, balanced_accuracy,
per-label precision/recall, LtH recall, Intermediate recall, LtH→NP collapse,
NP→LtH false-call, calibrated-margin behavior.

This is the image-LOO arena (fast iteration). Sibling-frame leakage caveat applies; the
leave-one-embryo-out / leave-one-experiment-out rungs come after a default is chosen.

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python run_time_aware_benchmark.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from calibration_core import (
    MAIN_LABELS,
    build_reference_loo_table,
    calibrate_q_time,
    calibrated_predictions,
    calibrated_margin,
    median_pairwise_q_distance,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

DATA_PATH = ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
OUT_DIR = HERE

LABEL_COL = "cluster_categories"
TIME_COL = "predicted_stage_hpf"
MIN_HPF = 30.0
MAX_HPF = 48.0
K_SMALL = 15
K_PRIOR = 100
K_Q = 50

SIGMA_T_GRID = [2.0, 3.0, 4.0, 6.0, np.inf]  # inf = time-agnostic control


def get_feature_cols(df):
    return sorted([c for c in df.columns if c.startswith("z_mu_b_")])


def _qblock(loo, labels):
    return loo[[f"q_{l}" for l in labels]].values.astype(float)


def evaluate(true_labels, pred_labels, labels, margins=None):
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
    if margins is not None:
        out["mean_margin"] = float(np.mean(margins))
        # margin separation: do correct calls have higher margins than incorrect?
        correct = (p == t)
        if correct.any() and (~correct).any():
            out["margin_correct"] = float(margins[correct].mean())
            out["margin_incorrect"] = float(margins[~correct].mean())
    return out


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df[LABEL_COL].notna()].copy()
    df = df[(df[TIME_COL] >= MIN_HPF) & (df[TIME_COL] <= MAX_HPF)].copy()
    df = df[df[LABEL_COL].isin(MAIN_LABELS)].copy().reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    print(f"  {len(df)} labeled images, {len(feature_cols)} features")

    ref_X = df[feature_cols].values.astype(float)
    ref_labels = df[LABEL_COL].values
    ref_hpf = df[TIME_COL].values.astype(float)

    print(f"Building time-aware reference LOO table (k_small={K_SMALL}) ...")
    loo = build_reference_loo_table(
        ref_X, ref_labels, labels=MAIN_LABELS,
        k_small=K_SMALL, k_prior=K_PRIOR, ref_hpf=ref_hpf,
    )
    loo.to_csv(OUT_DIR / "reference_loo_table_timeaware.csv", index=False)
    true = loo["true_label"].values
    q_ref = _qblock(loo, MAIN_LABELS)
    hpf = loo["ref_hpf"].values
    print(f"  Saved reference_loo_table_timeaware.csv ({len(loo)} rows)")

    results = []

    # ---- Baseline: raw argmax(q) ----
    print("\n=== BASELINE: raw argmax(q) ===")
    base = evaluate(true, loo["raw_pred_label"].values, MAIN_LABELS)
    results.append({"method": "RAW_BASELINE", "sigma_t": "-", **base})
    _print(base)

    # ---- Logistic regression baselines (the bar to clear) ----
    # Time-GLOBAL LR: one surface, hpf as a feature (the UNFAIR comparison — it cannot
    # represent a different q->label map per stage). Kept to show the gap.
    print("\n=== LOGISTIC REGRESSION baselines (5-fold CV preds) ===")
    for interactions in (False, True):
        feats = [q_ref, hpf[:, None]]
        if interactions:
            feats.append(q_ref * hpf[:, None])  # q×hpf
        Xlr = np.hstack(feats)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0),
        )
        pred = cross_val_predict(clf, Xlr, true, cv=5)
        m = evaluate(true, pred, MAIN_LABELS)
        tag = "LR_global[q,hpf,qxhpf]" if interactions else "LR_global[q,hpf]"
        results.append({"method": tag, "sigma_t": "-", **m})
        print(f"  {tag}:")
        _print(m, indent=4)

    # Time-LOCAL LR (windowed): fit a separate LR per hpf bin. The FAIR comparison —
    # the LR analogue of the kernel's per-stage localization. CV WITHIN each window.
    print("\n  Time-local LR (windowed, per-hpf-bin):")
    for width in (4.0, 6.0):
        pred = np.empty(len(true), dtype=object)
        edges = np.arange(MIN_HPF, MAX_HPF + width, width)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (hpf >= lo) & (hpf < hi if hi < MAX_HPF else hpf <= hi)
            if mask.sum() < 50:
                # too small for CV; fall back to global LR predictions later
                pred[mask] = None
                continue
            Xw = q_ref[mask]
            yw = true[mask]
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   multi_class="multinomial", C=1.0))
            # need >=2 classes per fold; if a window is single-class, assign that class
            if len(np.unique(yw)) < 2:
                pred[mask] = yw
            else:
                pred[mask] = cross_val_predict(clf, Xw, yw, cv=3)
        # fill any None (tiny windows) with the global LR over q+hpf
        none_mask = np.array([p is None for p in pred])
        if none_mask.any():
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   multi_class="multinomial", C=1.0))
            clf.fit(np.hstack([q_ref[~none_mask], hpf[~none_mask, None]]),
                    true[~none_mask])
            pred[none_mask] = clf.predict(np.hstack([q_ref[none_mask], hpf[none_mask, None]]))
        m = evaluate(true, pred.astype(str), MAIN_LABELS)
        tag = f"LR_local_win{width:g}"
        results.append({"method": tag, "sigma_t": "-", **m})
        print(f"    {tag}:")
        _print(m, indent=6)

    # Time-LOCAL LR (W_t-weighted): the TRUEST apples-to-apples — identical temporal
    # localization to the kernel (sample weights W_t), only linear-LR vs nonparametric
    # differs. Approx via leave-one-out is too slow for full LR refits; instead fit a
    # single weighted LR per representative stage and predict points near that stage,
    # using a coarse grid of anchor stages with W_t sample weights.
    print("\n  Time-local LR (W_t-weighted, anchored):")
    for sigma_t in (3.0, 4.0):
        anchors = np.arange(MIN_HPF + 1, MAX_HPF, 2.0)
        pred = np.empty(len(true), dtype=object)
        assigned = np.zeros(len(true), dtype=bool)
        # assign each point to its nearest anchor
        anchor_idx = np.argmin(np.abs(hpf[:, None] - anchors[None, :]), axis=1)
        for ai, a in enumerate(anchors):
            target = anchor_idx == ai
            if target.sum() == 0:
                continue
            w = np.exp(-((hpf - a) ** 2) / (2 * sigma_t ** 2))
            # train on all points, weighted by proximity to anchor; predict the target
            keep = w > 1e-3
            if len(np.unique(true[keep])) < 2:
                pred[target] = true[target]
                assigned[target] = True
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   multi_class="multinomial", C=1.0))
            clf.fit(q_ref[keep], true[keep], logisticregression__sample_weight=w[keep])
            pred[target] = clf.predict(q_ref[target])
            assigned[target] = True
        m = evaluate(true[assigned], pred[assigned].astype(str), MAIN_LABELS)
        tag = f"LR_local_wt{sigma_t:g}"
        results.append({"method": tag, "sigma_t": sigma_t, **m})
        print(f"    {tag}:")
        _print(m, indent=6)

    # ---- Time-kernel sweep ----
    sigma_q_js = median_pairwise_q_distance(q_ref, metric="jensenshannon")
    sigma_q_l1 = median_pairwise_q_distance(q_ref, metric="cityblock")
    sigma_q_eu = median_pairwise_q_distance(q_ref, metric="euclidean")
    print(f"\nσ_q defaults (median pairwise q-dist): JS={sigma_q_js:.3f} "
          f"L1={sigma_q_l1:.3f} Eu={sigma_q_eu:.3f}")

    estimators = [
        ("raw_q_knn",      dict(likelihood="raw_q_knn", k_q=K_Q)),
        ("balanced_q_knn", dict(likelihood="balanced_q_knn", k_q=K_Q)),
        ("kernel_js",      dict(likelihood="kernel", q_metric="jensenshannon", sigma_q=sigma_q_js)),
        ("kernel_l1",      dict(likelihood="kernel", q_metric="cityblock", sigma_q=sigma_q_l1)),
        ("kernel_eu",      dict(likelihood="kernel", q_metric="euclidean", sigma_q=sigma_q_eu)),
        ("kernelbal_js",   dict(likelihood="kernel_balanced", q_metric="jensenshannon", sigma_q=sigma_q_js)),
        ("kernelbal_l1",   dict(likelihood="kernel_balanced", q_metric="cityblock", sigma_q=sigma_q_l1)),
        ("kernelbal_eu",   dict(likelihood="kernel_balanced", q_metric="euclidean", sigma_q=sigma_q_eu)),
    ]

    print("\n=== TIME-KERNEL SWEEP (uniform prior, drop_self=True) ===")
    for est_name, kw in estimators:
        for sigma_t in SIGMA_T_GRID:
            post = calibrate_q_time(
                query_q=q_ref, query_hpf=hpf,
                ref_q=q_ref, ref_hpf=hpf, ref_labels=true,
                labels=MAIN_LABELS, sigma_t=sigma_t, drop_self=True, **kw,
            )
            pred, _ = calibrated_predictions(post, MAIN_LABELS)
            marg = calibrated_margin(post)
            m = evaluate(true, pred, MAIN_LABELS, margins=marg)
            st = "inf" if not np.isfinite(sigma_t) else sigma_t
            results.append({"method": est_name, "sigma_t": st, **m})
            print(f"  {est_name:<16s} σ_t={str(st):<4s} "
                  f"macroF1={m['macro_f1']:.3f} balAcc={m['balanced_accuracy']:.3f} "
                  f"LtH_rec={m['recall[Low_to_High]']:.3f} "
                  f"Int_rec={m['recall[Intermediate]']:.3f} "
                  f"LtH->NP={m['LtH->NP_collapse']:.3f} "
                  f"NP->LtH={m['NP->LtH_falsecall']:.3f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "time_aware_benchmark_results.csv", index=False)
    print(f"\nSaved time_aware_benchmark_results.csv ({len(res_df)} rows)")

    # ---- Ranked summary ----
    print("\n" + "=" * 100)
    print("RANKED BY macro_f1, then balanced_accuracy")
    print("=" * 100)
    show = ["method", "sigma_t", "accuracy", "balanced_accuracy", "macro_f1",
            "recall[Low_to_High]", "recall[Intermediate]",
            "LtH->NP_collapse", "NP->LtH_falsecall"]
    ranked = res_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(ranked[show].round(3).to_string(index=False))


def _print(m, indent=2):
    pad = " " * indent
    print(f"{pad}accuracy {m['accuracy']:.3f}  balAcc {m['balanced_accuracy']:.3f}  "
          f"macroF1 {m['macro_f1']:.3f}  "
          f"LtH_rec {m['recall[Low_to_High]']:.3f}  Int_rec {m['recall[Intermediate]']:.3f}  "
          f"LtH->NP {m['LtH->NP_collapse']:.3f}  NP->LtH {m['NP->LtH_falsecall']:.3f}")


if __name__ == "__main__":
    main()
