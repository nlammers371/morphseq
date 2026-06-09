"""
label_transfer.py
=================
Two-step label transfer pipeline. All predictions and quality metrics are
at the embryo level — image-level predictions are intermediate only.

Public API
----------
    ref_model = prepare_reference(ref_df, feature_cols, ...)
    result    = transfer_labels(ref_model, query_df)

    # quality-only mode (no query)
    ref_model = prepare_reference(ref_df, feature_cols, ...)
    plot_reference_quality(ref_model)

    # convenience one-liner
    out = run_label_transfer(ref_df, feature_cols, query_df=None, ...)
    # out is ref_model if query_df is None, else {"reference": ..., "transfer": ...}

Bundle schema (ref_model)
--------------------------
    config        : dict  — feature_cols, label_col, group_col, time_col, bin_width
    classes       : list  — sorted label list from reference
    final_model   : dict  — {"A": pipeline, "C": pipeline} trained on all ref data
    quality_report: dict  — per-class precision/recall by time bin, confusion matrix,
                            transferability flag, n_embryos (from LOEO CV)
    label_profile : dict  — class centroids, time distributions, purity
    diagnostics   : dict  — per-class warnings, transferability flags

Transferability flags
---------------------
    "ok"      — precision ≥ PREC_WARN and recall ≥ REC_WARN at full coverage
    "warn"    — one of precision or recall is marginal
    "skip"    — both are below threshold; label should not be transferred
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

# Shared binning/aggregation engine — same logic run_classification uses internally.
# Future improvement: run_classification should expose its fitted per-bin pipelines
# so label_transfer can reuse them directly instead of refitting here.
from ..engine.data_prep import _aggregate_binned
from ...utils.binning import add_time_bins

# ── thresholds ─────────────────────────────────────────────────────────────────
PREC_WARN = 0.50   # per-class precision floor for "ok" flag
REC_WARN  = 0.30   # per-class recall floor for "ok" flag
MIN_BIN_EMBRYOS = 3
MIN_BIN_EMBRYOS_PERBIN = 5  # minimum embryos per bin to fit a dedicated bin model

# ── model spec (matches classification module) ─────────────────────────────────
def _make_pipeline(random_state: int = 42) -> Any:
    return make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000, solver="liblinear",
                class_weight="balanced", random_state=random_state,
            )
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — prepare_reference
# ══════════════════════════════════════════════════════════════════════════════

def prepare_reference(
    ref_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "cluster_categories",
    group_col: str = "embryo_id",
    time_col: str = "predicted_stage_hpf",
    bin_width: float = 4.0,
    cv_group_col: str | None = None,
    n_folds: int = 5,
    random_state: int = 42,
    model_type: str = "global",
) -> dict:
    """Fit reference models and assess label quality via cross-validation.

    Parameters
    ----------
    ref_df       : Labeled reference images. Must have feature_cols, label_col,
                   group_col, time_col.
    feature_cols : Numeric embedding columns.
    label_col    : Phenotype label column.
    group_col    : Embryo identifier column (used for aggregation).
    time_col     : HPF column.
    bin_width    : Width of time bins in hpf.
    cv_group_col : Column to use for leave-one-group-out CV.
                   - Pass "embryo_id" (or group_col) → leave-one-embryo-out.
                   - Pass "experiment_id" → leave-one-experiment-out (recommended
                     when multiple experiments are present).
                   - Pass None → k-fold CV (n_folds used).
    n_folds      : Number of folds when cv_group_col is None (k-fold).
    random_state : Sklearn random seed.
    model_type   : "global" (default) — one model fit on all reference embryos.
                   "per_bin" — one model per time bin (bin_width) using embryo-mean
                   features; falls back to the global model for bins with fewer than
                   MIN_BIN_EMBRYOS_PERBIN embryos or fewer than 2 classes.
                   Per-bin is Mode C from the benchmark in
                   results/mcolon/20260601_label_transfer_method/.

    Returns
    -------
    ref_model dict — see module docstring for schema.
    """
    from sklearn.model_selection import StratifiedKFold

    ref = ref_df.dropna(subset=[label_col]).copy().reset_index(drop=True)
    classes = sorted(ref[label_col].dropna().unique())

    # Derive CV strategy label for display
    if cv_group_col is None:
        cv_strategy_label = f"{n_folds}-fold CV"
    elif cv_group_col == group_col:
        cv_strategy_label = f"leave-one-{group_col}-out"
    else:
        cv_strategy_label = f"leave-one-{cv_group_col}-out"

    if model_type not in ("global", "per_bin"):
        raise ValueError(f"model_type must be 'global' or 'per_bin', got {model_type!r}")

    config = dict(
        feature_cols=feature_cols, label_col=label_col, group_col=group_col,
        time_col=time_col, bin_width=bin_width,
        cv_group_col=cv_group_col, cv_strategy=cv_strategy_label,
        model_type=model_type,
    )

    # ── embryo-level aggregation for CV ──────────────────────────────────────
    emb = ref.groupby(group_col)[feature_cols].mean().reset_index()
    modal_label = (
        ref.groupby(group_col)[label_col]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    emb = emb.merge(modal_label, on=group_col)
    emb_time = ref.groupby(group_col)[time_col].median().reset_index()
    emb = emb.merge(emb_time, on=group_col)

    X_emb = emb[feature_cols].to_numpy(dtype=float)
    y_emb = emb[label_col].to_numpy()

    # CV splitter + groups
    if cv_group_col is None:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        groups = None
    else:
        if cv_group_col not in ref.columns:
            raise ValueError(f"cv_group_col='{cv_group_col}' not found in ref_df columns.")
        cv = LeaveOneGroupOut()
        grp_per_embryo = ref.groupby(group_col)[cv_group_col].first()
        groups = emb[group_col].map(grp_per_embryo).to_numpy()

    pipe = _make_pipeline(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_cv = cross_val_predict(
            pipe, X_emb, y_emb,
            groups=groups, cv=cv,
        )

    # ── per-class quality at embryo level ─────────────────────────────────────
    quality_per_class = {}
    for lbl in classes:
        prec = precision_score(y_emb, y_pred_cv, labels=[lbl], average="macro", zero_division=0)
        rec  = recall_score(y_emb, y_pred_cv, labels=[lbl], average="macro", zero_division=0)
        f1   = f1_score(y_emb, y_pred_cv, labels=[lbl], average="macro", zero_division=0)
        n_emb = int((y_emb == lbl).sum())
        if prec >= PREC_WARN and rec >= REC_WARN:
            flag = "ok"
        elif prec < PREC_WARN and rec < REC_WARN:
            flag = "skip"
        else:
            flag = "warn"
        quality_per_class[lbl] = dict(
            precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
            n_embryos=n_emb, flag=flag,
        )

    # ── per-class quality by time bin ─────────────────────────────────────────
    hpf_min = float(np.floor(emb[time_col].min() / bin_width) * bin_width)
    emb["_time_bin"] = (
        (emb[time_col] - hpf_min) // bin_width * bin_width + hpf_min + bin_width / 2
    )
    quality_by_timebin: dict[str, list[dict]] = {lbl: [] for lbl in classes}
    for b in sorted(emb["_time_bin"].unique()):
        mask = emb["_time_bin"] == b
        if mask.sum() < MIN_BIN_EMBRYOS:
            continue
        yt = y_emb[mask]
        yp = y_pred_cv[mask]
        for lbl in classes:
            if (yt == lbl).sum() == 0:
                continue
            quality_by_timebin[lbl].append(dict(
                time_bin=b,
                precision=round(precision_score(yt, yp, labels=[lbl], average="macro", zero_division=0), 4),
                recall=round(recall_score(yt, yp, labels=[lbl], average="macro", zero_division=0), 4),
                n_embryos=int((yt == lbl).sum()),
            ))

    # ── confusion matrix (embryo level, recall-normalized) ────────────────────
    cm = confusion_matrix(y_emb, y_pred_cv, labels=classes, normalize="true")

    # ── final model: fit on all reference embryos (global fallback) ──────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe = _make_pipeline(random_state)
        final_pipe.fit(X_emb, y_emb)

    # ── per-bin models (Mode C) ───────────────────────────────────────────────
    # Uses within-bin embryo aggregation via the same _aggregate_binned helper
    # that run_classification uses internally — one row per (embryo, bin) rather
    # than one row per embryo averaged across all time. This is the principled
    # approach: an embryo spanning 20–80 hpf contributes independently to each
    # bin it has images in, with features computed only from images in that window.
    bin_models: dict[float, Any] = {}  # bin_center → fitted pipeline
    if model_type == "per_bin":
        # Build within-bin embryo means from raw image rows using shared engine
        ref_binned = add_time_bins(ref, time_col=time_col, bin_width=bin_width, bin_col="_time_bin")
        # attach label for groupby (label_col is modal per embryo within bin)
        ref_binned["_label_for_agg"] = ref_binned[label_col]
        emb_binned = _aggregate_binned(
            ref_binned, id_col=group_col, feature_cols=feature_cols,
            bin_col="_time_bin", bin_width=bin_width,
        )
        # re-attach modal label per (embryo, bin)
        modal_per_bin = (
            ref_binned.groupby([group_col, "_time_bin"])[label_col]
            .agg(lambda s: s.mode().iloc[0])
            .reset_index()
        )
        emb_binned = emb_binned.merge(modal_per_bin, on=[group_col, "_time_bin"], how="left")

        for bin_center in sorted(emb_binned["time_bin_center"].unique()):
            sub = emb_binned[emb_binned["time_bin_center"] == bin_center].dropna(subset=[label_col])
            y_bin = sub[label_col].to_numpy()
            n_classes_bin = len(np.unique(y_bin))
            if len(sub) >= MIN_BIN_EMBRYOS_PERBIN and n_classes_bin >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe_bin = _make_pipeline(random_state)
                    pipe_bin.fit(sub[feature_cols].to_numpy(dtype=float), y_bin)
                bin_models[bin_center] = pipe_bin

    # ── label profile: class centroids + time distributions ───────────────────
    label_profile = {}
    for lbl in classes:
        mask = y_emb == lbl
        label_profile[lbl] = dict(
            centroid=X_emb[mask].mean(axis=0),
            n_embryos=int(mask.sum()),
            hpf_median=float(emb.loc[mask, time_col].median()),
            hpf_q25=float(emb.loc[mask, time_col].quantile(0.25)),
            hpf_q75=float(emb.loc[mask, time_col].quantile(0.75)),
        )

    # ── diagnostics / warnings ────────────────────────────────────────────────
    diagnostics = {"warnings": [], "flags": {}}
    for lbl, q in quality_per_class.items():
        diagnostics["flags"][lbl] = q["flag"]
        if q["flag"] == "skip":
            diagnostics["warnings"].append(
                f"'{lbl}': precision={q['precision']:.2f}, recall={q['recall']:.2f} — "
                f"below transferability threshold. Label will be excluded from transfer."
            )
        elif q["flag"] == "warn":
            diagnostics["warnings"].append(
                f"'{lbl}': precision={q['precision']:.2f}, recall={q['recall']:.2f} — "
                f"marginal separability. Transfer results may be unreliable."
            )

    for w in diagnostics["warnings"]:
        warnings.warn(w, UserWarning, stacklevel=2)

    return dict(
        config=config,
        classes=classes,
        final_model=final_pipe,
        bin_models=bin_models,   # empty dict when model_type="global"
        quality_report=dict(
            per_class=quality_per_class,
            by_timebin=quality_by_timebin,
            confusion_matrix=cm,
            confusion_labels=classes,
            cv_strategy=cv_strategy_label,
            n_embryos_total=len(emb),
            balanced_accuracy=round(balanced_accuracy_score(y_emb, y_pred_cv), 4),
            macro_f1=round(f1_score(y_emb, y_pred_cv, average="macro", zero_division=0), 4),
        ),
        label_profile=label_profile,
        diagnostics=diagnostics,
        _emb_df=emb,           # keep for plotting; prefixed _ = internal
        _feature_cols=feature_cols,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — transfer_labels
# ══════════════════════════════════════════════════════════════════════════════

def transfer_labels(
    ref_model: dict,
    query_df: pd.DataFrame,
    skip_flagged: bool = True,
) -> dict:
    """Transfer reference labels to query embryos.

    Parameters
    ----------
    ref_model    : Output of prepare_reference().
    query_df     : Query images. Must have feature_cols, group_col, time_col.
                   Labels are never read from query_df.
    skip_flagged : If True (default), classes flagged "skip" in the quality
                   report are excluded from the argmax prediction — the model
                   only chooses among transferable classes.

    Returns
    -------
    dict with:
        embryo_predictions : DataFrame — one row per query embryo.
            query_embryo_id, predicted_label, top_probability, argmax_margin,
            status ("assigned"/"warn"/"skip_flag"), per-class probabilities.
        image_predictions  : DataFrame — one row per query image with
            per-class probabilities and argmax label (intermediate, for diagnostics).
        skipped_classes    : list of class names excluded from argmax.
    """
    cfg = ref_model["config"]
    feature_cols = cfg["feature_cols"]
    group_col    = cfg["group_col"]
    time_col     = cfg["time_col"]
    bin_width    = cfg["bin_width"]
    model_type   = cfg.get("model_type", "global")
    classes      = ref_model["classes"]
    pipe         = ref_model["final_model"]
    bin_models   = ref_model.get("bin_models", {})
    flags        = ref_model["diagnostics"]["flags"]

    qry = query_df.copy().reset_index(drop=True)
    X_qry = qry[feature_cols].to_numpy(dtype=float)

    # ── image-level probabilities ──────────────────────────────────────────────
    # per_bin: route each image to its time-bin model; fall back to global
    if model_type == "per_bin" and bin_models:
        bin_centers = np.array(sorted(bin_models.keys()))
        hpf_vals = qry[time_col].to_numpy(dtype=float)
        proba = np.zeros((len(qry), len(classes)), dtype=float)
        bin_used = np.full(len(qry), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, hpf in enumerate(hpf_vals):
                nearest = bin_centers[int(np.argmin(np.abs(bin_centers - hpf)))]
                chosen = bin_models.get(nearest, pipe)
                p = chosen.predict_proba(X_qry[i : i + 1])
                # align to full class list (bin model may have seen fewer classes)
                p_full = np.zeros(len(classes))
                for j, cls in enumerate(classes):
                    if cls in chosen.classes_:
                        p_full[j] = p[0, list(chosen.classes_).index(cls)]
                proba[i] = p_full
                bin_used[i] = nearest
        qry = qry.copy()
        qry["_bin_used"] = bin_used
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = pipe.predict_proba(X_qry)   # (n_images, n_classes)
        qry["_bin_used"] = np.nan

    # Mask out skipped classes before argmax
    skipped = [c for c, f in flags.items() if f == "skip"] if skip_flagged else []
    active_idx = [i for i, c in enumerate(classes) if c not in skipped]
    active_classes = [classes[i] for i in active_idx]

    proba_active = proba[:, active_idx]
    argmax_idx = np.argmax(proba_active, axis=1)
    argmax_labels = np.array(active_classes)[argmax_idx]
    top1 = proba_active[np.arange(len(proba_active)), argmax_idx]
    sorted_proba = np.sort(proba_active, axis=1)[:, ::-1]
    margin = sorted_proba[:, 0] - sorted_proba[:, 1] if proba_active.shape[1] > 1 else sorted_proba[:, 0]

    img_df = pd.DataFrame({
        "query_snip_id": qry.get("snip_id", pd.Series(range(len(qry)))),
        group_col: qry[group_col],
        time_col: qry[time_col],
        "argmax_label": argmax_labels,
        "argmax_margin": margin,
        "top_probability": top1,
        "bin_used": qry["_bin_used"].to_numpy(),
    })
    for i, c in enumerate(classes):
        img_df[f"prob_{c}"] = proba[:, i]

    # ── embryo-level rollup: mean probabilities → argmax ──────────────────────
    prob_cols = [f"prob_{c}" for c in classes]
    emb_proba = (
        img_df.groupby(group_col)[prob_cols].mean().reset_index()
    )
    emb_proba_active = emb_proba[[f"prob_{c}" for c in active_classes]].to_numpy()
    emb_argmax_idx = np.argmax(emb_proba_active, axis=1)
    emb_labels = np.array(active_classes)[emb_argmax_idx]
    emb_top1 = emb_proba_active[np.arange(len(emb_proba_active)), emb_argmax_idx]
    emb_sorted = np.sort(emb_proba_active, axis=1)[:, ::-1]
    emb_margin = emb_sorted[:, 0] - emb_sorted[:, 1] if emb_proba_active.shape[1] > 1 else emb_sorted[:, 0]

    # consistency: fraction of images agreeing with embryo argmax
    img_df2 = img_df.merge(
        pd.DataFrame({group_col: emb_proba[group_col], "_emb_pred": emb_labels}),
        on=group_col, how="left"
    )
    consistency = (
        img_df2.groupby(group_col)
        .apply(lambda g: (g["argmax_label"] == g["_emb_pred"]).mean())
        .reset_index(name="consistency_score")
    )

    # modal bin used per embryo (for per_bin p-value lookup downstream)
    bin_modal = (
        img_df.groupby(group_col)["bin_used"]
        .agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().size > 0 else np.nan)
        .reset_index()
        .rename(columns={"bin_used": "bin_used"})
    )

    emb_df = emb_proba[[group_col] + prob_cols].copy()
    emb_df = emb_df.merge(bin_modal, on=group_col, how="left")
    emb_df["predicted_label"] = emb_labels
    emb_df["top_probability"] = emb_top1
    emb_df["argmax_margin"] = emb_margin
    emb_df = emb_df.rename(columns={group_col: "query_embryo_id"})
    emb_df = emb_df.merge(
        consistency.rename(columns={group_col: "query_embryo_id"}),
        on="query_embryo_id", how="left"
    )

    # image count per embryo
    n_img = img_df.groupby(group_col).size().reset_index(name="n_images")
    n_img = n_img.rename(columns={group_col: "query_embryo_id"})
    emb_df = emb_df.merge(n_img, on="query_embryo_id", how="left")

    # status
    def _status(row):
        if row["predicted_label"] in skipped:
            return "skip_flag"
        if row["argmax_margin"] < 0.1 or row["consistency_score"] < 0.5:
            return "warn"
        return "assigned"
    emb_df["status"] = emb_df.apply(_status, axis=1)

    return dict(
        embryo_predictions=emb_df,
        image_predictions=img_df,
        skipped_classes=skipped,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plots — public
# ══════════════════════════════════════════════════════════════════════════════

_CLASS_COLORS = {
    "Low_to_High":  "#4DAC26",
    "High_to_Low":  "#D6604D",
    "Intermediate": "#8073AC",
    "Not Penetrant":"#2166AC",
}
_FLAG_SUFFIX = {"ok": " [ok]", "warn": " [warn]", "skip": " [skip]"}


def plot_reference_quality(
    ref_model: dict,
    save_dir: str | None = None,
) -> list:
    """Generate reference quality figures and return them as a list.

    Returns
    -------
    [fig_timebin, fig_confusion]
        fig_timebin  : precision & recall by time bin, all classes on one plot
        fig_confusion: recall-normalised confusion matrix

    If save_dir is provided figures are saved to save_dir/label_transfer/.
    Caller owns the Figure objects.
    """
    import matplotlib
    matplotlib.use("Agg")

    figs = [
        _plot_timebin_quality(ref_model),
        _plot_confusion_matrix(ref_model),
    ]
    names = ["reference_quality_timebin.png", "reference_confusion_matrix.png"]
    if save_dir:
        import os
        out = os.path.join(save_dir, "label_transfer")
        os.makedirs(out, exist_ok=True)
        for fig, name in zip(figs, names):
            path = os.path.join(out, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")
    return figs


def plot_transfer_result(
    ref_model: dict,
    result: dict,
    save_dir: str | None = None,
) -> list:
    """Embryo-level transfer summary: label counts + margin distributions.

    Returns
    -------
    [fig_summary]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb     = result["embryo_predictions"]
    classes = ref_model["classes"]
    skipped = result["skipped_classes"]
    active  = [c for c in classes if c not in skipped]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: predicted label counts
    ax = axes[0]
    counts = emb["predicted_label"].value_counts()
    vals   = [counts.get(c, 0) for c in active]
    colors = [_CLASS_COLORS.get(c, "#888") for c in active]
    bars   = ax.bar(range(len(active)), vals, color=colors,
                    edgecolor="k", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(active)))
    ax.set_xticklabels(active, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Number of embryos")
    ax.set_title("Predicted label distribution")
    if skipped:
        ax.set_xlabel(f"Excluded (skip): {', '.join(skipped)}", fontsize=8)

    # Right: margin histogram per label
    ax = axes[1]
    for lbl in active:
        sub = emb[emb["predicted_label"] == lbl]["argmax_margin"]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=15, alpha=0.5,
                color=_CLASS_COLORS.get(lbl, "#888"),
                label=lbl, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Argmax margin (p_top1 − p_top2)")
    ax.set_ylabel("Embryo count")
    ax.set_title("Prediction confidence by label")
    ax.legend(fontsize=8)

    sc = emb["status"].value_counts().to_dict()
    fig.suptitle(
        f"Transfer result — {len(emb)} embryos  |  "
        f"assigned={sc.get('assigned', 0)}  warn={sc.get('warn', 0)}  "
        f"skip_flag={sc.get('skip_flag', 0)}",
        fontsize=10,
    )
    plt.tight_layout()

    figs = [fig]
    if save_dir:
        import os
        out = os.path.join(save_dir, "label_transfer")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "transfer_result_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    return figs


# ══════════════════════════════════════════════════════════════════════════════
# Plots — private helpers
# ══════════════════════════════════════════════════════════════════════════════

def _plot_timebin_quality(ref_model: dict):
    """1×2 figure: left=precision, right=recall, all classes as lines.

    Twin y-axis (secondary, right) shows total n_embryos per time bin
    as light gray bars — makes sparse bins visually obvious without
    cluttering the main lines.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    qr      = ref_model["quality_report"]
    classes = ref_model["classes"]
    cfg     = ref_model["config"]

    # Collect all time bins and total n_embryos per bin across all classes
    bin_totals: dict[float, int] = {}
    for lbl in classes:
        for r in qr["by_timebin"].get(lbl, []):
            b = r["time_bin"]
            bin_totals[b] = bin_totals.get(b, 0) + r["n_embryos"]
    all_bins = sorted(bin_totals)

    bar_vals = [bin_totals.get(b, 0) for b in all_bins]
    max_bar  = max(bar_vals) if bar_vals else 1
    bar_ylim = max_bar / 0.35   # bars reach ≤0.35 on primary [0,1] axis

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    legend_handles = []   # built once from the first panel, shared across figure

    for col, (metric_key, metric_title) in enumerate(
        [("precision", "Precision"), ("recall", "Recall")]
    ):
        ax = axes[col]
        ax2 = ax.twinx()

        ax2.bar(all_bins, bar_vals, width=cfg["bin_width"] * 0.8,
                color="lightgray", alpha=0.45, zorder=1)
        ax2.set_ylim(0, bar_ylim)
        ax2.set_ylabel("n embryos per bin", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelsize=7, colors="gray")

        for lbl in classes:
            rows = qr["by_timebin"].get(lbl, [])
            if not rows:
                continue
            bins_   = [r["time_bin"] for r in rows]
            vals_   = [r[metric_key] for r in rows]
            color   = _CLASS_COLORS.get(lbl, "#555555")
            # warn if either precision OR recall is marginal/skip
            flag    = qr["per_class"][lbl]["flag"]
            p_val   = qr["per_class"][lbl]["precision"]
            r_val   = qr["per_class"][lbl]["recall"]
            flag_str = " [warn]" if flag in ("warn", "skip") else " [ok]"
            line, = ax.plot(bins_, vals_, "o-", color=color, lw=2, ms=5,
                            alpha=0.85, zorder=3)
            if col == 0:   # collect handles only once
                legend_handles.append((line, f"{lbl}{flag_str}  P={p_val:.2f} R={r_val:.2f}"))

        ax.axhline(PREC_WARN, ls=":", color="gray", lw=1, alpha=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"{cfg['time_col']} bin center (hpf)", fontsize=9)
        ax.set_ylabel(metric_title, fontsize=10)
        ax.set_title(metric_title, fontsize=11)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(8))

    # single shared legend, anchored to right of figure
    handles, labels = zip(*legend_handles) if legend_handles else ([], [])
    fig.legend(
        handles, labels,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        borderaxespad=0,
        frameon=True,
        title="class [flag]  P  R",
        title_fontsize=7,
    )

    fig.suptitle(
        f"Reference label quality — {cfg['cv_strategy']}\n"
        f"bal_acc={qr['balanced_accuracy']:.3f}  macro_F1={qr['macro_f1']:.3f}  "
        f"n_embryos={qr['n_embryos_total']}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    return fig


def _plot_confusion_matrix(ref_model: dict):
    """Recall-normalised confusion matrix at embryo level."""
    import matplotlib.pyplot as plt

    qr     = ref_model["quality_report"]
    cm     = np.array(qr["confusion_matrix"])
    labels = qr["confusion_labels"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(
        "Confusion matrix — embryo-level CV\n(recall-normalised: rows sum to 1)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="Fraction of true-class embryos")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if v > 0.55 else "black")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def run_label_transfer(
    ref_df: pd.DataFrame,
    feature_cols: list[str],
    query_df: pd.DataFrame | None = None,
    **kwargs,
) -> dict:
    """Convenience wrapper around prepare_reference + transfer_labels.

    Returns ref_model if query_df is None, else
    {"reference": ref_model, "transfer": transfer_result}.
    """
    ref_model = prepare_reference(ref_df, feature_cols, **kwargs)
    if query_df is None:
        return ref_model
    return {
        "reference": ref_model,
        "transfer": transfer_labels(ref_model, query_df),
    }
