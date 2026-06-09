"""
perbin.py — per-bin label transfer (sibling of core.py; does not modify it).

Everything here is **per time bin**. There is no global model, no pooled fallback,
and no nearest-bin routing. This is the engine for the Sequencing Greenlight work
(see results/mcolon/20260607_sci_cilia_gene14_imaging_qc/PLAN.md, LOCKED DECISIONS).

Why a new module
----------------
``core.py`` aggregates one row per embryo across ALL time for CV
(``ref.groupby(group_col)[feature_cols].mean()``), which leaks across time bins and
hides per-bin class imbalance. ``core.py`` callers depend on its return schema, so it
is left untouched. This module owns a new two-layer contract and reuses the shared
``_make_pipeline`` (already ``class_weight="balanced"``), ``_aggregate_binned`` and
``add_time_bins`` helpers — no duplication.

Two-step API (same shape as core: fit -> read quality -> apply to query)
------------------------------------------------------------------------
    ref_model = prepare_reference_perbin(ref_df, feature_cols, ...)
    result    = transfer_labels_perbin(ref_model, query_df)

Substrate
---------
One row per ``(embryo, time_bin)`` via ``add_time_bins`` + ``_aggregate_binned``. An
embryo spanning many bins contributes one independent point to each bin it has images
in (prevents leakage).

CV (mode chosen explicitly by the caller — never inferred)
----------------------------------------------------------
``cv_mode="loeo"`` (default): leave-one-``cv_group_col``-out within each bin
(``cv_group_col="experiment_id"``). A bin with <=1 experiment, or <2 classes, cannot be
cross-validated -> it FAILS (recorded, never falls back; the message suggests kfold).
``cv_mode="kfold"``: stratified ``n_folds``-fold within each bin — for a single-experiment
reference (e.g. crispant) where LOEO-experiment is impossible. A query ``(embryo, bin)``
with no fitted bin model gets no prediction (missing support), never a pooled fallback.

Return contract (both functions emit this shape; ``config.schema_version == "1.0"``)
------------------------------------------------------------------------------------
    config{contract, schema_version, feature_cols, label_col, group_col, time_col,
           bin_width, cv_group_col}
    classes
    models{bin_models: {bin_center: pipeline}, bin_model_classes: {bin_center: [cls]}}
    per_bin{embryo_per_bin_prediction(df), support(df)}
    embryo_support{embryo_cross_bin_prediction(df), bin_coverage(df), n_bins_scored(int)}
    reference_performance{per_bin_metrics(df), per_bin_confusion(dict),
           embryo_support_precision(dict), embryo_support_recall(dict),
           embryo_support_confusion{matrix, labels}, transferability(dict)}
    missing_bins(df)        # bins with no model because CV could not run there
    missing_support(df)     # query (embryo, bin) rows with no exact-match model
    diagnostics{warnings, flags}

Grain of the two prediction frames (the names say it):
  - ``per_bin.embryo_per_bin_prediction`` — one row per (embryo, time_bin).
  - ``embryo_support.embryo_cross_bin_prediction`` — one row per embryo (aggregated).
  - ``embryo_support.bin_coverage`` — one row per embryo: which bins it had data in
    (``bins_present``), which were scorable (``bins_scored``), which were not
    (``bins_missing``).

``missing_bins`` = the time bins dropped because CV could not run (<=1 experiment or
<2 classes); distinct from ``missing_support`` (per query-embryo×bin with no model).

``prepare_reference_perbin`` fills ``reference_performance`` from the reference CV and
leaves ``missing_support`` empty. ``transfer_labels_perbin`` fills the query predictions,
top-level ``missing_support``, and carries ``models``, ``reference_performance`` and
``missing_bins`` through from the reference unchanged (the query has no ground truth).

This module does NO file I/O — saving the fitted ``ref_model`` is the caller's job.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

# Shared engine pieces — reuse, do not duplicate. _make_pipeline already uses
# class_weight="balanced"; PREC_WARN/REC_WARN keep the transferability thresholds
# in one place.
from .core import _make_pipeline, PREC_WARN, REC_WARN
from ..engine.data_prep import _aggregate_binned
from ...utils.binning import add_time_bins

SCHEMA_VERSION = "1.0"


# ── small helpers ────────────────────────────────────────────────────────────────

def _concat(rows: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Concat a list of frames; return an empty frame with `columns` if the list is empty."""
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=columns)


def _align_proba(
    proba: np.ndarray, model_classes: list, full_classes: list
) -> np.ndarray:
    """Expand a (n, len(model_classes)) proba matrix to (n, len(full_classes)), 0-filling.

    A per-bin model may have seen only a subset of the full class list; classes it never
    saw get probability 0. Column order follows `full_classes`.
    """
    out = np.zeros((proba.shape[0], len(full_classes)), dtype=float)
    idx = {c: j for j, c in enumerate(model_classes)}
    for k, c in enumerate(full_classes):
        if c in idx:
            out[:, k] = proba[:, idx[c]]
    return out


def _build_embryo_bin_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    time_col: str,
    bin_width: float,
    label_col: str | None = None,
    cv_group_col: str | None = None,
) -> pd.DataFrame:
    """One row per (embryo, bin): mean features + (optional) modal label + cv group.

    Aggregation is done WITHOUT label_col (so a mixed-label embryo-bin stays one row);
    the modal label per (embryo, bin) is merged back separately, mirroring core.py.
    """
    work = df.copy()
    if label_col is not None:
        work = work.dropna(subset=[label_col])
    work = work.reset_index(drop=True)

    binned = add_time_bins(work, time_col=time_col, bin_width=bin_width, bin_col="time_bin")
    eb = _aggregate_binned(
        binned, id_col=group_col, feature_cols=feature_cols,
        bin_col="time_bin", bin_width=bin_width,
    )  # columns: group_col, time_bin, time_bin_center, <mean features>

    if label_col is not None:
        modal = (
            binned.groupby([group_col, "time_bin"])[label_col]
            .agg(lambda s: s.mode().iloc[0])
            .reset_index()
        )
        eb = eb.merge(modal, on=[group_col, "time_bin"], how="left")

    if cv_group_col is not None:
        grp = (
            binned.groupby([group_col, "time_bin"])[cv_group_col]
            .first()
            .reset_index()
        )
        eb = eb.merge(grp, on=[group_col, "time_bin"], how="left")

    return eb


def _embryo_cross_bin_prediction(
    pred_df: pd.DataFrame, id_col: str, classes: list
) -> pd.DataFrame:
    """Per-embryo aggregate (one row per embryo): mean per-bin prob vectors -> argmax.

    ``bins_contributed`` lists the actual ``time_bin_center`` values that fed each
    embryo's aggregate, so coverage is explicit (not just a count).
    """
    prob_cols = [f"prob_{c}" for c in classes]
    out_cols = [id_col] + prob_cols + [
        "predicted_label", "top_probability", "argmax_margin",
        "n_bins_contributed", "bins_contributed",
    ]
    if pred_df.empty:
        return pd.DataFrame(columns=out_cols)

    g = pred_df.groupby(id_col)
    mean_p = g[prob_cols].mean()
    n_bins = g.size().rename("n_bins_contributed")
    bins_list = (
        g["time_bin_center"].agg(lambda s: sorted(s.unique()))
        .rename("bins_contributed")
    )

    P = mean_p.to_numpy()
    amax = P.argmax(axis=1)
    pred = np.array(classes)[amax]
    top1 = P[np.arange(len(P)), amax]
    srt = np.sort(P, axis=1)[:, ::-1]
    margin = srt[:, 0] - srt[:, 1] if P.shape[1] > 1 else srt[:, 0]

    out = mean_p.reset_index()
    out["predicted_label"] = pred
    out["top_probability"] = top1
    out["argmax_margin"] = margin
    out = out.merge(n_bins.reset_index(), on=id_col)
    out = out.merge(bins_list.reset_index(), on=id_col)
    return out[out_cols]


def _embryo_bin_coverage(
    all_eb: pd.DataFrame, scored_pred_df: pd.DataFrame, id_col: str,
    scored_centers: list[float],
) -> pd.DataFrame:
    """Per-embryo bin-coverage stats: which bins it had data in, which were scorable.

    One row per embryo. ``bins_present`` = every bin the embryo had images in;
    ``bins_scored`` = those that produced a prediction (bin was CV-scorable);
    ``bins_missing`` = present-but-not-scored (the bin failed CV / had no model).
    """
    out_cols = [id_col, "n_bins_present", "n_bins_scored", "n_bins_missing",
                "bins_present", "bins_scored", "bins_missing"]
    if all_eb.empty:
        return pd.DataFrame(columns=out_cols)

    present = (
        all_eb.groupby(id_col)["time_bin_center"]
        .agg(lambda s: sorted(s.unique())).rename("bins_present")
    )
    scored_set = set(scored_centers)
    rows = []
    for eid, bins_present in present.items():
        bins_scored = [b for b in bins_present if b in scored_set]
        bins_missing = [b for b in bins_present if b not in scored_set]
        rows.append(dict(
            **{id_col: eid},
            n_bins_present=len(bins_present), n_bins_scored=len(bins_scored),
            n_bins_missing=len(bins_missing),
            bins_present=bins_present, bins_scored=bins_scored, bins_missing=bins_missing,
        ))
    return pd.DataFrame(rows)[out_cols]


def _macro_over_bins(per_bin_metric: dict[float, dict]) -> dict:
    """Average each class's per-bin score across the bins where that class appeared."""
    acc: dict = defaultdict(list)
    for _, by_class in per_bin_metric.items():
        for c, v in by_class.items():
            acc[c].append(v)
    return {c: float(np.mean(vs)) for c, vs in acc.items()}


def _macro_confusion(per_bin_cm: dict[float, np.ndarray], classes: list) -> dict:
    """Mean of per-bin row-normalized confusion matrices over scored bins."""
    mats = list(per_bin_cm.values())
    if mats:
        matrix = np.nanmean(np.stack(mats), axis=0)
    else:
        matrix = np.zeros((len(classes), len(classes)), dtype=float)
    return {"matrix": matrix, "labels": list(classes)}


def _transferability(cb_prec: dict, cb_rec: dict, classes: list) -> dict:
    """ok/warn/skip per class from the macro-over-bins precision/recall."""
    flags = {}
    for c in classes:
        p = cb_prec.get(c, 0.0)
        r = cb_rec.get(c, 0.0)
        if p >= PREC_WARN and r >= REC_WARN:
            flags[c] = "ok"
        elif p < PREC_WARN and r < REC_WARN:
            flags[c] = "skip"
        else:
            flags[c] = "warn"
    return flags


# ── canonical empty-frame columns ────────────────────────────────────────────────

def _pred_columns(group_col: str, classes: list, *, query: bool) -> list[str]:
    base = [group_col, "time_bin", "time_bin_center"]
    if not query:
        base += ["cv_group", "true_label"]
    base += ["predicted_label"]
    return base + [f"prob_{c}" for c in classes]


_SUPPORT_COLS = ["time_bin_center", "n_embryos", "n_experiments", "status"]
_FAILED_COLS = ["time_bin_center", "n_embryos", "n_experiments", "reason"]
_METRIC_COLS = ["time_bin_center", "class", "precision", "recall", "n_embryos", "n_experiments"]
_MISSING_COLS = ["query_embryo_id", "time_bin", "time_bin_center"]


# ══════════════════════════════════════════════════════════════════════════════════
# Step 1 — prepare_reference_perbin
# ══════════════════════════════════════════════════════════════════════════════════

def prepare_reference_perbin(
    ref_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "cluster_categories",
    group_col: str = "embryo_id",
    time_col: str = "predicted_stage_hpf",
    bin_width: float = 4.0,
    cv_mode: str = "loeo",
    cv_group_col: str = "experiment_id",
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> dict:
    """Fit per-bin reference models and assess label quality via per-bin CV.

    ``cv_mode`` is chosen explicitly by the caller (no silent inference):
      - ``"loeo"`` (default): leave-one-``cv_group_col``-out within each bin (e.g.
        ``cv_group_col="experiment_id"``). A bin with <=1 group cannot be evaluated -> it
        FAILS, and the failure message suggests ``cv_mode="kfold"``. This is the honest
        analog of transfer (the query is always a new experiment); use it whenever the
        reference spans >=2 experiments.
      - ``"kfold"``: stratified ``n_folds``-fold within each bin. Use when the reference has
        a single experiment (e.g. crispant), so leave-one-experiment-out is impossible. A
        bin FAILS if it has <2 classes or any class with fewer than ``n_folds`` members.

    Everything is per-bin; there is no global-model path. See the module docstring for the
    full return contract.
    """
    from sklearn.model_selection import StratifiedKFold

    if cv_mode not in ("loeo", "kfold"):
        raise ValueError(f"cv_mode must be 'loeo' or 'kfold', got {cv_mode!r}.")
    if cv_mode == "loeo" and cv_group_col not in ref_df.columns:
        raise ValueError(f"cv_group_col={cv_group_col!r} not found in ref_df columns.")
    if cv_mode == "loeo" and ref_df[cv_group_col].nunique() < 2:
        raise ValueError(
            f"cv_mode='loeo' needs >=2 distinct {cv_group_col} values, but found "
            f"{ref_df[cv_group_col].nunique()}. This reference cannot do "
            f"leave-one-{cv_group_col}-out — pass cv_mode='kfold' instead."
        )

    ref = ref_df.dropna(subset=[label_col]).copy().reset_index(drop=True)
    classes = sorted(ref[label_col].dropna().unique())

    eb = _build_embryo_bin_frame(
        ref, feature_cols, group_col, time_col, bin_width,
        label_col=label_col,
        cv_group_col=(cv_group_col if cv_mode == "loeo" else None),
    )

    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    support_rows: list[dict] = []
    failed_rows: list[dict] = []
    confusion_by_bin: dict[float, dict] = {}

    per_bin_prec: dict[float, dict] = {}
    per_bin_rec: dict[float, dict] = {}
    per_bin_cm: dict[float, np.ndarray] = {}
    scored_centers: list[float] = []

    for bc in sorted(eb["time_bin_center"].unique()):
        sub = eb[eb["time_bin_center"] == bc]
        n_emb = len(sub)
        n_exp = int(sub[cv_group_col].nunique()) if cv_mode == "loeo" else np.nan
        y = sub[label_col].to_numpy()
        n_cls = int(len(np.unique(y)))
        min_class_count = int(pd.Series(y).value_counts().min()) if n_cls else 0

        # ── FAIL conditions: never fall back ──────────────────────────────────────
        if cv_mode == "loeo":
            # leave-one-group-out: need >=2 groups and >=2 classes
            fail_reason = (
                f"<=1 {cv_group_col} (try cv_mode='kfold')" if n_exp <= 1
                else "<2 classes" if n_cls < 2 else None
            )
        else:
            # stratified k-fold: need >=2 classes and each class >= n_folds members
            fail_reason = ("<2 classes" if n_cls < 2
                           else f"<{n_folds} per class for {n_folds}-fold"
                           if min_class_count < n_folds else None)
        if fail_reason is not None:
            failed_rows.append(dict(time_bin_center=bc, n_embryos=n_emb,
                                    n_experiments=n_exp, reason=fail_reason))
            support_rows.append(dict(time_bin_center=bc, n_embryos=n_emb,
                                     n_experiments=n_exp, status="failed"))
            if verbose:
                print(f"[perbin] bin {bc}: FAILED ({fail_reason}; n_emb={n_emb}, n_exp={n_exp})")
            continue

        X = sub[feature_cols].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cv_mode == "loeo":
                cv = LeaveOneGroupOut()
                groups = sub[cv_group_col].to_numpy()
                y_pred = cross_val_predict(_make_pipeline(random_state), X, y,
                                           groups=groups, cv=cv)
                proba = cross_val_predict(_make_pipeline(random_state), X, y,
                                          groups=groups, cv=cv, method="predict_proba")
            else:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=random_state)
                y_pred = cross_val_predict(_make_pipeline(random_state), X, y, cv=cv)
                proba = cross_val_predict(_make_pipeline(random_state), X, y,
                                          cv=cv, method="predict_proba")
        bin_classes = list(np.unique(y))  # sorted -> matches proba column order
        proba_full = _align_proba(proba, bin_classes, classes)

        if cv_mode == "loeo":
            block = sub[[group_col, "time_bin", "time_bin_center", cv_group_col]].copy()
            block = block.rename(columns={cv_group_col: "cv_group"})
        else:
            block = sub[[group_col, "time_bin", "time_bin_center"]].copy()
            block["cv_group"] = np.nan
        block["true_label"] = y
        block["predicted_label"] = y_pred
        for j, c in enumerate(classes):
            block[f"prob_{c}"] = proba_full[:, j]
        pred_rows.append(block)

        bin_prec, bin_rec = {}, {}
        for c in classes:
            n_c = int((y == c).sum())
            if n_c == 0:
                continue
            p = precision_score(y, y_pred, labels=[c], average="macro", zero_division=0)
            r = recall_score(y, y_pred, labels=[c], average="macro", zero_division=0)
            bin_prec[c], bin_rec[c] = p, r
            metric_rows.append(dict(time_bin_center=bc, **{"class": c},
                                    precision=round(p, 4), recall=round(r, 4),
                                    n_embryos=n_c, n_experiments=n_exp))
        per_bin_prec[bc] = bin_prec
        per_bin_rec[bc] = bin_rec

        cm = confusion_matrix(y, y_pred, labels=classes, normalize="true")
        confusion_by_bin[bc] = {"matrix": cm, "labels": list(classes)}
        per_bin_cm[bc] = cm

        support_rows.append(dict(time_bin_center=bc, n_embryos=n_emb,
                                 n_experiments=n_exp, status="scored"))
        scored_centers.append(bc)
        if verbose:
            print(f"[perbin] bin {bc}: scored (n_emb={n_emb}, n_exp={n_exp})")

    # ── final per-bin models: fit on ALL ref rows in each scored bin ──────────────
    bin_models: dict[float, Any] = {}
    bin_model_classes: dict[float, list] = {}
    for bc in scored_centers:
        sub = eb[eb["time_bin_center"] == bc].dropna(subset=[label_col])
        y = sub[label_col].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = _make_pipeline(random_state)
            pipe.fit(sub[feature_cols].to_numpy(dtype=float), y)
        bin_models[bc] = pipe
        bin_model_classes[bc] = list(pipe.classes_)

    predictions_df = _concat(pred_rows, _pred_columns(group_col, classes, query=False))

    # ── cross-bin prediction + per-embryo coverage (embryo_support layer) ─────────
    cross_bin_df = _embryo_cross_bin_prediction(predictions_df, group_col, classes)
    coverage_df = _embryo_bin_coverage(eb, predictions_df, group_col, scored_centers)
    cb_prec = _macro_over_bins(per_bin_prec)
    cb_rec = _macro_over_bins(per_bin_rec)
    cb_conf = _macro_confusion(per_bin_cm, classes)
    transferability = _transferability(cb_prec, cb_rec, classes)

    failed_df = (_concat([pd.DataFrame(failed_rows)], _FAILED_COLS)
                 if failed_rows else pd.DataFrame(columns=_FAILED_COLS))

    # ── diagnostics ───────────────────────────────────────────────────────────────
    warnings_list: list[str] = []
    if not scored_centers:
        warnings_list.append("No bins could be scored (every bin had <=1 experiment or <2 classes).")
    for c, flag in transferability.items():
        if flag == "skip":
            warnings_list.append(
                f"'{c}': cross-bin precision={cb_prec.get(c, 0.0):.2f}, "
                f"recall={cb_rec.get(c, 0.0):.2f} — below transferability threshold."
            )
        elif flag == "warn":
            warnings_list.append(
                f"'{c}': cross-bin precision={cb_prec.get(c, 0.0):.2f}, "
                f"recall={cb_rec.get(c, 0.0):.2f} — marginal separability."
            )
    for w in warnings_list:
        warnings.warn(w, UserWarning, stacklevel=2)

    return dict(
        config=dict(
            contract="perbin", schema_version=SCHEMA_VERSION,
            feature_cols=feature_cols, label_col=label_col, group_col=group_col,
            time_col=time_col, bin_width=bin_width,
            cv_mode=cv_mode, cv_group_col=cv_group_col, n_folds=n_folds,
        ),
        classes=classes,
        models=dict(bin_models=bin_models, bin_model_classes=bin_model_classes),
        per_bin=dict(
            embryo_per_bin_prediction=predictions_df,
            support=_concat([pd.DataFrame(support_rows)], _SUPPORT_COLS)
            if support_rows else pd.DataFrame(columns=_SUPPORT_COLS),
        ),
        embryo_support=dict(
            embryo_cross_bin_prediction=cross_bin_df,
            bin_coverage=coverage_df,
            n_bins_scored=len(scored_centers),
        ),
        reference_performance=dict(
            per_bin_metrics=_concat([pd.DataFrame(metric_rows)], _METRIC_COLS)
            if metric_rows else pd.DataFrame(columns=_METRIC_COLS),
            per_bin_confusion=confusion_by_bin,
            embryo_support_precision=cb_prec,
            embryo_support_recall=cb_rec,
            embryo_support_confusion=cb_conf,
            transferability=transferability,
        ),
        # bins absent from the model because CV could not run there (<=1 experiment
        # or <2 classes). Distinct from query missing_support (per query-embryo×bin).
        missing_bins=failed_df,
        missing_support=pd.DataFrame(columns=_MISSING_COLS),
        diagnostics=dict(warnings=warnings_list, flags=transferability),
    )


# ══════════════════════════════════════════════════════════════════════════════════
# Step 2 — transfer_labels_perbin
# ══════════════════════════════════════════════════════════════════════════════════

def transfer_labels_perbin(
    ref_model: dict,
    query_df: pd.DataFrame,
    verbose: bool = False,
) -> dict:
    """Apply per-bin reference models to query embryos (exact bin-center match only).

    A query (embryo, bin) with no matching bin model is reported under top-level
    ``missing_support`` and excluded from that embryo's cross-bin aggregate — never a
    silent pooled fallback. ``models`` and ``reference_performance`` are carried through
    from ``ref_model`` unchanged (the query has no ground truth).
    """
    if ref_model.get("config", {}).get("contract") != "perbin":
        raise ValueError("ref_model is not a perbin contract (config.contract != 'perbin').")

    cfg = ref_model["config"]
    feature_cols = cfg["feature_cols"]
    group_col = cfg["group_col"]
    time_col = cfg["time_col"]
    bin_width = cfg["bin_width"]
    classes = ref_model["classes"]
    bin_models = ref_model["models"]["bin_models"]
    bin_model_classes = ref_model["models"]["bin_model_classes"]

    qeb = _build_embryo_bin_frame(
        query_df, feature_cols, group_col, time_col, bin_width,
        label_col=None, cv_group_col=None,
    )

    pred_rows: list[pd.DataFrame] = []
    support_rows: list[dict] = []
    missing_rows: list[dict] = []

    for bc in sorted(qeb["time_bin_center"].unique()):
        sub = qeb[qeb["time_bin_center"] == bc]
        model = bin_models.get(bc)  # EXACT match only
        if model is None:
            for _, row in sub.iterrows():
                missing_rows.append(dict(
                    query_embryo_id=row[group_col],
                    time_bin=row["time_bin"], time_bin_center=bc,
                ))
            if verbose:
                print(f"[perbin-transfer] bin {bc}: NO model -> {len(sub)} embryo(s) missing support")
            continue

        X = sub[feature_cols].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = model.predict_proba(X)
        proba_full = _align_proba(proba, bin_model_classes.get(bc, list(model.classes_)), classes)

        block = sub[[group_col, "time_bin", "time_bin_center"]].copy()
        amax = proba_full.argmax(axis=1)
        block["predicted_label"] = np.array(classes)[amax]
        for j, c in enumerate(classes):
            block[f"prob_{c}"] = proba_full[:, j]
        pred_rows.append(block)
        support_rows.append(dict(time_bin_center=bc, n_embryos=len(sub),
                                 n_experiments=np.nan, status="scored"))

    predictions_df = _concat(pred_rows, _pred_columns(group_col, classes, query=True))
    predictions_df = predictions_df.rename(columns={group_col: "query_embryo_id"})

    cross_bin_df = _embryo_cross_bin_prediction(predictions_df, "query_embryo_id", classes)

    # Per query-embryo coverage: which bins it had data in, which got predicted.
    # qeb is per-(query embryo, bin); rename id col so coverage is keyed on query_embryo_id.
    qeb_named = qeb.rename(columns={group_col: "query_embryo_id"})
    scored_q_centers = (sorted(predictions_df["time_bin_center"].unique())
                        if not predictions_df.empty else [])
    coverage_df = _embryo_bin_coverage(qeb_named, predictions_df, "query_embryo_id",
                                        scored_q_centers)

    return dict(
        config=cfg,
        classes=classes,
        models=ref_model["models"],
        per_bin=dict(
            embryo_per_bin_prediction=predictions_df,
            support=_concat([pd.DataFrame(support_rows)], _SUPPORT_COLS)
            if support_rows else pd.DataFrame(columns=_SUPPORT_COLS),
        ),
        embryo_support=dict(
            embryo_cross_bin_prediction=cross_bin_df,
            bin_coverage=coverage_df,
            n_bins_scored=len(scored_q_centers),
        ),
        reference_performance=ref_model["reference_performance"],
        # Carried from the reference: bins with no model because CV could not run there.
        missing_bins=ref_model.get("missing_bins", pd.DataFrame(columns=_FAILED_COLS)),
        # Query-specific: (query embryo, bin) rows with no exact-match model.
        missing_support=_concat([pd.DataFrame(missing_rows)], _MISSING_COLS)
        if missing_rows else pd.DataFrame(columns=_MISSING_COLS),
        diagnostics=ref_model["diagnostics"],
    )
