from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flagging import (
    compute_confusion_enrichment,
    compute_time_localization,
    flag_consistently_misclassified,
)
from .io import (
    infer_class_labels_from_predictions,
    load_embryo_predictions,
    load_stage1_metadata,
    validate_stage2_inputs,
)
from .metrics import compute_per_embryo_metrics
from .null import (
    STREAK_SPEC,
    TOP_CONFUSED_SPEC,
    WRONG_RATE_SPEC,
    null_test_streak,
    null_test_top_confused_frac,
    null_test_wrong_rate,
)
from ..viz.misclassification import (
    plot_confusion_profile,
    plot_flagged_embryo_gallery,
    plot_wrong_rate_distributions,
    plot_wrongness_heatmap,
)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import false_discovery_control as _bh

        return _bh(pvals)
    except Exception:
        try:
            from statsmodels.stats.multitest import fdrcorrection

            return fdrcorrection(pvals, alpha=0.05, method="indep")[1]
        except Exception as exc:
            raise ImportError("Need SciPy>=1.11 or statsmodels for BH-FDR") from exc


def run_misclassification_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Run Stage 2 misclassification analysis on archived Stage 1 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    stage1_meta = None
    class_labels_source = "metadata"
    try:
        stage1_meta = load_stage1_metadata(input_dir)
        class_labels = list(stage1_meta.class_labels)
    except FileNotFoundError:
        class_labels = None
        class_labels_source = "inferred_union"

    df = load_embryo_predictions(input_dir)
    if class_labels is None:
        class_labels = infer_class_labels_from_predictions(df)

    validate_stage2_inputs(df, class_labels=class_labels)

    per_embryo, baseline_ct, baseline_c = compute_per_embryo_metrics(
        df,
        allow_mode_true_class=bool(config.get("allow_mode_true_class", False)),
    )

    idx_map = per_embryo.set_index("embryo_id")["embryo_idx"].to_dict()
    df["embryo_idx"] = df["embryo_id"].astype(str).map(idx_map).astype(int)

    per_embryo, run_wrong = null_test_wrong_rate(
        embryo_predictions=df,
        per_embryo_metrics=per_embryo,
        class_labels=class_labels,
        n_permutations=int(config.get("n_permutations", 1000)),
        random_state=int(config.get("random_state", 42)),
    )

    embryo_time_bins = df[["embryo_id", "time_bin"]].drop_duplicates()
    per_embryo, run_streak = null_test_streak(
        per_embryo_metrics=per_embryo,
        baseline_ct_df=baseline_ct,
        embryo_time_bins=embryo_time_bins,
        n_sim=int(config.get("n_sim", 10000)),
        random_state=int(config.get("random_state", 42)) + 1,
    )

    per_embryo, run_top = null_test_top_confused_frac(
        per_embryo_metrics=per_embryo,
        embryo_predictions=df,
        class_labels=class_labels,
        n_sim=int(config.get("n_sim", 10000)),
        random_state=int(config.get("random_state", 42)) + 2,
        require_n_wrong_min=int(config.get("require_n_wrong_min", 3)),
        loo_min_class_size=int(config.get("loo_min_class_size", 10)),
    )

    per_embryo["qval_wrong_rate"] = _bh_fdr(per_embryo["pval_wrong_rate"].to_numpy(dtype=float))
    per_embryo["qval_streak"] = _bh_fdr(per_embryo["pval_streak"].to_numpy(dtype=float))
    per_embryo["qval_top_confused_frac"] = _bh_fdr(
        per_embryo["pval_top_confused_frac"].to_numpy(dtype=float)
    )

    per_embryo_flagged = flag_consistently_misclassified(
        per_embryo,
        q_val_threshold=float(config.get("q_val_threshold", 0.05)),
        wrong_rate_z_threshold=float(config.get("wrong_rate_z_threshold", 2.0)),
        wrong_rate_delta_threshold=float(config.get("wrong_rate_delta_threshold", 0.20)),
        top_confused_frac_threshold=float(config.get("top_confused_frac_threshold", 0.80)),
        require_n_windows_min=int(config.get("require_n_windows_min", 3)),
        require_n_wrong_min=int(config.get("require_n_wrong_min", 3)),
    )

    flagged_only = per_embryo_flagged[per_embryo_flagged["is_flagged"]].copy()

    confusion_enrichment = compute_confusion_enrichment(df, flagged_only)
    time_localization = compute_time_localization(
        df,
        flagged_only,
        rolling_window_bins=int(config.get("rolling_window_bins", 3)),
        rolling_threshold=float(config.get("rolling_threshold", 0.60)),
    )

    # Write tables
    per_embryo_flagged.to_csv(tables_dir / "per_embryo_metrics.csv", index=False)
    per_embryo_flagged.to_csv(tables_dir / "per_embryo_null_pvals.csv", index=False)
    flagged_only.to_csv(tables_dir / "flagged_embryos.csv", index=False)
    confusion_enrichment.to_csv(tables_dir / "confusion_enrichment.csv", index=False)
    time_localization.to_csv(tables_dir / "time_localization.csv", index=False)
    baseline_ct.to_csv(tables_dir / "baseline_wrong_rate_by_class_time.csv", index=False)
    baseline_c.to_csv(tables_dir / "baseline_wrong_rate_by_class.csv", index=False)

    # Plots
    try:
        plot_wrongness_heatmap(df, per_embryo_flagged, plots_dir)
        plot_wrong_rate_distributions(per_embryo_flagged, plots_dir)
        plot_confusion_profile(df, flagged_only, plots_dir)
        plot_flagged_embryo_gallery(df, flagged_only, plots_dir, top_n=int(config.get("top_n_gallery", 20)))
    except Exception:
        # Keep pipeline robust in headless/test envs; tables are primary deliverable.
        pass

    metadata = {
        "null_specs": {
            "wrong_rate": WRONG_RATE_SPEC,
            "streak": STREAK_SPEC,
            "top_confused": TOP_CONFUSED_SPEC,
        },
        "n_permutations": int(config.get("n_permutations", 1000)),
        "n_sim": int(config.get("n_sim", 10000)),
        "random_state": int(config.get("random_state", 42)),
        "class_labels_source": class_labels_source,
        "class_labels": class_labels,
        "require_n_windows_min": int(config.get("require_n_windows_min", 3)),
        "require_n_wrong_min": int(config.get("require_n_wrong_min", 3)),
        "q_val_threshold": float(config.get("q_val_threshold", 0.05)),
        "wrong_rate_z_threshold": float(config.get("wrong_rate_z_threshold", 2.0)),
        "wrong_rate_delta_threshold": float(config.get("wrong_rate_delta_threshold", 0.20)),
        "top_confused_frac_threshold": float(config.get("top_confused_frac_threshold", 0.80)),
        "loo_min_class_size": int(config.get("loo_min_class_size", 10)),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(),
        "schema_version": "misclassification_v1",
        "stage1_schema_version": stage1_meta.schema_version if stage1_meta else "",
        "lite_runs": {
            "wrong_rate": run_wrong.spec,
            "streak": run_streak.spec,
            "top_confused": run_top.spec,
        },
        "q_C_per_class": run_top.metadata.get("q_C_per_class", {}),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {
        "per_embryo_metrics": per_embryo_flagged,
        "flagged_embryos": flagged_only,
        "confusion_enrichment": confusion_enrichment,
        "time_localization": time_localization,
        "baseline_ct_df": baseline_ct,
        "baseline_c_df": baseline_c,
    }
