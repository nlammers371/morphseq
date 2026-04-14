"""TFAP2 followup: trajectory condensation using classifier contrast scores.

Requires 02_run_all_pairs_classification.py to have been run first.

Runs two condensations per feature set:
  • supported_window  — rows filtered to the window from 01_support_table.py
                        (all genotypes have ≥3 embryos in every bin)
  • all_timepoints    — full time range, all embryos

Pipeline per (feature set, subset):
  1. Load raw_contrast_scores_long.parquet from all_pairs_classification/.
  2. Optionally filter to the supported time window.
  3. Pivot wide: rows = (embryo, time_bin), columns = comparison_id scores.
  4. Run trajectory condensation.
  5. Render time_slice.html.

Output layout:
  results/condensation/{feature_key}/{subset}/run/condensed_positions.npz
  figures/condensation/{feature_key}/{subset}/viz_genotype/time_slice.html

Run:
  conda run -n segmentation_grounded_sam --no-capture-output \\
      python results/mcolon/20260413_tfap2_followup/scripts/03_run_condensation.py

  Add --smoke for a fast end-to-end test (50 iters, 5 embryos/genotype).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="using precomputed metric", category=UserWarning)
warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)

import matplotlib
matplotlib.use("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_CACHE_ROOT = Path("/tmp") / "morphseq_20260413_tfap2_followup_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
for _name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
    Path(os.environ[_name]).mkdir(parents=True, exist_ok=True)

from analyze.trajectory_condensation import init_embedding, schema
from analyze.trajectory_condensation.condensation import CondensationConfig, StoppingConfig, run_condensation
from analyze.trajectory_condensation.viz.condensed_time_slice_viewer import time_slice_html
from analyze.viz.styling.color_utils import build_genotype_color_lookup

from common import load_supported_window

ID_COL = "embryo_id"
CLASS_COL = "genotype"
FEATURE_SET_SPECS = ["curvature", "embedding"]


# ---------------------------------------------------------------------------
# Condensation config (matches CEP290 production defaults)
# ---------------------------------------------------------------------------

def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def _condensation_config(*, n_iter: int, epsilon_r: float) -> CondensationConfig:
    return CondensationConfig(
        sigma=0.5,
        temporal_cohere_window=3,
        epsilon_r=float(epsilon_r),
        elastic_strength=16.0,
        elastic_mix=0.25,
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        void_strength=0.014,
        outlier_strength=16.0,
        outlier_cutoff_mode="robust",
        outlier_cutoff_value=3.0,
        attract_k=20,
        solver_lr=1e-4,
        solver_momentum=0.9,
        solver_max_iter=int(n_iter),
        coherence_cache_every=10,
    )


def _stopping_config() -> StoppingConfig:
    return StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--feature-sets", nargs="+", choices=FEATURE_SET_SPECS,
                   default=FEATURE_SET_SPECS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--epsilon-r", type=float, default=5e-4)
    p.add_argument("--smoke", action="store_true",
                   help="Fast test: 50 iters, 5 embryos/genotype.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pivot contrast scores → CondensationData
# ---------------------------------------------------------------------------

def _build_condensation_data(wide: pd.DataFrame) -> schema.CondensationData:
    """Convert pivoted wide scores DataFrame into a CondensationData tensor."""
    feature_cols = [c for c in wide.columns if c not in {ID_COL, "time_bin_center", CLASS_COL}]
    embryo_ids = np.array(sorted(wide[ID_COL].astype(str).unique()))
    time_values = np.array(sorted(wide["time_bin_center"].astype(float).unique()), dtype=float)
    n_e, t_count, k_count = len(embryo_ids), len(time_values), len(feature_cols)

    features = np.full((n_e, t_count, k_count), np.nan, dtype=float)
    mask = np.zeros((n_e, t_count), dtype=bool)
    labels = np.full(n_e, "", dtype=object)
    embryo_index = {str(e): i for i, e in enumerate(embryo_ids)}
    time_index = {float(t): j for j, t in enumerate(time_values)}

    for row in wide.itertuples(index=False):
        i = embryo_index[str(getattr(row, ID_COL))]
        j = time_index[float(row.time_bin_center)]
        vals = np.asarray([getattr(row, col) for col in feature_cols], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        features[i, j, :] = vals
        mask[i, j] = True
        label = str(getattr(row, CLASS_COL))
        if labels[i] == "":
            labels[i] = label
        elif labels[i] != label:
            raise ValueError(
                f"Label inconsistency for {getattr(row, ID_COL)!r}: {labels[i]!r} vs {label!r}"
            )

    data = schema.CondensationData(
        features=features,
        mask=mask,
        embryo_ids=embryo_ids,
        time_values=time_values,
        labels=labels,
        feature_names=feature_cols,
        embryo_index=embryo_index,
        time_index=time_index,
    )
    schema.validate(data, allow_feature_nans=True)
    return data


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render_view(
    *,
    positions: np.ndarray,
    x0: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    embryo_ids: np.ndarray,
    labels: np.ndarray,
    color_map: dict[str, str],
    output_dir: Path,
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_slice_html(
        positions, mask, time_values,
        labels=labels, color_map=color_map, embryo_ids=embryo_ids,
        output_path=output_dir / "time_slice.html", title=title,
    )


# ---------------------------------------------------------------------------
# Per-feature-set runner
# ---------------------------------------------------------------------------

def _run_feature_space(
    *,
    feature_key: str,
    subset_key: str,
    scores_long: pd.DataFrame,
    color_map: dict[str, str],
    run_dir: Path,
    seed: int,
    n_iter: int,
    save_every: int,
    epsilon_r: float,
) -> None:
    out_results = run_dir / "results" / "condensation" / feature_key / subset_key / "run"
    out_figures = run_dir / "figures" / "condensation" / feature_key / subset_key / "viz_genotype"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    # Filter to this feature set and pivot: rows=(embryo, time), cols=comparison scores
    fs_scores = scores_long[scores_long["feature_set"] == feature_key].copy()
    print(f"  {fs_scores['comparison_id'].nunique()} comparisons, "
          f"{fs_scores[ID_COL].nunique()} embryos, "
          f"{fs_scores['time_bin_center'].nunique()} time bins")

    wide = (
        fs_scores
        .pivot_table(
            index=[ID_COL, "time_bin_center", CLASS_COL],
            columns="comparison_id",
            values="class_signed_margin",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.sort_values([ID_COL, "time_bin_center"]).reset_index(drop=True)
    wide.to_parquet(out_results.parent.parent / "wide_scores.parquet", index=False)

    data = _build_condensation_data(wide)
    x0 = init_embedding.aligned_umap_init(data.features, data.mask, random_state=int(seed))

    print(f"  Running condensation ({n_iter} iters)...")
    result = run_condensation(
        x0=x0,
        mask=data.mask,
        config=_condensation_config(n_iter=int(n_iter), epsilon_r=float(epsilon_r)),
        stopping=_stopping_config(),
        log_every=max(1, int(n_iter) // 20),
        save_every=int(save_every) if int(save_every) > 0 else None,
        verbose=True,
    )

    payload = {
        "positions": result.positions, "x0": x0,
        "mask": data.mask, "time_values": data.time_values,
        "embryo_ids": data.embryo_ids, "labels": data.labels,
    }
    if result.position_history is not None:
        payload["position_history"] = result.position_history
        payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)
    np.savez(out_results / "condensed_positions.npz", **payload)
    pd.DataFrame(result.metrics_history).to_csv(out_results / "metrics.csv", index=False)

    present = sorted(set(data.labels.tolist()))
    viz_color_map = {k: v for k, v in color_map.items() if k in present}
    _render_view(
        positions=result.positions, x0=x0, mask=data.mask,
        time_values=data.time_values, embryo_ids=data.embryo_ids,
        labels=data.labels, color_map=viz_color_map,
        output_dir=out_figures,
        title=f"TFAP2 {feature_key} ({subset_key})",
    )
    print(f"  Done → {out_figures}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n_iter = 50
        args.save_every = min(args.save_every, 10)
        print("SMOKE MODE: n_iter=50, 5 embryos/genotype")

    run_dir = Path(__file__).resolve().parents[1]
    results_dir = run_dir / "results"
    scores_long_path = results_dir / "all_pairs_classification" / "raw_contrast_scores_long.parquet"

    if not scores_long_path.exists():
        raise FileNotFoundError(
            f"Contrast scores not found: {scores_long_path}\n"
            "Run 02_run_all_pairs_classification.py first."
        )

    # Load full contrast scores
    scores_all = pd.read_parquet(scores_long_path)

    # Smoke mode: subsample embryos (applied once, shared across both subsets)
    if args.smoke:
        _genotypes_all = sorted(scores_all[CLASS_COL].dropna().unique().tolist())
        keep: list[str] = []
        for gt, sub in scores_all.groupby(CLASS_COL, sort=True):
            eids = sorted(sub[ID_COL].astype(str).unique())
            keep.extend(eids[: min(5, len(eids))])
        scores_all = scores_all[scores_all[ID_COL].astype(str).isin(set(keep))].copy()
        print(f"Smoke: kept {scores_all[ID_COL].nunique()} embryos")

    # Supported-window subset
    window = load_supported_window(results_dir)
    t_min, t_max = float(window["t_min"]), float(window["t_max"])
    print(f"Supported window: {t_min} – {t_max} hpf")
    scores_window = scores_all[
        (scores_all["time_bin_center"] >= t_min) &
        (scores_all["time_bin_center"] <= t_max)
    ].copy()

    subsets = [
        ("supported_window", scores_window),
        ("all_timepoints",   scores_all),
    ]

    # Build color map from all genotypes across both subsets
    genotypes = sorted(scores_all[CLASS_COL].dropna().unique().tolist())
    print(f"Loaded {len(scores_all):,} rows (full), {scores_all[ID_COL].nunique()} embryos, "
          f"{len(genotypes)} genotypes")
    color_map = build_genotype_color_lookup(genotypes)

    for subset_key, scores_long in subsets:
        print(f"\n{'='*60}")
        print(f"Subset: {subset_key}  ({scores_long['time_bin_center'].nunique()} time bins, "
              f"{scores_long[ID_COL].nunique()} embryos)")
        for feature_key in args.feature_sets:
            if feature_key not in scores_long["feature_set"].unique():
                print(f"  Skipping {feature_key}: not in scores feature sets")
                continue
            print(f"\n  === Feature space: {feature_key} ===")
            _run_feature_space(
                feature_key=feature_key,
                subset_key=subset_key,
                scores_long=scores_long,
                color_map=color_map,
                run_dir=run_dir,
                seed=int(args.seed),
                n_iter=int(args.n_iter),
                save_every=int(args.save_every),
                epsilon_r=float(args.epsilon_r),
            )

    print(f"\nAll done. Results in: {results_dir / 'condensation'}")


if __name__ == "__main__":
    main()
