"""Axis-centering comparison through the standard trajectory-condensation flow.

Current policy:
- use the existing PBX-tuned condensation preset rather than raw package
  defaults, so runs remain comparable to the established PBX results-side
  condensation workflow

Follow-up if this works well:
- promote the PBX-tuned preset into the package-level default condensation path
- stop requiring results-side scripts to spell out the same
  ``CondensationConfig`` override repeatedly
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260410_axis_centering_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PBX_ANALYSIS_DIR = REPO_ROOT / "results" / "mcolon" / "20260407_pbx_analysis_cont"
PBX_ADDITIVITY_DIR = REPO_ROOT / "results" / "mcolon" / "20260409_pbx_additivity"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PBX_ANALYSIS_DIR))
sys.path.insert(0, str(PBX_ADDITIVITY_DIR))

from analyze.classification.directions.extract import extract_classifier_directions
from analyze.classification.engine.comparison_resolution import resolve_comparisons
from analyze.classification.engine.data_prep import _bin_and_aggregate, _build_binary_labels
from analyze.trajectory_condensation import init_embedding, schema
from analyze.trajectory_condensation.condensation import CondensationConfig, StoppingConfig, run_condensation
from analyze.trajectory_condensation.viz import api as tc_viz
from analyze.trajectory_condensation.viz import plotting as tc_plotting
from analyze.trajectory_condensation.viz.condensed_time_slice_viewer import time_slice_html
from phenotype_direction import (
    CENTERING_VARIANTS,
    center_metadata_row,
    compute_all_centered_scores,
)

from common import GENOTYPE_COLORS, SENSITIVITY_GENOTYPES, load_combined_pbx_dataframe


FEATURE_SET = "vae"
FEATURE_PREFIX = "z_mu_b"
TIME_COL = "stage_hpf"
CLASS_COL = "genotype"
ID_COL = "embryo_id"
BIN_WIDTH = 4.0

DEFAULT_COMPARISONS = "all_pairs"

DEFAULT_VARIANTS = list(CENTERING_VARIANTS)


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def _pbx_condensation_config(
    *,
    n_iter: int,
    epsilon_r: float,
    elastic_strength: float,
    elastic_mix: float,
    outlier_strength: float,
    cutoff_mode: str,
    cutoff_value: float,
) -> CondensationConfig:
    """Local PBX preset.

    We do not use raw package defaults here because the shared defaults are
    generic and differ substantially from the PBX trajectory settings used in
    the existing results-side condensation runners.

    If this preset performs well for the raw-direction centering workflow too,
    it should become the package-level default so callers can rely on
    ``CondensationConfig()`` without repeating results-side overrides.
    """
    return CondensationConfig(
        sigma=0.5,
        temporal_cohere_window=3,
        epsilon_r=float(epsilon_r),
        elastic_strength=float(elastic_strength),
        elastic_mix=float(elastic_mix),
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        void_strength=0.014,
        outlier_strength=float(outlier_strength),
        outlier_cutoff_mode=cutoff_mode,
        outlier_cutoff_value=float(cutoff_value),
        attract_k=20,
        solver_lr=1e-4,
        solver_momentum=0.9,
        solver_max_iter=int(n_iter),
    )


def _pbx_stopping_config() -> StoppingConfig:
    """Disable heuristic early stopping to match existing PBX runners."""
    return StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw-direction centered coordinate variants and run standard trajectory condensation."
    )
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--variants", nargs="+", choices=DEFAULT_VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--epsilon-r", type=float, default=5e-4)
    parser.add_argument("--elastic-strength", type=float, default=16.0)
    parser.add_argument("--elastic-mix", type=float, default=0.25)
    parser.add_argument("--outlier-strength", type=float, default=16.0)
    parser.add_argument("--outlier-cutoff-preset", choices=["q95", "q97", "q99", "robust3"], default="robust3")
    parser.add_argument("--skip-animations", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_cutoff_preset(name: str) -> tuple[str, float]:
    preset = str(name).strip().lower()
    if preset == "q95":
        return "quantile", 0.95
    if preset == "q97":
        return "quantile", 0.97
    if preset == "q99":
        return "quantile", 0.99
    if preset == "robust3":
        return "robust", 3.0
    raise ValueError(f"Unsupported outlier cutoff preset: {name!r}")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(c for c in df.columns if c.startswith(FEATURE_PREFIX))
    if not cols:
        raise ValueError(f"No feature columns found with prefix {FEATURE_PREFIX!r}")
    return cols


def _load_source_dataframe(*, smoke: bool) -> pd.DataFrame:
    df = load_combined_pbx_dataframe(genotypes=SENSITIVITY_GENOTYPES)
    if smoke:
        keep_ids: list[str] = []
        for genotype, sub in df.groupby(CLASS_COL, sort=True):
            embryo_ids = sorted(sub[ID_COL].astype(str).unique())
            keep_ids.extend(embryo_ids[: min(5, len(embryo_ids))])
        df = df[df[ID_COL].astype(str).isin(set(keep_ids))].copy()
    return df.reset_index(drop=True)


def _extract_directions(
    df: pd.DataFrame,
    *,
    comparisons: object,
    feature_cols: list[str],
    output_dir: Path,
) -> object:
    output_dir.mkdir(parents=True, exist_ok=True)
    return extract_classifier_directions(
        df=df,
        class_col=CLASS_COL,
        id_col=ID_COL,
        time_col=TIME_COL,
        comparisons=comparisons,
        features={FEATURE_SET: feature_cols},
        bin_width=BIN_WIDTH,
        min_samples_per_group=3,
        min_samples_per_member=2,
        random_state=42,
        class_weight="balanced",
        verbose=False,
        save_dir=output_dir,
        overwrite=True,
    )


def _build_variant_tables(
    df: pd.DataFrame,
    *,
    directions,
    comparisons: object,
    feature_cols: list[str],
    variants: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    available_labels = set(df[CLASS_COL].dropna().astype(str).unique())
    resolved = resolve_comparisons(
        positive=None,
        negative=None,
        comparisons=comparisons,
        available_labels=available_labels,
        class_col=CLASS_COL,
    )
    resolved_by_id = {rc.comparison_id: rc for rc in resolved}

    metadata = directions.metadata.copy()
    metadata = metadata[metadata["feature_set"] == FEATURE_SET].copy()
    metadata = metadata.sort_values(["comparison_id", "time_bin_center"]).reset_index(drop=True)
    if metadata.empty:
        raise ValueError(f"No direction metadata rows found for feature set {FEATURE_SET!r}")

    long_records: dict[str, list[dict[str, object]]] = {variant: [] for variant in variants}
    axis_meta_rows: list[dict[str, object]] = []
    diagnostics_frames: dict[str, list[pd.DataFrame]] = {variant: [] for variant in variants}

    for comparison_id, comp_meta in metadata.groupby("comparison_id", sort=False):
        rc = resolved_by_id.get(str(comparison_id))
        if rc is None:
            raise ValueError(f"Resolved comparison not found for metadata row {comparison_id!r}")

        labeled = _build_binary_labels(df, CLASS_COL, rc)
        binned = _bin_and_aggregate(labeled, ID_COL, TIME_COL, feature_cols, BIN_WIDTH)
        binned["genotype"] = np.where(
            binned["_y"].to_numpy(dtype=int) == 1,
            rc.positive_label,
            rc.negative_label,
        )

        for row in comp_meta.itertuples(index=False):
            axis = np.asarray(directions.vectors[str(row.vector_id)], dtype=float).ravel()
            sub = binned[np.isclose(binned["time_bin_center"], float(row.time_bin_center))].copy()
            if sub.empty:
                continue
            proj_bin = sub[[ID_COL, "time_bin_center", "genotype"]].copy()
            proj_bin["raw_score"] = sub[feature_cols].to_numpy(dtype=float) @ axis

            centered, center_stats = compute_all_centered_scores(
                proj_bin,
                intercept=float(row.intercept),
                coef_norm=float(row.coef_norm),
                pos_label=rc.positive_label,
                neg_label=rc.negative_label,
            )
            axis_meta_rows.append(
                center_metadata_row(
                    vector_id=str(row.vector_id),
                    comparison_id=str(row.comparison_id),
                    time_bin_center=float(row.time_bin_center),
                    time_bin=int(row.time_bin),
                    positive_label=rc.positive_label,
                    negative_label=rc.negative_label,
                    center_stats=center_stats,
                )
            )

            for variant in variants:
                diagnostics_frames[variant].append(
                    centered[[ID_COL, "time_bin_center", "genotype", "raw_score", variant]].assign(
                        variant=variant,
                        comparison_id=str(row.comparison_id),
                        vector_id=str(row.vector_id),
                    )
                )
                for entry in centered.itertuples(index=False):
                    long_records[variant].append(
                        {
                            "embryo_id": str(entry.embryo_id),
                            "time_bin_center": float(entry.time_bin_center),
                            "genotype": str(entry.genotype),
                            "vector_id": str(row.vector_id),
                            "score": float(getattr(entry, variant)),
                        }
                    )

    axis_metadata = pd.DataFrame(axis_meta_rows).sort_values(["comparison_id", "time_bin_center"]).reset_index(drop=True)
    wide_tables: dict[str, pd.DataFrame] = {}
    diagnostics_by_variant: dict[str, pd.DataFrame] = {}
    for variant in variants:
        long_df = pd.DataFrame(long_records[variant])
        if long_df.empty:
            raise ValueError(f"No long-form records generated for variant {variant!r}")
        wide = (
            long_df.pivot_table(
                index=["embryo_id", "time_bin_center", "genotype"],
                columns="vector_id",
                values="score",
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        feature_names = [c for c in wide.columns if c not in {"embryo_id", "time_bin_center", "genotype"}]
        wide = wide[["embryo_id", "time_bin_center", "genotype", *feature_names]].copy()
        wide_tables[variant] = wide.sort_values(["embryo_id", "time_bin_center"]).reset_index(drop=True)
        diagnostics_by_variant[variant] = pd.concat(diagnostics_frames[variant], ignore_index=True)
    return wide_tables, axis_metadata, diagnostics_by_variant


def _build_condensation_data(wide: pd.DataFrame) -> schema.CondensationData:
    feature_cols = [c for c in wide.columns if c not in {"embryo_id", "time_bin_center", "genotype"}]
    embryo_ids = np.array(sorted(wide["embryo_id"].astype(str).unique()))
    time_values = np.array(sorted(wide["time_bin_center"].astype(float).unique()), dtype=float)
    n_e, t_count, k_count = len(embryo_ids), len(time_values), len(feature_cols)

    features = np.full((n_e, t_count, k_count), np.nan, dtype=float)
    mask = np.zeros((n_e, t_count), dtype=bool)
    labels = np.full(n_e, "", dtype=object)
    embryo_index = {str(e): i for i, e in enumerate(embryo_ids)}
    time_index = {float(t): j for j, t in enumerate(time_values)}

    for row in wide.itertuples(index=False):
        i = embryo_index[str(row.embryo_id)]
        j = time_index[float(row.time_bin_center)]
        vals = np.asarray([getattr(row, col) for col in feature_cols], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        features[i, j, :] = vals
        mask[i, j] = True
        label = str(row.genotype)
        if labels[i] == "":
            labels[i] = label
        elif labels[i] != label:
            raise ValueError(
                f"Label inconsistency for embryo {row.embryo_id!r}: {labels[i]!r} vs {label!r}"
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


def _variant_summary(variant: str, wide: pd.DataFrame, axis_metadata: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in wide.columns if c not in {"embryo_id", "time_bin_center", "genotype"}]
    values = wide[feature_cols].to_numpy(dtype=float)
    frac_nonnan = float(np.isfinite(values).sum() / values.size) if values.size else 0.0
    per_feature_variance = (
        np.nanvar(values, axis=0).mean() if values.size and values.shape[1] else float("nan")
    )
    mean_embryos_per_bin = float(
        wide.groupby("time_bin_center")["embryo_id"].nunique().mean()
    ) if not wide.empty else 0.0
    low_n = ((axis_metadata["n_pos"] < 3) | (axis_metadata["n_neg"] < 3)).mean() if not axis_metadata.empty else 0.0
    return pd.DataFrame(
        [
            {
                "variant": variant,
                "n_embryos": int(wide["embryo_id"].nunique()),
                "n_timebins": int(wide["time_bin_center"].nunique()),
                "K_features": int(len(feature_cols)),
                "frac_nonnan": frac_nonnan,
                "mean_embryos_per_bin": mean_embryos_per_bin,
                "mean_abs_score": float(np.nanmean(np.abs(values))) if values.size else 0.0,
                "frac_bins_low_n": float(low_n),
                "per_feature_variance_mean": float(per_feature_variance),
            }
        ]
    )


def _plot_center_values(axis_metadata: pd.DataFrame, output_path: Path) -> None:
    comparisons = axis_metadata["comparison_id"].astype(str).unique().tolist()
    fig, axes = plt.subplots(
        len(comparisons),
        1,
        figsize=(8, max(3, 2.6 * len(comparisons))),
        squeeze=False,
        sharex=True,
    )
    for ax, comparison_id in zip(axes.ravel(), comparisons):
        sub = axis_metadata[axis_metadata["comparison_id"] == comparison_id].sort_values("time_bin_center")
        ax.plot(sub["time_bin_center"], sub["boundary_score"], label="boundary", lw=1.8)
        ax.plot(sub["time_bin_center"], sub["neg_mean"], label="neg_mean", lw=1.6)
        ax.plot(sub["time_bin_center"], sub["midpoint"], label="midpoint", lw=1.6)
        ax.set_title(comparison_id, fontsize=9)
        ax.set_ylabel("raw-score units", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend(frameon=False, fontsize=8, loc="best")
    axes.ravel()[-1].set_xlabel("time_bin_center (hpf)", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_score_distributions(diagnostics_by_variant: dict[str, pd.DataFrame], output_path: Path) -> None:
    all_df = pd.concat(diagnostics_by_variant.values(), ignore_index=True)
    bins = sorted(all_df["time_bin_center"].astype(float).unique())
    if not bins:
        return
    targets = [28.0, 48.0]
    selected = []
    for target in targets:
        selected.append(min(bins, key=lambda value: abs(value - target)))
    selected = list(dict.fromkeys(selected))
    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 4), squeeze=False)
    for ax, time_bin_center in zip(axes.ravel(), selected):
        sub = all_df[np.isclose(all_df["time_bin_center"], time_bin_center)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        positions = []
        data = []
        labels = []
        idx = 1
        for variant in diagnostics_by_variant:
            for genotype in sorted(sub["genotype"].astype(str).unique()):
                vals = sub[(sub["variant"] == variant) & (sub["genotype"] == genotype)][variant].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                positions.append(idx)
                data.append(vals)
                labels.append(f"{variant}\n{genotype}")
                idx += 1
        if data:
            ax.boxplot(data, positions=positions, widths=0.6, patch_artist=False)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{time_bin_center:.1f} hpf", fontsize=9)
        ax.set_ylabel("score", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_init_bundle(data: schema.CondensationData, x0: np.ndarray, output_dir: Path, title_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = tc_plotting.plot_trajectories(
        x0,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=GENOTYPE_COLORS,
        title=f"{title_prefix} init",
    )
    fig.savefig(output_dir / "plot_trajectories_init.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, _ = tc_plotting.plot_stacked_3d(
        x0,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=GENOTYPE_COLORS,
        title=f"{title_prefix} init stacked 3D",
    )
    fig.savefig(output_dir / "plot_stacked_3d_init.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_variant(
    *,
    variant: str,
    wide: pd.DataFrame,
    axis_metadata: pd.DataFrame,
    output_root: Path,
    n_iter: int,
    save_every: int,
    seed: int,
    epsilon_r: float,
    elastic_strength: float,
    elastic_mix: float,
    outlier_strength: float,
    cutoff_mode: str,
    cutoff_value: float,
    skip_animations: bool,
) -> tc_viz.RunDescriptor:
    variant_root = output_root / "results" / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    init_dir = variant_root / "init"
    run_dir = variant_root / "run"
    init_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    wide.to_parquet(variant_root / "wide_scores.parquet", index=False)
    axis_metadata.to_parquet(variant_root / "axis_metadata.parquet", index=False)
    _variant_summary(variant, wide, axis_metadata).to_csv(variant_root / "summary.csv", index=False)
    (variant_root / "data_manifest.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "bin_width": BIN_WIDTH,
                "feature_set": FEATURE_SET,
                "n_rows_wide": int(len(wide)),
                "n_vectors": int(len([c for c in wide.columns if c not in {"embryo_id", "time_bin_center", "genotype"}])),
            },
            indent=2,
        )
    )

    data = _build_condensation_data(wide)
    x0 = init_embedding.aligned_umap_init(data.features, data.mask, random_state=seed)
    np.savez(
        init_dir / "x0_init.npz",
        x0=x0,
        mask=data.mask,
        time_values=data.time_values,
        embryo_ids=data.embryo_ids,
        labels=data.labels,
    )
    _render_init_bundle(data, x0, init_dir, title_prefix=variant)

    config = _pbx_condensation_config(
        n_iter=int(n_iter),
        epsilon_r=float(epsilon_r),
        elastic_strength=float(elastic_strength),
        elastic_mix=float(elastic_mix),
        outlier_strength=float(outlier_strength),
        cutoff_mode=cutoff_mode,
        cutoff_value=float(cutoff_value),
    )
    stopping = _pbx_stopping_config()
    result = run_condensation(
        x0=x0,
        mask=data.mask,
        config=config,
        stopping=stopping,
        log_every=max(1, int(n_iter) // 20),
        save_every=int(save_every) if int(save_every) > 0 else None,
        verbose=True,
    )

    payload = {
        "positions": result.positions,
        "x0": x0,
        "mask": data.mask,
        "time_values": data.time_values,
        "embryo_ids": data.embryo_ids,
        "labels": data.labels,
    }
    if result.position_history is not None:
        payload["position_history"] = result.position_history
        payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)
    np.savez(run_dir / "condensed_positions.npz", **payload)
    pd.DataFrame(result.metrics_history).to_csv(run_dir / "metrics.csv", index=False)

    run = tc_viz.load_run(run_dir / "condensed_positions.npz", title=variant, color_map=GENOTYPE_COLORS)
    tc_viz.render_run(run, run_dir, title_prefix=variant, skip_animations=skip_animations)
    time_slice_html(
        run.positions,
        run.mask,
        run.time_values,
        labels=run.labels,
        color_map=GENOTYPE_COLORS,
        embryo_ids=run.embryo_ids,
        output_path=run_dir / "time_slice.html",
        title=variant,
    )
    return run


def _render_comparisons(runs: list[tc_viz.RunDescriptor], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = [runs]
    tc_viz.compare_run_grid(
        grid,
        mode="trajectories",
        output_path=output_dir / "compare_trajectories.png",
    )
    tc_viz.compare_run_grid(
        grid,
        mode="stacked_3d",
        output_path=output_dir / "compare_stacked_3d.png",
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_iter = 50
        args.save_every = min(args.save_every, 10)

    cutoff_mode, cutoff_value = parse_cutoff_preset(args.outlier_cutoff_preset)
    output_root = args.output_root.resolve()
    results_root = output_root / "results"
    figures_root = output_root / "figures"
    extracted_dir = results_root / "extracted_directions"
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    df = _load_source_dataframe(smoke=bool(args.smoke))
    feature_cols = _feature_columns(df)
    directions = _extract_directions(
        df,
        comparisons=DEFAULT_COMPARISONS,
        feature_cols=feature_cols,
        output_dir=extracted_dir,
    )
    wide_tables, axis_metadata, diagnostics_by_variant = _build_variant_tables(
        df,
        directions=directions,
        comparisons=DEFAULT_COMPARISONS,
        feature_cols=feature_cols,
        variants=[str(v) for v in args.variants],
    )

    axis_metadata.to_csv(figures_root / "axis_metadata_debug.csv", index=False)
    _plot_center_values(axis_metadata, figures_root / "center_values_diagnostic.png")
    _plot_score_distributions(diagnostics_by_variant, figures_root / "score_distributions.png")

    runs: list[tc_viz.RunDescriptor] = []
    for variant in args.variants:
        run = _run_variant(
            variant=str(variant),
            wide=wide_tables[str(variant)],
            axis_metadata=axis_metadata,
            output_root=output_root,
            n_iter=int(args.n_iter),
            save_every=int(args.save_every),
            seed=int(args.seed),
            epsilon_r=float(args.epsilon_r),
            elastic_strength=float(args.elastic_strength),
            elastic_mix=float(args.elastic_mix),
            outlier_strength=float(args.outlier_strength),
            cutoff_mode=cutoff_mode,
            cutoff_value=float(cutoff_value),
            skip_animations=bool(args.skip_animations),
        )
        runs.append(run)

    _render_comparisons(runs, figures_root)
    print(output_root)


if __name__ == "__main__":
    main()
