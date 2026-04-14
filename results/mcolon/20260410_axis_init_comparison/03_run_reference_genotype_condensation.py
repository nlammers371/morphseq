"""CEP290 reference genotype trajectory-condensation test for curvature and VAE.

This script:
1) Loads the CEP290 reference dataset with valid phenotype labels.
2) Filters to 24-120 hpf and excludes cep290_unknown.
3) Derives curvature from baseline_deviation_normalized.
4) Extracts genotype all-vs-rest classifier directions for one or more feature spaces.
5) Projects embryo-time observations onto those direction vectors.
6) Runs trajectory condensation on the projected score tables.
7) Renders two views of the same condensed run:
   - canonical genotype colors
   - genotype colors with homozygous embryos recolored by phenotype subtype

Run with the repo-mandated interpreter:
  /net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/conda run -n segmentation_grounded_sam --no-capture-output \
      python results/mcolon/20260302_NWDB_talk_figures_analysis/02_run_reference_genotype_condensation.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="using precomputed metric", category=UserWarning)
warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)
import pandas as pd


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260302_nwdb_reference_condensation_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

import matplotlib

matplotlib.use("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from analyze.classification.directions.extract import extract_classifier_directions
from analyze.classification.engine.comparison_resolution import resolve_comparisons
from analyze.classification.engine.data_prep import _bin_and_aggregate, _build_binary_labels
from analyze.trajectory_condensation import init_embedding, schema
from analyze.trajectory_condensation.condensation import CondensationConfig, StoppingConfig, run_condensation
from analyze.trajectory_condensation.viz import api as tc_viz
from analyze.trajectory_condensation.viz.condensed_time_slice_viewer import time_slice_html
from analyze.viz.styling.color_mapping_config import GENOTYPE_COLORS


GENOTYPES = [
    "cep290_wildtype",
    "cep290_heterozygous",
    "cep290_homozygous",
]
PHENOTYPE_COLORS = {
    "High_to_Low": "#E76FA2",
    "Low_to_High": "#2FB7B0",
}
CEP290_GENOTYPE_COLORS = {
    "cep290_wildtype": GENOTYPE_COLORS["wildtype"],
    "cep290_heterozygous": GENOTYPE_COLORS["heterozygous"],
    "cep290_homozygous": GENOTYPE_COLORS["homozygous"],
}
TIME_COL = "predicted_stage_hpf"
ID_COL = "embryo_id"
CLASS_COL = "genotype"
PHENOTYPE_COL = "cluster_categories"
FEATURE_SET_SPECS: dict[str, list[str] | str] = {
    "curvature": ["curvature"],
    "embedding": "z_mu_b",
}


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    out_dir: Path
    results_dir: Path
    figures_dir: Path
    cache_dir: Path
    directions_dir: Path


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run genotype all-vs-rest classifier-direction condensation for CEP290 reference data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument(
        "--data-dir",
        default="results/mcolon/20251229_cep290_phenotype_extraction/final_data",
        help="Directory containing embryo_data_with_labels.csv and embryo_cluster_labels.csv.",
    )
    p.add_argument("--t-min", type=float, default=10.0)
    p.add_argument("--t-max", type=float, default=120.0)
    p.add_argument("--bin-width", type=float, default=4.0)
    p.add_argument(
        "--feature-sets",
        nargs="+",
        choices=sorted(FEATURE_SET_SPECS.keys()),
        default=["curvature", "embedding"],
        help="Feature spaces to process under one script.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--epsilon-r", type=float, default=5e-4)
    p.add_argument("--skip-animations", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Paths:
    out_dir = Path(args.out_dir).resolve()
    data_dir = (_PROJECT_ROOT / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    results_dir = out_dir / "results" / "reference_genotype_condensation"
    figures_dir = out_dir / "figures" / "reference_genotype_condensation"
    cache_dir = figures_dir / "cache"
    directions_dir = results_dir / "classifier_directions"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    directions_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        project_root=_PROJECT_ROOT,
        data_dir=data_dir,
        out_dir=out_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
        cache_dir=cache_dir,
        directions_dir=directions_dir,
    )


def _load_reference_df(data_dir: Path, *, t_min: float, t_max: float, smoke: bool) -> pd.DataFrame:
    from analyze.utils.stats import normalize_arbitrary_feature

    labels_path = data_dir / "embryo_cluster_labels.csv"
    data_path = data_dir / "embryo_data_with_labels.csv"

    labels_valid = pd.read_csv(labels_path, usecols=["embryo_id", PHENOTYPE_COL], low_memory=False)
    labels_valid = labels_valid.drop_duplicates(subset="embryo_id")
    labels_valid = labels_valid[labels_valid[PHENOTYPE_COL].notna()].copy()
    labels_valid["embryo_id"] = labels_valid["embryo_id"].astype(str)

    keep_ids = set(labels_valid["embryo_id"].tolist())

    def usecols(col: str) -> bool:
        if col in {"embryo_id", CLASS_COL, TIME_COL, "baseline_deviation_normalized", PHENOTYPE_COL}:
            return True
        return col.startswith("z_mu_b")

    df = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    df["embryo_id"] = df["embryo_id"].astype(str)
    df = df[df["embryo_id"].isin(keep_ids)].copy()
    df = df[df[TIME_COL].notna()].copy()
    df = df[(df[TIME_COL] >= float(t_min)) & (df[TIME_COL] <= float(t_max))].copy()
    df = df[df[CLASS_COL].astype(str).isin(GENOTYPES)].copy()

    # Drop homozygous embryos that were never assigned a phenotype label —
    # they would appear as unlabeled red points in the viz.
    homo_mask = df[CLASS_COL].astype(str) == "cep290_homozygous"
    df = df[~(homo_mask & df[PHENOTYPE_COL].isna())].copy()

    df["curvature"] = normalize_arbitrary_feature(
        df["baseline_deviation_normalized"],
        low=0,
        high_percentile=100,
        clip=False,
    )
    df.loc[df[PHENOTYPE_COL] == "Intermediate", PHENOTYPE_COL] = "Low_to_High"

    if smoke:
        keep: list[str] = []
        for genotype, sub in df.groupby(CLASS_COL, sort=True):
            embryo_ids = sorted(sub[ID_COL].astype(str).unique())
            keep.extend(embryo_ids[: min(5, len(embryo_ids))])
        df = df[df[ID_COL].astype(str).isin(set(keep))].copy()

    return df.reset_index(drop=True)


def _feature_columns(df: pd.DataFrame, feature_spec: str | list[str]) -> list[str]:
    if isinstance(feature_spec, str):
        cols = sorted(c for c in df.columns if c.startswith(feature_spec))
        if not cols:
            raise ValueError(f"No feature columns found for prefix {feature_spec!r}")
        return cols
    missing = [c for c in feature_spec if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return list(feature_spec)


def _extract_directions(
    df: pd.DataFrame,
    *,
    feature_key: str,
    feature_cols: list[str],
    bin_width: float,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    return extract_classifier_directions(
        df=df,
        class_col=CLASS_COL,
        id_col=ID_COL,
        time_col=TIME_COL,
        comparisons="all_pairs",
        positive=GENOTYPES,
        features={feature_key: feature_cols},
        bin_width=float(bin_width),
        min_samples_per_group=3,
        min_samples_per_member=2,
        random_state=42,
        class_weight="balanced",
        verbose=False,
        save_dir=output_dir,
        overwrite=True,
    )


def _build_wide_scores(
    df: pd.DataFrame,
    *,
    directions,
    feature_key: str,
    feature_cols: list[str],
    bin_width: float,
    class_col: str,
    id_col: str,
    time_col: str,
    positive=None,   # subset comparisons to these labels (all_pairs within them)
    negative=None,   # subset comparisons to these labels
    comparisons=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_labels = set(df[class_col].dropna().astype(str).unique())
    resolved = resolve_comparisons(
        positive=positive,
        negative=negative,
        comparisons=comparisons,
        available_labels=available_labels,
        class_col=class_col,
    )
    resolved_by_id = {rc.comparison_id: rc for rc in resolved}

    metadata = directions.metadata.copy()
    metadata = metadata[metadata["feature_set"] == feature_key].copy()
    metadata = metadata.sort_values(["comparison_id", "time_bin_center"]).reset_index(drop=True)
    if metadata.empty:
        raise ValueError(f"No direction metadata rows found for feature set {feature_key!r}")

    embryo_meta = (
        df[[id_col, class_col]]
        .drop_duplicates(subset=[id_col])
        .assign(**{id_col: lambda x: x[id_col].astype(str)})
        .reset_index(drop=True)
    )
    if embryo_meta[id_col].duplicated().any():
        raise ValueError("Embryo metadata is not unique by embryo_id")
    embryo_class_map = embryo_meta.set_index(id_col)[class_col]

    records: list[dict[str, object]] = []
    axis_rows: list[dict[str, object]] = []

    for comparison_id, comp_meta in metadata.groupby("comparison_id", sort=False):
        rc = resolved_by_id[str(comparison_id)]
        labeled = _build_binary_labels(df, class_col, rc)
        binned = _bin_and_aggregate(labeled, id_col, time_col, feature_cols, float(bin_width))

        for row in comp_meta.itertuples(index=False):
            axis = np.asarray(directions.vectors[str(row.vector_id)], dtype=float).ravel()
            sub = binned[np.isclose(binned["time_bin_center"], float(row.time_bin_center))].copy()
            if sub.empty:
                continue
            scores = sub[feature_cols].to_numpy(dtype=float) @ axis
            for embryo_id, time_bin_center, score in zip(
                sub[id_col].astype(str),
                sub["time_bin_center"].astype(float),
                scores,
            ):
                records.append(
                    {
                        id_col: str(embryo_id),
                        "time_bin_center": float(time_bin_center),
                        class_col: str(embryo_class_map.loc[str(embryo_id)]),
                        "vector_id": str(row.vector_id),
                        "score": float(score),
                    }
                )
            axis_rows.append(
                {
                    "comparison_id": str(row.comparison_id),
                    "vector_id": str(row.vector_id),
                    "time_bin": int(row.time_bin),
                    "time_bin_center": float(row.time_bin_center),
                    "positive_label": rc.positive_label,
                    "negative_label": rc.negative_label,
                    "n_pos": int(row.n_pos),
                    "n_neg": int(row.n_neg),
                }
            )

    long_df = pd.DataFrame(records)
    if long_df.empty:
        raise ValueError(f"No projected records generated for feature set {feature_key!r}")

    wide = (
        long_df.pivot_table(
            index=[id_col, "time_bin_center", class_col],
            columns="vector_id",
            values="score",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    vector_cols = [c for c in wide.columns if c not in {id_col, "time_bin_center", class_col}]
    wide = wide[[id_col, "time_bin_center", class_col, *vector_cols]].copy()
    wide = wide.sort_values([id_col, "time_bin_center"]).reset_index(drop=True)
    axis_metadata = pd.DataFrame(axis_rows).sort_values(["comparison_id", "time_bin_center"]).reset_index(drop=True)
    return wide, axis_metadata


def _build_condensation_data(wide: pd.DataFrame, *, labels_col: str) -> schema.CondensationData:
    feature_cols = [c for c in wide.columns if c not in {ID_COL, "time_bin_center", CLASS_COL, "viz_label"}]
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
        label = str(getattr(row, labels_col))
        if labels[i] == "":
            labels[i] = label
        elif labels[i] != label:
            raise ValueError(f"Label inconsistency for embryo {getattr(row, ID_COL)!r}: {labels[i]!r} vs {label!r}")

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


def _make_homo_split_labels(wide: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    out["viz_label"] = out[CLASS_COL].astype(str)
    homo_mask = out[CLASS_COL].astype(str) == "cep290_homozygous"
    phenotype_map = (
        df[[ID_COL, PHENOTYPE_COL]]
        .drop_duplicates(subset=[ID_COL])
        .set_index(ID_COL)[PHENOTYPE_COL]
        .astype(str)
    )
    phenotype = out[ID_COL].astype(str).map(phenotype_map)
    subtype_mask = homo_mask & phenotype.isin(PHENOTYPE_COLORS.keys())
    out.loc[subtype_mask, "viz_label"] = phenotype[subtype_mask]
    return out


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
    skip_animations: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run = tc_viz.RunDescriptor(
        positions=positions,
        mask=mask,
        time_values=time_values,
        labels=labels,
        embryo_ids=embryo_ids,
        color_map=color_map,
        title=title,
        x0=x0,
    )
    # Always skip GIF animations — they are slow and not needed here.
    # Static PNGs + time_slice.html are the intended outputs.
    tc_viz.render_run(run, output_dir, title_prefix=title, skip_animations=True)
    fig = time_slice_html(
        run.positions,
        run.mask,
        run.time_values,
        labels=run.labels,
        color_map=color_map,
        embryo_ids=run.embryo_ids,
        output_path=output_dir / "time_slice.html",
        title=title,
    )
    fig.write_image(str(output_dir / "time_slice.png"), width=1500, height=700, scale=2)

    # Alpha sweep — static PNGs at different trajectory opacities so we can
    # pick the clearest value without re-running the full condensation.
    alpha_dir = output_dir / "alpha_sweep"
    alpha_dir.mkdir(exist_ok=True)
    for alpha in [0.03, 0.06, 0.10, 0.18, 0.30]:
        sweep_fig = time_slice_html(
            run.positions,
            run.mask,
            run.time_values,
            labels=run.labels,
            color_map=color_map,
            embryo_ids=run.embryo_ids,
            output_path=None,
            title=f"{title} (alpha={alpha})",
            trajectory_trace_alpha=alpha,
        )
        sweep_fig.write_image(
            str(alpha_dir / f"time_slice_alpha{int(alpha * 100):02d}.png"),
            width=1500, height=700, scale=2,
        )


def _run_feature_space(
    *,
    feature_key: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    paths: Paths,
    bin_width: float,
    seed: int,
    n_iter: int,
    save_every: int,
    epsilon_r: float,
    skip_animations: bool,
) -> tc_viz.RunDescriptor:
    feature_root = paths.results_dir / feature_key
    feature_root.mkdir(parents=True, exist_ok=True)
    directions_dir = paths.directions_dir / feature_key
    run_dir = feature_root / "run"
    viz_genotype_dir = feature_root / "viz_genotype"
    viz_homo_split_dir = feature_root / "viz_homo_split"
    run_dir.mkdir(parents=True, exist_ok=True)

    directions = _extract_directions(
        df,
        feature_key=feature_key,
        feature_cols=feature_cols,
        bin_width=bin_width,
        output_dir=directions_dir,
    )
    wide, axis_metadata = _build_wide_scores(
        df,
        directions=directions,
        feature_key=feature_key,
        feature_cols=feature_cols,
        bin_width=bin_width,
        class_col=CLASS_COL,
        id_col=ID_COL,
        time_col=TIME_COL,
        comparisons="all_pairs",
    )
    wide = _make_homo_split_labels(wide, df)
    wide.to_parquet(feature_root / "wide_scores.parquet", index=False)
    axis_metadata.to_parquet(feature_root / "axis_metadata.parquet", index=False)
    (feature_root / "manifest.json").write_text(
        json.dumps(
            {
                "feature_key": feature_key,
                "feature_columns": feature_cols,
                "bin_width": float(bin_width),
                "n_embryos": int(wide[ID_COL].nunique()),
                "n_timebins": int(wide["time_bin_center"].nunique()),
                "n_vectors": int(len([c for c in wide.columns if c not in {ID_COL, 'time_bin_center', CLASS_COL, 'viz_label'}])),
            },
            indent=2,
        )
    )

    data = _build_condensation_data(wide, labels_col=CLASS_COL)
    x0 = init_embedding.aligned_umap_init(data.features, data.mask, random_state=int(seed))
    np.savez(
        run_dir / "x0_init.npz",
        x0=x0,
        mask=data.mask,
        time_values=data.time_values,
        embryo_ids=data.embryo_ids,
        labels=data.labels,
    )

    config = _condensation_config(n_iter=int(n_iter), epsilon_r=float(epsilon_r))
    stopping = _stopping_config()
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

    genotype_labels = np.array(
        [str(wide.loc[wide[ID_COL].astype(str) == str(eid), CLASS_COL].iloc[0]) for eid in data.embryo_ids],
        dtype=object,
    )
    viz_labels = np.array(
        [str(wide.loc[wide[ID_COL].astype(str) == str(eid), "viz_label"].iloc[0]) for eid in data.embryo_ids],
        dtype=object,
    )
    genotype_color_map = {k: v for k, v in CEP290_GENOTYPE_COLORS.items() if k in set(genotype_labels.tolist())}
    homo_split_color_map = dict(genotype_color_map)
    homo_split_color_map.update(PHENOTYPE_COLORS)

    _render_view(
        positions=result.positions,
        x0=x0,
        mask=data.mask,
        time_values=data.time_values,
        embryo_ids=data.embryo_ids,
        labels=genotype_labels,
        color_map=genotype_color_map,
        output_dir=viz_genotype_dir,
        title=f"{feature_key} genotype",
        skip_animations=skip_animations,
    )
    _render_view(
        positions=result.positions,
        x0=x0,
        mask=data.mask,
        time_values=data.time_values,
        embryo_ids=data.embryo_ids,
        labels=viz_labels,
        color_map=homo_split_color_map,
        output_dir=viz_homo_split_dir,
        title=f"{feature_key} homo split",
        skip_animations=skip_animations,
    )

    return tc_viz.RunDescriptor(
        positions=result.positions,
        mask=data.mask,
        time_values=data.time_values,
        labels=genotype_labels,
        embryo_ids=data.embryo_ids,
        color_map=genotype_color_map,
        title=feature_key,
        x0=x0,
        position_history=result.position_history,
        snapshot_iters=result.snapshot_iters,
    )


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n_iter = 50
        args.save_every = min(args.save_every, 10)

    paths = _resolve_paths(args)
    cache_path = paths.cache_dir / f"cep290_reference_{int(args.t_min)}_{int(args.t_max)}.parquet"
    df = _load_reference_df(
        paths.data_dir,
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        smoke=bool(args.smoke),
    )
    df.to_parquet(cache_path, index=False)
    print(f"Wrote cache: {cache_path}")
    print(f"Rows: {len(df)}")
    print(f"Embryos by genotype: {df.groupby(CLASS_COL)[ID_COL].nunique().to_dict()}")

    runs: list[tc_viz.RunDescriptor] = []
    for feature_key in args.feature_sets:
        feature_cols = _feature_columns(df, FEATURE_SET_SPECS[feature_key])
        print(f"=== Running feature space: {feature_key} ({len(feature_cols)} columns) ===")
        run = _run_feature_space(
            feature_key=feature_key,
            df=df,
            feature_cols=feature_cols,
            paths=paths,
            bin_width=float(args.bin_width),
            seed=int(args.seed),
            n_iter=int(args.n_iter),
            save_every=int(args.save_every),
            epsilon_r=float(args.epsilon_r),
            skip_animations=bool(args.skip_animations),
        )
        runs.append(run)

    if len(runs) > 1:
        tc_viz.compare_run_grid([runs], mode="trajectories", output_path=paths.figures_dir / "compare_trajectories.png")
        tc_viz.compare_run_grid([runs], mode="stacked_3d", output_path=paths.figures_dir / "compare_stacked_3d.png")

    print(paths.results_dir)


if __name__ == "__main__":
    main()
