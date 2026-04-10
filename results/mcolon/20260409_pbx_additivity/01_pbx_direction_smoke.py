from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PBX_ANALYSIS_DIR = REPO_ROOT / "results" / "mcolon" / "20260407_pbx_analysis_cont"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PBX_ANALYSIS_DIR))

from analyze.classification import run_classification
from analyze.classification.engine.analysis import ClassificationAnalysis
from phenotype_direction import axis_alignment, cosine_alignment, load_classifier_directions

from common import load_combined_pbx_dataframe


EXPERIMENT_IDS = ["20251207_pbx", "20260304", "20260306"]
# wik_ab is loaded for normalization but is also a comparison class (inj_ctrl vs wik_ab sanity).
GENOTYPES = [
    "wik_ab",
    "inj_ctrl",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]
# All comparisons use wik_ab as negative (biological WT baseline).
# inj_ctrl vs wik_ab is the injection-effect sanity check.
COMPARISON_ROWS = [
    {"positive": "inj_ctrl",              "negative": "wik_ab"},
    {"positive": "pbx1b_crispant",        "negative": "wik_ab"},
    {"positive": "pbx4_crispant",         "negative": "wik_ab"},
    {"positive": "pbx1b_pbx4_crispant",   "negative": "wik_ab"},
]
COMPARISON_IDS = [
    "inj_ctrl__vs__wik_ab",
    "pbx1b_crispant__vs__wik_ab",
    "pbx4_crispant__vs__wik_ab",
    "pbx1b_pbx4_crispant__vs__wik_ab",
]
# PBX axis is built from the pbx4 vs wik_ab direction.
AXIS_SOURCE_COMPARISON = "pbx4_crispant__vs__wik_ab"
# Genotypes to project (all loaded genotypes).
PLOT_GENOTYPES = GENOTYPES
WT_GENOTYPE = "wik_ab"
NWT_MIN = 4  # minimum wik_ab embryos per bin for per-bin WT centroid

FEATURE_SET = "vae"
TIME_COL = "stage_hpf"
CLASS_COL = "genotype"
ID_COL = "embryo_id"
BIN_WIDTH = 4.0
RESULTS_DIR = SCRIPT_DIR / "results" / "pbx_direction_smoke_5class_bin4_perm0_wt_ref"
FIGURES_DIR = SCRIPT_DIR / "figures" / "pbx_direction_smoke_5class_bin4_perm0_wt_ref"
VECTOR_METADATA_FILE = "classifier_directions.parquet"
VECTOR_ARRAY_FILE = "classifier_directions_vectors.npz"
AXIS_AUROC_GATE = 0.55
VECTOR_NORM_TOL = 1e-6
NONNEG_TOL = 1e-8


def _load_raw_pbx_dataframe() -> pd.DataFrame:
    return load_combined_pbx_dataframe(
        experiment_ids=EXPERIMENT_IDS,
        genotypes=GENOTYPES,  # includes wik_ab
    )


def _run_smoke_classification(df: pd.DataFrame) -> ClassificationAnalysis:
    return run_classification(
        df=df,
        class_col=CLASS_COL,
        id_col=ID_COL,
        time_col=TIME_COL,
        comparisons=COMPARISON_ROWS,
        features={FEATURE_SET: "z_mu_b"},
        bin_width=BIN_WIDTH,
        n_permutations=0,
        n_splits=5,
        min_samples_per_group=3,
        min_samples_per_member=2,
        n_jobs=1,
        random_state=42,
        verbose=False,
        save_predictions=True,
        save_classifier_directions=True,
        save_dir=RESULTS_DIR,
        overwrite=True,
    )


def _ensure_classifier_direction_contract(
    analysis: ClassificationAnalysis,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    if not (RESULTS_DIR / VECTOR_METADATA_FILE).exists():
        raise FileNotFoundError(f"Missing direction metadata file: {RESULTS_DIR / VECTOR_METADATA_FILE}")
    if not (RESULTS_DIR / VECTOR_ARRAY_FILE).exists():
        raise FileNotFoundError(f"Missing direction vector file: {RESULTS_DIR / VECTOR_ARRAY_FILE}")

    directions = load_classifier_directions(RESULTS_DIR)
    metadata = directions.metadata.copy()
    metadata = metadata[
        (metadata["feature_set"] == FEATURE_SET)
        & (metadata["comparison_id"].isin(COMPARISON_IDS))
    ].copy()
    if metadata.empty:
        raise ValueError("No classifier direction rows found for the PBX smoke comparisons.")

    missing_ids = sorted(set(COMPARISON_IDS) - set(metadata["comparison_id"].unique()))
    if missing_ids:
        raise ValueError(f"Missing requested comparison IDs: {missing_ids}")

    feature_names = list(directions.feature_names.get(FEATURE_SET, []))
    if not feature_names:
        raise ValueError(f"Missing feature_names for feature set {FEATURE_SET!r}")

    vectors: dict[str, np.ndarray] = {}
    for row in metadata.itertuples(index=False):
        if row.direction_space != "raw_feature_space":
            raise ValueError(f"Unexpected direction_space for {row.vector_id}: {row.direction_space!r}")
        if row.vector_kind != "signed_unit_coef":
            raise ValueError(f"Unexpected vector_kind for {row.vector_id}: {row.vector_kind!r}")
        if row.refit_scope != "full_bin_after_cv":
            raise ValueError(f"Unexpected refit_scope for {row.vector_id}: {row.refit_scope!r}")
        if row.cv_scope != "as_scored":
            raise ValueError(f"Unexpected cv_scope for {row.vector_id}: {row.cv_scope!r}")
        vec = np.asarray(directions.vectors[row.vector_id], dtype=float).ravel()
        if len(vec) != len(feature_names):
            raise ValueError(
                f"Vector length mismatch for {row.vector_id}: {len(vec)} != {len(feature_names)}"
            )
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"Non-finite values found in vector {row.vector_id}")
        norm = float(np.linalg.norm(vec))
        if not np.isclose(norm, 1.0, atol=VECTOR_NORM_TOL):
            raise ValueError(f"Vector {row.vector_id} is not unit norm: {norm}")
        if float(row.centroid_dot) < -NONNEG_TOL:
            raise ValueError(
                f"centroid_dot is negative for {row.vector_id}: {row.centroid_dot}\n"
                "Check that positive class is crispant/inj_ctrl and negative class is wik_ab."
            )
        vectors[row.vector_id] = vec

    metadata = metadata.sort_values(["comparison_id", "time_bin_center"]).reset_index(drop=True)
    return metadata, vectors, feature_names


def _make_binned_feature_dataframe(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in raw dataframe: {missing}")
    work = df.copy()
    work["_time_bin"] = (np.floor(work[TIME_COL] / BIN_WIDTH) * BIN_WIDTH).astype(int)
    work["time_bin_center"] = work["_time_bin"].astype(float) + BIN_WIDTH / 2.0
    group_cols = [ID_COL, CLASS_COL, "_time_bin", "time_bin_center"]
    binned = work.groupby(group_cols, as_index=False)[feature_names].mean()
    ordered = binned[feature_names]
    if list(ordered.columns) != feature_names:
        raise ValueError("Projection feature order does not match saved feature_names.")
    return binned


def _build_alignment_table(metadata: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    pairings = [
        ("pbx4_crispant__vs__wik_ab",      "pbx1b_pbx4_crispant__vs__wik_ab"),
        ("pbx1b_crispant__vs__wik_ab",     "pbx1b_pbx4_crispant__vs__wik_ab"),
        ("inj_ctrl__vs__wik_ab",           "pbx4_crispant__vs__wik_ab"),   # injection sanity
    ]
    rows: list[dict[str, object]] = []
    for comparison_a, comparison_b in pairings:
        meta_a = metadata[metadata["comparison_id"] == comparison_a].copy()
        meta_b = metadata[metadata["comparison_id"] == comparison_b].copy()
        merged = meta_a.merge(
            meta_b,
            on="time_bin_center",
            suffixes=("_a", "_b"),
            how="inner",
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            vec_a = vectors[row.vector_id_a]
            vec_b = vectors[row.vector_id_b]
            rows.append(
                {
                    "time_bin": int(row.time_bin_a),
                    "time_bin_center": float(row.time_bin_center),
                    "comparison_a": comparison_a,
                    "comparison_b": comparison_b,
                    "signed_cosine": cosine_alignment(vec_a, vec_b),
                    "axis_alignment": axis_alignment(vec_a, vec_b),
                    "auroc_a": float(row.auroc_obs_a),
                    "auroc_b": float(row.auroc_obs_b),
                    "n_positive_a": int(row.n_positive_a),
                    "n_negative_a": int(row.n_negative_a),
                    "n_positive_b": int(row.n_positive_b),
                    "n_negative_b": int(row.n_negative_b),
                }
            )
    out = pd.DataFrame(rows).sort_values(["comparison_a", "comparison_b", "time_bin_center"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Alignment table is empty; no overlapping time bins across required comparison pairs.")
    return out


def _build_axis(
    metadata: pd.DataFrame,
    vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    axis_meta = metadata[metadata["comparison_id"] == AXIS_SOURCE_COMPARISON].copy()
    if axis_meta.empty:
        raise ValueError(f"Missing axis source comparison {AXIS_SOURCE_COMPARISON!r}")
    axis_meta["axis_weight"] = np.maximum(axis_meta["auroc_obs"].to_numpy(dtype=float) - 0.5, 0.0)
    axis_meta["passes_auroc_gate"] = axis_meta["auroc_obs"].to_numpy(dtype=float) >= AXIS_AUROC_GATE
    included = axis_meta[axis_meta["passes_auroc_gate"] & (axis_meta["axis_weight"] > 0.0)].copy()
    if included.empty:
        raise ValueError(
            f"No axis bins survived the AUROC gate for {AXIS_SOURCE_COMPARISON!r} "
            f"(threshold={AXIS_AUROC_GATE})."
        )
    stacked = np.vstack([vectors[row.vector_id] for row in included.itertuples(index=False)])
    weights = included["axis_weight"].to_numpy(dtype=float)
    axis = np.sum(stacked * weights[:, None], axis=0)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("Axis construction collapsed to a zero vector.")
    return axis / norm, axis_meta


def _compute_wt_centroid(
    projections: pd.DataFrame,
    sorted_bin_centers: list[float],
) -> tuple[dict[float, float], list[dict]]:
    """
    For each time bin, compute the per-bin WT centroid with NWT_MIN guardrail.

    If a bin has < NWT_MIN wik_ab embryos, expand symmetrically to adjacent bins
    until the pooled count >= NWT_MIN (or we exhaust all bins).

    Returns:
        mu_wt_by_bin: {time_bin_center -> mu_wt used for centering}
        diagnostics:  list of per-bin diagnostic dicts
    """
    # Build per-bin wt score arrays up front.
    wt_scores_by_bin: dict[float, np.ndarray] = {}
    for tbc in sorted_bin_centers:
        mask = (projections["time_bin_center"] == tbc) & (projections[CLASS_COL] == WT_GENOTYPE)
        wt_scores_by_bin[tbc] = projections.loc[mask, "score_raw"].to_numpy(dtype=float)

    mu_wt_by_bin: dict[float, float] = {}
    diagnostics: list[dict] = []
    n_bins = len(sorted_bin_centers)

    for i, tbc in enumerate(sorted_bin_centers):
        wt = wt_scores_by_bin[tbc]
        n_wt = len(wt)
        fallback_used = False
        window_bins: list[float] = [tbc]

        if n_wt >= NWT_MIN:
            mu_wt = float(np.mean(wt))
        else:
            # Expand window outward until we accumulate NWT_MIN WT embryos.
            pooled = list(wt)
            radius = 1
            while len(pooled) < NWT_MIN and radius <= n_bins:
                left_i  = i - radius
                right_i = i + radius
                if left_i >= 0:
                    tbc_l = sorted_bin_centers[left_i]
                    pooled.extend(wt_scores_by_bin[tbc_l].tolist())
                    window_bins.append(tbc_l)
                if right_i < n_bins:
                    tbc_r = sorted_bin_centers[right_i]
                    pooled.extend(wt_scores_by_bin[tbc_r].tolist())
                    window_bins.append(tbc_r)
                radius += 1
            mu_wt = float(np.mean(pooled)) if pooled else float("nan")
            fallback_used = True

        mu_wt_by_bin[tbc] = mu_wt
        diagnostics.append({
            "time_bin_center": tbc,
            "n_wt": n_wt,
            "mu_wt_used": mu_wt,
            "fallback_used": fallback_used,
            "window_bins_used": sorted(window_bins),
        })

    return mu_wt_by_bin, diagnostics


def _centre_projections(
    projections: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Center raw scores by per-bin WT centroid (with NWT_MIN guardrail).
    No division — scores remain in raw VAE projection units.
    """
    sorted_bins = sorted(projections["time_bin_center"].unique().tolist())
    mu_wt_by_bin, wt_diagnostics = _compute_wt_centroid(projections, sorted_bins)

    rows: list[pd.DataFrame] = []
    for _, grp in projections.groupby("time_bin_center", sort=True):
        grp = grp.copy()
        tbc = float(grp["time_bin_center"].iloc[0])
        mu_wt = mu_wt_by_bin[tbc]
        if np.isfinite(mu_wt):
            grp["score_wt_centered"] = grp["score_raw"].to_numpy(dtype=float) - mu_wt
        else:
            grp["score_wt_centered"] = np.nan
        rows.append(grp)

    return pd.concat(rows, ignore_index=True), wt_diagnostics


def _build_projection_table(
    binned: pd.DataFrame,
    axis: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    X = binned[feature_names]
    if list(X.columns) != feature_names:
        raise ValueError("Projection matrix columns do not match saved feature_names.")
    projections = binned[[ID_COL, CLASS_COL, "_time_bin", "time_bin_center"]].copy()
    projections = projections.rename(columns={"_time_bin": "time_bin"})
    projections["score_raw"] = X.to_numpy(dtype=float) @ np.asarray(axis, dtype=float)
    projections, wt_diagnostics = _centre_projections(projections)
    return projections.sort_values([CLASS_COL, ID_COL, "time_bin_center"]).reset_index(drop=True), wt_diagnostics


def _build_embryo_summary(projections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for embryo_id, grp in projections.groupby(ID_COL):
        rows.append(
            {
                ID_COL: str(embryo_id),
                CLASS_COL: str(grp[CLASS_COL].iloc[0]),
                "mean_score_wt_centered": float(grp["score_wt_centered"].dropna().mean()),
                "n_valid_bins": int(grp["score_wt_centered"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values([CLASS_COL, ID_COL]).reset_index(drop=True)


def _build_direction_sign_sanity(
    metadata: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    binned: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    X = binned[feature_names].to_numpy(dtype=float)
    for meta in metadata.itertuples(index=False):
        vec = vectors[meta.vector_id]
        mask_bin = binned["time_bin_center"].to_numpy(dtype=float) == float(meta.time_bin_center)
        mask_pos = binned[CLASS_COL].astype(str).to_numpy() == str(meta.positive_label)
        mask_neg = binned[CLASS_COL].astype(str).to_numpy() == str(meta.negative_label)
        scores = X @ vec
        pos_scores = scores[mask_bin & mask_pos]
        neg_scores = scores[mask_bin & mask_neg]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            raise ValueError(
                f"Missing positive or negative embryos for sanity check: {meta.comparison_id} @ {meta.time_bin_center}"
            )
        mean_pos = float(np.mean(pos_scores))
        mean_neg = float(np.mean(neg_scores))
        rows.append(
            {
                "comparison_id": str(meta.comparison_id),
                "time_bin": int(meta.time_bin),
                "time_bin_center": float(meta.time_bin_center),
                "mean_score_positive": mean_pos,
                "mean_score_negative": mean_neg,
                "centroid_score_delta": mean_pos - mean_neg,
                "centroid_dot": float(meta.centroid_dot),
                "sign_flipped": bool(meta.sign_flipped),
            }
        )
    out = pd.DataFrame(rows).sort_values(["comparison_id", "time_bin_center"]).reset_index(drop=True)
    if (out["centroid_score_delta"] < -NONNEG_TOL).any():
        raise ValueError("Negative centroid_score_delta found in sign sanity check.")
    if (out["centroid_dot"] < -NONNEG_TOL).any():
        raise ValueError("Negative centroid_dot found in sign sanity check.")
    return out


def _plot_alignment(df: pd.DataFrame, output_path: Path) -> None:
    pair_labels = [
        ("pbx4_crispant__vs__wik_ab",    "pbx1b_pbx4_crispant__vs__wik_ab", "pbx4 vs double",    "#B2182B"),
        ("pbx1b_crispant__vs__wik_ab",   "pbx1b_pbx4_crispant__vs__wik_ab", "pbx1b vs double",   "#2166AC"),
        ("inj_ctrl__vs__wik_ab",         "pbx4_crispant__vs__wik_ab",        "inj_ctrl vs pbx4", "#808080"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for comparison_a, comparison_b, label, color in pair_labels:
        sub = df[(df["comparison_a"] == comparison_a) & (df["comparison_b"] == comparison_b)]
        if sub.empty:
            continue
        axes[0].plot(sub["time_bin_center"], sub["signed_cosine"], marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(sub["time_bin_center"], sub["axis_alignment"], marker="o", linewidth=2.0, color=color, label=label)
    axes[0].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Signed cosine")
    axes[1].set_ylabel("Axis alignment |cosine|")
    axes[1].set_xlabel("Time bin center (hpf)")
    axes[0].set_ylim(-1.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[0].set_title("PBX direction alignment over time (ref: wik_ab)")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_panel(ax: plt.Axes, df: pd.DataFrame, value_col: str, title: str) -> None:
    color_map = {
        "wik_ab":               "#808080",
        "inj_ctrl":             "#2166AC",
        "pbx1b_crispant":       "#9467BD",
        "pbx4_crispant":        "#F7B267",
        "pbx1b_pbx4_crispant":  "#B2182B",
    }
    for genotype in PLOT_GENOTYPES:
        sub = df[df[CLASS_COL] == genotype].copy()
        if sub.empty:
            continue
        grouped = sub.groupby("time_bin_center")[value_col].agg(["mean", "std", "count"]).reset_index()
        grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
        x = grouped["time_bin_center"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, color=color_map[genotype], label=genotype)
        ax.fill_between(x, y - sem, y + sem, color=color_map[genotype], alpha=0.18)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time bin center (hpf)")
    ax.grid(True, alpha=0.2)


def _plot_projection_trajectories(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_trajectory_panel(ax, df.dropna(subset=["score_wt_centered"]), "score_wt_centered", "WT-centered projection (raw VAE units)")
    ax.set_ylabel("Score (raw VAE projection units, centered on wik_ab)")
    ax.legend(frameon=False, loc="best")
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, label="wik_ab reference")
    fig.suptitle("PBX axis projection trajectories (ref: wik_ab centroid per bin)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_embryo_summary(df: pd.DataFrame, output_path: Path) -> None:
    color_map = {
        "wik_ab":               "#808080",
        "inj_ctrl":             "#2166AC",
        "pbx1b_crispant":       "#9467BD",
        "pbx4_crispant":        "#F7B267",
        "pbx1b_pbx4_crispant":  "#B2182B",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(PLOT_GENOTYPES))
    series = [
        df.loc[df[CLASS_COL] == genotype, "mean_score_wt_centered"].dropna().to_numpy(dtype=float)
        for genotype in PLOT_GENOTYPES
    ]
    vp = ax.violinplot(series, positions=positions, showmeans=False, showextrema=False, widths=0.8)
    for body, genotype in zip(vp["bodies"], PLOT_GENOTYPES):
        body.set_facecolor(color_map[genotype])
        body.set_edgecolor(color_map[genotype])
        body.set_alpha(0.35)
    for idx, (genotype, values) in enumerate(zip(PLOT_GENOTYPES, series)):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, s=14, color=color_map[genotype], alpha=0.85)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions, [g.replace("_", "\n") for g in PLOT_GENOTYPES])
    ax.set_ylabel("Mean WT-centered score (raw VAE units)")
    ax.set_title("PBX axis embryo summary (wik_ab = 0 reference)")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_sign_sanity(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["centroid_score_delta"], df["centroid_dot"], s=30, color="#B2182B", alpha=0.9)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Projected centroid separation")
    ax.set_ylabel("Stored centroid_dot")
    ax.set_title("Direction sign sanity check")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary_json(
    metadata: pd.DataFrame,
    feature_names: list[str],
    axis_meta: pd.DataFrame,
    axis: np.ndarray,
    wt_diagnostics: list[dict],
    embryo_summary: pd.DataFrame,
) -> None:
    vector_norms = [float(np.linalg.norm(np.asarray(v, dtype=float))) for v in load_classifier_directions(RESULTS_DIR).vectors.values()]
    genotype_summary = (
        embryo_summary.groupby(CLASS_COL)[["mean_score_wt_centered"]]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    n_fallback_bins = sum(1 for d in wt_diagnostics if d["fallback_used"])
    payload = {
        "experiment_ids": EXPERIMENT_IDS,
        "genotypes": GENOTYPES,
        "wt_genotype": WT_GENOTYPE,
        "nwt_min": NWT_MIN,
        "normalization": "wt_centered_raw_vae_units",
        "class_col": CLASS_COL,
        "id_col": ID_COL,
        "time_col": TIME_COL,
        "bin_width": BIN_WIDTH,
        "n_permutations": 0,
        "n_splits": 5,
        "n_jobs": 1,
        "random_state": 42,
        "feature_set": FEATURE_SET,
        "feature_names": feature_names,
        "requested_comparison_ids": COMPARISON_IDS,
        "vector_count": int(len(metadata)),
        "time_bins_by_comparison": {
            cid: sorted(metadata.loc[metadata["comparison_id"] == cid, "time_bin"].astype(int).tolist())
            for cid in COMPARISON_IDS
        },
        "vector_norm_min": float(np.min(vector_norms)),
        "vector_norm_max": float(np.max(vector_norms)),
        "centroid_dot_min": float(metadata["centroid_dot"].min()),
        "centroid_dot_max": float(metadata["centroid_dot"].max()),
        "axis_source_comparison": AXIS_SOURCE_COMPARISON,
        "axis_auroc_gate": AXIS_AUROC_GATE,
        "axis_included_time_bins": sorted(axis_meta.loc[axis_meta["auroc_obs"] >= AXIS_AUROC_GATE, "time_bin"].astype(int).tolist()),
        "axis_excluded_time_bins": sorted(axis_meta.loc[axis_meta["auroc_obs"] < AXIS_AUROC_GATE, "time_bin"].astype(int).tolist()),
        "axis_weights": [
            {
                "time_bin": int(row.time_bin),
                "time_bin_center": float(row.time_bin_center),
                "auroc_obs": float(row.auroc_obs),
                "axis_weight": float(max(row.auroc_obs - 0.5, 0.0)),
                "passes_auroc_gate": bool(row.auroc_obs >= AXIS_AUROC_GATE),
            }
            for row in axis_meta.itertuples(index=False)
        ],
        "axis_norm": float(np.linalg.norm(axis)),
        "wt_normalization_diagnostics": wt_diagnostics,
        "n_fallback_normalization_bins": n_fallback_bins,
        "genotype_mean_summary": genotype_summary,
    }
    (RESULTS_DIR / "direction_smoke_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw_pbx_dataframe()
    analysis = _run_smoke_classification(raw_df)
    metadata, vectors, feature_names = _ensure_classifier_direction_contract(analysis)
    binned = _make_binned_feature_dataframe(raw_df, feature_names)

    alignment = _build_alignment_table(metadata, vectors)
    alignment.to_csv(RESULTS_DIR / "direction_alignment_over_time.csv", index=False)
    _plot_alignment(alignment, FIGURES_DIR / "direction_alignment_over_time.png")

    axis, axis_meta = _build_axis(metadata, vectors)
    projections, wt_diagnostics = _build_projection_table(binned, axis, feature_names)
    projections.to_csv(RESULTS_DIR / "pbx4_axis_projection_by_embryo_bin.csv", index=False)
    # Write WT normalization diagnostics as a separate CSV for easy inspection.
    pd.DataFrame(wt_diagnostics).to_csv(RESULTS_DIR / "wt_normalization_diagnostics.csv", index=False)
    _plot_projection_trajectories(projections, FIGURES_DIR / "pbx4_axis_projection_trajectories.png")

    embryo_summary = _build_embryo_summary(projections)
    embryo_summary.to_csv(RESULTS_DIR / "pbx4_axis_embryo_summary.csv", index=False)
    _plot_embryo_summary(embryo_summary, FIGURES_DIR / "pbx4_axis_embryo_summary.png")

    sign_sanity = _build_direction_sign_sanity(metadata, vectors, binned, feature_names)
    sign_sanity.to_csv(RESULTS_DIR / "direction_sign_sanity.csv", index=False)
    _plot_sign_sanity(sign_sanity, FIGURES_DIR / "direction_sign_sanity_scatter.png")

    _write_summary_json(metadata, feature_names, axis_meta, axis, wt_diagnostics, embryo_summary)

    print(RESULTS_DIR)
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
