from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from common import BUILD06_DIR, REPO_ROOT, resolve_embedding_roots
EXPERIMENT_IDS = ["20260304", "20260306"]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification import run_classification


RESULTS_DIR, FIGURES_DIR = resolve_embedding_roots()

DEFAULT_GENOTYPES = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]
TARGET_LABEL = "pbx4_crispant"
REFERENCE_LABEL = "inj_ctrl"
ENHANCER_LABEL = "pbx1b_pbx4_crispant"


def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    if g in {
        "ab_inj_ctrl",
        "wik-ab_inj_ctrl",
        "wik-ab_ctrl_inj",
        "wik_ab_inj_ctrl",
        "wik_ab_ctrl_inj",
    }:
        return "inj_ctrl"
    return g.replace("wik-ab", "wik_ab")


def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ")


def _short_name(label: str) -> str:
    label = str(label).strip().lower()
    if label == "inj_ctrl":
        return "inj_ctrl"
    if label == "wik_ab":
        return "wik_ab"
    if label == "pbx1b_crispant":
        return "pbx1b"
    if label == "pbx4_crispant":
        return "pbx4"
    if label == "pbx1b_pbx4_crispant":
        return "pbx1b+4"
    return _pretty_label(label)


def _pair_id(group1: str, group2: str) -> str:
    return f"{group1}_vs_{group2}"


def _pbx4_display_name(row: pd.Series, target_label: str) -> str:
    true_label = str(row["true_label"])
    pair_like = str(row["pair_like_label"])
    group1 = str(row["group1"])
    group2 = str(row["group2"])
    other = group2 if true_label == group1 else group1
    looks_like = true_label if pair_like.startswith(f"{true_label}_like_true_") else other
    return f"{_short_name(looks_like)}-like | {_short_name(group1)} vs {_short_name(group2)}"


def _pbx4_set_id(row: pd.Series, target_label: str) -> str:
    true_label = str(row["true_label"])
    pair_like = str(row["pair_like_label"])
    group1 = str(row["group1"])
    group2 = str(row["group2"])
    other = group2 if true_label == group1 else group1
    looks_like = true_label if pair_like.startswith(f"{true_label}_like_true_") else other
    return f"{looks_like}_like__{group1}_vs_{group2}__{true_label}_true"


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    vals = pvalues.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return pd.Series(out, index=pvalues.index)
    idx = np.where(mask)[0]
    m = len(idx)
    ranked = np.argsort(vals[idx])
    ranked_vals = vals[idx][ranked]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for rev_i in range(m - 1, -1, -1):
        rank = rev_i + 1
        curr = ranked_vals[rev_i] * m / rank
        prev = min(prev, curr)
        q[rev_i] = prev
    out_idx = idx[ranked]
    out[out_idx] = np.clip(q, 0.0, 1.0)
    return pd.Series(out, index=pvalues.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all PBX/control pairwise classifications and summarize PBX4 overlap structure.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--time-col", default="predicted_stage_hpf")
    parser.add_argument("--embedding-prefix", default="z_mu_b")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-time-bins", type=int, default=8)
    parser.add_argument("--late-threshold-hpf", type=float, default=74.0)
    parser.add_argument("--target-label", default=TARGET_LABEL)
    parser.add_argument("--reference-label", default=REFERENCE_LABEL)
    parser.add_argument("--enhancer-label", default=ENHANCER_LABEL)
    parser.add_argument("--genotypes", nargs="+", default=DEFAULT_GENOTYPES)
    return parser.parse_args()


def load_dataframe(genotypes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for exp_id in EXPERIMENT_IDS:
        data_path = BUILD06_DIR / f"df03_final_output_with_latents_{exp_id}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file: {data_path}")
        part = pd.read_csv(data_path, low_memory=False)
        if "experiment_id" in part.columns:
            part = part[part["experiment_id"].astype(str) == exp_id].copy()
        else:
            part["experiment_id"] = exp_id
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()
    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df = df[df["genotype"].isin(genotypes)].copy()
    if df.empty:
        raise ValueError("No rows remain after genotype filtering.")

    embryo_meta = (
        df[["embryo_id", "genotype", "experiment_id"]]
        .drop_duplicates()
        .rename(columns={"genotype": "true_label"})
        .reset_index(drop=True)
    )
    return df, embryo_meta


def embedding_features(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    if not cols:
        raise ValueError(f"No embedding columns found with prefix {prefix!r}")
    return cols


def run_pairwise_classification(
    df: pd.DataFrame,
    *,
    group1: str,
    group2: str,
    time_col: str,
    feature_cols: list[str],
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = run_classification(
        df=df.copy(),
        class_col="genotype",
        id_col="embryo_id",
        time_col=time_col,
        positive=group2,
        negative=group1,
        features={"embedding": feature_cols},
        bin_width=float(bin_width),
        n_permutations=int(n_permutations),
        n_splits=int(n_splits),
        min_samples_per_group=2,
        min_samples_per_member=2,
        n_jobs=1,
        random_state=int(random_state),
        verbose=True,
        save_predictions=True,
    )

    score_df = analysis.scores.copy().sort_values("time_bin_center").reset_index(drop=True)
    pred_df = analysis.layers["predictions"].copy().sort_values(["embryo_id", "time_bin_center"]).reset_index(drop=True)
    if score_df.empty or pred_df.empty:
        raise ValueError(f"No engine outputs produced for pair {group1} vs {group2}")

    score_df["time_bin"] = np.floor(score_df["time_bin_center"] - float(bin_width) / 2.0 + 1e-6).astype(int)
    score_df["pair_id"] = _pair_id(group1, group2)
    score_df["group1"] = group1
    score_df["group2"] = group2
    score_df["n_samples"] = score_df["n_positive"] + score_df["n_negative"]
    score_df["n_group1"] = score_df["n_negative"]
    score_df["n_group2"] = score_df["n_positive"]

    pred_df["time_bin"] = np.floor(pred_df["time_bin_center"] - float(bin_width) / 2.0 + 1e-6).astype(int)
    pred_df["true_label"] = np.where(pred_df["y_true"].astype(int) == 1, group2, group1)
    pred_df["predicted_label"] = np.where(pred_df["y_pred"].astype(int) == 1, group2, group1)
    pred_df["pred_prob_group2"] = pred_df["p_pos"].astype(float)
    pred_df["support_true"] = np.where(
        pred_df["y_true"].astype(int) == 1,
        pred_df["pred_prob_group2"],
        1.0 - pred_df["pred_prob_group2"],
    )
    pred_df["confidence"] = np.abs(pred_df["pred_prob_group2"] - 0.5)
    pred_df["signed_margin"] = np.where(
        pred_df["y_true"].astype(int) == 1,
        pred_df["pred_prob_group2"] - 0.5,
        0.5 - pred_df["pred_prob_group2"],
    )
    pred_df["pair_id"] = _pair_id(group1, group2)
    pred_df["group1"] = group1
    pred_df["group2"] = group2
    return score_df, pred_df


def label_pairwise_embryos(
    pred_df: pd.DataFrame,
    embryo_meta: pd.DataFrame,
    *,
    group1: str,
    group2: str,
    min_time_bins: int,
    late_threshold_hpf: float,
) -> pd.DataFrame:
    pair_id = _pair_id(group1, group2)
    pred_df = pred_df.copy()
    pred_df["window"] = np.where(pred_df["time_bin_center"] < late_threshold_hpf, "pre74", "post74")

    ranked = (
        pred_df.groupby(["embryo_id", "true_label"], as_index=False)
        .agg(
            mean_signed_margin=("signed_margin", "mean"),
            frac_correct=("is_correct", "mean"),
            n_time_bins=("time_bin_center", "nunique"),
            first_hpf=("time_bin_center", "min"),
            last_hpf=("time_bin_center", "max"),
        )
    )
    pre = (
        pred_df[pred_df["window"] == "pre74"]
        .groupby(["embryo_id", "true_label"])["signed_margin"]
        .mean()
        .rename("mean_signed_margin_pre74")
        .reset_index()
    )
    post = (
        pred_df[pred_df["window"] == "post74"]
        .groupby(["embryo_id", "true_label"])["signed_margin"]
        .mean()
        .rename("mean_signed_margin_post74")
        .reset_index()
    )
    ranked = ranked.merge(pre, on=["embryo_id", "true_label"], how="left").merge(post, on=["embryo_id", "true_label"], how="left")
    ranked["eligible_for_grouping"] = ranked["n_time_bins"] >= min_time_bins

    def classify(row: pd.Series) -> tuple[str, str]:
        true_label = str(row["true_label"])
        self_like = float(row["mean_signed_margin"]) >= 0.0
        if true_label == group1:
            like = group1 if self_like else group2
        else:
            like = group2 if self_like else group1
        return (
            f"{like}_like_true_{true_label}",
            f"{_short_name(like)}-like / true {_short_name(true_label)}",
        )

    labels = ranked.apply(classify, axis=1, result_type="expand")
    labels.columns = ["pair_like_label", "pair_like_display"]
    ranked = pd.concat([ranked, labels], axis=1)
    ranked["pair_id"] = pair_id
    ranked["group1"] = group1
    ranked["group2"] = group2
    ranked["pair_display"] = f"{_short_name(group1)} vs {_short_name(group2)}"
    ranked = ranked.merge(embryo_meta, on=["embryo_id", "true_label"], how="left")
    if "experiment_id_x" in ranked.columns:
        ranked = ranked.rename(columns={"experiment_id_x": "experiment_id"}).drop(columns=["experiment_id_y"])
    ranked["experiment_id"] = ranked["experiment_id"].astype(str)
    return ranked


def run_all_pairs(
    df: pd.DataFrame,
    embryo_meta: pd.DataFrame,
    genotypes: list[str],
    *,
    time_col: str,
    feature_cols: list[str],
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    random_state: int,
    min_time_bins: int,
    late_threshold_hpf: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auc_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []

    present_genotypes = [g for g in genotypes if g in set(df["genotype"].unique())]
    for idx, (group1, group2) in enumerate(combinations(present_genotypes, 2), start=1):
        print(f"[{idx}] {group1} vs {group2}")
        df_pair = df[df["genotype"].isin([group1, group2])].copy()
        auc_df, pred_df = run_pairwise_classification(
            df_pair,
            group1=group1,
            group2=group2,
            time_col=time_col,
            feature_cols=feature_cols,
            bin_width=bin_width,
            n_splits=n_splits,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        labels_df = label_pairwise_embryos(
            pred_df,
            embryo_meta,
            group1=group1,
            group2=group2,
            min_time_bins=min_time_bins,
            late_threshold_hpf=late_threshold_hpf,
        )
        auc_rows.append(auc_df)
        label_rows.append(labels_df)

    if not auc_rows or not label_rows:
        raise ValueError("No all-pairs classification results were produced.")
    return pd.concat(auc_rows, ignore_index=True), pd.concat(label_rows, ignore_index=True)


def summarize_pairwise_auroc(auc_df: pd.DataFrame) -> pd.DataFrame:
    return (
        auc_df.groupby(["pair_id", "group1", "group2"], as_index=False)
        .agg(
            n_time_bins=("time_bin", "nunique"),
            mean_auroc=("auroc_obs", "mean"),
            median_auroc=("auroc_obs", "median"),
            max_auroc=("auroc_obs", "max"),
            min_auroc=("auroc_obs", "min"),
        )
        .sort_values("mean_auroc", ascending=False)
        .reset_index(drop=True)
    )


def build_pbx4_label_sets(labels_df: pd.DataFrame, target_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbx4 = labels_df[(labels_df["true_label"] == target_label) & ((labels_df["group1"] == target_label) | (labels_df["group2"] == target_label))].copy()
    pbx4["set_display"] = pbx4.apply(_pbx4_display_name, axis=1, target_label=target_label)
    pbx4["set_id"] = pbx4.apply(_pbx4_set_id, axis=1, target_label=target_label)
    set_df = (
        pbx4[["set_id", "set_display", "pair_id", "group1", "group2", "pair_like_label"]]
        .drop_duplicates()
        .sort_values(["pair_id", "pair_like_label"])
        .reset_index(drop=True)
    )
    return pbx4, set_df


def compute_overlap_stats(pbx4_labels: pd.DataFrame, set_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = sorted(pbx4_labels["embryo_id"].astype(str).unique().tolist())
    n_universe = len(universe)
    if n_universe == 0:
        raise ValueError("No PBX4 embryos available for overlap analysis.")

    set_map = {
        set_id: set(
            pbx4_labels.loc[pbx4_labels["set_id"] == set_id, "embryo_id"].astype(str).tolist()
        )
        for set_id in set_df["set_id"]
    }

    ordered_rows: list[dict[str, object]] = []
    unique_rows: list[dict[str, object]] = []
    set_ids = set_df["set_id"].tolist()
    label_lookup = set_df.set_index("set_id")["set_display"].to_dict()

    for set_a in set_ids:
        embryos_a = set_map[set_a]
        for set_b in set_ids:
            embryos_b = set_map[set_b]
            overlap = len(embryos_a & embryos_b)
            size_a = len(embryos_a)
            size_b = len(embryos_b)
            expected = (size_a * size_b) / n_universe if n_universe else np.nan
            row_frac = overlap / size_a if size_a else np.nan
            col_frac = overlap / size_b if size_b else np.nan
            union = len(embryos_a | embryos_b)
            jaccard = overlap / union if union else np.nan
            ordered_rows.append(
                {
                    "set_a": set_a,
                    "set_b": set_b,
                    "set_a_display": label_lookup[set_a],
                    "set_b_display": label_lookup[set_b],
                    "n_universe": n_universe,
                    "size_a": size_a,
                    "size_b": size_b,
                    "observed_overlap": overlap,
                    "expected_overlap": expected,
                    "row_overlap_frac": row_frac,
                    "col_overlap_frac": col_frac,
                    "jaccard": jaccard,
                }
            )

    for idx_a, set_a in enumerate(set_ids):
        embryos_a = set_map[set_a]
        for idx_b in range(idx_a + 1, len(set_ids)):
            set_b = set_ids[idx_b]
            embryos_b = set_map[set_b]
            overlap = len(embryos_a & embryos_b)
            size_a = len(embryos_a)
            size_b = len(embryos_b)
            only_a = size_a - overlap
            only_b = size_b - overlap
            neither = n_universe - overlap - only_a - only_b
            table = np.array([[overlap, only_a], [only_b, neither]], dtype=int)
            _, p_depletion = fisher_exact(table, alternative="less")
            _, p_enrichment = fisher_exact(table, alternative="greater")
            expected = (size_a * size_b) / n_universe if n_universe else np.nan
            direction = "equal"
            if overlap < expected:
                direction = "depleted"
            elif overlap > expected:
                direction = "enriched"
            unique_rows.append(
                {
                    "set_a": set_a,
                    "set_b": set_b,
                    "set_a_display": label_lookup[set_a],
                    "set_b_display": label_lookup[set_b],
                    "n_universe": n_universe,
                    "size_a": size_a,
                    "size_b": size_b,
                    "observed_overlap": overlap,
                    "expected_overlap": expected,
                    "observed_minus_expected": overlap - expected,
                    "overlap_over_expected": np.nan if expected == 0 else overlap / expected,
                    "overlap_frac_min": np.nan if min(size_a, size_b) == 0 else overlap / min(size_a, size_b),
                    "jaccard": np.nan if (size_a + size_b - overlap) == 0 else overlap / (size_a + size_b - overlap),
                    "p_depletion": p_depletion,
                    "p_enrichment": p_enrichment,
                    "direction": direction,
                }
            )

    ordered_df = pd.DataFrame(ordered_rows)
    unique_df = pd.DataFrame(unique_rows)
    unique_df["q_depletion"] = _bh_qvalues(unique_df["p_depletion"])
    unique_df["q_enrichment"] = _bh_qvalues(unique_df["p_enrichment"])
    unique_df = unique_df.sort_values(["q_depletion", "p_depletion", "observed_minus_expected"]).reset_index(drop=True)
    return ordered_df, unique_df


def build_anchor_summary(overlap_stats: pd.DataFrame, *, reference_label: str, target_label: str, enhancer_label: str) -> pd.Series:
    anchor_a = f"{_short_name(reference_label)}-like | {_short_name(reference_label)} vs {_short_name(target_label)}"
    anchor_b = f"{_short_name(enhancer_label)}-like | {_short_name(target_label)} vs {_short_name(enhancer_label)}"
    mask = (
        ((overlap_stats["set_a_display"] == anchor_a) & (overlap_stats["set_b_display"] == anchor_b))
        | ((overlap_stats["set_a_display"] == anchor_b) & (overlap_stats["set_b_display"] == anchor_a))
    )
    if not mask.any():
        raise ValueError("Anchor overlap row not found in overlap stats.")
    return overlap_stats.loc[mask].iloc[0]


def plot_auroc_heatmap(summary_df: pd.DataFrame, genotypes: list[str], path: Path) -> None:
    n = len(genotypes)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i, g1 in enumerate(genotypes):
        matrix[i, i] = 1.0
        for j, g2 in enumerate(genotypes):
            if i >= j:
                continue
            row = summary_df[((summary_df["group1"] == g1) & (summary_df["group2"] == g2)) | ((summary_df["group1"] == g2) & (summary_df["group2"] == g1))]
            if row.empty:
                continue
            val = float(row["mean_auroc"].iloc[0])
            matrix[i, j] = val
            matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n), [_short_name(g) for g in genotypes], rotation=40, ha="right")
    ax.set_yticks(range(n), [_short_name(g) for g in genotypes])
    ax.set_title("All-pairs mean AUROC", fontsize=14, fontweight="bold")
    for i in range(n):
        for j in range(n):
            if np.isnan(matrix[i, j]):
                continue
            color = "white" if matrix[i, j] >= 0.72 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean AUROC")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pbx4_overlap_heatmap(ordered_overlap: pd.DataFrame, set_df: pd.DataFrame, path: Path) -> None:
    displays = set_df["set_display"].tolist()
    matrix = np.full((len(displays), len(displays)), np.nan, dtype=float)
    annot = [["" for _ in displays] for _ in displays]
    lookup = ordered_overlap.set_index(["set_a_display", "set_b_display"])

    for i, row_label in enumerate(displays):
        for j, col_label in enumerate(displays):
            row = lookup.loc[(row_label, col_label)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            matrix[i, j] = float(row["jaccard"])
            annot[i][j] = f"{float(row['jaccard']):.0%}\n({int(row['observed_overlap'])})"

    fig_w = max(8.0, len(displays) * 1.2)
    fig_h = max(7.0, len(displays) * 0.95)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(displays)), displays, rotation=45, ha="right")
    ax.set_yticks(range(len(displays)), displays)
    ax.set_title("PBX4 label-set overlap heatmap", fontsize=14, fontweight="bold")
    for i in range(len(displays)):
        for j in range(len(displays)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val >= 0.55 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent overlap (Jaccard)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    *,
    genotypes: list[str],
    auroc_summary: pd.DataFrame,
    anchor_row: pd.Series,
    overlap_stats: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "PBX all-pairs overlap analysis",
        "",
        f"genotypes: {', '.join(genotypes)}",
        f"n_pairs: {int(len(auroc_summary))}",
        "",
        "anchor_hypothesis:",
        f"- set_a: {anchor_row['set_a_display']}",
        f"- set_b: {anchor_row['set_b_display']}",
        f"- observed_overlap: {int(anchor_row['observed_overlap'])}",
        f"- expected_overlap: {float(anchor_row['expected_overlap']):.3f}",
        f"- observed_minus_expected: {float(anchor_row['observed_minus_expected']):.3f}",
        f"- overlap_over_expected: {float(anchor_row['overlap_over_expected']):.3f}",
        f"- p_depletion: {float(anchor_row['p_depletion']):.4g}",
        f"- q_depletion: {float(anchor_row['q_depletion']):.4g}",
        "",
        "top_depleted_overlaps:",
    ]
    depleted = overlap_stats.sort_values(["q_depletion", "observed_minus_expected"]).head(10)
    for _, row in depleted.iterrows():
        lines.append(
            f"- {row['set_a_display']} vs {row['set_b_display']}: "
            f"obs={int(row['observed_overlap'])}, exp={float(row['expected_overlap']):.2f}, "
            f"q_depletion={float(row['q_depletion']):.4g}"
        )

    lines.extend(["", "pairwise_mean_auroc:"])
    for _, row in auroc_summary.sort_values("mean_auroc", ascending=False).iterrows():
        lines.append(
            f"- {row['pair_id']}: mean_auroc={float(row['mean_auroc']):.3f}, "
            f"n_time_bins={int(row['n_time_bins'])}"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    genotypes = [_normalize_genotype(g) for g in args.genotypes]
    df, embryo_meta = load_dataframe(genotypes)
    feature_cols = embedding_features(df, args.embedding_prefix)

    auc_bins_df, labels_df = run_all_pairs(
        df,
        embryo_meta,
        genotypes,
        time_col=args.time_col,
        feature_cols=feature_cols,
        bin_width=args.bin_width,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        min_time_bins=args.min_time_bins,
        late_threshold_hpf=args.late_threshold_hpf,
    )
    auroc_summary = summarize_pairwise_auroc(auc_bins_df)
    pbx4_labels, set_df = build_pbx4_label_sets(labels_df, args.target_label)
    ordered_overlap, overlap_stats = compute_overlap_stats(pbx4_labels, set_df)
    anchor_row = build_anchor_summary(
        overlap_stats,
        reference_label=args.reference_label,
        target_label=args.target_label,
        enhancer_label=args.enhancer_label,
    )

    stem = "pbx_controls_embedding_all_pairs"
    labels_csv = args.results_dir / f"embryo_like_labels_{stem}.csv"
    auc_bins_csv = args.results_dir / f"pairwise_auroc_bins_{stem}.csv"
    auroc_summary_csv = args.results_dir / f"pairwise_auroc_summary_{stem}.csv"
    pbx4_set_csv = args.results_dir / f"pbx4_label_sets_{stem}.csv"
    overlap_long_csv = args.results_dir / f"pbx4_label_overlap_long_{stem}.csv"
    overlap_stats_csv = args.results_dir / f"pbx4_label_overlap_stats_{stem}.csv"
    summary_txt = args.results_dir / f"summary_{stem}.txt"

    auroc_heatmap_fig = args.figures_dir / f"pairwise_auroc_heatmap_{stem}.png"
    overlap_heatmap_fig = args.figures_dir / f"pbx4_label_overlap_heatmap_{stem}.png"

    labels_df.to_csv(labels_csv, index=False)
    auc_bins_df.to_csv(auc_bins_csv, index=False)
    auroc_summary.to_csv(auroc_summary_csv, index=False)
    set_df.to_csv(pbx4_set_csv, index=False)
    ordered_overlap.to_csv(overlap_long_csv, index=False)
    overlap_stats.to_csv(overlap_stats_csv, index=False)

    plot_auroc_heatmap(auroc_summary, genotypes, auroc_heatmap_fig)
    plot_pbx4_overlap_heatmap(ordered_overlap, set_df, overlap_heatmap_fig)
    write_summary_text(
        genotypes=genotypes,
        auroc_summary=auroc_summary,
        anchor_row=anchor_row,
        overlap_stats=overlap_stats,
        path=summary_txt,
    )

    print(f"Wrote labels: {labels_csv}")
    print(f"Wrote AUROC bins: {auc_bins_csv}")
    print(f"Wrote AUROC summary: {auroc_summary_csv}")
    print(f"Wrote PBX4 label sets: {pbx4_set_csv}")
    print(f"Wrote overlap long table: {overlap_long_csv}")
    print(f"Wrote overlap stats: {overlap_stats_csv}")
    print(f"Wrote summary: {summary_txt}")
    print(f"Wrote figure: {auroc_heatmap_fig}")
    print(f"Wrote figure: {overlap_heatmap_fig}")


if __name__ == "__main__":
    main()
