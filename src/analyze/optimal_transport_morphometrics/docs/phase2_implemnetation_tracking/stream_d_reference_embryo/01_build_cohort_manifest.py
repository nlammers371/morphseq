#!/usr/bin/env python3
"""Build cohort manifests for Phase-2 OT reference/deviation analyses.

Selection policy:
1) Maximize 24-48 hpf bin coverage (2-hour target bins)
2) Minimize curvature among tied coverage candidates

Outputs are written under stream_d_reference_embryo/cohort_selection/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CSV = (
    PROJECT_ROOT
    / "results"
    / "mcolon"
    / "20251229_cep290_phenotype_extraction"
    / "final_data"
    / "embryo_data_with_labels.csv"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "cohort_selection"


def _mode_or_na(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return "NA"
    mode = s.mode()
    if mode.empty:
        return "NA"
    return str(mode.iloc[0])


def _select_nearest_rows_for_bins(
    embryo_df: pd.DataFrame,
    bins_hpf: Sequence[float],
    stage_col: str,
    tolerance_hpf: float,
) -> Dict[float, pd.Series]:
    out: Dict[float, pd.Series] = {}
    for b in bins_hpf:
        idx = (embryo_df[stage_col] - b).abs().idxmin()
        row = embryo_df.loc[idx]
        err = abs(float(row[stage_col]) - float(b))
        if err <= tolerance_hpf:
            out[float(b)] = row
    return out


def _build_embryo_qc_table(
    df: pd.DataFrame,
    bins_hpf: Sequence[float],
    stage_col: str,
    curvature_col: str,
    tolerance_hpf: float,
) -> pd.DataFrame:
    rows: List[Dict] = []
    n_bins = len(bins_hpf)
    for embryo_id, g in df.groupby("embryo_id", sort=False):
        g = g.sort_values(stage_col)
        nearest = _select_nearest_rows_for_bins(
            g,
            bins_hpf=bins_hpf,
            stage_col=stage_col,
            tolerance_hpf=tolerance_hpf,
        )
        matched_rows = [nearest[b] for b in bins_hpf if b in nearest]
        matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame(columns=g.columns)

        rows.append(
            {
                "embryo_id": embryo_id,
                "genotype": _mode_or_na(g["genotype"]) if "genotype" in g.columns else "NA",
                "experiment_date": _mode_or_na(g["experiment_date"]) if "experiment_date" in g.columns else "NA",
                "well": _mode_or_na(g["well"]) if "well" in g.columns else "NA",
                "n_rows_window": int(len(g)),
                "n_frames_window": int(g["frame_index"].nunique()),
                "n_bins_covered": int(len(matched_rows)),
                "coverage_frac": float(len(matched_rows)) / float(n_bins),
                "missing_bins": int(n_bins - len(matched_rows)),
                "curvature_median": float(matched_df[curvature_col].median()) if not matched_df.empty else np.nan,
                "curvature_mean": float(matched_df[curvature_col].mean()) if not matched_df.empty else np.nan,
                "hpf_min_window": float(g[stage_col].min()),
                "hpf_max_window": float(g[stage_col].max()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["n_bins_covered", "curvature_median", "n_frames_window", "embryo_id"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    return out


def _pick_cohorts(
    qc_df: pd.DataFrame,
    wt_label: str,
    mutant_label: str,
    n_ref_wt: int,
    n_holdout_wt: int,
    n_mutants: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wt_df = qc_df[qc_df["genotype"] == wt_label].copy()
    mut_df = qc_df[qc_df["genotype"] == mutant_label].copy()

    if len(wt_df) < (n_ref_wt + n_holdout_wt):
        raise ValueError(
            f"Not enough WT embryos for refs+holdout: need {n_ref_wt + n_holdout_wt}, have {len(wt_df)}."
        )
    if len(mut_df) < n_mutants:
        raise ValueError(f"Not enough mutant embryos: need {n_mutants}, have {len(mut_df)}.")

    wt_sorted = wt_df.sort_values(
        ["n_bins_covered", "curvature_median", "n_frames_window", "embryo_id"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    mut_sorted = mut_df.sort_values(
        ["n_bins_covered", "curvature_median", "n_frames_window", "embryo_id"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    ref = wt_sorted.iloc[:n_ref_wt].copy()
    ref["set_type"] = "reference_wt"
    ref["set_rank"] = np.arange(1, len(ref) + 1)

    holdout = wt_sorted.iloc[n_ref_wt : n_ref_wt + n_holdout_wt].copy()
    holdout["set_type"] = "heldout_wt"
    holdout["set_rank"] = np.arange(1, len(holdout) + 1)

    mutant = mut_sorted.iloc[:n_mutants].copy()
    mutant["set_type"] = "mutant"
    mutant["set_rank"] = np.arange(1, len(mutant) + 1)

    selected = pd.concat([ref, holdout, mutant], ignore_index=True)
    selected = selected.sort_values(["set_type", "set_rank"]).reset_index(drop=True)
    return selected, wt_sorted


def _build_bin_frame_manifest(
    df_window: pd.DataFrame,
    selected: pd.DataFrame,
    bins_hpf: Sequence[float],
    stage_col: str,
    curvature_col: str,
    tolerance_hpf: float,
) -> pd.DataFrame:
    selected_by_id = selected.set_index("embryo_id")
    rows: List[Dict] = []

    for embryo_id, g in df_window.groupby("embryo_id", sort=False):
        if embryo_id not in selected_by_id.index:
            continue
        g = g.sort_values(stage_col)
        nearest = _select_nearest_rows_for_bins(
            g,
            bins_hpf=bins_hpf,
            stage_col=stage_col,
            tolerance_hpf=tolerance_hpf,
        )
        sel = selected_by_id.loc[embryo_id]
        for b in bins_hpf:
            if float(b) in nearest:
                row = nearest[float(b)]
                rows.append(
                    {
                        "embryo_id": embryo_id,
                        "set_type": sel["set_type"],
                        "set_rank": int(sel["set_rank"]),
                        "genotype": sel["genotype"],
                        "bin_hpf": float(b),
                        "frame_index": int(row["frame_index"]),
                        "matched_stage_hpf": float(row[stage_col]),
                        "stage_abs_err_hpf": abs(float(row[stage_col]) - float(b)),
                        "curvature": float(row[curvature_col]) if pd.notnull(row[curvature_col]) else np.nan,
                        "experiment_date": str(row["experiment_date"]) if "experiment_date" in row else "",
                        "well": str(row["well"]) if "well" in row else "",
                        "analysis_use": True,
                    }
                )
            else:
                rows.append(
                    {
                        "embryo_id": embryo_id,
                        "set_type": sel["set_type"],
                        "set_rank": int(sel["set_rank"]),
                        "genotype": sel["genotype"],
                        "bin_hpf": float(b),
                        "frame_index": np.nan,
                        "matched_stage_hpf": np.nan,
                        "stage_abs_err_hpf": np.nan,
                        "curvature": np.nan,
                        "experiment_date": str(sel["experiment_date"]),
                        "well": str(sel["well"]),
                        "analysis_use": True,
                    }
                )
    out = pd.DataFrame(rows).sort_values(["set_type", "set_rank", "bin_hpf"]).reset_index(drop=True)
    return out


def _build_transition_manifest(
    bin_manifest: pd.DataFrame,
    bins_hpf: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict] = []
    bins = list(bins_hpf)
    for embryo_id, g in bin_manifest.groupby("embryo_id", sort=False):
        g = g.set_index("bin_hpf")
        for i in range(len(bins) - 1):
            b0 = float(bins[i])
            b1 = float(bins[i + 1])
            if b0 not in g.index or b1 not in g.index:
                continue
            r0 = g.loc[b0]
            r1 = g.loc[b1]
            if pd.isna(r0["frame_index"]) or pd.isna(r1["frame_index"]):
                continue
            rows.append(
                {
                    "pair_id": f"{embryo_id}__{int(round(b0)):02d}to{int(round(b1)):02d}",
                    "embryo_id": embryo_id,
                    "set_type": str(r0["set_type"]),
                    "set_rank": int(r0["set_rank"]),
                    "genotype": str(r0["genotype"]),
                    "bin_src_hpf": b0,
                    "bin_tgt_hpf": b1,
                    "frame_src": int(r0["frame_index"]),
                    "frame_tgt": int(r1["frame_index"]),
                    "stage_src_hpf": float(r0["matched_stage_hpf"]),
                    "stage_tgt_hpf": float(r1["matched_stage_hpf"]),
                    "analysis_use": bool(r0["analysis_use"]),
                    "is_control_pair": False,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["set_type", "set_rank", "bin_src_hpf", "embryo_id"]
    ).reset_index(drop=True)


def _find_nearest_frame_for_embryo(
    df: pd.DataFrame,
    embryo_id: str,
    target_hpf: float,
    tolerance_hpf: float,
    stage_col: str,
) -> Tuple[int | None, float | None]:
    g = df[df["embryo_id"] == embryo_id]
    if g.empty:
        return None, None
    idx = (g[stage_col] - target_hpf).abs().idxmin()
    row = g.loc[idx]
    err = abs(float(row[stage_col]) - float(target_hpf))
    if err > tolerance_hpf:
        return None, None
    return int(row["frame_index"]), float(row[stage_col])


def _build_control_pair_manifest(
    df_window: pd.DataFrame,
    control_embryo_a: str,
    control_embryo_b: str,
    target_hpf: float,
    tolerance_hpf: float,
    stage_col: str,
) -> pd.DataFrame:
    fa, sa = _find_nearest_frame_for_embryo(
        df_window, control_embryo_a, target_hpf=target_hpf, tolerance_hpf=tolerance_hpf, stage_col=stage_col
    )
    fb, sb = _find_nearest_frame_for_embryo(
        df_window, control_embryo_b, target_hpf=target_hpf, tolerance_hpf=tolerance_hpf, stage_col=stage_col
    )
    if fa is None or fb is None:
        return pd.DataFrame(columns=["pair_id"])
    return pd.DataFrame(
        [
            {
                "pair_id": f"{control_embryo_a}__f{fa:04d}__to__{control_embryo_b}__f{fb:04d}",
                "embryo_id": f"{control_embryo_a}__to__{control_embryo_b}",
                "set_type": "control_pair",
                "set_rank": 1,
                "genotype": "control_cross_embryo",
                "bin_src_hpf": float(target_hpf),
                "bin_tgt_hpf": float(target_hpf),
                "frame_src": int(fa),
                "frame_tgt": int(fb),
                "stage_src_hpf": float(sa),
                "stage_tgt_hpf": float(sb),
                "analysis_use": False,
                "is_control_pair": True,
                "src_embryo_id": control_embryo_a,
                "tgt_embryo_id": control_embryo_b,
            }
        ]
    )


def _plot_qc_selection(
    qc_df: pd.DataFrame,
    selected: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    geno_colors = {
        "cep290_wildtype": "#1f77b4",
        "cep290_homozygous": "#d62728",
        "cep290_heterozygous": "#2ca02c",
    }
    default_color = "#999999"
    colors = [geno_colors.get(g, default_color) for g in qc_df["genotype"]]
    ax.scatter(
        qc_df["n_bins_covered"],
        qc_df["curvature_median"],
        c=colors,
        s=20,
        alpha=0.45,
        linewidths=0,
        label="All candidates",
    )

    marker_map = {"reference_wt": "o", "heldout_wt": "s", "mutant": "D"}
    for set_type, g in selected.groupby("set_type"):
        ax.scatter(
            g["n_bins_covered"],
            g["curvature_median"],
            s=90,
            marker=marker_map.get(set_type, "o"),
            edgecolor="black",
            linewidths=0.9,
            label=set_type,
        )
    ax.set_xlabel("2h Bins Covered (24-48 hpf)")
    ax.set_ylabel("Median Curvature (lower is straighter)")
    ax.set_title("Cohort Selection QC: Coverage First, Curvature Tie-break")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_selected_bin_heatmap(
    bin_manifest: pd.DataFrame,
    bins_hpf: Sequence[float],
    out_path: Path,
) -> None:
    selected = (
        bin_manifest[["embryo_id", "set_type", "set_rank"]]
        .drop_duplicates()
        .sort_values(["set_type", "set_rank"])
    )
    embryo_order = selected["embryo_id"].tolist()
    mat = np.zeros((len(embryo_order), len(bins_hpf)), dtype=np.float32)
    for i, eid in enumerate(embryo_order):
        g = bin_manifest[bin_manifest["embryo_id"] == eid].set_index("bin_hpf")
        for j, b in enumerate(bins_hpf):
            if float(b) in g.index:
                mat[i, j] = 0.0 if pd.isna(g.loc[float(b), "frame_index"]) else 1.0

    fig, ax = plt.subplots(figsize=(10, max(5, len(embryo_order) * 0.28)), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(bins_hpf)))
    ax.set_xticklabels([f"{int(b)}" for b in bins_hpf], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(embryo_order)))
    ax.set_yticklabels(embryo_order)
    ax.set_xlabel("Target Stage Bin (hpf)")
    ax.set_ylabel("Selected Embryo")
    ax.set_title("Selected Embryos: 2h-Bin Availability")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Has matched frame")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_manifest(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    usecols = [
        "embryo_id",
        "frame_index",
        "predicted_stage_hpf",
        "genotype",
        "mean_curvature_per_um",
        "experiment_date",
        "well",
    ]
    df = pd.read_csv(args.csv, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["embryo_id", "frame_index", args.stage_col]).copy()

    bins_hpf = np.arange(args.start_hpf, args.end_hpf + 1e-9, args.step_hpf, dtype=float)
    df_window = df[(df[args.stage_col] >= args.start_hpf) & (df[args.stage_col] <= args.end_hpf)].copy()

    qc_df = _build_embryo_qc_table(
        df_window,
        bins_hpf=bins_hpf,
        stage_col=args.stage_col,
        curvature_col=args.curvature_col,
        tolerance_hpf=args.match_tolerance_hpf,
    )

    selected, wt_ranked = _pick_cohorts(
        qc_df,
        wt_label=args.wt_label,
        mutant_label=args.mutant_label,
        n_ref_wt=args.n_ref_wt,
        n_holdout_wt=args.n_holdout_wt,
        n_mutants=args.n_mutants,
    )
    bin_manifest = _build_bin_frame_manifest(
        df_window=df_window,
        selected=selected,
        bins_hpf=bins_hpf,
        stage_col=args.stage_col,
        curvature_col=args.curvature_col,
        tolerance_hpf=args.match_tolerance_hpf,
    )
    transitions = _build_transition_manifest(bin_manifest, bins_hpf=bins_hpf)

    control_manifest = _build_control_pair_manifest(
        df_window=df,
        control_embryo_a=args.control_embryo_a,
        control_embryo_b=args.control_embryo_b,
        target_hpf=args.control_target_hpf,
        tolerance_hpf=args.control_tolerance_hpf,
        stage_col=args.stage_col,
    )
    if not control_manifest.empty:
        transitions_all = pd.concat([transitions, control_manifest], ignore_index=True, sort=False)
    else:
        transitions_all = transitions.copy()

    qc_path = output_root / "cohort_qc_table.csv"
    wt_ranked_path = output_root / "wt_ranked_candidates.csv"
    selected_path = output_root / "cohort_selected_embryos.csv"
    bin_manifest_path = output_root / "cohort_bin_frame_manifest.csv"
    transitions_path = output_root / "cohort_transition_manifest.csv"

    qc_df.to_csv(qc_path, index=False)
    wt_ranked.to_csv(wt_ranked_path, index=False)
    selected.to_csv(selected_path, index=False)
    bin_manifest.to_csv(bin_manifest_path, index=False)
    transitions_all.to_csv(transitions_path, index=False)

    _plot_qc_selection(qc_df=qc_df, selected=selected, out_path=output_root / "cohort_qc_scatter.png")
    _plot_selected_bin_heatmap(
        bin_manifest=bin_manifest,
        bins_hpf=bins_hpf,
        out_path=output_root / "cohort_selected_bin_heatmap.png",
    )

    print("Wrote cohort manifests:")
    print(f"  QC table: {qc_path}")
    print(f"  WT ranked: {wt_ranked_path}")
    print(f"  Selected embryos: {selected_path}")
    print(f"  Bin/frame manifest: {bin_manifest_path}")
    print(f"  Transition manifest: {transitions_path}")
    print("Selection summary:")
    for set_type, g in selected.groupby("set_type"):
        print(
            f"  {set_type}: n={len(g)} "
            f"mean_coverage={g['coverage_frac'].mean():.3f} "
            f"median_curvature={g['curvature_median'].median():.6f}"
        )
    if control_manifest.empty:
        print("Control pair: not found within target/tolerance.")
    else:
        row = control_manifest.iloc[0]
        print(
            "Control pair included: "
            f"{row['src_embryo_id']} f{int(row['frame_src']):04d} -> "
            f"{row['tgt_embryo_id']} f{int(row['frame_tgt']):04d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cohort manifests for Phase-2 OT analyses.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage-col", type=str, default="predicted_stage_hpf")
    parser.add_argument("--curvature-col", type=str, default="mean_curvature_per_um")
    parser.add_argument("--start-hpf", type=float, default=24.0)
    parser.add_argument("--end-hpf", type=float, default=48.0)
    parser.add_argument("--step-hpf", type=float, default=2.0)
    parser.add_argument("--match-tolerance-hpf", type=float, default=1.25)
    parser.add_argument("--wt-label", type=str, default="cep290_wildtype")
    parser.add_argument("--mutant-label", type=str, default="cep290_homozygous")
    parser.add_argument("--n-ref-wt", type=int, default=3)
    parser.add_argument("--n-holdout-wt", type=int, default=3)
    parser.add_argument("--n-mutants", type=int, default=20)
    parser.add_argument("--control-embryo-a", type=str, default="20251113_A05_e01")
    parser.add_argument("--control-embryo-b", type=str, default="20251113_E04_e01")
    parser.add_argument("--control-target-hpf", type=float, default=48.0)
    parser.add_argument("--control-tolerance-hpf", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    build_manifest(parse_args())
