from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


FOCAL_PROBES = [
    "inj_ctrl__vs__pbx1b_pbx4_crispant",
    "pbx1b_crispant__vs__pbx1b_pbx4_crispant",
    "pbx1b_pbx4_crispant__vs__pbx4_crispant",
    "pbx1b_pbx4_crispant__vs__wik_ab",
]
TARGET_GENOTYPES = ["inj_ctrl", "wik_ab"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast descriptive zoom-in on focal-reference probes for inj_ctrl vs wik_ab.")
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=Path("results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_positioning_pairwise_bin4_perm500"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_focal_probe_zoom"),
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=Path("results/mcolon/20260329_pbx_crispant_analysis_cont/shared/wikab_injctrl_focal_probe_zoom"),
    )
    parser.add_argument("--anchor-bins", type=str, default="26,54,78")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_anchor_bins(spec: str) -> list[float]:
    return [float(part.strip()) for part in spec.split(",") if part.strip()] or [26.0, 54.0, 78.0]


def nearest_available_bins(available: list[float], anchors: list[float]) -> list[float]:
    chosen: list[float] = []
    for anchor in anchors:
        nearest = min(available, key=lambda v: abs(v - anchor))
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["genotype"].isin(TARGET_GENOTYPES)].copy()
    df["embryo_id"] = df["embryo_id"].astype(str)
    df["experiment_id"] = df["embryo_id"].str.split("_", n=1).str[0]
    df["time_bin_center"] = pd.to_numeric(df["time_bin_center"], errors="coerce")
    df["time_bin"] = pd.to_numeric(df["time_bin"], errors="coerce")
    df = df.dropna(subset=["time_bin_center"]).reset_index(drop=True)
    return df


def single_probe_auroc(values: np.ndarray, labels: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    if np.allclose(vals, vals[0]):
        return 0.5
    return float(roc_auc_score(y, vals))


def summarize_probe_table(df: pd.DataFrame, *, table_name: str, anchor_bins: list[float]) -> pd.DataFrame:
    available = sorted(df["time_bin_center"].dropna().unique().tolist())
    chosen = nearest_available_bins(available, anchor_bins)
    rows: list[dict[str, object]] = []
    for requested_anchor, bin_center in zip(anchor_bins, chosen):
        sub = df[df["time_bin_center"] == bin_center].copy()
        y = sub["genotype"].map({"inj_ctrl": 0, "wik_ab": 1}).to_numpy(dtype=int)
        for probe in FOCAL_PROBES:
            vals = sub[probe].fillna(0.0).to_numpy(dtype=float)
            inj = sub.loc[sub["genotype"] == "inj_ctrl", probe].fillna(0.0).to_numpy(dtype=float)
            wik = sub.loc[sub["genotype"] == "wik_ab", probe].fillna(0.0).to_numpy(dtype=float)
            rows.append(
                {
                    "table": table_name,
                    "requested_anchor": float(requested_anchor),
                    "time_bin_center": float(bin_center),
                    "probe": probe,
                    "n_inj_ctrl": int((sub["genotype"] == "inj_ctrl").sum()),
                    "n_wik_ab": int((sub["genotype"] == "wik_ab").sum()),
                    "auroc": single_probe_auroc(vals, y),
                    "inj_mean": float(np.mean(inj)) if len(inj) else np.nan,
                    "wik_mean": float(np.mean(wik)) if len(wik) else np.nan,
                    "inj_median": float(np.median(inj)) if len(inj) else np.nan,
                    "wik_median": float(np.median(wik)) if len(wik) else np.nan,
                    "mean_diff_inj_minus_wik": float(np.mean(inj) - np.mean(wik)) if len(inj) and len(wik) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_probe_distributions(raw_df: pd.DataFrame, shrunk_df: pd.DataFrame, output_path: Path) -> None:
    anchors = sorted(raw_df["time_bin_center"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(anchors), len(FOCAL_PROBES), figsize=(4.1 * len(FOCAL_PROBES), 3.2 * len(anchors)), squeeze=False)
    palette = {"inj_ctrl": "#2166AC", "wik_ab": "#808080"}
    for row_idx, anchor in enumerate(anchors):
        for col_idx, probe in enumerate(FOCAL_PROBES):
            ax = axes[row_idx, col_idx]
            sub = raw_df[(raw_df["time_bin_center"] == anchor) & (raw_df["probe"] == probe)].copy()
            sub_shrunk = shrunk_df[(shrunk_df["time_bin_center"] == anchor) & (shrunk_df["probe"] == probe)].copy()
            for offset, (label, table) in enumerate([("raw", sub), ("shrunk", sub_shrunk)]):
                if table.empty:
                    continue
                inj = float(table["inj_mean"].iloc[0])
                wik = float(table["wik_mean"].iloc[0])
                ax.bar(offset - 0.15, inj, width=0.28, color=palette["inj_ctrl"])
                ax.bar(offset + 0.15, wik, width=0.28, color=palette["wik_ab"])
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
            if row_idx == 0:
                ax.set_title(probe, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{anchor:.0f} hpf")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["raw", "shrunk"], rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_path: Path, raw_summary: pd.DataFrame, shrunk_summary: pd.DataFrame) -> None:
    lines = [
        "# Wik_ab vs Inj_ctrl Focal Probe Zoom",
        "",
        "Descriptive-only audit: no permutations, no p-values.",
        "",
        "## Raw focal-reference probes",
        raw_summary.to_markdown(index=False),
        "",
        "## Shrunk focal-reference probes",
        shrunk_summary.to_markdown(index=False),
    ]
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_bins = parse_anchor_bins(args.anchor_bins)

    raw_coords = load_table(args.pairwise_dir / "raw_coordinates.csv")
    shrunk_coords = load_table(args.pairwise_dir / "shrunk_coordinates.csv")

    raw_summary = summarize_probe_table(raw_coords, table_name="raw", anchor_bins=anchor_bins)
    shrunk_summary = summarize_probe_table(shrunk_coords, table_name="shrunk", anchor_bins=anchor_bins)

    raw_summary.to_csv(args.output_dir / "raw_focal_probe_summary.csv", index=False)
    shrunk_summary.to_csv(args.output_dir / "shrunk_focal_probe_summary.csv", index=False)
    plot_probe_distributions(raw_summary, shrunk_summary, args.output_dir / "focal_probe_mean_bars.png")
    write_summary(args.output_dir / "FOCAL_PROBE_ZOOM_SUMMARY.md", raw_summary, shrunk_summary)

    manifest = {
        "anchor_bins_requested": anchor_bins,
        "focal_probes": FOCAL_PROBES,
    }
    with open(args.output_dir / "focal_probe_zoom_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    if args.shared_dir and not args.smoke:
        args.shared_dir.parent.mkdir(parents=True, exist_ok=True)
        if args.shared_dir.exists():
            shutil.rmtree(args.shared_dir)
        shutil.copytree(args.output_dir, args.shared_dir)

    print(args.output_dir)
    if args.shared_dir and not args.smoke:
        print(args.shared_dir)


if __name__ == "__main__":
    main()
