from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

cache_root = Path("/tmp") / "morphseq_compare_bridge_wik_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "results/mcolon/20260326_pbx_crispant_analysis/scripts"))

from phenotypic_positioning.plots import build_color_palette


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare bridge multiclass aligned UMAPs with and without wik_ab.")
    parser.add_argument(
        "--without-wik-path",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_multiclass_bridge_bin4_perm500" / "multiclass_aligned_umap_2d_coordinates.csv",
    )
    parser.add_argument(
        "--with-wik-path",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_multiclass_bridge_bin4_perm500_with_wik_ab" / "multiclass_aligned_umap_2d_coordinates.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "figures" / "bridge_with_without_wik_ab_aligned_umap_3x2.png",
    )
    parser.add_argument("--snapshot-times", nargs="+", type=float, default=[26.0, 54.0, 78.0])
    return parser.parse_args()


def _nearest_time(df: pd.DataFrame, target: float) -> float:
    centers = sorted(df["time_bin_center"].dropna().unique())
    if not centers:
        raise ValueError("No time_bin_center values found.")
    return min(centers, key=lambda x: abs(float(x) - float(target)))


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, time_center: float, palette: dict[str, str], title: str) -> None:
    panel = df[df["time_bin_center"] == time_center].copy()
    for genotype, sub in panel.groupby("genotype", observed=True):
        ax.scatter(
            sub["UMAP_1"],
            sub["UMAP_2"],
            s=18,
            alpha=0.8,
            c=palette.get(str(genotype), "#666666"),
            label=str(genotype),
            linewidths=0,
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.15)


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    without_df = pd.read_csv(args.without_wik_path)
    with_df = pd.read_csv(args.with_wik_path)

    class_labels = sorted(set(without_df["genotype"].dropna().astype(str)) | set(with_df["genotype"].dropna().astype(str)))
    palette = build_color_palette(class_labels)

    nearest_times = [(_nearest_time(without_df, t), _nearest_time(with_df, t)) for t in args.snapshot_times]

    fig, axes = plt.subplots(nrows=len(args.snapshot_times), ncols=2, figsize=(12, 15), squeeze=False)

    for row_idx, target in enumerate(args.snapshot_times):
        no_wik_time, with_wik_time = nearest_times[row_idx]
        _plot_panel(
            axes[row_idx][0],
            without_df,
            no_wik_time,
            palette,
            f"Without wik_ab | target {target:.0f} hpf | shown {no_wik_time:.0f} hpf",
        )
        _plot_panel(
            axes[row_idx][1],
            with_df,
            with_wik_time,
            palette,
            f"With wik_ab | target {target:.0f} hpf | shown {with_wik_time:.0f} hpf",
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(5, len(by_label)), frameon=False, title="Genotype")
    fig.suptitle("Bridge multiclass aligned UMAPs: without vs with wik_ab", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
