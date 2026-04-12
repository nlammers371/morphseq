"""
PBX no-yolk rerun heatmaps using the faceted heatmap API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import EXPERIMENT_LABEL, REPO_ROOT, resolve_bin_width_roots

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification.viz import plot_auroc_heatmaps

FEATURE_SETS = ["curvature", "length", "embedding"]
GENOTYPE_ORDER = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]


def _present_order(values: pd.Series, preferred_order: list[str]) -> list[str]:
    present = set(values.dropna().astype(str))
    ordered = [value for value in preferred_order if value in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def load_mode_scores(classification_dir: Path, mode_stem: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature_set in FEATURE_SETS:
        path = classification_dir / f"{EXPERIMENT_LABEL}_{mode_stem}_{feature_set}_comparisons.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "feature_set" not in df.columns:
            df["feature_set"] = feature_set
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "positive" in combined.columns and "positive_label" not in combined.columns:
        combined["positive_label"] = combined["positive"]
    if "negative" in combined.columns and "negative_label" not in combined.columns:
        combined["negative_label"] = combined["negative"]
    return combined


def save_mode_heatmaps(
    *,
    classification_dir: Path,
    output_path: Path,
    mode_stem: str,
    title: str,
    sig_threshold: float = 0.01,
) -> Path:
    scores = load_mode_scores(classification_dir, mode_stem)
    if scores.empty:
        raise FileNotFoundError(f"No comparison tables found for mode={mode_stem} in {classification_dir}")

    heatmap_row_order = _present_order(scores["positive_label"], GENOTYPE_ORDER)
    facet_row_order = _present_order(scores["feature_set"], FEATURE_SETS)

    fig = plot_auroc_heatmaps(
        scores,
        heatmap_row="positive_label",
        heatmap_col="time_bin_center",
        facet_row="feature_set",
        facet_col=None,
        heatmap_row_order=heatmap_row_order,
        facet_row_order=facet_row_order,
        title=title,
        x_label="Time (hpf)",
        y_label="Genotype",
        colorbar_label="AUROC",
        sig_threshold=sig_threshold,
        backend="matplotlib",
        cmap="BuPu",
        vmin=0.4,
        vmax=1.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rerun PBX classification heatmaps with the faceted heatmap API.")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Time bin width in hpf.")
    parser.add_argument("--results-subdir", default=None, help="Relative results subdir under the PBX analysis root.")
    parser.add_argument("--figures-subdir", default=None, help="Relative figures subdir under the PBX analysis root.")
    args = parser.parse_args()

    results_dir, figures_dir = resolve_bin_width_roots(
        bin_width=args.bin_width,
        results_subdir=args.results_subdir,
        figures_subdir=args.figures_subdir,
    )
    classification_dir = results_dir / "classification"
    figure_dir = figures_dir / "classification"

    out_path = save_mode_heatmaps(
        classification_dir=classification_dir,
        output_path=figure_dir / f"{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_heatmaps_v2.png",
        mode_stem="all_crispants_vs_wik_ab",
        title="All Crispants vs wik_ab",
    )
    print(out_path)


if __name__ == "__main__":
    main()
