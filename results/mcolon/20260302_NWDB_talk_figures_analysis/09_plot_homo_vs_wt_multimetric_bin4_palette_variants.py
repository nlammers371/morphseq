"""
NWDB talk: Homo vs WT AUROC overlays for curvature, length, and embedding.

This script renders one plot per palette variant, with three metric curves on the
same axes:
- Curvature
- Length
- VAE Embedding

If the requested 10-48 HPF bin-4 classification bundles are missing or do not
cover the requested window, the script reruns classification first and then
renders the figures from the refreshed results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PaletteVariant:
    name: str
    colors: dict[str, str]


FEATURE_SPECS = {
    "curvature": {
        "label": "Curvature",
        "results_dir_name": "curvature_het_homo_vs_wt",
    },
    "length": {
        "label": "Length",
        "results_dir_name": "length_het_homo_vs_wt",
    },
    "embedding": {
        "label": "VAE Embedding",
        "results_dir_name": "embedding_het_homo_vs_wt",
    },
}

PALETTES = (
    PaletteVariant(
        name="neutral_ink",
        colors={
            "Curvature": "#6E5A7E",
            "Length": "#556B7B",
            "VAE Embedding": "#A06A5F",
        },
    ),
    PaletteVariant(
        name="neutral_stone",
        colors={
            "Curvature": "#7C6278",
            "Length": "#5E6F7A",
            "VAE Embedding": "#9C7A68",
        },
    ),
    PaletteVariant(
        name="neutral_soft",
        colors={
            "Curvature": "#8A6F86",
            "Length": "#6F7E88",
            "VAE Embedding": "#AE8376",
        },
    ),
)

BRIGHTER_PALETTES = (
    PaletteVariant(
        name="neutral_ink",
        colors={
            "Curvature": "#B2182B",
            "Length": "#D990AE",
            "VAE Embedding": "#D7B46A",
        },
    ),
    PaletteVariant(
        name="neutral_stone",
        colors={
            "Curvature": "#A1122A",
            "Length": "#E1A3BA",
            "VAE Embedding": "#E1BF7A",
        },
    ),
    PaletteVariant(
        name="neutral_soft",
        colors={
            "Curvature": "#C21F39",
            "Length": "#E9B2C4",
            "VAE Embedding": "#E8CB90",
        },
    ),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render Homo vs WT multi-metric AUROC overlays with palette variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--t-min", type=float, default=10.0, help="Minimum HPF to display.")
    p.add_argument("--t-max", type=float, default=48.0, help="Maximum HPF to display.")
    p.add_argument("--bin-width", type=float, default=4.0, help="Classification bin width in HPF.")
    p.add_argument(
        "--palette-mode",
        choices=["neutral", "brighter"],
        default="neutral",
        help="Palette family to render.",
    )
    return p.parse_args()


def _intish_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _classification_subdir(*, t_min: float, t_max: float, bin_width: float) -> str:
    return f"classification_bin{_intish_token(bin_width)}_{_intish_token(t_min)}_{_intish_token(t_max)}"


def _target_results_root(*, t_min: float, t_max: float, bin_width: float) -> Path:
    here = Path(__file__).resolve().parent
    return here / "plot_dir" / _classification_subdir(t_min=t_min, t_max=t_max, bin_width=bin_width)


def _figure_root() -> Path:
    here = Path(__file__).resolve().parent
    return here / "figures" / "classification" / "homo_vs_wt_multimetric_palettes"


def _results_bundle_dir(feature_key: str, *, t_min: float, t_max: float, bin_width: float) -> Path:
    return _target_results_root(t_min=t_min, t_max=t_max, bin_width=bin_width) / str(
        FEATURE_SPECS[feature_key]["results_dir_name"]
    )


def _bundle_has_requested_support(bundle_dir: Path, *, t_min: float, t_max: float, bin_width: float) -> bool:
    comparisons_path = bundle_dir / "comparisons.parquet"
    metadata_path = bundle_dir / "metadata.json"
    if not comparisons_path.exists() or not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return False

    if float(metadata.get("bin_width", -1.0)) != float(bin_width):
        return False
    if list(metadata.get("groups", [])) != ["cep290_heterozygous", "cep290_homozygous"]:
        return False
    if str(metadata.get("reference", "")) != "cep290_wildtype":
        return False

    df = pd.read_parquet(comparisons_path)
    sub = df[
        (df["positive"].astype(str) == "cep290_homozygous")
        & (df["negative"].astype(str) == "cep290_wildtype")
    ].copy()
    if sub.empty:
        return False

    centers = sorted(sub["time_bin_center"].dropna().astype(float).unique().tolist())
    if not centers:
        return False

    # Bin centers are midpoint-like (e.g. 13, 15, ..., 47 for a 10-48 / bin-2 run),
    # so allow a half-bin offset at each edge while still rejecting late-starting bundles.
    lower_ok = float(min(centers)) <= float(t_min + (1.5 * bin_width))
    upper_ok = float(max(centers)) >= float(t_max - (0.5 * bin_width))
    return bool(lower_ok and upper_ok)


def _have_supported_results(*, t_min: float, t_max: float, bin_width: float) -> bool:
    for feature_key in FEATURE_SPECS:
        if not _bundle_has_requested_support(
            _results_bundle_dir(feature_key, t_min=t_min, t_max=t_max, bin_width=bin_width),
            t_min=t_min,
            t_max=t_max,
            bin_width=bin_width,
        ):
            return False
    return True


def _rerun_classification(*, t_min: float, t_max: float, bin_width: float) -> None:
    here = Path(__file__).resolve().parent
    runner = here / "01_run_reference_genotype_classification_curvature.py"
    cmd = [
        sys.executable,
        str(runner),
        "--t-min",
        str(float(t_min)),
        "--t-max",
        str(float(t_max)),
        "--bin-width",
        str(float(bin_width)),
        "--classification-subdir",
        _classification_subdir(t_min=t_min, t_max=t_max, bin_width=bin_width),
    ]
    print(f"Re-running classification for {t_min:g}-{t_max:g} HPF support at {bin_width:g}-HPF bins...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _drop_null_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("auroc_null_mean", "auroc_null_std"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _load_metric_curves(*, t_min: float, t_max: float, bin_width: float) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    for feature_key, spec in FEATURE_SPECS.items():
        bundle_dir = _results_bundle_dir(feature_key, t_min=t_min, t_max=t_max, bin_width=bin_width)
        comparisons_path = bundle_dir / "comparisons.parquet"
        df = pd.read_parquet(comparisons_path)
        sub = df[
            (df["positive"].astype(str) == "cep290_homozygous")
            & (df["negative"].astype(str) == "cep290_wildtype")
        ].copy()
        sub = _drop_null_cols(sub)
        sub = sub[sub["time_bin_center"].between(float(t_min), float(t_max), inclusive="both")].copy()
        curves[str(spec["label"])] = sub.sort_values("time_bin_center")
    return curves


def _curves_have_requested_support(
    curves: dict[str, pd.DataFrame], *, t_min: float, t_max: float, bin_width: float
) -> bool:
    for curve_df in curves.values():
        if curve_df.empty:
            return False
        centers = sorted(curve_df["time_bin_center"].dropna().astype(float).unique().tolist())
        if not centers:
            return False
        lower_ok = float(min(centers)) <= float(t_min + (1.5 * bin_width))
        upper_ok = float(max(centers)) >= float(t_max - (0.5 * bin_width))
        if not (lower_ok and upper_ok):
            return False
    return True


def _render_palette_variant(
    *,
    palette: PaletteVariant,
    curves: dict[str, pd.DataFrame],
    t_min: float,
    t_max: float,
    bin_width: float,
) -> None:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from analyze.classification.viz.classification import plot_multiple_aurocs

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_nwdb_genotype_classification_utils import apply_nwdb_legend, save_figure

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_multiple_aurocs(
        auroc_dfs_dict=curves,
        colors_dict=palette.colors,
        title="Homo vs WT (AUROC)",
        ax=ax,
        ylim=(0.3, 1.05),
        sig_threshold=0.01,
        show_null_band=False,
        show_significance=True,
        show_sig_legend=True,
        show_chance_line=True,
        chance_label="Random chance",
        chance_linestyle="--",
    )
    metric_labels = set(curves.keys())
    for line in ax.lines:
        if str(line.get_label()) in metric_labels:
            line.set_linewidth(3.2)
            line.set_markersize(7.5)
        elif str(line.get_label()) == "Random chance":
            line.set_linewidth(1.8)
    for collection in ax.collections:
        if hasattr(collection, "get_sizes") and collection.get_offsets().size:
            collection.set_sizes([260.0])
            if hasattr(collection, "set_linewidths"):
                collection.set_linewidths([3.0])
    ax.set_xlim((float(t_min), float(t_max)))
    ax.set_ylabel("Detection Accuracy", fontsize=22)
    ax.set_xlabel("Hours Post Fertilization (hpf)", fontsize=22)
    ax.set_title("Homo vs WT (AUROC)", fontsize=24, pad=10)
    ax.tick_params(axis="both", labelsize=18)
    apply_nwdb_legend(ax, outside=True)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(18)

    fig_dir = _figure_root()
    out_base = fig_dir / (
        f"cep290_ref_homo_vs_wt_curvature_length_embedding_bin{_intish_token(bin_width)}_"
        f"{_intish_token(t_min)}_{_intish_token(t_max)}_"
        f"no_null_sig_legend_outside_{palette.name}"
    )
    save_figure(
        fig,
        out_png=out_base.with_suffix(".png"),
        out_pdf=out_base.with_suffix(".pdf"),
        tight_layout_rect=(0.0, 0.0, 0.80, 1.0),
        use_tight_layout=True,
    )
    print(f"Saved: {out_base.with_suffix('.png')}")


def main() -> None:
    args = _parse_args()
    t_min = float(args.t_min)
    t_max = float(args.t_max)
    bin_width = float(args.bin_width)
    palettes = BRIGHTER_PALETTES if str(args.palette_mode) == "brighter" else PALETTES

    if not _have_supported_results(t_min=t_min, t_max=t_max, bin_width=bin_width):
        _rerun_classification(t_min=t_min, t_max=t_max, bin_width=bin_width)

    curves = _load_metric_curves(t_min=t_min, t_max=t_max, bin_width=bin_width)
    if not _curves_have_requested_support(curves, t_min=t_min, t_max=t_max, bin_width=bin_width):
        raise RuntimeError(
            "Loaded classification curves do not span the requested display window "
            "for one or more metrics."
        )
    for palette in palettes:
        _render_palette_variant(
            palette=palette,
            curves=curves,
            t_min=t_min,
            t_max=t_max,
            bin_width=bin_width,
        )


if __name__ == "__main__":
    main()
