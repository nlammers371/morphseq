"""
NWDB talk (selected): 4 hpf bins, no null, random chance baseline, Het/Homo vs WT.

Outputs (per metric):
- ..._bin4_no_null_sig_legend_outside.(png|pdf)
- ..._bin4_no_null_sig_legend_outside_hpf30line.(png|pdf)  # includes a vertical line at 30 hpf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _drop_null_cols(df):
    df = df.copy()
    for col in ("auroc_null_mean", "auroc_null_std"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _plot_one(
    *,
    res_dir: Path,
    title: str,
    out_base: Path,
    add_hpf30_line: bool,
) -> None:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from analyze.classification.results import MulticlassOVRResults
    from analyze.classification.viz.classification import plot_multiple_aurocs
    from analyze.viz.styling.color_mapping_config import GENOTYPE_SUFFIX_COLORS

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_nwdb_genotype_classification_utils import (
        SIG_THRESHOLD,
        apply_nwdb_axis_overrides,
        apply_nwdb_legend,
        save_figure,
    )

    res = MulticlassOVRResults.from_dir(res_dir)

    sig_threshold = SIG_THRESHOLD
    curves = [
        (("cep290_heterozygous", "cep290_wildtype"), "Het vs WT", GENOTYPE_SUFFIX_COLORS["heterozygous"]),
        (("cep290_homozygous", "cep290_wildtype"), "Homo vs WT", GENOTYPE_SUFFIX_COLORS["homozygous"]),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 5))
    auroc_dfs = {}
    colors = {}
    for (posneg, label, color) in curves:
        auroc_dfs[label] = _drop_null_cols(res[posneg])
        colors[label] = str(color)

    plot_multiple_aurocs(
        auroc_dfs_dict=auroc_dfs,  # type: ignore[arg-type]
        colors_dict=colors,
        title=title,
        ax=ax,
        ylim=(0.3, 1.05),
        sig_threshold=float(sig_threshold),
        show_null_band=False,
        show_significance=True,
        show_sig_legend=True,
        show_chance_line=True,
        chance_label="Random chance",
        chance_linestyle="--",
    )

    if add_hpf30_line:
        ax.axvline(30.0, color="#4d4d4d", linestyle="-", linewidth=2.0, alpha=0.9, label="_nolegend_")

    apply_nwdb_axis_overrides(ax)
    apply_nwdb_legend(ax, outside=True)
    save_figure(fig, out_png=out_base.with_suffix(".png"), out_pdf=out_base.with_suffix(".pdf"), use_tight_layout=False)
    print(f"Saved: {out_base.with_suffix('.png')}")


def main() -> None:
    here = Path(__file__).resolve().parent
    class_root = here / "plot_dir" / "classification_bin4"
    fig_dir = here / "figures" / "classification"

    feature_sets = {
        "curvature": {"label": "Curvature", "results_dir": class_root / "curvature_het_homo_vs_wt"},
        "length": {"label": "Total Length", "results_dir": class_root / "length_het_homo_vs_wt"},
        "embedding": {"label": "VAE Embedding", "results_dir": class_root / "embedding_het_homo_vs_wt"},
    }

    for key, spec in feature_sets.items():
        res_dir = Path(spec["results_dir"])
        if not res_dir.exists():
            raise FileNotFoundError(
                f"Missing classification results dir: {res_dir}. "
                "Run 01_run_reference_genotype_classification_curvature.py with "
                "--bin-width 4 --classification-subdir classification_bin4 first."
            )

        base = fig_dir / f"cep290_ref_{key}_het_and_homo_vs_wt_bin4_no_null_sig_legend_outside"
        _plot_one(
            res_dir=res_dir,
            title=f"{spec['label']} — Het/Homo vs WT (AUROC)",
            out_base=base,
            add_hpf30_line=False,
        )
        _plot_one(
            res_dir=res_dir,
            title=f"{spec['label']} — Het/Homo vs WT (AUROC)",
            out_base=fig_dir / f"{base.name}_hpf30line",
            add_hpf30_line=True,
        )


if __name__ == "__main__":
    main()
