"""
NWDB backup: genotype classification plots (with null + significance) — Het vs WT.

Generates one figure per feature set with:
- permutation null band (mean ± std)
- open-circle markers at p ≤ 0.01
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from analyze.classification.results import MulticlassOVRResults
    from analyze.classification.viz.classification import plot_multiple_aurocs
    from analyze.viz.styling.color_mapping_config import GENOTYPE_SUFFIX_COLORS

    here = Path(__file__).resolve().parent
    class_root = here / "plot_dir" / "classification"
    fig_dir = here / "figures" / "classification"

    sys.path.insert(0, str(here))
    from _plot_nwdb_genotype_classification_utils import (
        SIG_THRESHOLD,
        apply_nwdb_axis_overrides,
        apply_nwdb_legend,
        save_figure,
    )

    comparisons = ("cep290_heterozygous", "cep290_wildtype")
    sig_threshold = SIG_THRESHOLD

    feature_sets = {
        "curvature": {
            "label": "Curvature",
            "color": GENOTYPE_SUFFIX_COLORS["heterozygous"],
            "results_dir": class_root / "curvature_het_homo_vs_wt",
        },
        "length": {
            "label": "Total Length",
            "color": GENOTYPE_SUFFIX_COLORS["heterozygous"],
            "results_dir": class_root / "length_het_homo_vs_wt",
        },
        "embedding": {
            "label": "VAE Embedding",
            "color": GENOTYPE_SUFFIX_COLORS["heterozygous"],
            "results_dir": class_root / "embedding_het_homo_vs_wt",
        },
    }

    for key, spec in feature_sets.items():
        res_dir = Path(spec["results_dir"])
        if not res_dir.exists():
            raise FileNotFoundError(f"Missing classification results dir: {res_dir}")

        res = MulticlassOVRResults.from_dir(res_dir)
        df = res[comparisons].copy()

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_multiple_aurocs(
            auroc_dfs_dict={"Het vs WT": df},
            colors_dict={"Het vs WT": str(spec["color"])},
            title=f"{spec['label']} — Het vs WT (AUROC, null band; p ≤ {sig_threshold:g})",
            ax=ax,
            ylim=(0.3, 1.05),
            sig_threshold=float(sig_threshold),
            show_chance_line=True,
            chance_label="Random chance",
            chance_linestyle="--",
        )
        apply_nwdb_axis_overrides(ax)
        for outside, rect, close in (
            (False, None, False),
            (True, (0.0, 0.0, 0.80, 1.0), True),
        ):
            apply_nwdb_legend(ax, outside=outside)
            extra = "_legend_outside" if outside else ""
            out_png = fig_dir / f"cep290_ref_{key}_het_vs_wt_with_null{extra}.png"
            out_pdf = fig_dir / f"cep290_ref_{key}_het_vs_wt_with_null{extra}.pdf"
            save_figure(fig, out_png=out_png, out_pdf=out_pdf, tight_layout_rect=rect, close=close)
            print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
