"""
Visual test of plot_auroc_heatmaps using real pbx crispant classification results.

Generates several faceted heatmap variants to confirm the engine works end-to-end:

  Plot 1: Single-feature heatmap (embedding), all crispants vs wik_ab
  Plot 2: Faceted by feature_set (embedding + curvature + length), rows=feature_set
  Plot 3: Fixed negative (wik_ab), compare crispants — heatmap_row=positive_label, facet_col forced off
  Plot 4: Annotations on (show_annotations=True)
  Plot 5: Significance borders only (show_significance=True, pval threshold tightened)
  Plot 6: Multi-feature combined df, explicit facet_row=feature_set
"""

import sys
from pathlib import Path

morphseq_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(morphseq_root / "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze.classification.viz import plot_auroc_heatmaps

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "results" / "bin_width_4.0hpf" / "classification"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

emb = pd.read_csv(DATA_DIR / "20260304_20260306_all_crispants_vs_wik_ab_embedding_comparisons.csv")
curv = pd.read_csv(DATA_DIR / "20260304_20260306_all_crispants_vs_wik_ab_curvature_comparisons.csv")
length = pd.read_csv(DATA_DIR / "20260304_20260306_all_crispants_vs_wik_ab_length_comparisons.csv")

# Combined multi-feature dataframe
all_features = pd.concat([emb, curv, length], ignore_index=True)

print(f"Embedding: {emb.shape}, features: {emb['feature_set'].unique()}")
print(f"Combined:  {all_features.shape}, features: {all_features['feature_set'].unique()}")
print(f"Genotypes (positive_label): {sorted(emb['positive_label'].unique())}")
print(f"Time bins: {sorted(emb['time_bin_center'].unique())}")
print()

# ---------------------------------------------------------------------------
# Plot 1: Single feature set — embedding only, all crispants vs wik_ab
# ---------------------------------------------------------------------------
print("Plot 1: Single feature (embedding) heatmap...")
fig = plot_auroc_heatmaps(
    emb,
    heatmap_row="positive_label",
    heatmap_col="time_bin_center",
    facet_row=None,
    facet_col=None,
    title="AUROC: All Crispants vs wik_ab (Embedding)",
    sig_threshold=0.05,
)
out = FIG_DIR / "heatmap_01_single_feature_embedding.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 2: Multi-feature, facet_row=feature_set (auto-inferred)
# ---------------------------------------------------------------------------
print("Plot 2: Multi-feature, auto-inferred facet_row=feature_set...")
fig = plot_auroc_heatmaps(
    all_features,
    heatmap_row="positive_label",
    heatmap_col="time_bin_center",
    # facet_row/col not set → auto-inferred: feature_set→rows
    title="AUROC: All Crispants vs wik_ab (All Features)",
    sig_threshold=0.05,
)
out = FIG_DIR / "heatmap_02_multi_feature_auto_facet.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 3: Explicit facet_row=feature_set, explicit row/col ordering
# ---------------------------------------------------------------------------
print("Plot 3: Explicit facet_row, custom ordering...")
feature_order = ["embedding", "curvature", "length"]
time_order = sorted(all_features["time_bin_center"].unique())
genotype_order = sorted(all_features["positive_label"].unique())

fig = plot_auroc_heatmaps(
    all_features,
    heatmap_row="positive_label",
    heatmap_col="time_bin_center",
    facet_row="feature_set",
    facet_col=None,
    heatmap_row_order=genotype_order,
    heatmap_col_order=time_order,
    facet_row_order=feature_order,
    title="AUROC: All Crispants vs wik_ab (Explicit Ordering)",
    sig_threshold=0.05,
)
out = FIG_DIR / "heatmap_03_explicit_ordering.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 4: With annotations (show_annotations=True)
# ---------------------------------------------------------------------------
print("Plot 4: Annotations enabled...")
fig = plot_auroc_heatmaps(
    emb,
    heatmap_row="positive_label",
    heatmap_col="time_bin_center",
    facet_row=None,
    facet_col=None,
    show_annotations=True,
    annotation_fmt="{:.2f}",
    title="AUROC with Annotations (Embedding)",
    sig_threshold=0.05,
)
out = FIG_DIR / "heatmap_04_with_annotations.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 5: Tight significance threshold (p < 0.01) to see border density
# ---------------------------------------------------------------------------
print("Plot 5: Tight significance threshold (p<0.01)...")
fig = plot_auroc_heatmaps(
    all_features,
    heatmap_row="positive_label",
    heatmap_col="time_bin_center",
    facet_row="feature_set",
    facet_col=None,
    facet_row_order=feature_order,
    sig_threshold=0.01,
    show_significance=True,
    title="AUROC: Significance p<0.01 (All Features)",
)
out = FIG_DIR / "heatmap_05_tight_significance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 6: Curvature only — swap axes (row=time, col=feature) to test flexibility
# ---------------------------------------------------------------------------
print("Plot 6: Swapped axes (heatmap_row=time_bin_center, heatmap_col=positive_label)...")
fig = plot_auroc_heatmaps(
    emb,
    heatmap_row="time_bin_center",
    heatmap_col="positive_label",
    facet_row=None,
    facet_col=None,
    title="AUROC: Transposed (Time × Genotype)",
    sig_threshold=0.05,
    x_label="Genotype",
    y_label="Time (hpf)",
)
out = FIG_DIR / "heatmap_06_transposed_axes.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 7: facet_col=positive_label, heatmap_row=feature_set
#   Each panel = one genotype; rows inside = feature set; cols = time
# ---------------------------------------------------------------------------
print("Plot 7: facet_col=positive_label, heatmap_row=feature_set...")
fig = plot_auroc_heatmaps(
    all_features,
    heatmap_row="feature_set",
    heatmap_col="time_bin_center",
    facet_row=None,
    facet_col="positive_label",
    heatmap_row_order=feature_order,
    heatmap_col_order=time_order,
    sig_threshold=0.05,
    title="AUROC by Genotype (Feature × Time)",
    x_label="Time (hpf)",
    y_label="Feature Set",
)
out = FIG_DIR / "heatmap_07_facet_col_genotype_row_feature.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 8: Both facet_row=feature_set AND facet_col=positive_label
#   Each cell = one feature × genotype combination; matrix = (single-row) × time
# ---------------------------------------------------------------------------
print("Plot 8: facet_row=feature_set, facet_col=positive_label, heatmap_row=negative_label...")
fig = plot_auroc_heatmaps(
    all_features,
    heatmap_row="negative_label",
    heatmap_col="time_bin_center",
    facet_row="feature_set",
    facet_col="positive_label",
    heatmap_col_order=time_order,
    facet_row_order=feature_order,
    sig_threshold=0.05,
    title="AUROC Grid: Feature × Genotype",
    x_label="Time (hpf)",
    y_label="Negative",
)
out = FIG_DIR / "heatmap_08_full_grid_feature_by_genotype.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Colormap + border style comparison
# Each cmap tested with the white+black-halo border (new default)
# ---------------------------------------------------------------------------
print("Colormap + border style comparison plots...")

from analyze.viz.plotting.faceting_engine.ir import HeatmapStyle, ColorbarSpec
from analyze.viz.plotting.faceting_engine.heatmap import build_heatmap_figure
from analyze.viz.plotting.faceting_engine import FacetSpec, render, default_style

cmap_candidates = {
    "magma":       dict(cmap="magma",        vmin=0.4, vmax=1.0),
    "inferno":     dict(cmap="inferno",      vmin=0.4, vmax=1.0),
    "cividis":     dict(cmap="cividis",      vmin=0.4, vmax=1.0),
    "YlOrRd":      dict(cmap="YlOrRd",       vmin=0.4, vmax=1.0),
    "plasma":      dict(cmap="plasma",       vmin=0.4, vmax=1.0),
}

for name, cmap_kwargs in cmap_candidates.items():
    hm_style = HeatmapStyle(
        **cmap_kwargs,
        vcenter=None,
        # White border + black halo — works on any background
        sig_border_color='#ffffff',
        sig_border_width=2.5,
        sig_halo_color='#000000',
        sig_halo_width=4.5,
    )
    fig_data = build_heatmap_figure(
        all_features,
        heatmap_row="positive_label",
        heatmap_col="time_bin_center",
        facet_row="feature_set",
        facet_row_order=feature_order,
        value_col="auroc_obs",
        sig_col="pval",
        sig_threshold=0.05,
        heatmap_style=hm_style,
        colorbar=ColorbarSpec(label="AUROC"),
        title=f"AUROC — {name}  (white border + black halo)",
        subtitle="negative_label: wik_ab",
        x_label="time_bin_center",
        y_label="positive_label",
    )
    fig = render(fig_data, backend="matplotlib", facet=FacetSpec(sharex=False, sharey=False), style=default_style())
    out = FIG_DIR / f"heatmap_border_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
print(f"\nAll figures saved to: {FIG_DIR}")
print("Done.")
