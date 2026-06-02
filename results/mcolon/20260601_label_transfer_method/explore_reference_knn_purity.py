"""
Explore KNN purity of the reference label structure.

For each reference image, find its K nearest reference neighbors (excluding self),
and ask: are those neighbors the same label as the focal image?

This tells us:
- How separable the labels are in embedding space
- Where mixing occurs (which label pairs bleed into each other)
- What the "natural" purity ceiling is — i.e., even with perfect label transfer,
  what accuracy could we expect if labels are mixed in feature space?
- Whether mixing is symmetric (LtH <-> NP blends both ways) or asymmetric

This informs whether a null distribution is needed and what form it should take.

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python explore_reference_knn_purity.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

DATA_PATH = ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
OUT_DIR = HERE / "plots"
OUT_DIR.mkdir(exist_ok=True)

LABEL_COL = "cluster_categories"
EMBRYO_COL = "embryo_id"
SNIP_COL = "snip_id"
TIME_COL = "predicted_stage_hpf"
MAIN_LABELS = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
MIN_HPF = 30.0
MAX_HPF = 48.0
K = 15


def get_feature_cols(df):
    return sorted([c for c in df.columns if c.startswith("z_mu_b_")])


def compute_knn_purity(X, labels, k):
    """
    For each point, find k nearest neighbors (excluding self).
    Returns array of shape (n,) with fraction of neighbors sharing the same label.
    Also returns the full neighbor label matrix for confusion analysis.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    dists, indices = nn.kneighbors(X)

    # Drop self (rank 0)
    neighbor_indices = indices[:, 1:]   # (n, k)
    neighbor_labels = labels[neighbor_indices]  # (n, k)

    focal_labels = labels[:, np.newaxis]  # (n, 1)
    purity = (neighbor_labels == focal_labels).mean(axis=1)  # (n,)

    return purity, neighbor_labels


def neighbor_label_confusion(focal_labels, neighbor_labels, all_labels):
    """
    For each focal label, what fraction of its neighbors have each label?
    Returns a DataFrame: rows=focal label, cols=neighbor label, values=fraction.
    """
    records = []
    for fl in all_labels:
        mask = focal_labels == fl
        if mask.sum() == 0:
            continue
        nl = neighbor_labels[mask].ravel()
        total = len(nl)
        row = {fl2: (nl == fl2).sum() / total for fl2 in all_labels}
        row["focal_label"] = fl
        row["n_images"] = mask.sum()
        records.append(row)
    return pd.DataFrame(records).set_index("focal_label")


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df[LABEL_COL].notna()].copy()
    df = df[(df[TIME_COL] >= MIN_HPF) & (df[TIME_COL] <= MAX_HPF)].copy()
    df = df[df[LABEL_COL].isin(MAIN_LABELS)].copy()
    df = df.reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    print(f"  {len(df)} labeled images, {len(feature_cols)} features, {df[LABEL_COL].nunique()} labels")
    print(f"  Label distribution:\n{df[LABEL_COL].value_counts()}\n")

    X = df[feature_cols].values.astype(float)
    labels = df[LABEL_COL].values

    # --- KNN purity ---
    print(f"Computing KNN purity (k={K}) ...")
    purity, neighbor_labels = compute_knn_purity(X, labels, K)
    df["knn_purity"] = purity

    print("\n=== KNN purity by label (mean fraction of neighbors sharing same label) ===")
    purity_by_label = df.groupby(LABEL_COL)["knn_purity"].describe()
    print(purity_by_label.round(3))

    # --- Neighbor label confusion ---
    print("\n=== Neighbor label confusion matrix (row=focal, col=neighbor, values=fraction) ===")
    confusion = neighbor_label_confusion(labels, neighbor_labels, MAIN_LABELS)
    print(confusion[MAIN_LABELS].round(3))

    # --- What fraction of images have majority-wrong neighbors? ---
    print("\n=== Images with purity < 0.5 (neighbors disagree with own label) ===")
    for lbl in MAIN_LABELS:
        mask = labels == lbl
        low_purity = (purity[mask] < 0.5).sum()
        total = mask.sum()
        print(f"  {lbl}: {low_purity}/{total} ({100*low_purity/total:.1f}%) have purity < 0.5")

    # --- Purity at the embryo level ---
    print("\n=== Embryo-level purity (mean image purity per embryo) ===")
    df["embryo_knn_purity"] = df.groupby(EMBRYO_COL)["knn_purity"].transform("mean")
    embryo_df = df.groupby(EMBRYO_COL).agg(
        label=(LABEL_COL, "first"),
        mean_knn_purity=("knn_purity", "mean"),
        min_knn_purity=("knn_purity", "min"),
    ).reset_index()
    print(embryo_df.groupby("label")[["mean_knn_purity","min_knn_purity"]].describe().round(3))

    # --- Key question: are LtH/NP mixed? ---
    print("\n=== LtH <-> NP mixing detail ===")
    lth_mask = labels == "Low_to_High"
    np_mask = labels == "Not Penetrant"
    lth_neighbors = neighbor_labels[lth_mask]
    np_neighbors = neighbor_labels[np_mask]
    print(f"  LtH images: fraction of neighbors that are NP: {(lth_neighbors == 'Not Penetrant').mean():.3f}")
    print(f"  NP images:  fraction of neighbors that are LtH: {(np_neighbors == 'Low_to_High').mean():.3f}")
    print(f"  LtH images: fraction of neighbors that are LtH: {(lth_neighbors == 'Low_to_High').mean():.3f}")
    print(f"  NP images:  fraction of neighbors that are NP:  {(np_neighbors == 'Not Penetrant').mean():.3f}")

    # --- Save purity CSV ---
    out_csv = HERE / "reference_knn_purity.csv"
    df[[EMBRYO_COL, SNIP_COL, LABEL_COL, TIME_COL, "knn_purity"]].to_csv(out_csv, index=False)
    print(f"\nSaved image-level purity to {out_csv}")

    # --- Plots ---
    _plot_purity_distributions(df, purity, labels)
    _plot_confusion_heatmap(confusion)
    _plot_purity_vs_hpf(df, labels)

    print(f"\nPlots saved to {OUT_DIR}/")


def _plot_purity_distributions(df, purity, labels):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    axes = axes.ravel()
    colors = {"Low_to_High": "#E6194B", "High_to_Low": "#3CB44B",
               "Intermediate": "#4363D8", "Not Penetrant": "#F58231"}
    for ax, lbl in zip(axes, MAIN_LABELS):
        mask = labels == lbl
        ax.hist(purity[mask], bins=20, range=(0, 1), color=colors.get(lbl, "gray"), edgecolor="white")
        ax.axvline(purity[mask].mean(), color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"{lbl}\n(n={mask.sum()}, mean={purity[mask].mean():.2f})", fontsize=9)
        ax.set_xlabel("KNN purity (fraction same-label neighbors)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
    fig.suptitle(f"Reference KNN purity distributions (k={K}, 30–48 hpf)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reference_knn_purity_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved reference_knn_purity_distributions.png")


def _plot_confusion_heatmap(confusion):
    mat = confusion[MAIN_LABELS].values.astype(float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(MAIN_LABELS)))
    ax.set_yticks(range(len(MAIN_LABELS)))
    ax.set_xticklabels(MAIN_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(MAIN_LABELS, fontsize=8)
    ax.set_xlabel("Neighbor label")
    ax.set_ylabel("Focal label")
    ax.set_title(f"Reference neighbor label mixing (k={K})\nfraction of focal-label neighbors by label", fontsize=9)
    for i in range(len(MAIN_LABELS)):
        for j in range(len(MAIN_LABELS)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reference_neighbor_label_mixing.png", dpi=150)
    plt.close(fig)
    print("  Saved reference_neighbor_label_mixing.png")


def _plot_purity_vs_hpf(df, labels):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    colors = {"Low_to_High": "#E6194B", "High_to_Low": "#3CB44B",
               "Intermediate": "#4363D8", "Not Penetrant": "#F58231"}
    for ax, lbl in zip(axes, MAIN_LABELS):
        sub = df[df[LABEL_COL] == lbl]
        ax.scatter(sub[TIME_COL], sub["knn_purity"], alpha=0.3, s=8,
                   color=colors.get(lbl, "gray"), rasterized=True)
        # Running mean
        sub_sorted = sub.sort_values(TIME_COL)
        rolling = sub_sorted["knn_purity"].rolling(50, min_periods=5, center=True).mean()
        ax.plot(sub_sorted[TIME_COL].values, rolling.values, color="black", linewidth=1.5)
        ax.set_title(lbl, fontsize=9)
        ax.set_ylabel("KNN purity")
        ax.set_xlabel("predicted_stage_hpf")
        ax.set_ylim(0, 1.05)
    fig.suptitle("KNN purity vs developmental time (30–48 hpf)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reference_knn_purity_vs_hpf.png", dpi=150)
    plt.close(fig)
    print("  Saved reference_knn_purity_vs_hpf.png")


if __name__ == "__main__":
    main()
