"""
Visualization of leave-one-experiment-out (LOEO) validation results.

Reads:
  leave_one_experiment_out_predictions.csv
  leave_one_experiment_out_summary.csv

Saves plots to: plots/ subdirectory

Usage:
    conda run -n morphseq-env --no-capture-output python plot_loeo_results.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOT_DIR = HERE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

LABELS = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
STATUSES = ["assigned", "ambiguous", "low_density", "not_evaluated"]


def load_data():
    pred_path = HERE / "leave_one_experiment_out_predictions.csv"
    sum_path = HERE / "leave_one_experiment_out_summary.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Run run_loeo_validation.py first. Missing: {pred_path}")
    pred_df = pd.read_csv(pred_path)
    sum_df = pd.read_csv(sum_path)
    return pred_df, sum_df


def plot_per_experiment_accuracy(sum_df):
    fig, ax = plt.subplots(figsize=(9, 4))
    exps = sum_df["heldout_experiment_id"].tolist()
    x = np.arange(len(exps))
    w = 0.35
    ax.bar(x - w/2, sum_df["accuracy_all_evaluated"], w, label="All evaluated", color="#4878CF")
    ax.bar(x + w/2, sum_df["accuracy_assigned_only"], w, label="Assigned only", color="#6ACC65")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("LOEO accuracy per held-out experiment")
    ax.legend()
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="chance (4 classes)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loeo_accuracy_per_experiment.png", dpi=150)
    plt.close(fig)
    print(f"  Saved loeo_accuracy_per_experiment.png")


def plot_confusion_matrix(pred_df, subset_name, subset_mask=None):
    df = pred_df.copy()
    if subset_mask is not None:
        df = df[subset_mask]
    df = df[df["true_label"].notna() & df["predicted_label"].notna()]
    df = df[df["predicted_label"] != "not_assigned"]

    valid_labels = [l for l in LABELS if (df["true_label"] == l).any() or (df["predicted_label"] == l).any()]
    n = len(valid_labels)
    mat = np.zeros((n, n), dtype=int)
    label_idx = {l: i for i, l in enumerate(valid_labels)}

    for _, row in df.iterrows():
        true_i = label_idx.get(row["true_label"])
        pred_i = label_idx.get(row["predicted_label"])
        if true_i is not None and pred_i is not None:
            mat[true_i, pred_i] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(valid_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(valid_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {subset_name}")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() * 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fname = f"confusion_matrix_{subset_name.replace(' ', '_')}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_confidence_by_correctness(pred_df):
    df = pred_df[
        pred_df["true_label"].notna()
        & pred_df["predicted_label"].notna()
        & pred_df["embryo_confidence"].notna()
        & (pred_df["predicted_label"] != "not_assigned")
    ].copy()
    df["correct"] = df["true_label"] == df["predicted_label"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, mask, color in [("Correct", df["correct"], "#6ACC65"), ("Incorrect", ~df["correct"], "#D65F5F")]:
        vals = df.loc[mask, "embryo_confidence"].dropna()
        ax.hist(vals, bins=20, alpha=0.6, color=color, label=f"{label} (n={len(vals)})")
    ax.set_xlabel("embryo_confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution by prediction correctness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confidence_by_correctness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved confidence_by_correctness.png")


def plot_accuracy_by_confidence_bin(pred_df):
    df = pred_df[
        pred_df["true_label"].notna()
        & pred_df["predicted_label"].notna()
        & pred_df["embryo_confidence"].notna()
        & (pred_df["predicted_label"] != "not_assigned")
    ].copy()
    df["correct"] = df["true_label"] == df["predicted_label"]
    df["conf_bin"] = pd.cut(df["embryo_confidence"], bins=np.linspace(0, 1, 6), include_lowest=True)

    bin_acc = df.groupby("conf_bin")["correct"].agg(["mean", "count"]).reset_index()
    bin_labels = [str(b) for b in bin_acc["conf_bin"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(bin_acc)), bin_acc["mean"], color="#4878CF")
    ax.set_xticks(range(len(bin_acc)))
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by embryo_confidence bin")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8)
    for i, (acc, n) in enumerate(zip(bin_acc["mean"], bin_acc["count"])):
        ax.text(i, acc + 0.02, f"n={n}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "accuracy_by_confidence_bin.png", dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy_by_confidence_bin.png")


def plot_accuracy_by_status(pred_df):
    df = pred_df[
        pred_df["true_label"].notna()
        & pred_df["predicted_label"].notna()
        & (pred_df["predicted_label"] != "not_assigned")
    ].copy()
    df["correct"] = df["true_label"] == df["predicted_label"]

    statuses = [s for s in STATUSES if s in df["status"].values and s != "not_evaluated"]
    accs = []
    ns = []
    for s in statuses:
        sub = df[df["status"] == s]
        accs.append(sub["correct"].mean() if len(sub) > 0 else np.nan)
        ns.append(len(sub))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(statuses)), accs, color=["#6ACC65", "#D65F5F", "#FFB347"])
    ax.set_xticks(range(len(statuses)))
    ax.set_xticklabels(statuses, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by status")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8)
    for i, (acc, n) in enumerate(zip(accs, ns)):
        if not np.isnan(acc):
            ax.text(i, acc + 0.02, f"n={n}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "accuracy_by_status.png", dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy_by_status.png")


def plot_n_images_distribution(pred_df):
    df = pred_df[pred_df["n_images"].notna()].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    max_n = int(df["n_images"].max())
    bins = np.arange(0.5, max_n + 1.5, 1)
    ax.hist(df["n_images"], bins=bins, color="#4878CF", edgecolor="white")
    ax.set_xlabel("n_images per embryo")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of images per query embryo")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "n_images_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved n_images_distribution.png")


def main():
    print("Loading LOEO results ...")
    pred_df, sum_df = load_data()
    print(f"  Predictions: {len(pred_df)} embryo-folds")
    print(f"  Summary: {len(sum_df)} folds")

    print("Generating plots ...")
    plot_per_experiment_accuracy(sum_df)
    plot_confusion_matrix(pred_df, "all_evaluated")
    plot_confusion_matrix(pred_df, "assigned_only", pred_df["status"] == "assigned")
    plot_confidence_by_correctness(pred_df)
    plot_accuracy_by_confidence_bin(pred_df)
    plot_accuracy_by_status(pred_df)
    plot_n_images_distribution(pred_df)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
