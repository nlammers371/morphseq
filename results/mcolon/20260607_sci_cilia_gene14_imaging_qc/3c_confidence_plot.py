"""
3c - Confidence plot: THE KEY sequencing-greenlight artifact (v1).

v1 = homozygous-phenotype BINARY, per gene:
    b9d2   -> CE vs HTA
    cep290 -> High_to_Low vs Low_to_High
Multi-class (genotype / crispant) is deferred.

WHY homozygous-only: genotype isn't clean (some b9d2 heterozygotes look homozygous), so
the greenlight restricts to the homozygous cohort we actually plan to sequence.

LAYOUT — 5 rows x PER-GENE columns. Columns = collection x support (plot_config.PHENOTYPE_COLUMNS[gene]):
    b9d2   : 14, 18, 30, 48-snapshot(plate02), 48-timeseries(plate01)
    cep290 : 18, 24, 30, 48-snapshot(plate02), 48-timeseries(plate01)
The 48 hpf collection appears TWICE on purpose: plate02 single snapshot vs plate01 30->48
timeseries. Rows 4/5 (reference metrics) should look BETTER for the timeseries column —
that contrast is the entire point of the plot. ALL of a gene's columns are rendered; a
column with no data draws blank with an `n=0` note (the missing-support story is visible).

ROWS:
  1. argmax model prediction  -> bar plot (query predicted-class counts).
  2. query  sequenced prediction probabilities (confidence in what we sequenced).
  3. reference prediction probabilities, stripped with TRUE classes on y (separability).
  4. reference precision & recall per class (key metrics).
  5. reference confusion matrix (where the mass goes / imbalance).
Rows 2 vs 3 are the query/reference pair.

Reference time-bins (4 hpf) do NOT line up 1:1 with the collection-time columns. Mapping
(LOCKED): snapshot column -> the ref bin nearest its collection time (snap_to_design_stage,
+/-2 hpf); timeseries column -> macro-average the ref metrics across the bins it spans
(its own query collection window). Early columns the reference never saw (b9d2 14/18) draw
empty reference rows — honest.

Reads:
    predictions/sequenced_homozygous_phenotype_per_bin.csv  (query rows)
    models/<gene>_homozygous_phenotype_reference_cv.csv     (reference CV strip + confusion)
    models/<gene>_homozygous_phenotype.pkl                  (reference_performance metrics)
Writes:
    plots/confidence/<gene>_confidence.png

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3c_confidence_plot.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from plot_config import PHENOTYPE_COLORS, PHENOTYPE_COLUMNS, snap_to_design_stage  # noqa: E402

PRED_DIR = RUN_DIR / "predictions"
MODEL_DIR = RUN_DIR / "models"
CONF_PLOT_DIR = RUN_DIR / "plots" / "confidence"
CONF_PLOT_DIR.mkdir(parents=True, exist_ok=True)

PER_BIN_PATH = PRED_DIR / "sequenced_homozygous_phenotype_per_bin.csv"

# Per gene: (left class, right class). right = the positive class whose P(.) is the x-axis.
# The spectrum runs left-class color -> gray -> right-class color, pulled from the SHARED
# plot_config.PHENOTYPE_COLORS so the confidence plot matches the 3d feature plot and the
# earlier first-pass figures (b9d2: CE green -> HTA orange; cep290: High_to_Low pink ->
# Low_to_High teal). No per-script color redefinition.
GENE_SPEC = {
    "b9d2":   {"model_id": "b9d2_homozygous_phenotype", "left": "CE", "right": "HTA"},
    "cep290": {"model_id": "cep290_homozygous_phenotype",
               "left": "High_to_Low", "right": "Low_to_High"},
}


def gene_cmap_stops(left: str, right: str) -> list[str]:
    """Spectrum endpoints from the shared PHENOTYPE_COLORS: left -> gray -> right."""
    return [PHENOTYPE_COLORS.get(left, "#808080"), "#E6E6E6", PHENOTYPE_COLORS.get(right, "#808080")]


def ref_bins_for_column(collection_hpf: int, data_source: str,
                        query_sub: pd.DataFrame, bin_centers: list[float]) -> list[float]:
    """Which reference 4-hpf bins map to this collection x support column.

    snapshot   -> the single ref bin nearest the collection time (snap_to_design_stage).
    timeseries -> every ref bin overlapping the query collection window for this column
                  (i.e. the bins the timeseries query rows actually occupy), so the
                  timeseries column aggregates the whole 30->48 window.
    Bins the reference never saw (early collection times) -> [] (empty reference rows).
    """
    bin_centers = sorted(bin_centers)
    if data_source == "timeseries":
        # Use the query timeseries rows' own bins, intersected with what the ref has.
        ts_bins = sorted(query_sub["time_bin_center"].dropna().unique())
        return [b for b in bin_centers if b in set(ts_bins)]
    # snapshot: nearest ref bin to the collection time, within +/- (bin half-width + 2)
    if not bin_centers:
        return []
    nearest = min(bin_centers, key=lambda b: abs(b - collection_hpf))
    return [nearest] if abs(nearest - collection_hpf) <= 4.0 else []


def make_confidence_plot(gene: str, per_bin_all: pd.DataFrame) -> None:
    spec = GENE_SPEC[gene]
    left, right = spec["left"], spec["right"]
    prob_col = f"prob_{right}"
    classes = [left, right]
    columns = PHENOTYPE_COLUMNS[gene]

    print(f"\n[{gene}] confidence plot  (left={left}, right={right})")

    # --- query rows for this gene ---
    q = per_bin_all[per_bin_all["gene"] == gene].copy()
    if prob_col not in q.columns:
        print(f"  {prob_col} missing from per-bin table — skip")
        return

    # --- reference CV + performance ---
    cv = pd.read_csv(MODEL_DIR / f"{spec['model_id']}_reference_cv.csv", low_memory=False)
    with (MODEL_DIR / f"{spec['model_id']}.pkl").open("rb") as fh:
        model = pickle.load(fh)
    perf = model["reference_performance"]
    metrics = perf["per_bin_metrics"]
    confusion = perf["per_bin_confusion"]
    ref_bin_centers = sorted(confusion.keys())

    cmap = LinearSegmentedColormap.from_list("spectrum", gene_cmap_stops(left, right), N=256)
    bar_colors = [PHENOTYPE_COLORS.get(left, "#808080"), PHENOTYPE_COLORS.get(right, "#808080")]
    rng = np.random.default_rng(0)
    ncol = len(columns)

    fig, axes = plt.subplots(5, ncol, figsize=(2.7 * ncol + 1.6, 11.5), squeeze=False)
    row_titles = ["argmax\nprediction", "query P\n(sequenced)",
                  "reference P\n(true class)", "reference\nP / R", "reference\nconfusion"]

    for col, (coll_hpf, source, col_label) in enumerate(columns):
        cell_q = q[(q["collection_time_hpf"] == coll_hpf) & (q["data_source"] == source)]
        ref_bins = ref_bins_for_column(coll_hpf, source, cell_q, ref_bin_centers)

        # ── Row 0: argmax bar (query predicted-class counts) ──────────────────────
        ax = axes[0][col]
        counts = cell_q["predicted_label"].astype(str).value_counts()
        vals = [int(counts.get(c, 0)) for c in classes]
        ax.bar(range(len(classes)), vals,
               color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=7, rotation=20, ha="right")
        ax.set_title(col_label, fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(vals + [1]) * 1.25)
        if col == 0:
            ax.set_ylabel(row_titles[0], fontsize=8)

        # ── Row 1: query probability strip ────────────────────────────────────────
        ax = axes[1][col]
        ax.axvspan(0.45, 0.55, color="#EEEEEE", zorder=0)
        ax.axvline(0.5, color="#777777", lw=0.8, ls=":", zorder=1)
        if not cell_q.empty:
            x = cell_q[prob_col].astype(float).to_numpy()
            y = rng.uniform(-0.18, 0.18, size=len(cell_q))
            ax.scatter(x, y, c=x, cmap=cmap, vmin=0, vmax=1,
                       s=34, edgecolors="black", linewidths=0.25, alpha=0.9, zorder=3)
        ax.text(1.0, 1.0, f"n={len(cell_q)}", ha="right", va="top",
                fontsize=7, transform=ax.transAxes, color="#555555")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.45, 0.45)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
        if col == 0:
            ax.set_ylabel(row_titles[1], fontsize=8)

        # ── Row 2: reference probability strip, true class on y ───────────────────
        ax = axes[2][col]
        ax.axvspan(0.45, 0.55, color="#EEEEEE", zorder=0)
        ax.axvline(0.5, color="#777777", lw=0.8, ls=":", zorder=1)
        ref_cell = cv[cv["time_bin_center"].isin(ref_bins)] if ref_bins else cv.iloc[0:0]
        y_levels = {left: 0.22, right: -0.22}
        for cls in classes:
            sub_cls = ref_cell[ref_cell["true_label"] == cls]
            if sub_cls.empty:
                continue
            x = sub_cls[prob_col].astype(float).to_numpy()
            yy = y_levels[cls] + rng.uniform(-0.12, 0.12, size=len(sub_cls))
            ax.scatter(x, yy, c=x, cmap=cmap, vmin=0, vmax=1,
                       s=22, edgecolors="black", linewidths=0.15, alpha=0.35, zorder=3)
        ax.text(1.0, 1.0, f"n={len(ref_cell)}", ha="right", va="top",
                fontsize=7, transform=ax.transAxes, color="#555555")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.45, 0.45)
        ax.set_yticks([0.22, -0.22])
        ax.set_yticklabels([f"t:{left[:6]}", f"t:{right[:6]}"] if col == 0 else ["", ""],
                           fontsize=6)
        ax.set_xlabel(f"{left} ← P({right}) → {right}", fontsize=6)
        ax.tick_params(axis="x", labelsize=7)
        if col == 0:
            ax.set_ylabel(row_titles[2], fontsize=8)

        # ── Row 3: reference precision & recall (macro across the column's ref bins)
        ax = axes[3][col]
        met = metrics[metrics["time_bin_center"].isin(ref_bins)] if ref_bins else metrics.iloc[0:0]
        if not met.empty:
            agg = met.groupby("class")[["precision", "recall"]].mean().reindex(classes)
            xpos = np.arange(len(classes))
            ax.bar(xpos - 0.18, agg["precision"].fillna(0), width=0.36,
                   label="P", color="#4D4D4D")
            ax.bar(xpos + 0.18, agg["recall"].fillna(0), width=0.36,
                   label="R", color="#B0B0B0")
            for i, c in enumerate(classes):
                p, r = agg.loc[c, "precision"], agg.loc[c, "recall"]
                if not pd.isna(p):
                    ax.text(i - 0.18, p, f"{p:.2f}", ha="center", va="bottom", fontsize=6)
                if not pd.isna(r):
                    ax.text(i + 0.18, r, f"{r:.2f}", ha="center", va="bottom", fontsize=6)
            ax.set_xticks(xpos)
            ax.set_xticklabels(classes, fontsize=7, rotation=20, ha="right")
            ax.set_ylim(0, 1.15)
            if col == 0:
                ax.legend(fontsize=6, loc="lower left", ncol=2, frameon=False)
        else:
            ax.text(0.5, 0.5, "no ref bins", ha="center", va="center",
                    fontsize=7, color="#999999", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_titles[3], fontsize=8)

        # ── Row 4: reference confusion (mean of the column's ref bins, row-normalized)
        ax = axes[4][col]
        mats = [confusion[b]["matrix"] for b in ref_bins if b in confusion]
        if mats:
            cm = np.nanmean(np.stack(mats), axis=0)
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.55 else "black")
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([f"p:{c[:6]}" for c in classes], fontsize=6,
                               rotation=30, ha="right")
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels([f"t:{c[:6]}" for c in classes] if col == 0 else ["", ""],
                               fontsize=6)
        else:
            ax.text(0.5, 0.5, "no ref bins", ha="center", va="center",
                    fontsize=7, color="#999999", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_titles[4], fontsize=8)

        if cell_q.empty:
            print(f"  column '{col_label.replace(chr(10), ' ')}': n=0 query (rendered blank)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.subplots_adjust(left=0.10, right=0.90, top=0.93, bottom=0.05, wspace=0.30, hspace=0.45)
    cax = fig.add_axes([0.925, 0.18, 0.014, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"P({right})", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1"], fontsize=7)
    fig.suptitle(f"{gene} homozygous-phenotype confidence — query vs reference "
                 f"by collection × support", fontsize=11)
    out = CONF_PLOT_DIR / f"{gene}_confidence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/confidence/{out.name}")


print("3c - confidence plot (KEY): homozygous-phenotype binary, per gene")
per_bin_all = pd.read_csv(PER_BIN_PATH, low_memory=False)
print(f"Loaded {len(per_bin_all)} query per-bin rows; genes {sorted(per_bin_all['gene'].unique())}")

for gene in GENE_SPEC:
    make_confidence_plot(gene, per_bin_all)

print(f"\nWrote confidence plots under: {CONF_PLOT_DIR.relative_to(RUN_DIR)}/")
