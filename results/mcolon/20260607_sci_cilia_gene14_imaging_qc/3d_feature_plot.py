"""
3d - Feature plot (physical-reality check), 24->48 hpf window, per gene.

Does the model's predicted phenotype line up with real morphology over development?
We plot TWO real morphology features (NOT PC1, NOT model probability):
    baseline_deviation_normalized  (curvature)
    total_length_um                (length)
as a function of predicted_stage_hpf (true age), with three visual tiers:

  - reference TRAINING points -> low-alpha trajectory backdrop (the labeled embryos the
    model was fit on; colored faintly by their true phenotype class).
  - timeseries query embryos  -> high-alpha trajectory LINE per embryo, ending in a CIRCLE
    colored by that embryo's cross-bin predicted class.
  - snapshot query embryos    -> a SQUARE at its collection time (30 & 48 hpf have
    snapshots), colored by its predicted class.

The trajectory layers (reference backdrop + timeseries lines) are drawn by the shared
plot_feature_over_time plotter (same call shape as
../20260605_sci_cilia_qc_first_pass/make_trajectory_plots_sci.py). The single-point
snapshot squares and the timeseries end-circles are overlaid on the returned matplotlib
axes (the generic line plotter can't express single-point markers / per-embryo end caps).

SEQUENCED HOMOZYGOUS ONLY (mirrors the confidence plot's cohort): query rows are the
sequenced homozygous embryos whose predicted class comes from
sequenced_homozygous_phenotype_cross_bin.csv.

Reads:
    tables/reference_<gene>_clean.csv            (backdrop, cluster_categories truth)
    tables/query_all_rows_clean.csv              (query per-frame features + data_source)
    predictions/sequenced_homozygous_phenotype_cross_bin.csv  (predicted_label per embryo)
Writes:
    plots/feature/<gene>_feature_24_48.png

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3d_feature_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from plot_config import PHENOTYPE_COLORS  # noqa: E402
from src.analyze.viz.plotting import plot_feature_over_time  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
PRED_DIR = RUN_DIR / "predictions"
FEAT_PLOT_DIR = RUN_DIR / "plots" / "feature"
FEAT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["baseline_deviation_normalized", "total_length_um"]
TIME_COL = "predicted_stage_hpf"
ID_COL = "embryo_id"

# Per gene: the two homozygous phenotype classes; the reference truth column is
# cluster_categories (the reference's own labels), normalized to these classes.
GENE_SPEC = {
    "b9d2":   {"left": "CE", "right": "HTA"},
    "cep290": {"left": "High_to_Low", "right": "Low_to_High"},
}

QUERY_USE = {
    ID_COL, TIME_COL, "gene", "zygosity", "data_source", "collection_time_hpf",
    "physical_embryo_id", *FEATURES,
}
REF_USE = {ID_COL, TIME_COL, "cluster_categories", *FEATURES}


def load_reference(gene: str, classes: list[str]) -> pd.DataFrame:
    """Reference homozygous-phenotype embryos for the backdrop (true class = cluster_categories)."""
    df = pd.read_csv(TABLE_DIR / f"reference_{gene}_clean.csv",
                     usecols=lambda c: c in REF_USE, low_memory=False)
    df = df.dropna(subset=["cluster_categories", TIME_COL])
    # normalize reference label aliases to the binary classes (mirror trajectory script)
    if gene == "cep290":
        df.loc[df["cluster_categories"] == "Intermediate", "cluster_categories"] = "Low_to_High"
    if gene == "b9d2":
        df["cluster_categories"] = df["cluster_categories"].replace("BA_rescue", "HTA")
    df = df[df["cluster_categories"].isin(classes)].copy()
    df["true_label"] = df["cluster_categories"]
    return df


def load_query(gene: str, predicted: pd.Series) -> pd.DataFrame:
    """Sequenced homozygous query frames with the cross-bin predicted class attached."""
    df = pd.read_csv(TABLE_DIR / "query_all_rows_clean.csv",
                     usecols=lambda c: c in QUERY_USE, low_memory=False)
    df = df[(df["gene"] == gene) & (df["zygosity"] == "homozygous")].copy()
    df["predicted_label"] = df[ID_COL].map(predicted)
    df = df.dropna(subset=[TIME_COL, "predicted_label", *FEATURES])
    return df


def make_feature_plot(gene: str, cross_bin: pd.DataFrame) -> None:
    spec = GENE_SPEC[gene]
    classes = [spec["left"], spec["right"]]
    colors = {c: PHENOTYPE_COLORS.get(c, "#808080") for c in classes}

    predicted = (cross_bin[cross_bin["gene"] == gene]
                 .set_index("query_embryo_id")["predicted_label"])
    qry = load_query(gene, predicted)
    ref = load_reference(gene, classes)
    if qry.empty:
        print(f"  [{gene}] no sequenced homozygous query rows — skip")
        return

    # x-window from the query (24->48 collection window, padded)
    hpf_min = float(qry[TIME_COL].min()) - 1
    hpf_max = float(qry[TIME_COL].max()) + 1
    ref = ref[ref[TIME_COL].between(hpf_min, hpf_max)].copy()

    qry_ts = qry[qry["data_source"] == "timeseries"].copy()
    qry_snap = qry[qry["data_source"] == "snapshot"].copy()
    n_ts = qry_ts[ID_COL].nunique()
    n_snap = qry_snap["physical_embryo_id"].nunique()
    n_ref = ref[ID_COL].nunique()
    print(f"  [{gene}] ref n={n_ref} | timeseries n={n_ts} | snapshot n={n_snap} | "
          f"window [{hpf_min:.0f}, {hpf_max:.0f}]")

    # ── Trajectory layers via the shared plotter ───────────────────────────────────
    # Backdrop: reference, VERY faint and thin, in back (true class color).
    # Sequenced timeseries: DARK, thick lines in front (predicted class color), each
    # tracing to its end circle (overlaid below). We render both in ONE plotter call and
    # set EVERY embryo's style explicitly via id_style_lookup — otherwise the plotter's
    # default individual_alpha (0.18) makes the sequenced lines as faint as the reference.
    ref_layer = ref.assign(color_group=ref["true_label"])
    ts_layer = qry_ts.assign(color_group=qry_ts["predicted_label"])
    combined = pd.concat([ref_layer, ts_layer], ignore_index=True)

    id_styles = {eid: {"alpha": 0.08, "width": 0.5, "zorder": 1}
                 for eid in ref_layer[ID_COL].unique()}
    # sequenced timeseries: dark + bold, well above the backdrop
    id_styles.update({eid: {"alpha": 0.95, "width": 1.8, "zorder": 4}
                      for eid in ts_layer[ID_COL].unique()})

    fig = plot_feature_over_time(
        combined,
        features=FEATURES,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="color_group",
        color_lookup=colors,
        backend="matplotlib",
        output_path=None,
        title=f"{gene} morphology over development — query prediction vs reference",
        xlim=(hpf_min, hpf_max),
        show_individual=True,
        show_trend=False,
        id_style_lookup=id_styles,
        legend_loc="upper left",
    )
    fig = fig[0] if isinstance(fig, (list, tuple)) else fig

    # ── Overlays: timeseries end-circles + snapshot squares, per feature axis ──────
    # Map each returned axis to its feature via the y-label the plotter set.
    ax_by_feature = {ax.get_ylabel(): ax for ax in fig.axes if ax.get_ylabel() in FEATURES}
    for feat, ax in ax_by_feature.items():
        # timeseries: a circle at each embryo's LAST frame, colored by predicted class
        for eid, sub in qry_ts.groupby(ID_COL):
            last = sub.loc[sub[TIME_COL].idxmax()]
            ax.scatter(last[TIME_COL], last[feat], marker="o", s=46,
                       color=colors.get(last["predicted_label"], "#808080"),
                       edgecolors="black", linewidths=0.5, zorder=6)
        # snapshots: one square per physical embryo (single frame), colored by predicted class
        if not qry_snap.empty:
            ax.scatter(qry_snap[TIME_COL], qry_snap[feat], marker="s", s=44,
                       c=qry_snap["predicted_label"].map(colors).fillna("#808080"),
                       edgecolors="black", linewidths=0.5, zorder=5)

    out = FEAT_PLOT_DIR / f"{gene}_feature_24_48.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/feature/{out.name}")


print("3d - feature plot (curvature + length) over development, per gene")
cross_bin = pd.read_csv(PRED_DIR / "sequenced_homozygous_phenotype_cross_bin.csv", low_memory=False)

for gene in GENE_SPEC:
    print(f"\n[{gene}]")
    make_feature_plot(gene, cross_bin)

print(f"\nWrote feature plots under: {FEAT_PLOT_DIR.relative_to(RUN_DIR)}/")
