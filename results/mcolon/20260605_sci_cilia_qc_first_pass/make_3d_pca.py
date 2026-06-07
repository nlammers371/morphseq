"""
3D PCA view of the cilia QC data — one interactive HTML per dataset with a DROPDOWN to toggle
between coloring views (the main use: spot batch effects vs real biology):

    • HPF        — continuous developmental stage (predicted_stage_hpf)
    • genotype   — true genotype/zygosity (gene class for crispant)
    • phenotype  — TRANSFERRED phenotype prediction (cep290/b9d2)
    • experiment — experiment_id (BATCH view: do query plates cluster apart from reference?)
    • source     — reference vs query (does the new data overlay the old?)

PCA is fit on the REFERENCE+QUERY combined (shared 80-dim z_mu_b space) so query points land in the
same coordinates as the reference — that is what makes a batch effect visible.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/make_3d_pca.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.utils.pca import fit_pca_on_embeddings, transform_embeddings_to_pca  # noqa: E402

import build_reference_and_transfer as T  # noqa: E402
import make_plots as M  # reuse loaders, palettes, transfer  # noqa: E402

OUT = RUN_DIR / "plots" / "pca_3d"
OUT.mkdir(parents=True, exist_ok=True)
PCA_COLS = ["PCA_1", "PCA_2", "PCA_3"]

# discrete-color palettes per categorical view
ZYG = M.ZYG_COLORS
CRISPANT = M.CRISPANT_GENE_COLORS
PHENO = M._lt_core._CLASS_COLORS


def _cat_palette(name: str) -> dict:
    return {**ZYG, **CRISPANT, **PHENO,
            "reference": "#bbbbbb", "query": "#000000"}


def _add_categorical_traces(fig, df, color_by, palette, visible):
    """One Plotly trace per category value. Returns the count of traces added."""
    vals = [v for v in df[color_by].dropna().unique()]
    # stable, palette-ordered first
    order = [v for v in palette if v in vals] + [v for v in sorted(map(str, vals)) if v not in palette]
    n = 0
    for v in order:
        sub = df[df[color_by].astype(str) == str(v)]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=sub[PCA_COLS[0]], y=sub[PCA_COLS[1]], z=sub[PCA_COLS[2]],
            mode="markers",
            marker=dict(size=3, color=palette.get(v, "#888"), opacity=0.6),
            # unique legendgroup per category -> clicking a legend entry toggles ONLY that class
            name=f"{v}", legendgroup=f"{color_by}::{v}", showlegend=True, visible=visible,
            text=sub["_hover"], hoverinfo="text",
        ))
        n += 1
    return n


def _add_continuous_trace(fig, df, color_by, title, visible):
    fig.add_trace(go.Scatter3d(
        x=df[PCA_COLS[0]], y=df[PCA_COLS[1]], z=df[PCA_COLS[2]],
        mode="markers",
        marker=dict(size=3, color=df[color_by], colorscale="Viridis", opacity=0.6,
                    colorbar=dict(title=title), showscale=True),
        name=title, showlegend=False, visible=visible, text=df["_hover"], hoverinfo="text",
    ))
    return 1


def build_dataset_figure(name: str, cfg: dict):
    qpaths = [(e, T.B6 / f"df03_final_output_with_latents_{e}.csv") for e in cfg["queries"]]
    qpaths = [(e, p) for e, p in qpaths if p.exists()]
    if not qpaths:
        print(f"[{name}] no queries — skip"); return
    feat = T.resolve_feature_cols([cfg["ref"], *[p for _, p in qpaths]])
    gene_hint = name if name in ("cep290", "b9d2") else None

    # reference
    ref = T._load(cfg["ref"], feat, gene_hint=gene_hint)
    ref = ref.dropna(subset=feat)
    ref["source"] = "reference"
    ref["query_experiment"] = ref.get("experiment_id", "reference")

    # query (+ transferred phenotype prediction, if any)
    res = M.run_dataset(name, cfg)
    qry = res["qry"].dropna(subset=feat).copy()
    qry["source"] = "query"
    # phenotype prediction per embryo -> map onto query rows
    if "pemb" in res:
        pmap = res["pemb"].set_index("query_embryo_id")["predicted_label"]
        qry["pheno_pred"] = qry[T.GROUP_COL].map(pmap)
    ref["pheno_pred"] = ref.get(T.PHENO_COL, np.nan)  # reference carries its true phenotype

    # combined for shared PCA
    keep = feat + [T.GENO_COL, T.GROUP_COL, T.TIME_COL, "experiment_id",
                   "query_experiment", "source", "pheno_pred"]
    comb = pd.concat([ref.reindex(columns=keep), qry.reindex(columns=keep)], ignore_index=True)
    comb = comb.dropna(subset=feat)

    pca, scaler, _ = fit_pca_on_embeddings(comb, z_mu_cols=feat, n_components=3, scale=True)
    comb = transform_embeddings_to_pca(comb, pca, scaler, z_mu_cols=feat)
    evr = pca.explained_variance_ratio_
    print(f"[{name}] PCA EVR={evr.round(3).tolist()}  combined n={len(comb)} "
          f"(ref={int((comb.source=='reference').sum())}, query={int((comb.source=='query').sum())})")

    # zygosity (for cep290/b9d2 the categorical genotype view = zygosity; crispant = gene class)
    comb["zygosity"] = comb[T.GENO_COL].map(T.to_zygosity)
    comb["genotype_view"] = comb["zygosity"] if name != "crispant" else comb[T.GENO_COL]
    comb["batch"] = comb["query_experiment"].astype(str)

    comb["_hover"] = (
        "exp: " + comb["query_experiment"].astype(str)
        + "<br>geno: " + comb[T.GENO_COL].astype(str)
        + "<br>pheno: " + comb["pheno_pred"].astype(str)
        + "<br>hpf: " + comb[T.TIME_COL].round(1).astype(str)
        + "<br>" + comb["source"]
    )

    # ── build views ──
    fig = go.Figure()
    views = []  # (label, n_traces)

    n = _add_continuous_trace(fig, comb, T.TIME_COL, "hpf", visible=True)
    views.append(("HPF", n))

    gp = _cat_palette(name)
    n = _add_categorical_traces(fig, comb.dropna(subset=["genotype_view"]),
                                "genotype_view", gp, visible=False)
    views.append(("genotype", n))

    has_pheno = comb["pheno_pred"].notna().any()
    if has_pheno:
        n = _add_categorical_traces(fig, comb.dropna(subset=["pheno_pred"]),
                                    "pheno_pred", PHENO, visible=False)
        views.append(("phenotype", n))

    # batch view: distinct color per experiment (categorical, auto colors)
    import plotly.express as px
    batches = sorted(comb["batch"].unique())
    bpal = {b: px.colors.qualitative.Dark24[i % 24] for i, b in enumerate(batches)}
    n = _add_categorical_traces(fig, comb, "batch", bpal, visible=False)
    views.append(("experiment (batch)", n))

    n = _add_categorical_traces(fig, comb, "source",
                                {"reference": "#cccccc", "query": "#d62728"}, visible=False)
    views.append(("source (ref vs query)", n))

    # ── dropdown toggling trace visibility per view ──
    total = sum(c for _, c in views)
    buttons = []
    start = 0
    for label, cnt in views:
        vis = [False] * total
        for k in range(start, start + cnt):
            vis[k] = True
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis},
                                  {"title": f"{name} — 3D PCA — colored by {label}"}]))
        start += cnt

    fig.update_layout(
        title=f"{name} — 3D PCA — colored by HPF",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.01, xanchor="left", y=0.99, yanchor="top")],
        scene=dict(
            xaxis_title=f"PCA_1 ({evr[0]:.0%})",
            yaxis_title=f"PCA_2 ({evr[1]:.0%})",
            zaxis_title=f"PCA_3 ({evr[2]:.0%})",
        ),
        # groupclick=toggleitem -> a legend click toggles only its own item, not a whole group
        legend=dict(title="", itemsizing="constant", groupclick="toggleitem"),
        width=1100, height=850,
    )
    out = OUT / f"{name}_pca_3d.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  wrote {out.relative_to(RUN_DIR)}  (views: {[v for v,_ in views]})")


def main():
    for name, cfg in T.DATASETS.items():
        build_dataset_figure(name, cfg)
    print(f"\nInteractive HTMLs under: {OUT.relative_to(RUN_DIR)}/  — open in a browser, use the dropdown.")


if __name__ == "__main__":
    main()
