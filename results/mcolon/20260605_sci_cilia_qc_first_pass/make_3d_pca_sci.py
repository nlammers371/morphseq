"""
3D PCA of the sci_ SNAPSHOT plates — MIRRORS the transfer-workflow PCA (make_3d_pca.py).

The sci_ (sky / sequenced) plates are handled SEPARATELY from the main cilia QC transfer analysis,
but the PCA QC question is the SAME: project each sci_ plate into its gene's REFERENCE PCA space and
look for batch effects — does the new snapshot overlay the old reference, or sit apart?

So PCA is fit on REFERENCE + sci-plate combined (shared z_mu_b space), exactly like make_3d_pca.py.
One interactive HTML per sci_ plate with a DROPDOWN to toggle coloring views:
    • HPF         — continuous predicted_stage_hpf
    • genotype    — zygosity (sci-specific map; T.to_zygosity collapses *_homozygous etc.)
    • experiment  — experiment_id (BATCH view: does the sci plate cluster apart from the reference?)
    • source      — reference vs query (does the new snapshot overlay the old reference?)
    • sequenced   — sequenced(1/2) / reference / not_sequenced  (from the plate Excel `sequenced`
                    sheet, joined by well — the sheet is NOT in build06, so we read it directly)

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/make_3d_pca_sci.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.utils.pca import fit_pca_on_embeddings, transform_embeddings_to_pca  # noqa: E402

import build_reference_and_transfer as T  # reference paths, genotype standardization  # noqa: E402

PLATE_META = PROJECT_ROOT / "metadata/plate_metadata"
OUT = RUN_DIR / "plots" / "pca_3d_sci"
PCA_COLS = ["PCA_1", "PCA_2", "PCA_3"]
WELLS = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]

# each sci_ snapshot plate -> the reference it is projected against (its gene)
SCI_PLATES = {
    "20260414_sci_b9d2_48hpf_plate01": "b9d2",
    "20260415_sci_cep290_48hpf_plate01": "cep290",
}

ZYG_COLORS = {  # colorblind-safe, matches GENOTYPE_SUFFIX_COLORS conventions
    "wildtype": "#2166AC", "heterozygous": "#F7B267", "homozygous": "#B2182B",
    "unknown": "#808080",
}
SEQ_COLORS = {
    "reference": "#cccccc",
    "not_sequenced": "#d62728",
    "seq_wt(1)": "#2166AC",
    "seq_mutant(2)": "#B2182B",
}


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"z_mu_b_\d+", c)]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def sequenced_map(exp: str) -> dict[str, str]:
    """well -> {'seq_wt(1)','seq_mutant(2)','not_sequenced'} from the plate Excel `sequenced` sheet.

    Coding (verified): 0/blank = not sequenced, 1 = wildtype-confirmed, 2 = mutant-allele-confirmed
    (het OR homo). The sheet is NOT parsed by Build01, so we read the 8x12 grid directly and join
    by well (same parse as patch_predicted_stage_hpf.py).
    """
    xl = PLATE_META / f"{exp}_well_metadata.xlsx"
    with pd.ExcelFile(xl) as xlf:
        if "sequenced" not in xlf.sheet_names:
            return {}
        df = xlf.parse("sequenced", header=0)
        block = df.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
        arr = block.to_numpy(dtype=str).ravel()
    out: dict[str, str] = {}
    for w, v in zip(WELLS, arr):
        s = v.strip()
        try:
            code = int(float(s)) if s not in ("", "nan") else 0
        except ValueError:
            code = 0
        if code == 1:
            out[w] = "seq_wt(1)"
        elif code == 2:
            out[w] = "seq_mutant(2)"
        else:
            out[w] = "not_sequenced"
    return out


def _add_categorical_traces(fig, df, color_by, palette, visible):
    vals = list(df[color_by].dropna().unique())
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


def build_plate_figure(exp: str, gene: str):
    qpath = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
    ref_path = T.DATASETS[gene]["ref"]
    if not qpath.exists():
        print(f"[{exp}] missing build06 csv — skip"); return

    feat = T.resolve_feature_cols([ref_path, qpath])

    # reference (old, labeled) — same loader + genotype standardization as the transfer workflow
    ref = T._load(ref_path, feat, gene_hint=gene).dropna(subset=feat).copy()
    ref["source"] = "reference"
    ref["experiment_id"] = ref.get("experiment_id", "reference").astype(str)
    ref["sequenced_view"] = "reference"

    # query = the sci_ snapshot plate
    qry = T._load(qpath, feat, gene_hint=gene).dropna(subset=feat).copy()
    qry["source"] = "query"
    qry["experiment_id"] = exp
    seqmap = sequenced_map(exp)
    # join sequenced status by well (well column lives in build06 but T._load drops it -> reload)
    wells = pd.read_csv(qpath, usecols=["snip_id", "well"], low_memory=False)
    well_by_snip = wells.set_index("snip_id")["well"]
    qry["well"] = qry["snip_id"].map(well_by_snip) if "snip_id" in qry.columns else np.nan
    qry["sequenced_view"] = qry["well"].map(seqmap).fillna("not_sequenced")

    keep = feat + [T.GENO_COL, T.ZYG_COL, T.TIME_COL, "experiment_id", "source", "sequenced_view"]
    comb = pd.concat([ref.reindex(columns=keep), qry.reindex(columns=keep)], ignore_index=True)
    comb = comb.dropna(subset=feat)

    pca, scaler, _ = fit_pca_on_embeddings(comb, z_mu_cols=feat, n_components=3, scale=True)
    comb = transform_embeddings_to_pca(comb, pca, scaler, z_mu_cols=feat)
    evr = pca.explained_variance_ratio_
    nref = int((comb.source == "reference").sum())
    nq = int((comb.source == "query").sum())
    print(f"[{exp}] vs {gene} ref — PCA EVR={evr.round(3).tolist()}  "
          f"combined n={len(comb)} (ref={nref}, query={nq})")

    comb["zygosity"] = comb[T.ZYG_COL]  # already het/homo/wt/None from T._load
    comb["_hover"] = (
        "exp: " + comb["experiment_id"].astype(str)
        + "<br>geno: " + comb[T.GENO_COL].astype(str)
        + "<br>zyg: " + comb["zygosity"].astype(str)
        + "<br>seq: " + comb["sequenced_view"].astype(str)
        + "<br>hpf: " + comb[T.TIME_COL].round(1).astype(str)
        + "<br>" + comb["source"]
    )

    fig = go.Figure()
    views = []  # (label, n_traces)

    views.append(("HPF", _add_continuous_trace(fig, comb, T.TIME_COL, "hpf", visible=True)))

    views.append(("genotype (zygosity)",
                  _add_categorical_traces(fig, comb.dropna(subset=["zygosity"]),
                                          "zygosity", ZYG_COLORS, visible=False)))

    batches = sorted(comb["experiment_id"].astype(str).unique())
    bpal = {b: px.colors.qualitative.Dark24[i % 24] for i, b in enumerate(batches)}
    views.append(("experiment (batch)",
                  _add_categorical_traces(fig, comb, "experiment_id", bpal, visible=False)))

    views.append(("source (ref vs query)",
                  _add_categorical_traces(fig, comb, "source",
                                          {"reference": "#cccccc", "query": "#d62728"},
                                          visible=False)))

    views.append(("sequenced",
                  _add_categorical_traces(fig, comb, "sequenced_view", SEQ_COLORS, visible=False)))

    total = sum(c for _, c in views)
    buttons, start = [], 0
    for label, cnt in views:
        vis = [False] * total
        for k in range(start, start + cnt):
            vis[k] = True
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis},
                                  {"title": f"{exp} — 3D PCA — colored by {label}"}]))
        start += cnt

    fig.update_layout(
        title=f"{exp} — 3D PCA (vs {gene} ref) — colored by HPF",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.01, xanchor="left", y=0.99, yanchor="top")],
        scene=dict(
            xaxis_title=f"PCA_1 ({evr[0]:.0%})",
            yaxis_title=f"PCA_2 ({evr[1]:.0%})",
            zaxis_title=f"PCA_3 ({evr[2]:.0%})",
        ),
        legend=dict(title="", itemsizing="constant", groupclick="toggleitem"),
        width=1100, height=850,
    )
    out = OUT / f"{exp}_pca_3d.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  wrote {out.relative_to(RUN_DIR)}  (views: {[v for v, _ in views]})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for exp, gene in SCI_PLATES.items():
        build_plate_figure(exp, gene)
    print(f"\nInteractive HTMLs under: {OUT.relative_to(RUN_DIR)}/  — open in a browser, use the dropdown.")


if __name__ == "__main__":
    main()
