"""
LABEL TRANSFER — isolated, publication-critical step (genotype + phenotype).

This is THE label-transfer code. It is deliberately separated from all plotting so it can be audited
on its own: it runs the transfer and writes prediction CSVs; nothing here draws a figure. Plotting
lives in make_plots*.py and consumes ONLY the CSVs this writes.

Supersedes build_reference_and_transfer.py (kept as legacy). Behavior of the transfer itself is
unchanged; what's new is that every output row carries a `sequenced` column (the raw value from the
plate Excel `sequenced` sheet, joined by well) and a `stratum` tag, so downstream analysis can split
SEQUENCED vs not without re-running anything. All query embryos go through the transfer — we tag, we
don't pre-filter. The data all comes from the same place; the `sequenced` column just makes
provenance explicit so snapshots (these query plates) and the time-lapse experiments can be analyzed
separately while sharing identical metadata.

`sequenced` sheet coding (verified): 0/blank = not sequenced · 1 = wildtype-confirmed (incl. AB) ·
2 = mutant-allele-confirmed (het OR homo). NOT parsed by Build01 -> read from the Excel here.

Strata (sequenced>0 only): homozygous · heterozygous · wildtype_sibling (code 1, non-AB) · AB.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/label_transfer_snapshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# reuse the audited loaders / references / genotype standardization (single source of truth for
# WHERE data comes from); only the transfer + sequenced tagging are driven from here.
import build_reference_and_transfer as T  # noqa: E402

PLATE_META = PROJECT_ROOT / "metadata/plate_metadata"
WELLS = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]
OUT = RUN_DIR / "transfer_results"


# ── sequenced join (Excel `sequenced` sheet -> well -> embryo) ──────────────────
def sequenced_grid(exp: str) -> dict[str, int] | None:
    """well -> sequenced code {0,1,2}. None if the plate has no Excel / no sequenced sheet.

    Same 8x12 grid parse as patch_predicted_stage_hpf.py / export_utils.py.
    """
    for cand in (f"{exp}_well_metadata.xlsx", f"{exp}.xlsx"):
        p = PLATE_META / cand
        if not p.exists():
            continue
        with pd.ExcelFile(p) as xlf:
            if "sequenced" not in xlf.sheet_names:
                return None
            df = xlf.parse("sequenced", header=0)
            block = df.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
            arr = block.to_numpy(dtype=str).ravel()
        out: dict[str, int] = {}
        for w, v in zip(WELLS, arr):
            s = v.strip()
            try:
                out[w] = int(float(s)) if s not in ("", "nan") else 0
            except ValueError:
                out[w] = 0
        return out
    return None


def well_by_embryo(exp: str) -> pd.Series:
    """embryo_id -> well, from build06 (embryo_id maps to exactly one well)."""
    p = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
    df = pd.read_csv(p, usecols=[T.GROUP_COL, "well"], low_memory=False)
    return df.drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)["well"]


def build_sequenced_lookup(queries: list[str]) -> pd.DataFrame:
    """embryo_id -> (query_experiment, well, sequenced) across all query plates."""
    rows = []
    for exp in queries:
        grid = sequenced_grid(exp)
        wmap = well_by_embryo(exp)
        for emb, well in wmap.items():
            code = (grid or {}).get(str(well), np.nan if grid is None else 0)
            rows.append((emb, exp, well, code))
    return pd.DataFrame(rows, columns=[T.GROUP_COL, "query_experiment", "well", "sequenced"])


def assign_stratum(row) -> str:
    """4 strata for sequenced>0; '' otherwise. Uses standardized genotype + sequenced code."""
    seq = row["sequenced"]
    if pd.isna(seq) or seq == 0:
        return ""  # not sequenced
    g = str(row.get("true_genotype", "")).lower()
    if g.endswith("ab_wildtype") or g == "ab" or "_ab" in g:
        return "AB"
    if seq == 1:
        return "wildtype_sibling"   # wt-confirmed, non-AB
    if seq == 2:
        if g.endswith("homozygous"):
            return "homozygous"
        if g.endswith("heterozygous"):
            return "heterozygous"
        return "mutant_unresolved"  # code-2 but genotype not homo/het (rare)
    return ""


# ── transfer (delegates to the audited functions in build_reference_and_transfer) ─
def _tag(emb: pd.DataFrame, seqlk: pd.DataFrame) -> pd.DataFrame:
    """Add `sequenced` + `stratum` columns onto an embryo-prediction frame, joined by embryo_id."""
    s = seqlk.set_index(T.GROUP_COL)["sequenced"]
    emb = emb.copy()
    emb["sequenced"] = emb["query_embryo_id"].map(s)
    emb["stratum"] = emb.apply(assign_stratum, axis=1)
    return emb


def main() -> None:
    OUT.mkdir(exist_ok=True)
    geno_all, pheno_all, seq_all = [], [], []

    for name, cfg in T.DATASETS.items():
        queries = [e for e in cfg["queries"]
                   if (T.B6 / f"df03_final_output_with_latents_{e}.csv").exists()]
        if not queries:
            print(f"[{name}] no query build06 yet — skipping"); continue
        qpaths = [T.B6 / f"df03_final_output_with_latents_{e}.csv" for e in queries]
        feat = T.resolve_feature_cols([cfg["ref"], *qpaths])

        print(f"\n######## {name} (feature dims={len(feat)}) ########")
        gene_hint = name if name in ("cep290", "b9d2") else None
        ref = T._load(cfg["ref"], feat, gene_hint=gene_hint)
        qparts = []
        for e, p in zip(queries, qpaths):
            q = T._load(p, feat, gene_hint=gene_hint)
            q["query_experiment"] = e
            qparts.append(q)
        qry = pd.concat(qparts, ignore_index=True)

        seqlk = build_sequenced_lookup(queries)
        seqlk["dataset"] = name
        seq_all.append(seqlk)
        nseq = int((seqlk["sequenced"] > 0).sum())
        print(f"  ref={len(ref)} rows | query={len(qry)} rows "
              f"({qry['query_experiment'].nunique()} exps) | sequenced embryos={nseq}")

        # GENOTYPE (benchmarked) + PHENOTYPE (predictions only) via the audited transfer fns
        g = T.run_genotype_transfer(name, ref, qry, feat)
        if not g.empty:
            geno_all.append(_tag(g, seqlk))
        if cfg["phenotype"]:
            ph = T.run_phenotype_transfer(name, ref, qry, feat)
            pheno_all.append(_tag(ph, seqlk))

    # ── registry: one row per query embryo, with sequenced code + stratum ──
    seqlk_all = pd.concat(seq_all, ignore_index=True)

    # genotype predictions (carry sequenced + stratum)
    geno = pd.concat(geno_all, ignore_index=True)
    geno.to_csv(OUT / "genotype_transfer_predictions.csv", index=False)

    # phenotype predictions
    if pheno_all:
        pheno = pd.concat(pheno_all, ignore_index=True)
        pheno.to_csv(OUT / "phenotype_transfer_predictions.csv", index=False)

    # sequenced registry = the sequenced-only truth set, with stratum + the genotype call.
    gmeta = geno.set_index("query_embryo_id")
    reg = seqlk_all[seqlk_all["sequenced"] > 0].copy()
    reg["true_genotype"] = reg[T.GROUP_COL].map(gmeta["true_genotype"]) if not geno.empty else np.nan
    reg["true_zygosity"] = reg[T.GROUP_COL].map(gmeta["true_zygosity"]) if not geno.empty else np.nan
    reg["stratum"] = reg.apply(assign_stratum, axis=1)
    reg["predicted_zygosity"] = reg[T.GROUP_COL].map(gmeta["predicted_label"]) if not geno.empty else np.nan
    if pheno_all:
        pmeta = pheno.set_index("query_embryo_id")["predicted_label"]
        reg["predicted_phenotype"] = reg[T.GROUP_COL].map(pmeta)
    reg = reg.rename(columns={T.GROUP_COL: "embryo_id"})
    reg.to_csv(OUT / "sequenced_registry.csv", index=False)

    # ── console summary ──
    print("\n================ SEQUENCED REGISTRY ================")
    print(f"sequenced embryos: {len(reg)}")
    print("\nby gene x stratum:")
    print(reg.groupby(["dataset", "stratum"]).size().to_string())

    bench = geno[geno["benchmarkable"] & (geno["sequenced"] > 0)]
    if not bench.empty:
        print("\n=== GENOTYPE label-transfer accuracy on SEQUENCED embryos ===")
        print(bench.groupby(["dataset", "stratum"])["correct"].agg(["mean", "size"]).to_string())

    print(f"\nWrote: {OUT}/genotype_transfer_predictions.csv, phenotype_transfer_predictions.csv, "
          f"sequenced_registry.csv")


if __name__ == "__main__":
    main()
