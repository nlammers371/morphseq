"""
Cilia QC first-pass — PER-DATASET label transfer.

Each new query experiment is transferred against ITS OWN dataset's reference (no cross-dataset
mixing — we already know which gene each experiment is):

    cep290 query  -> cep290 reference
    b9d2   query  -> b9d2   reference
    crispant query-> crispant reference (20260202)

Two labels, two roles (see LABEL_STANDARDIZATION.md):
  • GENOTYPE  — known ground truth on reference AND query. Transferred as a BENCHMARK:
    predict zygosity (homozygous/heterozygous/wildtype) and score against the known query
    genotype. THIS IS THE EMPHASIS — it validates the transfer machinery + embedding quality.
  • PHENOTYPE — NO ground truth on the query (cep290: trajectory classes; b9d2: CE/HTA/wt).
    Transferred to PRODUCE predictions only; verified later by looking at images. Not scored.
    Crispants have no phenotype label -> genotype only.

Uses the production module src/analyze/classification/label_transfer.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/build_reference_and_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.classification.label_transfer import prepare_reference, transfer_labels  # noqa: E402

B6 = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
GROUP_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
GENO_COL = "genotype"          # standardized genotype
ZYG_COL = "zygosity"           # bare het/homo/wt (the genotype-transfer label)
PHENO_COL = "cluster_categories"

# ── genotype standardization (in place; see LABEL_STANDARDIZATION.md) ───────────
GENOTYPE_RENAME = {
    "cep290_homo": "cep290_homozygous", "cep290_homozygous": "cep290_homozygous",
    "cep290_het": "cep290_heterozygous", "cep290_heterozygous": "cep290_heterozygous",
    "cep290_wt": "cep290_wildtype", "cep290_wildtype": "cep290_wildtype",
    "cep290_unknown": "cep290_unknown", "cep290_uncertain": "cep290_unknown",
    "b9d2_homo": "b9d2_homozygous", "b9d2_homozygous": "b9d2_homozygous",
    "b9d2_het": "b9d2_heterozygous", "b9d2_heterozygous": "b9d2_heterozygous",
    "b9d2_wt": "b9d2_wildtype", "b9d2_wildtype": "b9d2_wildtype",
    "b9d2_unknown": "b9d2_unknown", "b9d2_uncertain": "b9d2_unknown",
    "foxj1a_crispant": "foxj1a_crispant", "ift88_crispant": "ift88_crispant",
    "sspo_crispant": "sspo_crispant", "scospondin_crispant": "sspo_crispant",
    "if88_ift74_crispant": "ift88_ift74_crispant",
    "ab": "ab_wildtype", "inj-ctrl": "injection_control",
}

# bare zygosity extracted from the standardized genotype (None = not a zygosity class)
def to_zygosity(genotype: str) -> str | None:
    for z in ("homozygous", "heterozygous", "wildtype", "unknown"):
        if genotype.endswith(z):
            return z
    return None  # crispants / controls have no zygosity


# ── reference sources (old, labeled) ────────────────────────────────────────────
CEP290_REF = PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
B9D2_REF = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
CRISPANT_REF = B6 / "df03_final_output_with_latents_20260202.csv"

# ── query experiments: read from the canonical list, route to a gene by name ─────
# Single source of truth = the experiment-list file. `sci_` plates (full time-series, different
# acquisition mode) are EXCLUDED here — analyzed separately. Only build06 CSVs that exist are kept.
EXPERIMENT_LIST = RUN_DIR / "20260605_sci_cilia_qc_first_pass.txt"


def _route_gene(exp: str) -> str | None:
    if "cep290" in exp:
        return "cep290"
    if "b9d2" in exp:
        return "b9d2"
    if "crispant" in exp:
        return "crispant"
    return None


def _load_queries() -> dict[str, list[str]]:
    by_gene: dict[str, list[str]] = {"cep290": [], "b9d2": [], "crispant": []}
    for line in EXPERIMENT_LIST.read_text().splitlines():
        exp = line.strip()
        if not exp or "_sci_" in exp or exp.startswith("sci_"):
            continue  # skip sci plates (analyzed separately)
        gene = _route_gene(exp)
        if gene is None:
            continue
        if not (B6 / f"df03_final_output_with_latents_{exp}.csv").exists():
            print(f"[warn] no build06 for {exp} — skipping")
            continue
        by_gene[gene].append(exp)
    return by_gene


_QUERIES = _load_queries()

# ── per-dataset config: reference + which query experiments + which labels apply ─
DATASETS = {
    "cep290": {"ref": CEP290_REF, "queries": _QUERIES["cep290"], "phenotype": True},
    "b9d2": {"ref": B9D2_REF, "queries": _QUERIES["b9d2"], "phenotype": True},
    "crispant": {"ref": CRISPANT_REF, "queries": _QUERIES["crispant"], "phenotype": False},
}


def resolve_feature_cols(paths: list[Path]) -> list[str]:
    """Intersection of z_mu_b_* columns across all files (reference + query share feature space)."""
    sets = [{c for c in pd.read_csv(p, nrows=0).columns if c.startswith("z_mu_b_")} for p in paths]
    shared = sorted(set.intersection(*sets)) if sets else []
    if not shared:
        raise ValueError("No shared z_mu_b_* feature columns.")
    return shared


def _load(path: Path, feat: list[str], *, gene_hint: str | None = None) -> pd.DataFrame:
    use = set(feat) | {GENO_COL, GROUP_COL, TIME_COL, PHENO_COL, "experiment_id", "snip_id"}
    df = pd.read_csv(path, usecols=lambda c: c in use, low_memory=False)
    for req in feat + [GENO_COL, GROUP_COL, TIME_COL]:
        if req not in df.columns:
            raise ValueError(f"{path.name} missing required col: {req}")
    # standardize genotype in place
    g = df[GENO_COL].astype(str)
    if gene_hint is not None:
        g = g.where(~g.isin(["uncertain", "unknown"]), f"{gene_hint}_unknown")
    df[GENO_COL] = g.map(lambda v: GENOTYPE_RENAME.get(v, v))
    df[ZYG_COL] = df[GENO_COL].map(to_zygosity)
    if "experiment_id" in df.columns:
        df["experiment_id"] = df["experiment_id"].astype(str)
    return df


def run_genotype_transfer(name: str, ref: pd.DataFrame, qry: pd.DataFrame,
                          feat: list[str]) -> pd.DataFrame:
    """Transfer zygosity (het/homo/wt) and BENCHMARK against known query genotype."""
    # reference: only embryos with a real zygosity (drop unknown/crispant/control)
    r = ref.dropna(subset=[ZYG_COL, TIME_COL])
    r = r[r[ZYG_COL] != "unknown"]
    if name == "crispant":
        return pd.DataFrame()  # crispants have no zygosity
    n_exp = r["experiment_id"].nunique()
    cv = "experiment_id" if n_exp >= 3 else None
    model = prepare_reference(r, feat, label_col=ZYG_COL, group_col=GROUP_COL,
                              time_col=TIME_COL, cv_group_col=cv)
    res = transfer_labels(model, qry, skip_flagged=False)
    emb = res["embryo_predictions"]
    meta = qry.drop_duplicates(GROUP_COL).set_index(GROUP_COL)
    emb["dataset"] = name
    emb["query_experiment"] = emb["query_embryo_id"].map(meta["query_experiment"])
    emb["true_zygosity"] = emb["query_embryo_id"].map(meta[ZYG_COL])
    emb["true_genotype"] = emb["query_embryo_id"].map(meta[GENO_COL])
    # benchmark only where the query has a known zygosity (exclude unknown/uncertain)
    emb["benchmarkable"] = emb["true_zygosity"].notna() & (emb["true_zygosity"] != "unknown")
    emb["correct"] = emb["benchmarkable"] & (emb["predicted_label"] == emb["true_zygosity"])
    return emb


def run_phenotype_transfer(name: str, ref: pd.DataFrame, qry: pd.DataFrame,
                           feat: list[str]) -> pd.DataFrame:
    """Transfer phenotype (no ground truth -> predictions only, verified later via images)."""
    r = ref.dropna(subset=[PHENO_COL, TIME_COL])
    r = r[~r[PHENO_COL].astype(str).isin(["unlabeled", "nan"])]
    # cep290: fold Intermediate -> Low_to_High (per condensation script)
    if name == "cep290":
        r.loc[r[PHENO_COL] == "Intermediate", PHENO_COL] = "Low_to_High"
    n_exp = r["experiment_id"].nunique()
    cv = "experiment_id" if n_exp >= 3 else None
    model = prepare_reference(r, feat, label_col=PHENO_COL, group_col=GROUP_COL,
                              time_col=TIME_COL, cv_group_col=cv)
    res = transfer_labels(model, qry, skip_flagged=False)
    emb = res["embryo_predictions"]
    meta = qry.drop_duplicates(GROUP_COL).set_index(GROUP_COL)
    emb["dataset"] = name
    emb["query_experiment"] = emb["query_embryo_id"].map(meta["query_experiment"])
    emb["true_genotype"] = emb["query_embryo_id"].map(meta[GENO_COL])
    return emb


def main() -> None:
    out = RUN_DIR / "transfer_results"
    out.mkdir(exist_ok=True)

    geno_all, pheno_all = [], []
    for name, cfg in DATASETS.items():
        qpaths = [B6 / f"df03_final_output_with_latents_{e}.csv" for e in cfg["queries"]]
        qpaths = [p for p in qpaths if p.exists()]
        if not qpaths:
            print(f"[{name}] no query build06 yet — skipping"); continue
        feat = resolve_feature_cols([cfg["ref"], *qpaths])

        print(f"\n######## {name} (feature dims={len(feat)}) ########")
        gene_hint = name if name in ("cep290", "b9d2") else None
        ref = _load(cfg["ref"], feat, gene_hint=gene_hint)
        qparts = []
        for e, p in zip([q for q in cfg["queries"] if (B6 / f"df03_final_output_with_latents_{q}.csv").exists()], qpaths):
            q = _load(p, feat, gene_hint=gene_hint)
            q["query_experiment"] = e
            qparts.append(q)
        qry = pd.concat(qparts, ignore_index=True)
        print(f"  ref={len(ref)} rows | query={len(qry)} rows ({qry['query_experiment'].nunique()} exps)")

        # ── GENOTYPE (benchmarked) ──
        g = run_genotype_transfer(name, ref, qry, feat)
        if not g.empty:
            geno_all.append(g)
        # ── PHENOTYPE (predictions only) ──
        if cfg["phenotype"]:
            pheno_all.append(run_phenotype_transfer(name, ref, qry, feat))

    # ===== GENOTYPE BENCHMARK (the headline) =====
    geno = pd.concat(geno_all, ignore_index=True)
    geno.to_csv(out / "genotype_transfer_predictions.csv", index=False)
    bench = geno[geno["benchmarkable"]]
    summ = (bench.groupby(["dataset", "query_experiment", "true_zygosity"])
            .agg(n=("query_embryo_id", "size"), accuracy=("correct", "mean"),
                 mean_top_prob=("top_probability", "mean"))
            .reset_index())
    summ.to_csv(out / "genotype_benchmark_by_class.csv", index=False)

    print("\n================ GENOTYPE TRANSFER BENCHMARK ================")
    print(summ.to_string(index=False))
    overall = (bench.groupby("dataset")["correct"].mean())
    print("\nper-dataset genotype accuracy:")
    print(overall.to_string())
    print(f"\noverall (benchmarkable embryos): {bench['correct'].mean():.1%} "
          f"({int(bench['correct'].sum())}/{len(bench)})")

    # ===== PHENOTYPE (predictions, not scored) =====
    if pheno_all:
        pheno = pd.concat(pheno_all, ignore_index=True)
        pheno.to_csv(out / "phenotype_transfer_predictions.csv", index=False)
        print("\n========= PHENOTYPE TRANSFER (predictions; verify via images) =========")
        for name in pheno["dataset"].unique():
            sub = pheno[pheno["dataset"] == name]
            print(f"\n[{name}] predicted phenotype distribution:")
            print(sub["predicted_label"].value_counts().to_string())

    print(f"\nWrote outputs to: {out}/")


if __name__ == "__main__":
    main()
