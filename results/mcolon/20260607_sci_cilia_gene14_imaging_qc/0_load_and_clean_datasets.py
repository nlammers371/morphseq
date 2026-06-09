"""
0 - Load and clean datasets for sequenced-only SCI cilia QC.

This script does not fit models and does not make plots.

It answers:
    1. Which reference files are we using?
    2. Which query plates are we using?
    3. Which plate metadata Excel files were copied here for provenance?
    4. Which embryos are sequenced?
    5. What labels will later scripts use?

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/0_load_and_clean_datasets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]

BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
PLATE_METADATA_DIR = PROJECT_ROOT / "metadata" / "plate_metadata"
EXPERIMENT_LIST = PROJECT_ROOT / "src/run_morphseq_pipeline/run_experiment_lists/20260605_sci_cilia_qc_first_pass.txt"

TABLE_DIR = RUN_DIR / "tables"
EXCEL_COPY_DIR = RUN_DIR / "source_plate_metadata_excels"
TABLE_DIR.mkdir(exist_ok=True)
EXCEL_COPY_DIR.mkdir(exist_ok=True)

# This is intentionally a plain string so provenance is stable in the output table.
COPIED_ON = "2026-06-09"
COPY_EXCEL_FILES = False  # Set True only when intentionally refreshing provenance copies.


# -----------------------------------------------------------------------------
# Reference datasets
# -----------------------------------------------------------------------------

REFERENCE_FILES = {
    "b9d2": PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv",
    "cep290": PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv",
    "crispant": BUILD06_DIR / "df03_final_output_with_latents_20260202.csv",
}


# -----------------------------------------------------------------------------
# Label conventions
# -----------------------------------------------------------------------------

GENOTYPE_RENAME = {
    "b9d2_homo": "b9d2_homozygous",
    "b9d2_homozygous": "b9d2_homozygous",
    "b9d2_het": "b9d2_heterozygous",
    "b9d2_heterozygous": "b9d2_heterozygous",
    "b9d2_wt": "b9d2_wildtype",
    "b9d2_wildtype": "b9d2_wildtype",
    "b9d2_unknown": "b9d2_unknown",
    "b9d2_uncertain": "b9d2_unknown",
    "cep290_homo": "cep290_homozygous",
    "cep290_homozygous": "cep290_homozygous",
    "cep290_het": "cep290_heterozygous",
    "cep290_heterozygous": "cep290_heterozygous",
    "cep290_wt": "cep290_wildtype",
    "cep290_wildtype": "cep290_wildtype",
    "cep290_unknown": "cep290_unknown",
    "cep290_uncertain": "cep290_unknown",
    "foxj1a_crispant": "foxj1a_crispant",
    "ift88_crispant": "ift88_crispant",
    "sspo_crispant": "sspo_crispant",
    "scospondin_crispant": "sspo_crispant",
    "if88_ift74_crispant": "ift88_ift74_crispant",
    "ab": "ab_wildtype",
    "inj-ctrl": "injection_control",
    "unknown": "unknown",
    "uncertain": "unknown",
}

HOMO_PHENOTYPE_LABELS = {
    "b9d2": ["CE", "HTA"],
    "cep290": ["High_to_Low", "Low_to_High"],
}

WELLS = [f"{row}{col:02}" for row in "ABCDEFGH" for col in range(1, 13)]


def gene_for_experiment(experiment: str) -> str | None:
    """Route an experiment name to the gene-specific analysis group."""
    if "b9d2" in experiment:
        return "b9d2"
    if "cep290" in experiment:
        return "cep290"
    if "crispant" in experiment:
        return "crispant"
    return None


def zygosity_from_genotype(genotype: str) -> str | None:
    """Extract the zygosity-only label from a gene-specific genotype label."""
    genotype = str(genotype)
    for label in ["homozygous", "heterozygous", "wildtype", "unknown"]:
        if genotype.endswith(label):
            return label
    return None


def standardize_genotype(raw_genotype: object, gene: str | None) -> str:
    """Convert raw genotype calls into gene-specific labels like b9d2_homozygous."""
    value = str(raw_genotype).strip()
    if gene in {"b9d2", "cep290"} and value in {"unknown", "uncertain"}:
        value = f"{gene}_unknown"
    return GENOTYPE_RENAME.get(value, value)


def clean_phenotype(gene: str, phenotype: object) -> object:
    """Pool phenotype labels used downstream: b9d2 BA_rescue -> HTA; cep290 Intermediate -> Low_to_High."""
    if pd.isna(phenotype):
        return phenotype
    value = str(phenotype)
    if gene == "b9d2" and value == "BA_rescue":
        return "HTA"
    if gene == "cep290" and value == "Intermediate":
        return "Low_to_High"
    return value


def read_experiment_list() -> pd.DataFrame:
    """Read the query plate list and mark gene, sci timelapse status, and build06 availability."""
    rows = []
    for line in EXPERIMENT_LIST.read_text().splitlines():
        experiment = line.strip()
        if not experiment:
            continue
        gene = gene_for_experiment(experiment)
        if gene is None:
            continue
        build06_path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        rows.append(
            {
                "experiment": experiment,
                "gene": gene,
                "is_sci_timelapse": experiment.startswith("sci_") or "_sci_" in experiment,
                "build06_path": str(build06_path),
                "build06_exists": build06_path.exists(),
            }
        )
    return pd.DataFrame(rows)


def find_plate_metadata_excel(experiment: str) -> Path | None:
    """Find the source plate metadata Excel for one experiment."""
    for path in [
        PLATE_METADATA_DIR / f"{experiment}_well_metadata.xlsx",
        PLATE_METADATA_DIR / f"{experiment}.xlsx",
    ]:
        if path.exists():
            return path
    return None


def copy_plate_metadata_excels(experiment_manifest: pd.DataFrame) -> pd.DataFrame:
    """Copy source plate metadata Excels into this folder and return a provenance table."""
    rows = []
    for experiment in experiment_manifest["experiment"]:
        source = find_plate_metadata_excel(experiment)
        if source is None:
            rows.append(
                {
                    "experiment": experiment,
                    "source_excel": "",
                    "copied_excel": "",
                    "copied_on": COPIED_ON,
                    "status": "missing",
                }
            )
            continue
        copied = EXCEL_COPY_DIR / source.name
        shutil.copy2(source, copied)
        rows.append(
            {
                "experiment": experiment,
                "source_excel": str(source),
                "copied_excel": str(copied),
                "copied_on": COPIED_ON,
                "status": "copied",
            }
        )
    return pd.DataFrame(rows)


def read_sequenced_grid(experiment: str) -> dict[str, int] | None:
    """Read the Excel sequenced sheet as well -> code, where 0 is not sequenced and 1 or 2 is sequenced."""
    path = find_plate_metadata_excel(experiment)
    if path is None:
        return None
    with pd.ExcelFile(path) as xlf:
        if "sequenced" not in xlf.sheet_names:
            return None
        sheet = xlf.parse("sequenced", header=0)
    block = sheet.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
    values = block.to_numpy(dtype=str).ravel()
    out = {}
    for well, raw in zip(WELLS, values):
        text = raw.strip()
        try:
            out[well] = int(float(text)) if text not in {"", "nan"} else 0
        except ValueError:
            out[well] = 0
    return out


def load_query_table(row: pd.Series) -> pd.DataFrame:
    """Load one query build06 table and add genotype_clean, zygosity, phenotype_clean, and sequencing labels."""
    experiment = row["experiment"]
    gene = row["gene"]
    path = Path(row["build06_path"])
    df = pd.read_csv(path, low_memory=False)
    df["experiment"] = experiment
    df["gene"] = gene
    # genotype_clean keeps the gene name plus zygosity, e.g. b9d2_homozygous.
    # zygosity is the zygosity-only helper column used for genotype QC.
    df["genotype_clean"] = df["genotype"].map(lambda x: standardize_genotype(x, gene))
    df["zygosity"] = df["genotype_clean"].map(zygosity_from_genotype)
    if "cluster_categories" in df.columns:
        df["phenotype_clean"] = df["cluster_categories"].map(lambda x: clean_phenotype(gene, x))
    else:
        df["phenotype_clean"] = np.nan

    sequenced_grid = read_sequenced_grid(experiment)
    if sequenced_grid is None:
        df["sequenced"] = np.nan
    else:
        df["sequenced"] = df["well"].map(sequenced_grid).fillna(0).astype(int)

    df["sequenced_stratum"] = ""
    df.loc[(df["sequenced"] > 0) & (df["genotype_clean"] == "ab_wildtype"), "sequenced_stratum"] = "AB"
    df.loc[(df["sequenced"] == 1) & (df["sequenced_stratum"] == ""), "sequenced_stratum"] = "wildtype_sibling"
    df.loc[(df["sequenced"] == 2) & (df["zygosity"] == "heterozygous"), "sequenced_stratum"] = "heterozygous"
    df.loc[(df["sequenced"] == 2) & (df["zygosity"] == "homozygous"), "sequenced_stratum"] = "homozygous"
    df.loc[(df["sequenced"] == 2) & (df["sequenced_stratum"] == ""), "sequenced_stratum"] = "mutant_unresolved"
    return df


def load_reference_table(gene: str, path: Path) -> pd.DataFrame:
    """Load one reference table and apply the same genotype and phenotype label cleanup."""
    df = pd.read_csv(path, low_memory=False)
    df["gene"] = gene
    # genotype_clean keeps the gene name plus zygosity, e.g. b9d2_homozygous.
    # zygosity is the zygosity-only helper column used for genotype QC.
    df["genotype_clean"] = df["genotype"].map(lambda x: standardize_genotype(x, gene))
    df["zygosity"] = df["genotype_clean"].map(zygosity_from_genotype)
    if "cluster_categories" in df.columns:
        df["phenotype_clean"] = df["cluster_categories"].map(lambda x: clean_phenotype(gene, x))
    else:
        df["phenotype_clean"] = np.nan
    return df


def main() -> None:
    print("0 - load and clean datasets")
    print("Reference: all valid labeled reference embryos")
    print("Query: sequenced embryos only for downstream analysis")

    experiment_manifest = read_experiment_list()
    experiment_manifest.to_csv(TABLE_DIR / "experiment_manifest.csv", index=False)

    print(f"\nQuery experiments in manifest: {len(experiment_manifest)}")
    print(experiment_manifest.groupby(["gene", "is_sci_timelapse", "build06_exists"]).size().to_string())

    if COPY_EXCEL_FILES:
        excel_manifest = copy_plate_metadata_excels(experiment_manifest)
        excel_manifest.to_csv(TABLE_DIR / "copied_plate_metadata_excels.csv", index=False)
        print(f"\nCopied plate metadata Excel files on {COPIED_ON}:")
        print(excel_manifest["status"].value_counts().to_string())
    else:
        print("\nCOPY_EXCEL_FILES is False; not copying or overwriting plate metadata Excels.")
        print(f"Existing copied files, if any, stay in: {EXCEL_COPY_DIR.relative_to(RUN_DIR)}/")

    query_tables = []
    for _, row in experiment_manifest[experiment_manifest["build06_exists"]].iterrows():
        print(f"  loading query {row['experiment']}")
        query_tables.append(load_query_table(row))
    query = pd.concat(query_tables, ignore_index=True)

    sequenced_embryos = (
        query[query["sequenced"] > 0]
        .drop_duplicates("embryo_id")
        [[
            "embryo_id",
            "experiment",
            "gene",
            "well",
            "sequenced",
            "sequenced_stratum",
            "genotype_clean",
            "zygosity",
        ]]
        .sort_values(["gene", "experiment", "well"])
    )

    query.to_csv(TABLE_DIR / "query_all_rows_clean.csv", index=False)
    sequenced_embryos.to_csv(TABLE_DIR / "query_sequenced_embryos.csv", index=False)

    reference_tables = []
    for gene, path in REFERENCE_FILES.items():
        print(f"  loading reference {gene}")
        ref = load_reference_table(gene, path)
        reference_tables.append(ref)
        ref.to_csv(TABLE_DIR / f"reference_{gene}_clean.csv", index=False)
    reference = pd.concat(reference_tables, ignore_index=True)
    reference.to_csv(TABLE_DIR / "reference_all_clean.csv", index=False)

    label_rows = []
    for gene in ["b9d2", "cep290"]:
        label_rows.append(
            {
                "gene": gene,
                "model": "genotype_qc",
                "reference_labels": "wildtype, heterozygous, homozygous",
                "query_scope": "sequenced embryos only",
            }
        )
        label_rows.append(
            {
                "gene": gene,
                "model": "homozygous_phenotype",
                "reference_labels": ", ".join(HOMO_PHENOTYPE_LABELS[gene]),
                "query_scope": "sequenced homozygous embryos for main interpretation",
            }
        )
    label_rows.append(
        {
            "gene": "crispant",
            "model": "genotype_qc",
            "reference_labels": "crispant/control labels",
            "query_scope": "sequenced embryos only if present",
        }
    )
    pd.DataFrame(label_rows).to_csv(TABLE_DIR / "label_plan.csv", index=False)

    print("\nSequenced embryo counts:")
    print(sequenced_embryos.groupby(["gene", "sequenced_stratum"]).size().to_string())

    print("\nReference embryo counts by gene x zygosity:")
    ref_counts = reference.drop_duplicates("embryo_id").groupby(["gene", "zygosity"]).size()
    print(ref_counts.to_string())

    print(f"\nWrote cleaned tables to: {TABLE_DIR.relative_to(RUN_DIR)}/")
    if COPY_EXCEL_FILES:
        print(f"Copied source Excel files to: {EXCEL_COPY_DIR.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
