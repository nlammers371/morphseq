# SCI cilia gene14 imaging QC

This folder is for sequenced-only QC and phenotype-prediction analysis.

Core rule:
- Reference data can use all valid labeled reference embryos.
- Query analysis is restricted to embryos with `sequenced > 0`.
- Genotype outputs are QC against sequencing truth.
- Phenotype outputs are predictions only; query embryos do not have phenotype truth.
- Unsequenced query embryos are not part of the main analysis.

## Script Flow

Run scripts in order. The scripts should read like an analysis pipeline, not a software package.

0. `0_load_and_clean_datasets.py`
   - Define reference files, query plates, labels, and sequenced embryos.
   - Optionally copy source plate metadata Excel files into `source_plate_metadata_excels/` for provenance.
   - Excel copying is controlled by `COPY_EXCEL_FILES`; leave it `False` unless intentionally refreshing the provenance snapshot.
   - When copying is enabled, write `tables/copied_plate_metadata_excels.csv` with the copy date.
   - Write cleaned query/reference tables to `tables/`.

1. `1_fit_reference_models.py`
   - Fit genotype and homozygous phenotype reference models.
   - Save models and simple model summaries.

2. `2_predict_sequenced_embryos.py`
   - Load saved models.
   - Predict only sequenced query embryos.
   - Save genotype QC predictions and homozygous phenotype predictions.

3. `3_plot_qc_and_phenotype_predictions.py`
   - Read saved sequenced-only predictions.
   - Make genotype QC plots and homozygous phenotype/time-series plots.

## Labels

Genotype QC labels for `b9d2` and `cep290`:
- `wildtype`
- `heterozygous`
- `homozygous`

Genotype QC labels for cilia crispants use `genotype_clean` directly:
- `ab_wildtype`
- `foxj1a_crispant`
- `ift88_crispant`
- `ift88_ift74_crispant`
- `sspo_crispant`

Sequenced strata:
- `AB`
- `wildtype_sibling`
- `heterozygous`
- `homozygous`

Homozygous phenotype labels:
- `b9d2`: `CE` vs `HTA`; `BA_rescue` is pooled into `HTA`.
- `cep290`: `High_to_Low` vs `Low_to_High`; `Intermediate` is pooled into `Low_to_High`.

## Provenance

Step `0` can copy the plate metadata Excel files used for sequencing status into this folder. The current snapshot has already been copied; do not refresh it unless provenance intentionally changes. The copy manifest records:
- experiment name
- original Excel path
- copied Excel path
- copy date
- copy status

## Outputs

- `source_plate_metadata_excels/`: copied source plate metadata Excel files.
- `tables/`: cleaned manifests and cleaned data tables.
- `models/`: fitted reference models.
- `predictions/`: sequenced-only prediction CSVs.
- `plots/`: final figures.
