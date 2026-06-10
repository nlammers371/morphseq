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

The plotting layer is split into focused `3x` scripts (the old monolith
`3_plot_qc_and_phenotype_predictions.py` has been retired). Shared colors, the per-gene
collection×support column spec, and `snap_to_design_stage` live in `plot_config.py`.

3a. `3a_audit_sequenced_coverage.py`
   - Data-sanity (not a figure): Excel `sequenced` sheet vs build04 QC, per non-sci
     b9d2/cep290 plate. Classifies every sequenced well `OK / QC_EXCLUDED / ABSENT /
     NO_BUILD04`.
   - Writes `MISSING_SEQUENCED_AUDIT.md`, `tables/sequenced_coverage_audit.csv`, and
     `plots/audit/sequenced_coverage_heatmap.png`.

3b. `3b_genotype_qc_per_plate.py`
   - Genotype QC from `predictions/sequenced_genotype_qc_cross_bin.csv` vs sequencing
     truth (`zygosity` / `genotype_clean`). Per gene: confusion, accuracy-by-class,
     plate×design-stage accuracy heatmap → `plots/genotype_qc/`. NOT the key plot.

3c. `3c_confidence_plot.py` — THE KEY greenlight artifact.
   - Homozygous-phenotype binary, per gene (b9d2 CE/HTA, cep290 High_to_Low/Low_to_High).
   - 5 rows (argmax bar / query P / reference P by true class / reference P&R / reference
     confusion) × per-gene collection×support columns. The 48 hpf timeseries column should
     out-perform the 48 hpf snapshot column. → `plots/confidence/<gene>_confidence.png`.

3d. `3d_feature_plot.py`
   - Physical-reality check: curvature (`baseline_deviation_normalized`) and length
     (`total_length_um`) over `predicted_stage_hpf`. Reference low-alpha backdrop,
     timeseries lines ending in circles colored by predicted class, snapshots as squares.
     → `plots/feature/<gene>_feature_24_48.png`.

3e/3f (DEFER/REUSE): 3D PCA batch-effect check and the per-embryo image portfolio reuse
   the working scripts in `../20260605_sci_cilia_qc_first_pass/` (`make_3d_pca*.py`,
   `make_embryo_portfolio.py`, `make_sequenced_portfolio_views.py`).

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
- `plots/audit/`: sequenced-well coverage heatmap (plate × status).
- `plots/genotype_qc/`: confusion matrices, per-class accuracy bars, and plate×stage accuracy heatmaps.
- `plots/confidence/`: the key 5-row × collection×support homozygous-phenotype confidence plot, per gene.
- `plots/feature/`: curvature + length over development (reference backdrop, timeseries lines, snapshot squares), per gene.

Note: the `.csv` outputs under `models/` and `predictions/` are gitignored — regenerate by
running scripts `1` then `2`.

## Engine

Label transfer uses the per-bin two-layer engine
`src/analyze/classification/label_transfer/perbin.py` (`prepare_reference_perbin` /
`transfer_labels_perbin`). Reference CV probabilities are saved as
`models/<model_id>_reference_cv.csv`; per-bin precision/recall/confusion live in the
`reference_performance` block of `models/<model_id>.pkl`.
