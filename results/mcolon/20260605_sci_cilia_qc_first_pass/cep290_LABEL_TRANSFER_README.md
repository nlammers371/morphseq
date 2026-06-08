# cep290 Label Transfer Notes

## Purpose

This file records the cep290-specific label-transfer policy for the 20260605 sci/cilia QC first-pass
analysis. It is intentionally separate from b9d2 because the phenotype label spaces now diverge.

## Genotype benchmark

- Genotype transfer predicts zygosity: `wildtype`, `heterozygous`, `homozygous`.
- Query genotype is treated as ground truth when it is known.
- Sequenced-only genotype plots score against known zygosity on embryos with `sequenced > 0`.
- `AB` sequenced controls are treated as wildtype-like for genotype scoring.

## Phenotype transfer

cep290 Phase A keeps three phenotype classes:

- `High_to_Low`
- `Low_to_High`
- `Not Penetrant`

`Intermediate` is merged into `Low_to_High`. `Not Penetrant` is not dropped in Phase A because it is
the phenotype analogue of a mutant embryo being predicted wildtype: if known mutant/sequenced embryos
land in `Not Penetrant`, that is a diagnostic outcome, not a class to hide.

The query has no phenotype ground truth, so query phenotype figures are predicted distributions only.
Reference-CV phenotype confusion is a real confusion matrix because the reference has true phenotype
labels.

## Phase plan

- Phase A: keep `Not Penetrant`, run 3-class cep290 phenotype transfer, inspect reference-CV confusion
  overall and over stage, and compare query/sequenced predicted phenotype distributions over stage.
- Phase B: if Phase A confirms that `Low_to_High` versus `Not Penetrant` is the main issue, optionally
  subset to `High_to_Low`/`Low_to_High` and rerun as a second analysis.

## Script and outputs

- Script: `cep290_phenotype_transfer.py`
- Transfer CSVs:
  - `transfer_results/cep290_phenotype_3class_predictions.csv`
  - `transfer_results/cep290_phenotype_3class_reference_cv_predictions.csv`
  - `transfer_results/cep290_genotype_same_model_context.csv`
- Plots:
  - `plots/cep290_phaseA_3class/cep290_phenotype_3class_reference_confusion.png`
  - `plots/cep290_phaseA_3class/cep290_phenotype_3class_reference_confusion_by_stage.png`
  - `plots/cep290_phaseA_3class/cep290_phenotype_3class_query_predicted_by_stage.png`
  - `plots/cep290_phaseA_3class/cep290_phenotype_3class_sequenced_predicted_by_stage.png`

## Conventions

- Reference CV uses `experiment_id` as the leave-one-experiment-out group.
- The main transfer workflow excludes `sci_` plates; those are analyzed by dedicated sci scripts.
- Existing two-directional cep290 plots in `make_plots.py` are preserved for continuity and are not
  overwritten by Phase A.
