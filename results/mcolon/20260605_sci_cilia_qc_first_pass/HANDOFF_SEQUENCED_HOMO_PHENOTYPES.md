# Handoff - sequenced homozygous phenotype focus

Date: 2026-06-08  
Owner: mdcolon / Codex

## Current goal

We moved from broad dataset QC to sequenced-embryo interpretation. The workflow is now organized around:

1. **Everybody sequenced** - genotype QC and phenotype composition for all sequenced embryos.
2. **Homozygous versus not homozygous** - compare true genotype groups (`AB`, gene wildtype, het, homo) and how they map.
3. **Homozygous-only sequenced** - conditional phenotype models trained only on homozygous mutant reference embryos and only on the two mutant phenotypes.

## Key interpretation

The genotype-transfer plots are QC: they tell us whether sequenced embryos map to their known genotype labels. The phenotype-transfer plots are not scored against query truth; they show predicted phenotype composition.

For the homozygous-only phase, the model is conditional: given a homozygous mutant, classify only between the two mutant phenotypes.

- cep290: `High_to_Low` vs `Low_to_High`
- b9d2: `CE` vs `HTA`, with `BA_rescue -> HTA`

This is intentionally separate from the earlier 3-class cep290 model that kept `Not Penetrant`.

## Scripts added/updated

- `cep290_phenotype_transfer.py`
  - 3-class cep290 Phase-A model: `High_to_Low`, `Low_to_High`, `Not Penetrant`.
  - Includes target-hpf +/-2 reference CV confusion.

- `cep290_homo_low_to_high.py`
  - Homozygous-conditional cep290 model: `High_to_Low` vs `Low_to_High`.
  - Generates reference CV target-window confusion, sequenced homo mini-bars, and sequenced probability spectrum.

- `b9d2_homo_ce_hta.py`
  - Homozygous-conditional b9d2 model: `CE` vs `HTA`, with `BA_rescue -> HTA`.
  - Uses 14/18/30/48 hpf target windows. There is no b9d2 sequenced 24 hpf cohort in this run.
  - Uses LOEO CV where >=2 experiments exist; falls back to stratified k-fold for 14/18 hpf where only one reference experiment exists but both classes are present.
  - Generates reference CV target-window confusion, sequenced homo mini-bars, and sequenced probability spectrum.

- `make_plots_sequenced.py`
  - Sequenced-focus plots now use true genotype group facets and expected stage columns.
  - b9d2 phenotype plotting merges `BA_rescue -> HTA`.

## Important outputs

### Sequenced-focus genotype/phenotype outputs

- `plots/sequenced_focus/cep290/`
- `plots/sequenced_focus/b9d2/`

Key files per gene:

- `<gene>_seq_true_genotype_accuracy_heatmap_plate_stage.png`
- `<gene>_seq_true_genotype_mapping_composition_plate_stage.png`
- `<gene>_seq_true_genotype_phenotype_minibars_plate_stage.png`
- `<gene>_seq_genotype_assignment_confusion_by_plate.png`

### Homozygous-only sequenced focus

cep290:

- `plots/sequenced_focus/cep290/homozygous_focus/cep290_homo_low_to_high_predicted_phenotype_minibars_homo_only.png`
- `plots/sequenced_focus/cep290/homozygous_focus/cep290_homo_low_to_high_predicted_phenotype_minibars_all_true_genotypes.png`
- `plots/sequenced_focus/cep290/homozygous_focus/cep290_homo_low_to_high_homo_only_probability_spectrum_sequenced.png`

b9d2:

- `plots/sequenced_focus/b9d2/homozygous_focus/b9d2_homo_ce_hta_predicted_phenotype_minibars_homo_only.png`
- `plots/sequenced_focus/b9d2/homozygous_focus/b9d2_homo_ce_hta_predicted_phenotype_minibars_all_true_genotypes.png`
- `plots/sequenced_focus/b9d2/homozygous_focus/b9d2_homo_ce_hta_homo_only_probability_spectrum_sequenced.png`

### Homozygous-only reference CV outputs

cep290:

- `plots/cep290/cep290_homo_only_and_its_phenotypes/cep290_homo_low_to_high_reference_confusion_target_hpf_pm2.png`

b9d2:

- `plots/b9d2/b9d2_homo_only_and_its_phenotypes/b9d2_homo_ce_hta_reference_confusion_target_hpf_pm2.png`

## Current numeric summaries

cep290 homo-only sequenced true-homozygous split:

- `High_to_Low`: 31
- `Low_to_High`: 43

b9d2 homo-only sequenced true-homozygous split:

- `CE`: 16
- `HTA`: 48

b9d2 homo-only reference CV target windows:

- 14 hpf: stratified 5-fold fallback, 17 embryos
- 18 hpf: stratified 5-fold fallback, 17 embryos
- 30 hpf: leave-one-experiment-out, 37 embryos
- 48 hpf: leave-one-experiment-out, 35 embryos

The fallback is necessary because 14/18 hpf b9d2 homo-only reference data has both CE/HTA classes but only one experiment (`20251125`), so LOEO CV is impossible there.

## Plot style notes

- Keep the b9d2 phenotype colors used here going forward:
  - `CE`: green (`#1b9e77`)
  - `HTA`: orange (`#d95f02`)
- Probability-spectrum plots use a diverging gradient: phenotype A color -> gray at 0.5 -> phenotype B color.
- The spectrum colorbar has been moved to a dedicated right-side axis to avoid overlapping panels.

## Next likely step

Inspect the sequenced embryos/images themselves using these phenotype assignments, especially embryos where:

- the same well/time series changes predicted phenotype across timepoints,
- homozygous mutants land near the 0.5 uncertainty region,
- wildtype/AB controls project strongly into mutant phenotype space.

This should help distinguish real phenotype dynamics/rescue from batch effects or model uncertainty.
