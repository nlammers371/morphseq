# Cilia QC first-pass — label transfer

**Date:** 2026-06-06 · **Owner:** mdcolon

Project the NEW QC-first-pass experiments (`20260605_sci_cilia_qc_first_pass.txt`, built into
`build06_output/df03_final_output_with_latents_*.csv`) onto our OLD labeled reference data, to
check whether the new data is high quality.

See `LABEL_STANDARDIZATION.md` for the label conventions.

## Design (per-dataset, no cross-dataset mixing)

Each query experiment is transferred against **its own dataset's reference** — we already know the
gene, so we never ask "is this b9d2 or cep290." Two labels, two roles:

| Label | Ground truth on query? | Role |
|---|---|---|
| **genotype** (zygosity: homozygous/heterozygous/wildtype) | **YES** (known) | **BENCHMARK** — transfer it, score against the known query genotype. Tests the transfer machinery + embedding quality. **This is the emphasis.** |
| **phenotype** (cep290 trajectory classes; b9d2 CE/HTA/wt) | NO | **predictions only** — produced for visual verification against images; not scored. |

Crispants (foxj1a/ift88/sspo) have **genotype only** (no phenotype label).

References:
- cep290 → `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
- b9d2 → `results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv`
- crispant → `build06_output/df03_final_output_with_latents_20260202.csv`

Method: `src/analyze/classification/label_transfer` (`prepare_reference` + `transfer_labels`).

## Run

```
conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260605_sci_cilia_qc_first_pass/build_reference_and_transfer.py
```

Outputs (`transfer_results/`):
- `genotype_transfer_predictions.csv` — per-embryo zygosity prediction + true genotype + correct flag
- `genotype_benchmark_by_class.csv` — accuracy per (dataset, experiment, true zygosity)
- `phenotype_transfer_predictions.csv` — per-embryo phenotype prediction (unscored)

## First-pass findings (2026-06-06)

**Genotype benchmark — wildtype recovers reliably; het/homo are hard.**
- **wildtype** transfers well across datasets (cep290 wt 0.67–0.92; b9d2 wt 0.69–0.80) — the
  cleanest, most separable class, and the main signal that the transfer machinery + embeddings work.
- **heterozygous / homozygous** are weak and inconsistent (many 0.0; occasional spikes e.g.
  cep290 homo 0.90 on `..._plate01_t02`). Overall ~38% on benchmarkable embryos.
- This is **biologically reasonable**: het vs homo vs wt is hard to separate by morphology at early
  stages (heterozygotes look wildtype-like). Reliable wildtype recovery is the QC win.

**Phenotype — predictions emitted for visual check (not scored):**
- cep290: mostly `Not Penetrant` (290), then `High_to_Low` (112), `Low_to_High` (78).
- b9d2: `wildtype` (61), `CE` (43), `HTA` (8).
- **Next:** mdcolon verifies these against the actual images.

## ⚠️ BLOCKER FOUND & FIXED — `predicted_stage_hpf` all-NaN on new data (2026-06-06)

The per-time-window analysis was blocked because **`predicted_stage_hpf` is all NaN** on every
new query experiment (reference has full hpf: cep290 12–140, b9d2 11–125).

**Root cause (plate-metadata sheet-name drift):**
- `predicted_stage_hpf` is computed in **Build03** (`_ensure_predicted_stage_hpf`,
  `build03A_process_images.py:1640`) via the Kimmel formula
  `start_age_hpf + (Time Rel/3600)·(0.055·temp − 0.57)`.
- `start_age_hpf` is read by Build01 from a **sheet literally named `start_age_hpf`** in the
  plate-metadata Excel (`export_utils.py:120,149`).
- The **new (2026) Excels renamed that sheet to `start_stage_hpf`** — populated correctly, but the
  reader only knows `start_age_hpf`, so it silently reads nothing → `start_age_hpf` empty → formula
  yields NaN. (Made silent by a column-EXISTS-only guard + a bare `except: pass`.)
- **Blast radius:** exactly the **22 new 2026 QC-era experiments**. The 135 older experiments
  (incl. all reference data) use `start_age_hpf` and stage fine.

**Fix applied (data, not code):** renamed the `start_stage_hpf` sheet → `start_age_hpf` in all 22
affected `*_well_metadata.xlsx` (values preserved; backups in
`metadata/plate_metadata/_backup_before_sheet_rename_20260606/`).

**Still required:** re-run Build01→Build06 on the affected experiments to backfill
`predicted_stage_hpf`, THEN redo the per-time-window transfer analysis. Until then the genotype/
phenotype numbers above ran with NaN/degenerate time and should not be trusted for the
time-resolved view.

**Notes:** `20260324_cep290_18hpf_24hpf_plate02` is multi-stage (`start_age` = 18 AND 24).
`20260414_sci_b9d2_48hpf_plate01` is named `48hpf` but its sheet value is `30` — flagged, not
altered (verify which is correct).

## Caveats / open

- The het/homo accuracy partly reflects **reference label noise** (reference genotypes were
  themselves assigned, some morphologically), not only embedding quality — low scores are not a
  pure pipeline failure.
- Feature space is the **intersection** of `z_mu_b_*` cols (reference has `z_mu_b_00..99`; newer
  query build06 only carry `z_mu_b_20..99`) → 80 shared dims used. If the missing 20 dims matter,
  rebuild query embeddings on the full latent set.
- `unknown`/`uncertain` query embryos are excluded from the genotype benchmark (no trusted truth);
  resolving them needs strain from plate metadata (not in build06).
