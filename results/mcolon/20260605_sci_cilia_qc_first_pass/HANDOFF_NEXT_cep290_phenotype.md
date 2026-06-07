# Handoff — NEXT PHASE: cep290 phenotype transfer (keep Not Penetrant) + sequenced per-plate views

**Created:** 2026-06-07 · mdcolon · resume-after-clear doc for the NEXT work.
Read this + `HANDOFF.md` (Phase-0/Phase-1 state) + `LABEL_STANDARDIZATION.md`.

This phase was DISCUSSED and DECIDED but NOT yet implemented (context was getting bloated, so we
handed off here). Nothing below is built yet except where noted.

---

## Why we're here
Phase 1 (sequenced-focus QC) is DONE. Looking at the cep290 phenotype results, mdcolon noticed
**Low_to_High is being predicted as Not Penetrant**, and that the current handling (which DROPS Not
Penetrant from the reference) hides this. Key insight:

> If a known-homozygous embryo gets predicted as **Not Penetrant**, that is the phenotype-analog of
> predicting it **wildtype** for genotype. So keeping Not Penetrant in makes the phenotype transfer a
> FAIR baseline directly comparable to the genotype predictions.

cep290 reference class counts (`cluster_categories`, 45,767 rows, 7 experiments):
`Not Penetrant 21187 · High_to_Low 11725 · Low_to_High 8775 · Intermediate 3170 · (NaN 910)`.
**Not Penetrant is the LARGEST class** — dropping it materially changes the model. That's the crux.

## DECISIONS (confirmed with mdcolon 2026-06-07)
1. **cep290 phenotype reference = KEEP 3 classes:** `High_to_Low` / `Low_to_High` / `Not Penetrant`.
   Still MERGE `Intermediate` → `Low_to_High` (so 3, not 4). Stop DROPPING Not Penetrant. This is the
   honest baseline. (Current `make_plots.py` drops Not Penetrant + merges Intermediate→LtH → 2 class.)
2. **Scope = NEW separate cep290 script.** Do NOT change `make_plots.py` or `make_plots_sequenced.py`
   yet. Build the keep-Not-Penetrant analysis standalone so existing plots are untouched while we
   investigate. (b9d2 stays as-is for now — see split below.)
3. **Confusion over time = BOTH views:**
   - **Reference-CV confusion** (REAL confusion — reference true phenotype is known): overall + faceted
     by `predicted_stage_hpf` windows, to show WHERE Low_to_High→Not Penetrant arises over development.
   - **Query (sequenced) companion**: predicted-phenotype distribution over stage (no query truth →
     not a true confusion, just the predicted mix over time).
4. **Two-phase plan for cep290 phenotype:**
   - **Phase A (THIS phase):** rerun with all 3 classes (keep Not Penetrant) → confusion + confusion-
     over-time → establish that known-homozygotes are mispredicted as Not Penetrant, and that
     Low_to_High↔Not Penetrant is the main confusion. Use the SAME model to look at genotype preds so
     the two are directly comparable.
   - **Phase B (LATER):** once established, SUBSET to High_to_Low / Low_to_High and redo the transfer.
5. **`cep290_LABEL_TRANSFER_README.md`** (NEW, gene-specific): document this. mdcolon: *"this is where
   cep290 and b9d2 split"* — the genes need DIFFERENT phenotype handling, so cep290 gets its own README
   and b9d2 will get its own later. Cross-link from `LABEL_STANDARDIZATION.md`.

## TASKS (not started)
### 1. New cep290 phenotype script (e.g. `cep290_phenotype_transfer.py`)
- Reuse loaders from `label_transfer_snapshots.py` / `build_reference_and_transfer.py` (single source
  for WHERE data comes from). The transfer step stays ISOLATED from plotting per the atomization rule.
- cep290 phenotype reference: keep `High_to_Low`/`Low_to_High`/`Not Penetrant`, merge Intermediate→LtH.
  (vs `build_reference_and_transfer.run_phenotype_transfer` which also drops Not Penetrant — DON'T here.)
- CV: `cv_group_col="experiment_id"` (7 exps ≥ 3, so leave-one-experiment-out, like the rest).
- Outputs: prediction CSVs (genotype + this 3-class phenotype, sequenced/stratum-tagged) so plotting
  consumes CSVs only.

### 2. Plots
- **Reference-CV confusion** (3-class) — overall + faceted by stage window. This is the diagnostic that
  shows LtH→NotPen over time. (The core `plot_reference_quality` confusion is a starting point but we
  want it faceted over `predicted_stage_hpf` windows — likely a custom plot.)
- **Query sequenced predicted-phenotype over stage** companion.
- **Same-model genotype view** for direct comparison (homo predicted as NotPen ≈ homo predicted as wt).

### 3. Two NEW sequenced-focus plots (mirror existing, add to `make_plots_sequenced.py`)
mdcolon explicitly asked for these (modeled on `plots/b9d2/b9d2_accuracy_heatmap.png` +
`plots/cep290/cep290_phenotype_reference_confusion.png`):
- **(a) per-plate × stage genotype-ACCURACY heatmap, SEQUENCED only** — for each plate, which embryos
  did we sequence and how well did we ASSIGN genotype (accuracy / what we predicted correctly — NOT
  F1). "This lets us know, for each plate, the ones that we sequenced: what are they?"
- **(b) per-plate genotype-ASSIGNMENT view, SEQUENCED only** — how genotypes get assigned per plate.
  mdcolon praised the confusion plot ("beautiful to look at what the label predictions were"). LAYOUT
  NOT FINALIZED — Claude proposed BOTH: a per-plate confusion grid (the 'beautiful' one) AND a
  per-plate stacked predicted-genotype composition. mdcolon didn't pick → default to BOTH, prune later.

### 4. `cep290_LABEL_TRANSFER_README.md`
Per-gene phenotype handling, the two-phase cep290 plan, CV scheme, genotype=zygosity, sequenced/
stratum conventions. State clearly: **this is the cep290/b9d2 split point.**

## Gotchas (carry-over)
- **Excel `genotype` = GROUND TRUTH** — don't reconcile het/homo vs sequenced; low het counts are real.
- Atomization rule: label transfer in its own script, plotting consumes CSVs only.
- sci_ plates (time-lapse) are analyzed SEPARATELY — this phase is QUERY plates only.
- Phenotype has NO query truth → predicted distributions only, no phenotype F1. Genotype F1 is real.

## State / where things are
- Scripts (committed, main): `label_transfer_snapshots.py` (transfer), `make_plots.py` (general),
  `make_plots_sequenced.py` (sequenced-focus), `make_3d_pca_sci.py`, `build_reference_and_transfer.py`
  (LEGACY loader lib), `patch_predicted_stage_hpf.py`.
- Outputs (local, gitignored): `transfer_results/*.csv` (incl. `sequenced_registry.csv`),
  `plots/sequenced_focus/*`, `plots/{b9d2,cep290,crispant}/*`, `plots/pca_3d*/`.
- cep290 phenotype currently 2-class (NotPen dropped) in `make_plots.py` — the thing this phase fixes.
