# Handoff — cilia QC first-pass label transfer

**Last updated:** 2026-06-07 (late) · mdcolon
Resume-after-clear doc. Read this + `README.md` + `LABEL_STANDARDIZATION.md`.

Goal: project the NEW cilia QC experiments onto OLD labeled reference data, per gene, to QC the new
data. GENOTYPE transfer = the benchmark (scored vs known truth); PHENOTYPE = predictions only.

---

## Scripts (run order, all in this dir)
1. `patch_predicted_stage_hpf.py --apply` — backfills `start_age_hpf` + `predicted_stage_hpf` into
   existing build03/build06 CSVs from the plate Excel (interim fix; `.bak_prestage` backups). DONE.
2. `fill_excel_start_age.py --apply` — filled 3 cep290 plates' genotyped-but-unstaged wells in the
   Excel source (`.bak_excel_fill` backups). DONE.
3. `build_reference_and_transfer.py` — defines `DATASETS` + loaders. Queries are READ FROM
   `20260605_sci_cilia_qc_first_pass.txt`, routed to gene by name, `sci_` plates EXCLUDED (analyzed
   separately), only-existing-build06 kept. Single source of truth for config.
4. `make_plots.py` — all static plots (see Outputs). Reuses #3's loaders.
5. `make_3d_pca.py` — interactive 3D PCA HTMLs with a dropdown to toggle views.

Env: `conda run -n segmentation_grounded_sam --no-capture-output python <script>`.
(Note: CLAUDE.md says `morphseq-env` but that env does NOT exist on this machine.)

## Label conventions (FINAL — set 2026-06-07)
- **cep290 phenotype:** Intermediate **MERGED INTO** Low_to_High; Not Penetrant DROPPED.
  Final classes = `High_to_Low`, `Low_to_High`. Colors = NWDB "talk"
  (`src/analyze/viz/presets/nwdb.py`): High_to_Low=#E76FA2 (pink), Low_to_High=#2FB7B0 (teal).
- **b9d2 phenotype:** `BA_rescue` **MERGED INTO `HTA`** (head-trunk angle). Final = `CE`/`HTA`/`wildtype`.
- **genotype benchmark:** cep290/b9d2 = ZYGOSITY (wt/het/homo), leave-one-experiment-out CV.
  crispant = GENE CLASS (foxj1a/ift88/sspo/ab), **k-fold CV** (single-experiment reference).
- Feature space = INTERSECTION of `z_mu_b_*` = 80 dims (ref has 100: z_mu_b_00..99; query has 80:
  z_mu_b_20..99). PCA confirmed to use ONLY these 80.

## Query plate set (from the list, sci_ excluded)
- cep290: 10 plates · b9d2: 6 plates (becomes 7 once plate02_t02 builds) · crispant: 4 plates.
- Headline genotype accuracy (current): cep290 42.1%, b9d2 30.7% on benchmarkable embryos.
  wildtype recovers well; het/homo weak — partly biology, partly reference-label noise.

## Outputs (`plots/`)
- `<gene>/<gene>_genotype_reference_quality_timebin.png`, `_reference_confusion.png`
- `<gene>/<gene>_genotype_transfer_result.png`, `_genotype_label_composition_by_plate.png`
- `<gene>/<gene>_accuracy_heatmap.png` (red=high/blue=low, plate×discrete-stage, faceted by class),
  `_accuracy_bars.png`
- `<gene>/per_plate/{genotype,phenotype}_transfer_<NN_stage>_<plate>.png` (stage-ordered)
- same set for phenotype (cep290/b9d2)
- `homozygous_focus/<gene>_homozygous_predicted_by_stage.png` + `_predictions.csv` — for known-homo
  embryos, what zygosity AND phenotype they're predicted as, by stage.
- `pca_3d/<gene>_pca_3d.html` — dropdown views: HPF / genotype / phenotype / experiment(batch) /
  source(ref-vs-query). PCA fit on ref+query combined so batch effects are visible. Per-label legend
  clicks now toggle individual classes.

---

## IN FLIGHT (started this session)
- **Build job `20785806` task 18** running (t001): builds `20260415_b9d2_30to48hpf_plate02_t02`
  (b9d2 30→48 plate02 — raw existed in `raw_image_data/Keyence/`, was never built). Its Excel was
  fixed: renamed to `..._well_metadata.xlsx` AND sheet `start_stage_hpf`→`start_age_hpf` (it had
  reintroduced the staging bug). `.xlsx.orig` backup kept.
  - **When done:** confirm
    `build06_output/df03_final_output_with_latents_20260415_b9d2_30to48hpf_plate02_t02.csv` exists +
    `predicted_stage_hpf` non-null, then re-run `make_plots.py` + `make_3d_pca.py` (auto-picks it up
    via the list — it's already line 18). Verify build with `qstat`/the `logs/` dir.
- The two experiment lists are now **symlinked**: results-folder copy → canonical
  `src/run_morphseq_pipeline/run_experiment_lists/20260605_sci_cilia_qc_first_pass.txt`. Edit only
  the source.

## NEXT TASK — sequenced-sample QC (the real goal)
**Finding (verified):** the Build01 metadata reader (`src/build/export_utils.py:116` `well_sheets`)
parses only `medium, genotype, chem_perturbation, start_age_hpf, embryos_per_well, temperature, pair`.
It **does NOT parse the `sequenced` sheet** (nor `strain`, `notes`, `qc`). Confirmed: no seq/strain
column in built metadata OR build06 for query plates. So `sequenced` never reaches the analysis.

**`sequenced` sheet coding (verified on b9d2 plates):**
`0`/blank = not sequenced · `1` = wildtype-confirmed (b9d2_wt, ab) · `2` = mutant-allele-confirmed
(covers BOTH het AND homo — NOT homo-only). Coverage is patchy: e.g. b9d2_18hpf_plate01 has 32
sequenced wells; cep290_24hpf_plate01 has ZERO.

**Plan (bypass the rebuild — do a 1:1 well join, like the start_age patch):**
1. Add a `sequenced` (and maybe `strain`) loader that reads the plate Excel `sequenced` sheet as the
   8×12 grid (same parse as `fill_excel_start_age.py` / `export_utils.py`), keyed by `well`, and
   joins onto build06 by well. (Do NOT construct well_id — keep the existing well-grid approach.)
   - Optionally also fix `export_utils.py` `well_sheets` to include `sequenced`+`strain` for future
     builds, but the join bypass unblocks analysis now without a rerun.
2. **Restrict the QC to sequenced>0 embryos** — that's the high-confidence truth set. For those:
   - genotype-transfer accuracy (does the embedding agree with the sequenced call: 1↔wt, 2↔het/homo?)
   - phenotype-prediction distribution
   - the 3D PCA quality check (do sequenced samples sit cleanly, or look low-quality / batch-shifted?)
3. **Report the genotype×phenotype diversity of the sequenced samples** — how many wt vs mutant, and
   what phenotypes they land in, per gene/stage. This is the deliverable mdcolon asked for.

## Gotchas
- New plate Excels keep arriving with sheet `start_stage_hpf` (the bug) and without the
  `_well_metadata` filename suffix — always check+fix before building.
- crispant has no zygosity benchmark column and no phenotype label; its accuracy heatmap may be empty
  if its query plates lack `start_age_hpf` staging.
- `20260324_cep290_18hpf_24hpf_plate02` is multi-stage (18 AND 24).
