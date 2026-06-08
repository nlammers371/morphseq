# Handoff — cilia QC first-pass label transfer

**Last updated:** 2026-06-07 (sci PCA + sequenced-focus plan) · mdcolon
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
6. `make_3d_pca_sci.py` — SAME, but for the sci_ snapshot plates (handled separately); projects each
   into its gene's reference PCA space; adds a `sequenced` view. Outputs `plots/pca_3d_sci/`.
7. `make_plots_sequenced.py` — (NEXT, not yet written) sequenced-FOCUS plots; see NEXT TASK below.

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

## DONE this session (was IN FLIGHT)
- **b9d2 plate02_t02 built** (`20260415_b9d2_30to48hpf_plate02_t02`). build06 CSV exists,
  `predicted_stage_hpf` 35/35 non-null. `make_plots.py` + `make_3d_pca.py` re-ran with it in (b9d2
  now 7 plates). DONE.
- **sci_ snapshot plates staged + PCA'd.** Both `20260414_sci_b9d2_48hpf_plate01` and
  `20260415_sci_cep290_48hpf_plate01` are built. Added them to `patch_predicted_stage_hpf.py`'s list
  → `start_age_hpf`/`predicted_stage_hpf` now 100% non-null (`.bak_prestage` backups).
  - **`make_3d_pca_sci.py`** (NEW, standalone): projects each sci_ plate into its gene's REFERENCE
    PCA space (ref+query combined, like `make_3d_pca.py`) to check batch effects. Views: HPF /
    genotype(zygosity) / experiment(batch) / source(ref vs query) / **sequenced**. Outputs in
    `plots/pca_3d_sci/`. **VERDICT: no batch effects** — sci snapshots overlay the reference cleanly.
  - The `sequenced` PCA view reads the plate Excel `sequenced` sheet directly (NOT in build06) and
    joins by well via `snip_id`→`well`. Coding seen: code 1 = wt-confirmed, code 2 = mutant-confirmed.
- Committed: `d158bdd6` (sci PCA + stage patch). Followed repo convention = commit straight to `main`.
- The two experiment lists are **symlinked**: results-folder copy → canonical
  `src/run_morphseq_pipeline/run_experiment_lists/20260605_sci_cilia_qc_first_pass.txt`. Edit the source.

## sci_ snapshots are handled SEPARATELY (do not fold into the transfer scripts)
The sci_ (sky/sequenced) plates are snapshots processed differently from the transfer workflow:
no reference/transfer step, `phenotype`==`genotype` (no transferred labels), and a genotype
vocabulary (`b9d2_het`, `cep290_homo`, `ab`, `uncertain`, ...) that `T.to_zygosity` doesn't map.
They are EXCLUDED from `build_reference_and_transfer.py` DATASETS and from `make_plots.py`/
`make_3d_pca.py`. Their analysis lives in dedicated `*_sci.py` scripts.

---

## CODE ARCHITECTURE (atomized 2026-06-07)
The label-transfer step is ISOLATED from plotting (mdcolon: it's the publication-critical part, must
be auditable on its own). Data flow:
  `label_transfer_snapshots.py`  → writes CSVs → `make_plots*.py` (plotting only, no transfer).
- **`label_transfer_snapshots.py`** = THE transfer script (genotype + phenotype). Runs on ALL query
  embryos; tags each output row with `sequenced` (raw 0/1/2 from the Excel `sequenced` sheet, joined
  by well) + `stratum`. Writes `transfer_results/{genotype,phenotype}_transfer_predictions.csv` +
  `sequenced_registry.csv`. Reuses loaders/refs from `build_reference_and_transfer.py` (now LEGACY —
  kept as the audited loader library; prefer the new script).
- **`make_plots.py`** = general per-plate plots (re-runs transfer to get model objects for the core
  reference_quality/transfer_result figures).
- **`make_plots_sequenced.py`** = sequenced-FOCUS plots; consumes the CSVs only (no transfer/model).

## Phase 1 DONE — sequenced-FOCUS QC
No batch effects (Phase-0 PCA), so we honed in on SEQUENCED embryos across the QUERY plates (the two
sci_ time-lapse experiments are analyzed SEPARATELY, not here). 487 sequenced embryos, FOUR strata.

**Genotype label-transfer is the benchmark with a REAL F1** (vs the sequenced call, 3-class zygosity,
AB→wildtype). Headline (sequenced embryos): b9d2 macro-F1 0.31, cep290 macro-F1 0.35. wildtype
recovers best; homozygous under-called (embedding calls many homo as wt/het). cep290 has 0 het in the
sequenced set, b9d2 only ~4 — that is the REAL genotype data (Excel genotype = ground truth, do NOT
reconcile against sequenced; see the gotcha). **Phenotype = predicted distributions only, NO F1.**

Outputs `plots/sequenced_focus/`: per gene — `_seq_genotype_confusion.png`, `_seq_genotype_f1.png`,
`_seq_predicted_{genotype,phenotype}_by_stratum.png`, `_seq_phenotype_distribution_by_stratum.png`;
plus `seq_genotype_f1_summary.csv`, `seq_stratum_counts.csv`.

## (superseded) original NEXT TASK — sequenced-FOCUS QC plan
We checked the projection (no batch effects), so now hone in on the SEQUENCED embryos. This mirrors
the existing per-plate/general plots ONE-TO-ONE, but restricted to sequenced embryos and stratified.

**The `sequenced` sheet (verified):** `0`/blank = not sequenced · `1` = wildtype-confirmed
(includes AB) · `2` = mutant-allele-confirmed (BOTH het AND homo — not homo-only). NOT parsed by
Build01 (`src/build/export_utils.py:116` `well_sheets`) → must be joined from the Excel by well.

**FOUR strata** (decided 2026-06-07) — built from sequenced>0 embryos:
`homozygous` · `heterozygous` · `wildtype-sibling` (code 1, non-AB) · `AB` (the Trachnal AB).
Split code-2 into homo vs het using the `genotype` column. The four are what we care about; there
turn out to be many heterozygotes, hence 4 not 3.

**Decisions (confirmed with mdcolon):**
- **Registry: CSV here only** for now — `sequenced_registry.csv` (embryo_id, plate, well, sequenced
  code, genotype, zygosity, stratum, predicted_stage_hpf). Do NOT touch the Excel / `export_utils.py`
  yet.
- **New script `make_plots_sequenced.py`**, mirroring `make_plots.py` ~1:1, writing to a new
  `plots/sequenced_focus/` subtree. Reproduce the same plot set as the general/per-plate ones
  (composition-by-plate, transfer-result, reference-confusion, quality-by-timebin, accuracy
  heatmap/bars, per-plate transfer panels) but RESTRICTED to sequenced embryos and faceted/colored by
  the 4 strata + sequenced-vs-not where it makes sense. (One-to-one unless a plot clearly needs a
  tweak — use judgment.)
- **F1 is REAL here** (we have sequenced truth): score the genotype/zygosity label-transfer against
  the **sequenced genotype call** (3-class homo/het/wt; AB→wildtype). Report per-stratum F1.
- **Phenotype** still has NO truth → predictions only, NO F1. Report the predicted-phenotype
  DISTRIBUTION per stratum (and per plate/stage). The walk-through wants, per stratum:
  (a) phenotype distribution, (b) what we collectively predict them as, (c) the label-transfer result
  per embryo, (d) genotype F1 (phenotype has none, by design — we'll "work our way to it" later).

**Open scope notes:** sequenced coverage is patchy (some plates have 0 sequenced wells); the sci_
plates have the richest sequenced grids. Decide whether Phase 1 covers ALL query plates' sequenced
wells, or just the sci_ snapshots — likely both, but the sci_ plates are the dense ones.

## THEN — Phase 2 (raw-snapshot canvas; after Phase 1 is reviewed)
Using the honed-in sequenced set, build an image canvas: a grid where each cell is the raw snapshot
image of an embryo, captioned with **actual (sequenced) genotype · predicted genotype · predicted
phenotype**. Organize the canvas by what Phase 1 reveals (by stratum / predicted class). This is the
visual QC of whether the embeddings' calls match the images. NOT started — Phase 1 first.

## Sequenced embryo audit (2026-06-07) — why embryos are missing from sequenced_registry

Total: 604 sequenced wells in Excel across all non-sci plates. 491 in registry after fixes below.
Breakdown by cause (sci_ plates excluded throughout — analyzed separately):

**Fixed bugs (already resolved):**
- `20260414_b9d2_14hpf_plate01` — sequenced sheet header cols 6–9 were typed as `0` instead of
  `6,7,8,9`. Parser skipped those columns → E06, E07, G07, H07 silently dropped. Fixed in Excel
  (`.bak_sequenced_header_fix`). Registry now has 491 (was 487).
- `plate01_t02` B01 genotype changed `b9d2_wt` → `b9d2_unknown` (clearly not wt visually). Note
  added to Excel notes sheet. Backup: `.bak_b01_genotype_fix`.
- `frame_flag` removed from `use_embryo_flag` exclusion in `src/build/qc/embryo_flags.py` (too many
  false positives on snapshot plates — embryos with a tail near the well edge). Now informational
  only. qc_staged CSVs patched for all 23 named cilia plates. Recovery takes effect on next build06
  rerun (build06 was generated before the patch).

**Genuine imaging gaps (not bugs — no fix possible):**
- **Crispant plates imaged in columns 1–7 only** (56-well layout). Cols 8–12 were never used.
  - `20260319_cilia_crispant_24hpf` D01, E02: empty wells at collection, no embryo present.
  - `20260320_cilia_crispant_48hpf` row H (H01–H05, H07): embryos not detected due to height error
    in the microscope acquisition; H06 only survived. Lost to imaging QC.
- **Other plates with absent wells**: embryos were plated + sequenced but not imaged (dead/lost at
  collection, or the plate wasn't filled to 96 wells). Confirmed by cross-checking: genotype=None
  in Excel = template artifact (not a real embryo); genotype present = real embryo, just not imaged.
  Only 2 template artifacts found: `20260324_cep290_18hpf_24hpf_plate02` G08 and H08.

**Hard QC exclusions (frame_flag + sam2_qc_flag both True — not recoverable):**
- Both flags together = SAM2 segmentation genuinely failed (not just frame-clipping). These embryos
  don't have valid masks and can't contribute to the analysis.

**sa_outlier_flag:**
- `20260415_b9d2_30to48hpf_plate02_t02` B01: embryo looks fine visually but has unusual orientation
  → triggers SA outlier. Needs sa_outlier threshold review + rebuild to recover.

## Gotchas
- **Excel `genotype` = GROUND TRUTH.** Genotyping had some pre-pipeline errors; what's in the Excel
  (and flowed through to build06) is the corrected/known call. That's why there are many
  unknown/uncertain entries and why sequenced-code-2 wells may be labeled mostly homozygous with few
  het. Do NOT cross-check genotype vs sequenced to "reconcile" het/homo counts, and do NOT flag a low
  het count as a bug — it's real. (mdcolon, 2026-06-07.)
- New plate Excels keep arriving with sheet `start_stage_hpf` (the bug) and without the
  `_well_metadata` filename suffix — always check+fix before building.
- crispant has no zygosity benchmark column and no phenotype label; its accuracy heatmap may be empty
  if its query plates lack `start_age_hpf` staging.
- `20260324_cep290_18hpf_24hpf_plate02` is multi-stage (18 AND 24).
