# Sequencing Greenlight Plan

## The workflow has two major goals and plots should be aimed towards that for sequenced embryos:
1) Confirm that image data are complete enough, no batch effects good quality, and passing through the pipeline
   - this is largley QC step
2) Confirm that the cohort contains a useful distribution of phenotypes with sufficient label-transfer confidence.
   - this is Largely a label transfer/confidence problem 
  

## Dataset trickiness 
If the data set has multiple different kinds of problems 
- some time points only have snapshots (single image)
- some time have time series (multiple images over time) 
- Some embryo has snapshots across different times and also has time series data
  - as a consequence there is some slightly redundand data 
  - this is because we wanted have hbackups,
    -  this is primarily a confusion points for cep290 and b9d2 who each had "plate01" with both at 30hpf and 48hpf (given the t01 or t02 suffix) as well as time lapse, labeld _sci (not the best naming i know...)
    -  plate02 for cep290 and b9d20 48hpf collected data was ONLY snapshots, so no time series data for those plates
- also metadta has to be From the Excel spreadsheets to the main file in a particular way that aligns with the collection  reality 

- to help with this, we have a "sequenced" column in the query metadata that is >0 (1 is seq wt and 2 is seq mutant) for embryos that have sequencing data and 0 for those that do not. This allows us to easily filter the query dataset to only include embryos that have sequencing data for the main analysis, while still keeping the unsequenced embryos in the dataset for potential future analyses or reference.
- To help with some of this I realize that to help with this disambiguity when we're processing the data, we need to add a collections section to the metadata to help organize things. 
- This is a bit of a mess but I think we can get through it with careful documentation and organization. The key is to be clear about which embryos have sequencing data and which do not, and to make sure that the analysis pipeline is designed to handle this appropriately.
- Overall, the main thing is to be clear about the structure of the dataset and to design the analysis pipeline in a way that can handle the complexity of the data. With careful organization and documentation, we should be able to navigate this successfully.
- The main thing is to be clear about the structure of the dataset and to design the analysis pipeline in a way that can handle the complexity of the data. With careful organization and documentation, we should be able to navigate this successfully.
  - We need to add collection time at data. Specifically, what this means is that we will use the start age HPF plate for all of the experiment IDs, which are plate IDs. Except in all caps for the plates labeled 30 to 48, as they are labeled that because data was collected over those time points and they represent the same embryos. Such plates should be labeled collection at 48 HPF. 
  - as such we can utill use the prediscted stage hpf as the true age of these embryos for the purposes of the analysis, and we can use the collection time to help us understand the structure of the dataset and to design our analysis pipeline accordingly. This will help us to ensure that we are analyzing the data in a way that is consistent with the reality of the data collection process, and that we are able to make meaningful conclusions from our analysis.


## First Steps 
- Check for missing data Discrepancy between sequenced but not getting through the model, so not present in our data 
- Fit model on reference and then apply labels to query. 
  - note there are different splits deending on question being ask (e.g. homzyous only then homzyous only phenotypes)

## Plots 

### portfolio plots
Labeled by true genotype, predicted genotype, predicted phenotype, and QC status 
- helps us see if image data going in is high quality, great for spot checking (what was used for results/mcolon/20260605_sci_cilia_qc_first_pass/portfolio/sequenced_views.)
### 3d  pca plots
-  For seeing if there's batch effects 
   -  results/mcolon/20260605_sci_cilia_qc_first_pass/make_3d_pca_sci.py and results/mcolon/20260605_sci_cilia_qc_first_pass/make_3d_pca.py
### sequenced vs get through pipleine plots
### Label Transfer plots
note that for palte01 for cep290 and b9d2 we will use these we wont use redundant "_t01" and "_t02" labels for the time points as we should use the time series for these embryos 
#### genotype predication q plots 
- results/mcolon/20260607_sci_cilia_gene14_imaging_qc/plots/genotype_qc/b9d2_sequenced_accuracy_heatmap.png this class of plots prettymuich does it well!

#### label transfer confidence plots 
Each column is a collection time and support time
- , this is relevant because 48hpf collection has plate02 single snapshots (support for only at  48hpf)  and plate01 we did capture snapshots but we should use the time series data for the label transfer, so we should have support across all time bins (30 to 48)
- thus for us there will be 5 cols for the cep290 and b9d2 plots and 4 for the cilia crispant plots 
##### wahts in each row 
- row 1 a model prediction (based on argmax) bar plot 
- row 2 model query sequenced prediction probababilities (too se level of confidence)
- row 3 model rery prediction probababilities (strip based true classes on y) (visually see prdistributiuon of predction )
- row 4 reference or each class/time bin Precision and Recall  (summnarize into key metrics)
- row 5 rereference model confusion matrit (confusion matric to see label prediction imbalance (were the mass is going) )

this is the key plot Defer getting this plot to work for multi-class for now. For example, when we're doing the late-onset type prediction, we might need to think about how we do that. 

#### Time series label transfer plots 
We need a plot to help us actually see physically that our model predictions make sense with a known biological reality. However, we also need to do it in a way that respects that some points have time, that are references of time series, but we have a mixture of snapshots and time series data. 
- So first we're going to do a plot features plot from 24 to 48 hpf, and we're going to make the reference points that we used to train the model on. I'm going to make them have a low alpha. 
- Then, for the time series data, we're going to give them a high alpha line, and they're going to end with a circle colored by their predicted class. 
- Then for the snapshots, we're just going to do a square for the time snapshot. 
- Note that two of the later time points have snapshots.    
  - so 30 hpf and 48hpf has snopshots 
this way we can see if predictions actually line up with the expected phenotypes. 

# Label Transfer Improvement 
- The issue with our current label transfer is that we need it in order to work by per time bin. It aggregates embryo information and to generate class label and probabilities. Specifically, for a given time bin, aggregate images and predict per-embryo information. This is physical aggregation, so it should only be one data point per embryo within a time bin. This prevents data leakage. Then we also needed to aggregate support or classification decisions across time bins, and this would be descriptive statistical support. 

- Fundamentally, we just need to know how confident we should be in our predictions and what the actual predictions are(hence why we return reference model). Also note that this should work for the multi-class approach as well. 
  - note that we just dont fit a model if we dont have enough support points for a given class in a time bin
  - TRIPPLE check we are using balanced models as we keep seing model collapse towards one label or another 
## Goal

Run label transfer while respecting embryo and time-bin structure.:
- inputs are embryo_ids , feature for each embryo_id and time_bin, and class labels for each embryo_id 
  - first you first you fit a reference (making sure to save reference model CV probabilities for later plotting and QC) generating per embryo predictiosn within a time bin and aggregating across time bins., also save final reference model on ALL data 
  - then generate per-embryo classification for the querry (the model fit on ALL data of course ), simialrly save within a time bin and aggregating across time bins (same data strtcutrue as before )
- outputs should be per time bin predictions 
The method should be feature-agnostic.


---

# LOCKED DECISIONS (interview, 2026-06-09)

Everything above is the original intent. This section is the binding spec agreed in
the planning interview. Where the two disagree, **this section wins**. Each decision
records the *why* so a future reader does not re-litigate it.

## ⚠️ LOUD RULE — plate01 redundancy (cep290 & b9d2)

> **plate01 has BOTH the `_sci` timeseries AND redundant `_t01`/`_t02` snapshots for the
> same embryos. The snapshots were made only as backups.**
>
> - **Label transfer uses the TIMESERIES ONLY for plate01.**
> - **`_t01`/`_t02` snapshots are for portfolio / QC spot-checking ONLY — never the model.**
> - **For label PREDICTION, plate01 uses only the `_t02` (48 hpf) snapshot, never `_t01`** —
>   so plate01's snapshot handling matches plate02 (a single 48 hpf snapshot). Some plates
>   have more than one snapshot; pulling only the 48 hpf one is the clean, uniform rule.
>
> This must be stated at the top of the relevant scripts AND in their print statements.
> Document every such decision explicitly.

## Features

- **`feature_cols` = the 80 `z_mu_b_*` columns** (`z_mu_b_20` … `z_mu_b_99`) — the
  **biological** embedding. Confirmed 80 columns in the reference tables.
- **EXCLUDE `z_mu_n_*` (nuisance latent, 20 dims).** Using bare `z_mu*` would drag the
  nuisance dims into the model — wrong. Select via `c.startswith("z_mu_b_")`.
- Reference and query must expose the **same `z_mu_b_*` set in the same order**.

## Metadata — `collection_time_hpf` (added in `0_load_and_clean_datasets.py`)

- New **per-row** column derived by a dedicated helper. Plates are not one-to-one: a
  single plate can carry multiple time points and multiple genotypes, so the mapping is
  per-row (well/embryo aware), not just per-experiment-name.
- Rule: collection time = the plate's **start-age** from the experiment name, **EXCEPT
  `30to48` plates (caps in name) → `collection_time_hpf = 48`** (one collection spanning
  that window).
- `predicted_stage_hpf` stays the per-frame **true age** for binning / x-axes. Collection
  time carries the disambiguation that is genuinely ambiguous for timeseries data.
- Parse spec must handle messy names explicitly and **surface ambiguous names rather than
  guessing silently**, e.g. `20260324_cep290_18hpf_24hpf_plate02` (two ages in one name)
  and `20260415_b9d2_30to48hpf_plate01_t02`.

## `bin_width = 4.0` hpf

- Kept at 4 hpf. Collection times (18/24/30/48) do not land in surprising bins at this width.

## Label-transfer engine — per-bin, two-layer (NEW MODULE)

**Everything is per-bin.** There is no global-model path in the new engine.

- **Location:** a **new module** `src/analyze/classification/label_transfer/perbin.py`.
  Do **not** edit `core.py` in place — existing callers (`cep290_homo_low_to_high.py`,
  etc.) depend on its current return schema and would break. The new module reuses the
  shared `_aggregate_binned`, `add_time_bins`, and `_make_pipeline` (no duplication) but
  owns the new contract. `core.py` stays untouched; migrate callers later, deliberately.
- **Substrate:** per-`(embryo, time_bin)` rows via `_aggregate_binned` — one point per
  embryo per bin (prevents leakage; an embryo spanning many bins contributes to each).
- **CV:** **leave-one-experiment-out (LOEO-experiment)**, default `cv_group_col =
  experiment_id`. Rationale: no embryo is shared across experiments, so LOEO-experiment
  *is* leave-one-embryo-out **and stricter** (no batch in both train & test). It is the
  honest analog of transfer itself — the query is always a new experiment.
- **Insufficient support:** a bin with **≤1 experiment** for the CV cannot be evaluated →
  **FAIL that bin (do not fake/fallback). Print the failure when `verbose=True`.** A query
  `(embryo, bin)` with no model gets **no prediction** (reported as missing support) and
  does not contribute to that embryo's cross-bin aggregate — never a silent pooled fallback.
- **Balanced models:** `class_weight="balanced"` everywhere; collapse-to-one-label is
  verified by **reading the per-bin CV** (a collapsed bin shows recall≈1 / ≈0), not a
  separate detector.

### Two reported layers (coexist — neither replaces the other)

1. **Per-bin layer:** per-`(embryo, bin)` predictions; per-bin precision/recall/confusion;
   **per-bin support** (n_embryos, n_experiments) so each per-bin number is interpretable.
2. **`cross_bin_summary`:** aggregates the per-bin layer. Names prefixed **`cross_bin_*`**:
   - `cross_bin_pred` — per-embryo aggregate = **mean of the embryo's per-bin probability
     vectors over ALL bins it appears in → argmax**.
   - `cross_bin_precision` / `cross_bin_recall` / `cross_bin_confusion` — **score per bin,
     then average across bins (macro over bins)**. Rationale: let class imbalance surface
     at the **time-bin** level, not be hidden by per-embryo abundance; the bin is the unit
     of measurement, so it is also the unit of averaging.
   - The `skip` / `warn` transferability gate derives from these macro-over-bins numbers.

- Both `prepare_reference` (reference CV) and `transfer_labels` (query) emit this same
  two-layer shape.

## Confidence plot — the KEY plot (v1)

- **v1 = homozygous-phenotype BINARY, per gene** (cep290 `High_to_Low` vs `Low_to_High`;
  b9d2 `CE` vs `HTA`). Multi-class deferred.
- **Why homozygous-only:** genotype is *not* clean — some b9d2 heterozygotes show
  homozygous-like phenotypes — so the key plot restricts to homozygous embryos, which are
  the cohort we actually plan to sequence.
- **Columns = collection × support.** cep290/b9d2 → 5 columns:
  `18, 24, 30, 48-snapshot(plate02), 48-timeseries(plate01)`. The 48 hpf collection appears
  TWICE on purpose: plate02 = 48 hpf snapshot support; plate01 = 30→48 hpf timeseries
  support. The contrast between these two support regimes at the same collection time is
  the entire point — rows 4/5 should look **better** for the timeseries-support column.
  Crispant → 4 columns (`18, 24, 30, 48`), no split (no timeseries).
- **Rows (5):**
  1. argmax model prediction — **bar plot** (predicted-class counts/fractions).
  2. **query** sequenced prediction probabilities (level of confidence in the embryos we
     actually sequenced).
  3. **reference** prediction probabilities, **stripped with true classes on y** (see the
     distribution of predictions per true class).
  4. **reference** precision & recall per class·time-bin (key metrics).
  5. **reference** confusion matrix (where the prediction mass goes / imbalance).
  Rows 2 vs 3 are the query/reference pair: row 2 = "how confident about what we
  sequenced", row 3 = "how separable are the reference classes that justify it".

## Genotype QC plots

- Still produced, but **not** the key plot. Existing `*_sequenced_accuracy_heatmap.png`
  style already does this job well (e.g.
  `plots/genotype_qc/b9d2_sequenced_accuracy_heatmap.png`).

## Crispant

- **In scope**, as a **genotype-only multi-class** model (labels: `ab_wildtype`,
  `foxj1a_crispant`, `ift88_crispant`, `ift88_ift74_crispant`, `sspo_crispant`). Rides the
  same per-bin engine. The multi-class confidence plot follows once the binary v1 works.

## Time-series label-transfer plot

- Feature plot 24→48 hpf. Reference training points = **low alpha**. Timeseries = **high
  alpha line ending in a circle colored by predicted class**. Snapshots = **square** (30 &
  48 hpf have snapshots). Lets us see whether predictions line up with biological reality.


# Resolved ambiguities (2026-06-09)

Additive clarifications agreed after the interview. These record what the prose above left
implicit so a future reader does not have to re-derive it from the data. They refine, they
do not override, the LOCKED DECISIONS.

- **Which `experiment_id` is the plate01 timeseries.** It is the `_sci_..._plate01`
  experiment — `20260414_sci_b9d2_48hpf_plate01` (4437 rows, stage 30→48 hpf) and
  `20260415_sci_cep290_48hpf_plate01` (3337 rows). Despite the `48hpf` in the name, these
  ARE the long timeseries. The `..._30to48hpf_plate01_t01/_t02` and `plate02_t01/_t02`
  experiments are the redundant ~40–90-row **48 hpf snapshots**. (The LOUD rule said
  "plate01 timeseries" without naming the string; this is the string.)

- **CV group column = `experiment_id`** (build06-native; identical values to the
  `experiment` column written by `0_load_and_clean_datasets.py`). Matches what
  `1_fit_reference_models.py` already passes.

- **plate01 redundancy is kept, not dropped.** The timeseries and snapshot rows already
  carry **distinct `embryo_id`s** (the id embeds the experiment_id, e.g.
  `...sci...A01_e01` ≠ `...30to48..._t02_A01_e01`). Two new columns disambiguate them:
  - `data_source` ∈ {`timeseries`, `snapshot`}.
  - `physical_embryo_id = {gene}_{collection_time_hpf}_{plate}_{well}` — keyed on
    **biological collection time, NOT `experiment_id`** — so the plate01 timeseries row and
    its `_t02` 48 hpf snapshot row for one physical embryo join cleanly for portfolio/QC,
    across all cohorts. `experiment_id` stays the analysis/source id and is kept OUT of
    `physical_embryo_id`; analysis rows stay distinct via `experiment_id` + `data_source`.
  - Reference/label-transfer uses `data_source == "timeseries"` for plate01; prediction
    uses the `_t02` (48 hpf) snapshot.

- **Ambiguous experiment names are surfaced, never silently guessed** (e.g.
  `20260324_cep290_18hpf_24hpf_plate02`, two ages in one name): the collection-time parser
  warns with the offending name and documents the chosen age.

- **Label-transfer engine lives in a NEW module** `label_transfer/perbin.py` (sibling of
  `core.py`, which is untouched). Two-step API mirrors core: `prepare_reference_perbin`
  (fit + reference performance) → read quality → `transfer_labels_perbin` (apply to query).
  Return contract is the two-layer per-bin shape (`per_bin` / `embryo_support` /
  `reference_performance` / `missing_support`).


# Code we were went looking around 
results/mcolon/20260605_sci_cilia_qc_first_pass 
 

# PLOT PLAN

A cold-start brief for an agent building the plotting layer. The engine + data layer
(scripts 0/1/2) are DONE and verified. This section is the contract for the plots that
sit on top. Build them as **separate scripts `3a`…`3f`** in this directory (retire the
current monolith `3_plot_qc_and_phenotype_predictions.py` once `3a/3b` cover its job).

## Environment & conventions
- Run: `conda run -n segmentation_grounded_sam --no-capture-output python <script>`.
  Never bare `python` / `conda activate`.
- Each script is standalone: `RUN_DIR = Path(__file__).resolve().parent`, then
  `sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT/"src"))`,
  and `sys.path.insert(0, str(RUN_DIR))` to import `cilia_qc_helpers`.
- Save PNGs under `plots/<family>/` (e.g. `plots/genotype_qc/`, `plots/confidence/`),
  `dpi=150, bbox_inches="tight"`. Mirror the existing save/print pattern.
- **Sequenced-only**: query plots read the `predictions/` tables, which are already
  sequenced-only (script 2 filters once, up front).
- **Genotype Excel is ground truth** for true labels (zygosity / genotype_clean). Do not
  reconcile against sequencing codes.

## Shared facts the plots depend on

**Features**: 80 `z_mu_b_*` only (biological latents). Plots don't refit; they read the
saved predictions. (Engine excludes `z_mu_n_*` nuisance.)

**The 5 columns (cep290 & b9d2) = collection × support**, driven by two per-row columns
on every query table (`collection_time_hpf`, `data_source`):
`18, 24, 30, 48-snapshot(plate02), 48-timeseries(plate01)`. The 48 hpf collection appears
TWICE on purpose — plate02 = single 48 hpf snapshot; plate01 = 30→48 timeseries. Rows 4/5
of the confidence plot should look BETTER for the timeseries column. Crispant → 4 columns
(`18, 24, 30, 48`), no split (no timeseries).
- Split by: `data_source == "timeseries"` (the `_sci_` plate01) vs `"snapshot"`; and
  `collection_time_hpf ∈ {14,18,24,30,48}`.
- LOUD plate01 rule already applied in script 2 via `select_for_label_transfer` (timeseries
  wins per `physical_embryo_id`); the redundant `_t02` backup is dropped from prediction.

## Input tables (all under this run dir, produced by scripts 1 & 2)

Reference CV (one row per **reference embryo × bin**) — `models/<model_id>_reference_cv.csv`:
`embryo_id, time_bin, time_bin_center, cv_group, true_label, predicted_label, prob_<CLASS>…`
Models: `b9d2_homozygous_phenotype`, `cep290_homozygous_phenotype`,
`b9d2_genotype_qc`, `cep290_genotype_qc`, `cilia_crispant_genotype_qc`.

Full `ref_model` pickle — `models/<model_id>.pkl` (the per-bin contract): keys
`reference_performance.per_bin_metrics` (df: `time_bin_center, class, precision, recall,
n_embryos, n_experiments`), `reference_performance.per_bin_confusion` (`{bin: {matrix,
labels}}`), `reference_performance.transferability` (`{class: ok/warn/skip}`),
`missing_bins` (df of bins that failed CV), `embryo_support.n_bins_scored`,
`config.cv_mode` (`loeo`|`kfold`). Read these for rows 4/5 rather than recomputing.

Query per-bin (one row per **query embryo × bin**) — split by kind:
`predictions/sequenced_homozygous_phenotype_per_bin.csv`,
`predictions/sequenced_genotype_qc_per_bin.csv`. Non-latent cols:
`model_id, prediction_kind, query_embryo_id, time_bin, time_bin_center, predicted_label,
prob_<CLASS>…, embryo_id, experiment, gene, well, sequenced, sequenced_stratum,
genotype_clean, zygosity, phenotype_clean, predicted_stage_hpf, collection_time_hpf,
data_source, physical_embryo_id`.

Query cross-bin (one row per **query embryo**) — `…_cross_bin.csv`: adds
`top_probability, argmax_margin, n_bins_contributed, bins_contributed`.

Missing support — `predictions/sequenced_missing_support.csv`:
`model_id, query_embryo_id, time_bin, time_bin_center` (query bins with no model).

> CLEANUP: `predictions/sequenced_embryo_predictions.csv`, `*_image_predictions.csv`, and
> `*_predictions.csv` are STALE outputs from the old core-engine script. The new script 2
> writes `*_cross_bin.csv` / `*_per_bin.csv`. Delete the stale files when convenient; do
> not build plots against them.

## Class conventions (binary v1)
- b9d2 phenotype: `CE` vs `HTA` (homozygous only). cep290 phenotype: `High_to_Low` vs
  `Low_to_High` (homozygous only). Confidence plot v1 is **homozygous-only binary** because
  genotype isn't clean (some b9d2 hets look homozygous); multi-class deferred.
- Genotype QC labels: `wildtype/heterozygous/homozygous` (b9d2/cep290),
  `ab_wildtype/foxj1a_crispant/ift88_crispant/ift88_ift74_crispant/sspo_crispant` (crispant).
- Colors + constants: **`plot_config.py`** in this dir (shared so all 3x plots match the
  earlier published figures). Holds `PHENOTYPE_COLORS` (CE=`#1b9e77` green, HTA=`#d95f02`
  orange, High_to_Low=`#E76FA2`, Low_to_High=`#2FB7B0`), `GENOTYPE_COLORS` (wt=`#2166AC`,
  het=`#F7B267`, homo=`#B2182B`, + crispants), `STATUS_COLORS`, `DESIGN_STAGES_HPF`,
  `PHENOTYPE_COLUMNS` / `CRISPANT_COLUMNS` (the collection×support column spec), and a
  `color_for(label, kind)` lookup. Import it; don't redefine colors per script. Keep it
  lean — add shared *functions* only when a 2nd script repeats them.

## The scripts (build order = QC → decision → deep-dive)

### 3a — sequenced-vs-pipeline coverage audit
"Excel says sequenced vs what actually made it through the pipeline." Port + retarget
`results/mcolon/20260605_sci_cilia_qc_first_pass/audit_sequenced_coverage.py` (produced
`MISSING_SEQUENCED_AUDIT.md`). For each non-`sci_` plate it reads the `sequenced` Excel
sheet and the build04 `qc_staged_<exp>.csv` and classifies every sequenced well: `OK` /
`QC_EXCLUDED` (with flags) / `ABSENT` / `NO_BUILD04`. Update the `EXPS` list to this
cohort; emit a markdown audit (`MISSING_SEQUENCED_AUDIT.md`) + a CSV. This is data-sanity,
not a matplotlib figure (a table/heatmap of coverage per plate is the deliverable).

### 3b — genotype QC per plate (accuracy heatmap + confusion)
The `*_sequenced_accuracy_heatmap.png` style already does this well (keep it). Reads
`predictions/sequenced_genotype_qc_cross_bin.csv` (one row per embryo) + truth
(`zygosity` / `genotype_clean`). Per gene: accuracy heatmap (plate × design-stage, snap
`collection_time_hpf`), confusion matrix, accuracy-by-class. Reuse the existing functions
in the monolith `3_plot_qc_and_phenotype_predictions.py`: `plot_accuracy_heatmap`,
`plot_genotype_qc`, `snap_to_design_stage`. NOT the key plot.

### 3c — confidence plot (THE KEY GREENLIGHT ARTIFACT)
v1 = homozygous-phenotype binary, per gene. **Columns = collection × support** (5 for
cep290/b9d2, 4 crispant — see "5 columns" above). **Rows (5):**
1. argmax model prediction — **bar plot** (predicted-class counts/fractions) from the query
   per-bin / cross-bin predictions.
2. **query** sequenced prediction probabilities (confidence in the embryos we sequenced) —
   from `…_homozygous_phenotype_per_bin.csv` `prob_<CLASS>`.
3. **reference** prediction probabilities, **stripped with true classes on y** — from
   `models/<model_id>_reference_cv.csv` (`true_label`, `prob_<CLASS>`).
4. **reference** precision & recall per class·bin — from
   `ref_model["reference_performance"]["per_bin_metrics"]`.
5. **reference** confusion matrix — from
   `ref_model["reference_performance"]["per_bin_confusion"]`.
Rows 2 vs 3 = the query/reference pair. Group each column by `collection_time_hpf` +
`data_source`. Reference material: `plot_phenotype_spectrum_with_ref` (3-row version) in
the monolith — this is its 5-row, multi-column generalization. Multi-class deferred.

### 3d — feature plot (24→48 hpf)
Physical-reality check. Reference TRAINING points = **low alpha**. Timeseries embryos =
**high-alpha line ending in a circle colored by predicted class**. Snapshots = **square**
(30 & 48 hpf have snapshots). Uses query per-bin (`predicted_stage_hpf` x a feature/score,
colored by `predicted_label`) + the reference CV as the low-alpha backdrop. Distinguish
timeseries vs snapshot via `data_source`. Lets us see predictions line up with biology.

### 3e — 3D PCA (batch-effect check)  [DEFER / REUSE]
Existing working scripts: `…/20260605_sci_cilia_qc_first_pass/make_3d_pca_sci.py` and
`make_3d_pca.py` (interactive plotly HTML; dropdowns for hpf/genotype/experiment/source/
sequenced). Reuse as-is or retarget to this cohort's tables later. Not MVP.

### 3f — portfolio (per-embryo image grid)  [DEFER / REUSE]
End goal: per-embryo image grid with seq genotype / predicted genotype / predicted
phenotype / QC-exclusion status above each photo. Existing:
`…/20260605_sci_cilia_qc_first_pass/make_embryo_portfolio.py` +
`make_sequenced_portfolio_views.py`. Snapshots feed the portfolio (you can't put a
timeseries in a contact sheet); join timeseries↔snapshot rows by `physical_embryo_id`.
Not MVP.

## Status snapshot (what's done vs to-build)
- DONE: `0_load_and_clean_datasets.py` (collection_time_hpf, data_source,
  physical_embryo_id), `src/analyze/classification/label_transfer/perbin.py` (engine),
  `1_fit_reference_models.py` (all 5 models per-bin; loeo for b9d2/cep290, kfold for
  crispant), `2_predict_sequenced_embryos.py` (sequenced-only, timeseries-priority dedup,
  cross-bin + per-bin outputs), `cilia_qc_helpers.py` (`select_for_label_transfer`),
  `plot_config.py` (shared colors + column spec).
- TO BUILD: `3a` (audit), `3b` (genotype per plate), `3c` (confidence — KEY), `3d`
  (feature). REUSE/DEFER: `3e` (PCA), `3f` (portfolio).
- Current `3_plot_qc_and_phenotype_predictions.py` is the MONOLITH to mine for reusable
  functions, then retire.
