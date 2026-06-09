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

### Label Transfer 
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


# Code we were went looking around 
results/mcolon/20260605_sci_cilia_qc_first_pass 
 

