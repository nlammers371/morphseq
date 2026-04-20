MorphSeq Cross-Bin Architecture (Consolidated Spec v3 for Audit)
===============================================================

Purpose / Why we are doing this
-------------------------------
MorphSeq analyses historically loop over time bins and run per-time-bin computations on raw per-frame
samples. This is convenient and often works when phenotypes are phase-aligned and embryos share similar
time support.

However, many phenotypes are fundamentally stochastic in time:
  - signals can rise, fall, then rise again
  - peak/trough timing (phase) varies across embryos
  - signal may appear in different bins for different embryos

In this regime, per-time-bin analyses can become phase-sensitive and unstable. In addition, embryos often
have unequal time support (some cover only early bins, others late). Comparing embryos without enforcing
shared time support can induce spurious class separation driven by baseline developmental differences.

We therefore introduce a cross-bin architecture that:
  - keeps the default per-bin workflow ergonomic (BinObject exposes binned features directly)
  - makes cross-bin/trajectory-level comparisons explicit, deterministic, and auditable
  - eliminates complex “policy objects” in favor of physical dials with clear math
  - uses strict, typed “data levels” plus declarative input handles (InputRef) to route data
  - explicitly defines how features are added and how binned keys are produced

Summary of the core idea
------------------------
1) BinObject is created from raw samples and defines time bins.
2) BinObject exposes a binned level (per embryo × bin) using within-bin reducers (defaults).
3) Cross-bin reductions are performed by an engine that:
     - selects bins in scope via time_window
     - filters embryos by a deterministic required_bins rule (bin_fract/min_bins + reducer floor)
     - fetches all required inputs by reading reducer "consumes" manifests (InputRefs)
     - computes meta features and returns an audit SupportReport

No abstract SupportPolicy objects are required for the user.

Key principles
--------------
A) Deterministic missingness filtering via physical dials
   - No boolean allow_missing flags
   - No user-instantiated “support policy” objects
   - Missingness behavior is fully governed by bin_fract + min_bins + reducer.math_min_bins

B) Strict separation of data by temporal level (“levels”)
   - Prevents naming collisions and mathematical errors
   - Enables automatic routing of inputs without boilerplate

C) Reducers must declare exactly what they consume (InputRef)
   - “Smart defaults” are only allowed when grounded in declared consumption
   - Missing required inputs must fail fast with explicit errors

D) Cross-bin reductions operate on BINNED features as targets
   - cross_bin_reduce targets binned level keys (e.g., "bin__vae__radius")
   - It does NOT target raw feature families directly
   - Within-bin reduction choices are encoded in the binned column names

E) Reducers do NOT know bin size
   - Binning is defined by BinObject
   - Reducers may log [t_min, t_max] for provenance but must not infer bin width
   - Any weighting based on bin duration or frame count is provided via bin_meta inputs

F) Target semantics are pinned down (no ambiguity)
   - InputRef(level="binned", key="target") resolves to exactly ONE binned feature key.
   - Multi-feature reducers must declare multiple explicit InputRefs and may not use "target".

Data levels (“levels”)
----------------------
The engine separates all inputs into four levels:

1) binned
   - Per-(embryo_id, bin_id) aggregated biological measurements
   - Examples:
       bin__vae__radius
       bin__curvature__mean
       bin__vae__mean_latent_003

2) raw
   - Per-frame underlying recordings
   - Examples:
       timestamp
       x_centroid, y_centroid
       qc_pass_flag

3) bin_meta
   - Per-(embryo_id, bin_id) metadata about bin composition/weights
   - Examples:
       n_frames          (how many frames survived into this bin)
       bin_width_seconds (duration of the bin in seconds)
       bin_center_time   (bin center timestamp/hpf)

4) embryo_meta
   - Per-embryo static identifiers and covariates
   - Examples:
       genotype
       batch_id
       temperature

Cross-bin outputs as a first-class level (add to Spec v3)
========================================================

New level: cross_bin
--------------------
We introduce a fifth level to store cross-bin reduction outputs:

5) cross_bin
   - Grain: (embryo_id)
   - Contents: embryo-level meta features produced by cross_bin_reduce / cross_bin_reduce_batch
   - Rationale:
       * cross-bin outputs are derived (depend on time_window, bin_fract/min_bins, reducer identity)
       * they must not be mixed into embryo_meta (which is reserved for static covariates)
       * they must not be mixed into binned/bin_meta (which are (embryo_id, bin_id) grain)

Naming convention for cross-bin keys
------------------------------------
Cross-bin keys MUST be prefixed with:
  xbin__

Cross-bin keys MUST NOT include the "bin__" prefix inside the name.

Canonical format:
  xbin__{feature_id}__{cross_reducer_id}__{tmin}_{tmax}

Where:
  - feature_id is the binned feature identifier without the "bin__" prefix
    (and may include the within-bin reducer identifier when relevant),
    e.g.:
      vae__radius
      curvature__mean
      vae__velocity_energy
  - cross_reducer_id is the cross-bin reducer identity (e.g., max, top2, auc, mean_equal_bin)
  - tmin/tmax are the requested time_window bounds used to select bins (for human readability)

Examples:
  - xbin__vae__radius__max__30_70
  - xbin__curvature__mean__top2__30_70
  - xbin__vae__velocity_energy__auc__30_70

Note on uniqueness:
  - If multiple results with different bin_fract/min_bins must coexist simultaneously in the same
    cross_bin dataframe, append optional tags, e.g.:
      __bf0p8  (bin_fract=0.8)
      __mb5    (min_bins=5)
    Otherwise, bin_fract/min_bins are stored in provenance only.

Storage contract: CrossBinResult record
---------------------------------------
Each cross-bin computation produces a CrossBinResult record:
  - df: DataFrame at grain (embryo_id) with one or more xbin__... columns
  - support_report: SupportReport (mandatory)
  - record/provenance (mandatory):
      * reducer name/version
      * input binned key(s) actually consumed (e.g., source_key="bin__vae__radius")
      * time_window, selected bin_ids, bins_in_scope, required_bins
      * bin_fract, min_bins, math_min_bins
      * confounding_threshold and any emitted ConfoundingRiskWarning
      * timestamp/run_id (optional but recommended)

Levels introspection updates
----------------------------
bo.levels.inspect() must include cross_bin and print:
  - grain (embryo_id)
  - available xbin__... keys
  - count of stored CrossBinResult records (or run_ids) and how to retrieve them

End.

Naming and routing invariants:
  - A key may exist in only one level namespace to avoid ambiguity.
  - Reducers specify which level a key comes from via InputRef(level=..., key=...).

Levels introspection (debugging / audit ergonomics):
  - BinObject must expose bo.levels.inspect() which prints, for each level:
      - grain (index columns; e.g., (embryo_id, bin_id))
      - available keys
      - row counts / coverage stats
  - Registration of a key that already exists in a level is an error unless overwrite=True.

Bin selection rule (deterministic)
----------------------------------
The BinObject must define a deterministic mapping from time_window to bins_in_scope.
Recommended convention:
  - A bin is in scope if its bin_center_time lies within [t_min, t_max].
This avoids edge effects from partial overlaps.

Support policy (“missingness dials”)
------------------------------------
Cross-bin operations must avoid skew due to unequal time support. Rather than policy objects, the engine
uses a single deterministic required_bins rule.

Inputs:
  - time_window = [t_min, t_max] (requested cross-bin window)
  - bin_fract: float in [0,1] (fractional coverage requirement; default 1.0)
  - min_bins: int | None (user-provided absolute floor; default None)
  - reducer.math_min_bins: int (reducer-provided mathematical floor; required)

Definitions:
  - bins_in_scope: number of bin_ids selected by time_window
  - required_bins = max(
        reducer.math_min_bins,
        ceil(bin_fract * bins_in_scope),
        (min_bins or 0)
    )

Filtering rule:
  - For each embryo, count bins_present within the selected bins.
  - Keep embryo iff bins_present >= required_bins.
  - If required_bins > bins_in_scope: raise a clear error.

Default behavior:
  - bin_fract = 1.0 implies strict intersection across the window with no new concept.
  - min_bins defaults to None so the reducer floor governs unless the user explicitly sets a floor.

Dial effectiveness warning:
  - If the user sets bin_fract < 1.0 but required_bins == bins_in_scope anyway,
    emit a warning that the relaxation had no effect (due to reducer.math_min_bins and/or min_bins).

Handles: InputRef and ReducerSpec (“what a reducer eats”)
---------------------------------------------------------
To keep the public API minimal, reducers declare their required inputs.

InputRef
  - A reference to an input on a specific level.
  - Fields:
      level: one of {"binned","raw","bin_meta","embryo_meta"}
      key: string identifier of the input

Special placeholder key:
  - key="target" is a reserved alias bound at runtime to the user-provided binned feature key.
  - "target" MUST be a single binned key.
  - Multi-feature reducers must not use "target"; they must declare multiple explicit binned keys.

ReducerSpec (for cross-bin reducers)
  - name: string
  - consumes: list[InputRef]               (explicit declaration; mandatory)
  - output_schema: list[str]               (deterministic output columns; mandatory)
  - math_min_bins: int                     (absolute minimum bins required; mandatory)
  - notes: optional                        (human-readable description)

Weighting is encoded by reducer identity and consumption, not by public knobs:
  - If a reducer needs weights, it declares them in consumes (e.g., bin_meta:n_frames).
  - If it does not need weights, it does not declare them.
  - Different “mean” variants are different reducer names:
      mean_equal_bin: consumes only target
      mean_frame_weighted: consumes target + bin_meta:n_frames
      mean_time_weighted: consumes target + bin_meta:bin_width_seconds

This eliminates ambiguous “mean-of-means vs weighted mean” behavior from the user surface area.

Alignment and broadcasting contract across levels
-------------------------------------------------
Cross-bin reduction is performed relative to a canonical grain:

  Canonical grain = (embryo_id, bin_id) for selected bins in scope.

Input alignment rules:
  - binned inputs:
      * must be indexed by (embryo_id, bin_id)
      * joined on both keys
  - bin_meta inputs:
      * must be indexed by (embryo_id, bin_id)
      * joined on both keys
  - embryo_meta inputs:
      * indexed by (embryo_id)
      * broadcast across bin_id for embryos retained after filtering
  - raw inputs in cross-bin reducers:
      * disallowed by default for cross_bin_reduce
      * required time information must be pre-materialized into bin_meta (e.g., bin_center_time,
        bin_width_seconds, n_frames)

Rationale:
  - This keeps cross-bin reducers deterministic and prevents ambiguous raw-to-bin joins.
  - If raw-to-bin consumption is needed in the future, it must be introduced as an explicit, declared
    raw->bin aggregation step (not part of this minimal spec).

Feature adding policy (how binned keys are created)
---------------------------------------------------
Cross-bin reducers operate on binned level keys. Therefore the system must specify how binned keys are
produced from raw inputs.

Feature registration is two-part:
  A) FeatureSpec (raw extraction): declares how to obtain per-frame values from raw level.
  B) WithinBinReducer (within-bin collapse): declares how to convert per-(embryo,bin) raw samples into
     flat binned columns.

Default behavior at BinObject creation:
  - Auto-detect common feature families from raw level (configurable detectors):
      * scalar columns (e.g., curvature, length)
      * embeddings (e.g., vae_0..vae_{K-1} or vector-valued column)
  - Attach default WithinBinReducers:
      * scalar -> mean
      * embedding(time,latent) -> mean_vector
  - Materialize binned outputs into levels.binned with deterministic names:
      bin__{feature_family}__{within_bin_reducer_id}[__{axis_index_if_expanded}]

User-added features:
  - Users may register a new FeatureSpec and optionally provide a WithinBinReducer.
  - If no WithinBinReducer is provided and the feature is not a supported default kind,
    registration fails fast.

Materialization modes (initially shipped; keep minimal):
  - mode="reduced" (default): compute and store binned columns; raw access remains via levels.raw.
  - mode="lazy": raw extraction is available on-demand; binned columns remain the primary stored products.
Note:
  - Additional modes (raw/both) are intentionally deferred to avoid over-engineering.

All binned keys must be unique within levels.binned.
Collisions are errors unless overwrite=True.

Public interface (front door)
-----------------------------
Cross-bin reduction is invoked with a minimal call:

    meta_df, report = bo.cross_bin_reduce(
        features="bin__vae__radius",
        reducer="max",
        time_window=[30, 70],
        bin_fract=0.8,
        min_bins=None,
        verbose=True
    )

User provides:
  - ONE binned feature key (string)
  - reducer (name or object)
  - time_window
  - optionally adjusts missingness dials bin_fract/min_bins

Engine responsibilities:
  1) Identify bins_in_scope using bin_center_time convention
  2) Load reducer spec and compute required_bins using the deterministic rule
  3) Filter embryos by bins_present >= required_bins
  4) Resolve all reducer inputs via InputRef across levels and align them on canonical grain
  5) Execute reducer math and return meta_df (one row per embryo) + SupportReport

Audit trail: SupportReport
--------------------------
SupportReport is the mandatory audit artifact returned alongside meta_df.
It must include:

  - time_window and selected bins (bin_ids, bin_center_time range)
  - bins_in_scope and required_bins formula inputs:
      bin_fract, min_bins (or None), reducer.math_min_bins
  - number of embryos kept/dropped and reasons for dropping
  - per-class drop rates (if labels available)
  - ConfoundingRiskWarning if exclusions are class-imbalanced:
      * default threshold: max_drop_rate - min_drop_rate > confounding_threshold
      * confounding_threshold is an engine parameter (default 0.15), not hardcoded
  - warning if bin_fract relaxation had no effect (dial effectiveness warning)
  - reducer name and consumed inputs (InputRefs)

Reducer authoring ergonomics (implementation guidance)
------------------------------------------------------
To minimize boilerplate while preserving tight contracts, the system should provide:
  - a decorator/factory for simple reducers that still requires:
      consumes, output_schema, math_min_bins, name/version
  - bo.validate_reducer(reducer, feature_key, time_window=...) that dry-runs:
      input resolution via InputRef, bins_in_scope selection, output schema generation,
      collision checks, required_bins viability, and global feasibility checks:
        * if reducer.math_min_bins exceeds total bins available in this BinObject,
          validation must fail with a clear error.

Multi-feature cross-bin convenience (batch wrapper)
---------------------------------------------------
The core cross_bin_reduce API targets a single binned key, and InputRef(key="target") resolves to exactly
one binned feature. For convenience and cohort consistency, an optional wrapper may be provided:

  cross_bin_reduce_batch(features=[...], reducer=..., time_window=..., bin_fract=..., min_bins=...)

Behavior:
  - Computes bins_in_scope and required_bins once.
  - Filters embryos once using the deterministic rule, with feature-aware validity:
      * A bin counts toward bins_present only if ALL requested feature columns are non-missing
        for that embryo/bin (default behavior).
  - Applies the same reducer independently to each requested binned feature on the retained cohort.
  - Returns:
      * a single dataframe with one row per embryo and one output column per input feature
      * a single SupportReport for the shared cohort

Rationale:
  - Prevents accidental cohort drift when users apply the same reducer to many features via multiple calls.
  - Keeps reducer specs simple (single-target) while enabling efficient batch usage.

Expected usage patterns
-----------------------

A) Default per-bin analysis (ergonomic)
  - Users operate on bo.levels.binned directly (one row per embryo×bin).
  - Per-bin classification loops bins as today using binned features.

B) Cross-bin meta features (explicit and strict)
  - Users call bo.cross_bin_reduce(...) on a binned feature key to produce embryo-level meta features.
  - They adjust bin_fract/min_bins when partial coverage is acceptable.
  - They read the SupportReport to understand cohort changes and confounding risk.

C) Batch cross-bin meta features (cohort-consistent)
  - Users call bo.cross_bin_reduce_batch(...) to apply the same reducer across several binned features
    under the same window/dials, guaranteeing a single filtered cohort.

D) Extending reducers and levels (low boilerplate, fail fast)
  - New reducers must declare consumes + math_min_bins + output_schema.
  - Level registration must be unambiguous; collisions fail fast unless overwrite=True.
  - Missing consumed inputs fail fast with clear errors:
      e.g., "mean_time_weighted requires bin_meta:bin_width_seconds, but it is missing."

Tight contracts and restrictions (summary)
------------------------------------------
1) cross_bin_reduce targets a single binned feature key (string).
2) "target" InputRef resolves to exactly one binned key; multi-feature reducers must be explicit.
3) Bins are selected deterministically by bin_center_time within time_window.
4) Missingness filtering is governed only by required_bins rule (bin_fract/min_bins + reducer floor).
5) Reducers declare all consumed inputs via InputRef and fail fast if unresolved.
6) Weighting is encoded by reducer identity and consumption; no hidden weighting defaults.
7) SupportReport is always returned and warns deterministically on class-imbalanced exclusions.
8) Cross-bin reducers do not consume raw inputs by default; time/weights must come from bin_meta.
9) Feature registration must produce deterministic, unique binned keys; collisions are errors by default.

End.