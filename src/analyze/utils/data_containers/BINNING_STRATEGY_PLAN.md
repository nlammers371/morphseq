# Binning Strategy Plan

## Decision

Move binning out of the object-level and out of downstream classification code.

Instead of letting `BinObject` or classification functions decide how to bin raw rows, the workflow should be:

1. bin once in an explicit preprocessing step
2. store or hand off a prepared, bin-aware DataFrame
3. let classification consume that prepared table directly

## Why this is the right move

### 1) Removes hidden behavior

If a function bins internally, the caller cannot easily see:
- which bin width was used
- whether the binning rule changed
- whether multiple calls used different resolutions
- whether classification re-binned data differently from feature construction

That makes the pipeline harder to audit.

### 2) Keeps one source of truth

Binning is a data contract, not an implementation detail.

If raw rows are binned once and stored explicitly, then:
- every downstream function sees the same grain
- feature creation and classification operate on the same table
- bin resolution is not recomputed implicitly in multiple places

### 3) Makes multiple resolutions possible

The current object model is simpler if it does not own binning logic.

That allows us to support:
- `bw2`
- `bw4`
- other future resolutions

as separate prepared tables, rather than having one object silently choose a resolution at runtime.

### 4) Simplifies function responsibilities

A function that bins internally is doing two jobs:
- transforming the grain
- analyzing the features

Splitting those responsibilities makes the code easier to test and review.

The preferred split is:
- preprocessing owns binning
- `BinObject` owns storage, grain checks, and feature attachment
- classification owns model fitting only

### 5) Reduces the chance of subtle mismatch bugs

Internal binning makes it easier to accidentally compare:
- one feature set binned at one width
- another feature set binned at a different width
- classification features that were re-binned independently from the container

Explicit prepared inputs avoid that class of bug.

## What changes in practice

### Before

- raw data enters a function
- the function bins internally
- the function computes features
- the function passes the result to classification

### After

- raw data is binned explicitly once
- the binned table is stored or passed forward
- feature computation attaches to the correct grain
- classification consumes the prepared DataFrame

## Where this pattern already appears

These are concrete examples of the current mixed model:

- [src/analyze/difference_detection/penetrance_threshold.py](src/analyze/difference_detection/penetrance_threshold.py) imports `add_time_bins` from `src/analyze/utils/binning.py` and bins observation-level data before penetrance aggregation.
- [src/analyze/classification/engine/data_prep.py](src/analyze/classification/engine/data_prep.py) owns `_bin_and_aggregate(...)`, which floors time into bins and then groups by `(id, bin, label)`.
- [src/analyze/classification/run_classification.py](src/analyze/classification/run_classification.py) uses the shared `_bin_and_aggregate(...)` path for binary runs and reimplements the same floor/bin step in the multiclass fast path.
- [src/analyze/difference_detection/classification_test.py](src/analyze/difference_detection/classification_test.py) and [src/analyze/classification/classification_test.py](src/analyze/classification/classification_test.py) also perform their own time-bin construction directly with `np.floor(...)`.

Taken together, these show the exact problem this plan is addressing: binning is spread across multiple analysis functions, so the resolution becomes an implementation detail instead of an explicit preprocessing contract.

## Expected benefits

- easier auditing
- less duplicated logic
- fewer hidden assumptions
- cleaner support for multiple bin widths
- simpler classification API
- more reproducible results

## Contract we want

- binning happens explicitly, not implicitly
- classification does not own binning
- `BinObject` does not need to infer or re-create bin grain on demand
- feature attachment must match the existing grain exactly
- downstream consumers receive a plain pandas DataFrame with known resolution

## Summary

This is a simplification move.

It trades a little convenience for a lot of clarity:
- fewer hidden steps
- fewer places where bin width can drift
- easier auditing
- cleaner data flow from preprocessing to classification

That makes the pipeline easier to trust and easier to extend later.
