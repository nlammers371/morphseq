# Identifier and Wildcard Contract

## TL;DR

The pipeline is tracking a simple hierarchy:

- experiments
- wells
- images
- channels
- embryos
- snips

The wildcards should reflect that hierarchy directly:

- `experiment_id` identifies the experiment
- `well_id` identifies the well within the experiment
- `channel_id` identifies the imaging channel
- `time_int` identifies the acquisition timepoint
- `embryo_id` identifies a tracked embryo within a well
- `snip_id` identifies one embryo at one timepoint

Use `time_int` everywhere as the canonical temporal field. Do not reintroduce `frame_index` in the active pipeline.

## How To Read It

Start from the top of the hierarchy and move down:

1. Pick an experiment.
2. Inside that experiment, pick a well.
3. Inside that well, pick a channel.
4. Inside that channel, pick a timepoint.
5. After tracking, each embryo gets a stable embryo ID.
6. After tracking and snip processing, each snip is that embryo at one timepoint.

The wildcards are just a compact encoding of that hierarchy. They are not extra metadata. They are the names we use to keep the pipeline paths, tables, and joins consistent.

## Canonical Identifiers

### Image ID

`image_id = "{experiment_id}_{well_id}_{channel_id}_t{time_int:04d}"`

Example:

- `20240418_A01_BF_t0003`

Read it as:

- experiment: `20240418`
- well: `A01`
- channel: `BF`
- timepoint: `t0003`

So this is the BF image from well A01 in experiment 20240418 at timepoint 3.

### Embryo ID

`embryo_id = "{experiment_id}_{well_id}_e{local_track_id:02d}"`

Rules:

- `local_track_id` is assigned by segmentation and tracking.
- `local_track_id` is one-based.
- `embryo_id` is stable across all `time_int` values for that tracked embryo.

Example:

- `20240418_A01_e07`

Read it as:

- experiment: `20240418`
- well: `A01`
- tracked embryo: `e07`

So this is the seventh tracked embryo in well A01 of experiment 20240418.

### Snip ID

`snip_id = "{embryo_id}_t{time_int:04d}"`

Example:

- `20240418_A01_e07_t0003`

Read it as:

- embryo: `20240418_A01_e07`
- timepoint: `t0003`

So this is embryo 07 in well A01 at timepoint 3.

## Concrete Example

If you start with:

- `experiment_id = 20240418`
- `well_id = A01`
- `channel_id = BF`
- `time_int = 3`
- `local_track_id = 7`

Then the identifiers are:

- `image_id = 20240418_A01_BF_t0003`
- `embryo_id = 20240418_A01_e07`
- `snip_id = 20240418_A01_e07_t0003`

That is the pattern we want the pipeline and Snakemake wildcards to follow.

## Wildcard Contract

Use the smallest possible wildcard set in Snakemake:

- acquisition/materialization rules: `experiment_id`, `well_id`, `channel_id`, `time_int`
- tracking rules: `experiment_id`, `well_id`, `time_int`, `embryo_id`
- feature rules: `experiment_id`, `well_id`, `time_int`, `embryo_id`, `snip_id`

## Naming Rules

- Use `time_int` in paths, tables, and wildcard names.
- Use `tNNNN` only as the formatted suffix of derived identifiers and filenames.
- Use `eNN` for embryo IDs.
- Do not invent alternate temporal names like `frame`, `frame_index`, or `T` for the active pipeline.

## Contract Summary

- `time_int` = when
- `image_id` = which image
- `embryo_id` = which tracked embryo
- `snip_id` = which embryo at which timepoint
