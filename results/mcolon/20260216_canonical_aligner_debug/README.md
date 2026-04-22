# Canonical aligner debug workspace

This directory is an experiment-local validation area for canonical alignment.
It is not production code.

## What it contains

- alignment plans and implementation notes
- one-off debug scripts for specific embryos and failure modes
- generated plots and CSVs in `debug_results/`

## Current interpretation

The current aligner investigation is about cleanly separating:

1. single-embryo canonical alignment
2. optional src→tgt registration
3. downstream OT consumption of already-canonical masks

The most relevant plan notes are:

- [PLAN_two_stage_canonical_alignment.md](PLAN_two_stage_canonical_alignment.md)
- [PLAN_simplify_canonical_aligner_yolk_orientation.md](PLAN_simplify_canonical_aligner_yolk_orientation.md)

## Suggested entry points

- [debug_coarse_scoring_fix.py](debug_coarse_scoring_fix.py) — validation of the stage split
- [debug_alignment_pairs.py](debug_alignment_pairs.py) — paired canonical overlays
- [debug_alignment_by_embryo_id.py](debug_alignment_by_embryo_id.py) — per-embryo inspection
- [debug_yolk_pivot_rotation.py](debug_yolk_pivot_rotation.py) — yolk-pivot rotation checks

## Status

The debug folder should be read as a record of the refactor path and failure
analysis, not as the source of truth for the production API.
