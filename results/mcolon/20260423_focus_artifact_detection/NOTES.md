# Focus Artifact Detection — Notes

Date: 2026-04-23

## Goal

Use entropy-based signal quality inside the embryo mask to find a stable focus threshold.

## Primary score

- `rel_entropy_mean`

This is the stack-level summary we want to rank by.

## Contract

The focus workflow should preserve the same shape as the motion workflow:

- `t`, `p`, `well`, `label`, `color` in the ranked-figure example dicts
- per-stack CSV summaries
- decile sampling for exemplar selection
- reusable ranked figure with metric bars

## Intended usage

- lower `rel_entropy_mean` should correspond to worse focus
- ranked examples should expose good, bad, and borderline cases
- the figure should make threshold selection visually obvious

## Implementation note

The motion QC module already exposes the needed entropy summary helper:

- `entropy_stack_summary`
- `rel_entropy_summary`

The focus folder should build on those helpers instead of re-implementing them.
