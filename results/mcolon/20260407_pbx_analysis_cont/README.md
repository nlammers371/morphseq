# 20260407 PBX Analysis Continuation

Canonical workflow:

1. `01_pairwise_all_pairs_positioning.py`
   Builds the combined `20251207_pbx + 20260304 + 20260306` all-pairs contrast-coordinate bundle.
2. `02_pairwise_trajectory_condensation.py`
   Runs trajectory condensation on the sparse NaN-aware pairwise vectors.

Production defaults:

- canonical class set: `inj_ctrl`, `pbx1b_crispant`, `pbx4_crispant`, `pbx1b_pbx4_crispant`
- canonical representation: `pairwise_raw_vectors.csv`
- optional experimental representation: `pairwise_shrunk_vectors.csv`
- `wik_ab` is sensitivity-only via `--include-wik-ab`

Current PBX decision:

- raw pairwise coordinates are the working default
- shrunk/weighted coordinates remain available, but are currently treated as experimental
- shrinkage has not yet been validated to preserve real-world trajectory geometry
- in the PBX comparisons here, shrunk coordinates degraded tree structure relative to raw

Age handling:

- `20260304` and `20260306`: use `predicted_stage_hpf`
- `20251207_pbx`: use `start_age_hpf + relative_time_s / 3600`

## Force-Fixing Work

This folder also became the active PBX force-diagnostics sandbox for
trajectory condensation. The goal was to fix visually obvious 60-80 hpf /
70 hpf outliers without breaking the overall manifold geometry.

Main diagnostic scripts:

1. `03_force_diagnostics.py`
   Runs audited condensation sweeps and writes summary tables, tracked-ID
   tables, and comparison figures.
2. `04_outlier_force_sensing.py`
   Single-slice sensing on `x0` or final positions to measure how strongly
   the explicit outlier force is actually activating.
3. `05_force_vector_audit.py`
   Per-force gradient decomposition at selected IDs/time bins to check solver
   balance rather than guessing from plots.

## Problem Statement

The combined PBX runs showed strong detached outliers around 62-70 hpf and
especially 70 hpf. These were most visible in:

- `inj_ctrl`
- `pbx1b_pbx4_crispant`

The key question was whether this was caused by:

- missing support in the pairwise inputs
- bad initialization
- elasticity being too weak or shaped incorrectly
- an outlier force that was not activating
- solver balance, especially repulsion overwhelming corrective forces

## What We Checked

### Input QC

Findings from the pairwise input audits:

- the flagged embryos generally had full non-NaN pairwise support at the bad
  slices, so this was not primarily a missing-support issue
- several bad embryos already had unusually large pairwise-vector norms
- some double-crispant offenders were true feature-level outliers
- some `inj_ctrl` offenders were more like coordinated moderate shifts across
  multiple pairwise dimensions

Conclusion:

- this was not fixed by simply filtering on missing coordinates
- some bad behavior is already present in the inputs / initialization

### Temporal Persistence

We checked whether the flagged embryos were only isolated slice-level issues
or persisted across adjacent bins.

Conclusion:

- several double-crispant offenders were persistent across adjacent bins
- some `inj_ctrl` offenders were more weakly persistent or isolated

This argued against blindly dropping them as obvious garbage.

## Force-Family Refactor

We refactored the public force API to make the force families explicit:

- `local_scale_strength`
- `elastic_strength`
- `elastic_mix`
- `void_strength`
- `void_bandwidth`
- `outlier_strength`

We also added force-balance reporting so runs print the resolved values and
the geometry references used to calibrate them.

Important implementation detail:

- elasticity strength is geometry-aware through `s_step` and `s_bend`
- public aliases are additive; conflicting legacy and public settings raise
  instead of blending silently

## Elasticity Work

We first tested whether elasticity alone could solve the outliers.

What we tried:

- stronger quadratic elasticity
- ratio-hinge elasticity

What happened:

- both helped somewhat
- `ratio_hinge` was modestly better than plain quadratic
- neither solved the problem

Interpretation:

- elasticity improves smoothness
- but it was not enough to pull detached points back onto the slice manifold

## Explicit Outlier Force

We added an explicit slice-relative outlier force:

- implemented in `src/analyze/trajectory_condensation/condensation/forces/slice_outlier.py`
- initially based on slice distance to centroid
- smooth softplus-quartic tail

### Initial Failure Mode

The first version used a `q99` cutoff computed from `x0`.

Single-slice sensing at 70 hpf showed:

- the worst initialized offenders were sitting almost exactly at the `q99`
  boundary they themselves helped define
- activation was extremely weak
- turning up `outlier_strength` did almost nothing

Conclusion:

- this was a cutoff-definition failure, not a strength failure

### Cutoff Comparisons

We compared:

- `q99`
- `q97`
- `q95`
- `robust3` = `median + 3 * 1.4826 * MAD`

Conclusion from sensing:

- `q99` was asleep
- `q97` woke up a little
- `q95` woke up meaningfully
- `robust3` woke up strongly

This led to real condensation tests with `robust3`.

## Solver-Balance Audit

We then stopped guessing and measured per-force gradient norms on the tracked
IDs at `x0`, iteration 0.

Reusable source utility:

- `src/analyze/trajectory_condensation/force_diagnostics.py`

PBX wrapper:

- `05_force_vector_audit.py`

Main result:

- fidelity was not the issue at iteration 0; it is exactly zero at `x0`
- repulsion was dominating the update by orders of magnitude
- the outlier force under `robust3` was awake, but still much smaller than
  repulsion under the original repulsion kernel

Conclusion:

- solver balance was real
- the early problem was not “outlier force asleep” anymore
- it was “repulsion acting like a long-range wind”

## Repulsion Kernel Change

To address the Summation Trap, we changed the default soft-core repulsion from:

- old: `epsilon_r / (r^2 + eta)`
- new: `epsilon_r / (r^4 + eta)`

This keeps short-range exclusion while making the tail decay much faster.

Implementation:

- `src/analyze/trajectory_condensation/condensation/forces/repulsion.py`

This was the first change that materially fixed the PBX outlier problem.

### Effect With `robust3`

With quartic repulsion, `robust3`, `outlier_strength=2`, and `epsilon_r=0.005`:

- focus-window `z > 10` outliers dropped to `0`
- focus-window `z > 8` outliers dropped to `0`
- the worst tracked 70 hpf offender fell from `z=21.12` to about `z=7.32`

This is currently the best-performing force configuration tested in this
folder.

## Repulsion Strength Sweep

After the quartic repulsion fix, we tested whether reducing `epsilon_r`
further would improve trunk compactness without reintroducing outliers.

Compared values:

- `0.005`
- `0.0025`
- `0.0005`
- `0.00025`

Conclusion at that stage:

- reducing `epsilon_r` below `0.005` did not help when `outlier_strength`
  was still low
- `0.0025` was slightly worse than `0.005`
- `0.0005` and `0.00025` clearly reintroduced outlier pathology under the
  weaker outlier-force settings

That was not the end state. After increasing `outlier_strength` and retuning
elasticity, the preferred PBX visual regime shifted to a smaller
`epsilon_r = 0.0005`.

## Current Working Interpretation

What worked:

- explicit slice outlier force with a robust cutoff
- quartic-decay repulsion

What did not solve it on its own:

- stronger elasticity alone
- ratio-hinge elasticity alone
- outlier-strength sweeps with the original `q99` cutoff
- lowering `epsilon_r` below `0.005` after the quartic repulsion change

## Current Selected PBX Configuration

For the PBX 4-class combined run, the currently selected configuration in this
folder is:

- quartic soft-core repulsion
- `epsilon_r = 0.0005`
- explicit slice outlier force
- `outlier_cutoff = robust3`
- `outlier_strength = 16`
- `elastic_strength = 16`
- `elastic_mix = 0.25`

Reasoning:

- `outlier_strength = 16` was the first setting that cleanly removed the
  `z > 8` focus-window outliers under the preferred lower-repulsion visual
  regime
- `elastic_mix = 0.25` reduced visible kinks relative to the balanced
  `0.5` split
- the user-selected comparison target for presentation was the
  `quadratic_plus_outlier | 16` run under this `mix=0.25` sweep

Important note:

- the targeted `mix=0.25` elasticity sweep showed `elastic_strength = 8` as
  slightly better on the aggregate outlier summary than `16`
- however, the currently selected presentation/default PBX setting is the
  `16` variant because that is the run chosen for direct init-vs-final
  comparison and review

## What Remains Unresolved

The major outlier failure mode is fixed, and the clusters are materially
cleaner. Residual detached behavior may still remain in some views.

The important causal point is:

- the practical PBX fix came from a stronger explicit outlier force together
  with stronger, stretch-biased elasticity
- the longer-run / higher-iteration checks were secondary diagnostics, not the
  main reason the outliers were corrected

So the current open question is narrower:

- whether any remaining detached points need a better inward direction than the
  current centroid-relative outlier pull
