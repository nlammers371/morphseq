# Emergence Algorithm

Reference-rooted phenotype emergence timeline. Separates two questions that
pairwise onset data conflates:

1. **When does a phenotype emerge from a baseline?** (emergence from reference)
2. **When do mutant phenotypes become distinguishable from each other?** (within-block resolution)

---

## Input

An **onset matrix** — a symmetric DataFrame where entry `[i, j]` is the first
time point (hpf) at which class `i` is durably statistically separated from
class `j`, or NaN if they never separate.

A **reference set** — one or more genotypes that serve as the baseline (e.g.
injection controls, wildtype).

---

## Step 1 — Reference validation (`validate_reference`)

Checks whether the reference members are internally coherent — i.e., they
don't separate from each other. Computes:

    coherence_score = n_NaN_pairs / n_total_pairs

Three-state status:
- **valid** — all internal pairs NaN (perfect baseline)
- **ambiguous** — coherence ≥ 0.5 (some internal separation, but majority are NaN)
- **invalid** — coherence < 0.5 (reference is tearing itself apart)

The pipeline always proceeds regardless of status. Invalid/ambiguous references
are flagged in the UI but not blocked.

---

## Step 2 — Emergence scoring (`compute_emergence_scores`)

For each non-reference class `c`:

    emergence_time(c) = median{ onset(c, r) : r ∈ reference, finite }

Also records `emergence_min`, `emergence_max`, and per-reference-member
breakdown.

Classes that never separate from any reference member → `emergence_time = NaN`.

---

## Step 3 — Block formation (`form_emergence_blocks`)

Groups non-reference classes into time bins:

    bin_key = floor(emergence_time / bin_width) * bin_width

The **displayed** time for each block is the raw median of its members'
emergence times — never the floored bin key. Classes with NaN emergence → a
single "unresolved from reference" block at the end.

Default `bin_width = 4.0 hpf` (matches the data bin width).

---

## Step 4 — Recursive block resolution (`resolve_block`)

For each multi-member block, asks: among these classes that all emerged at the
same time from the reference, when can *they* be distinguished from each other?

The reference is ignored at this step. Only within-block pairwise onsets are used.

### Bipartition search

For a block with members `M`, enumerate all non-trivial bipartitions `B1 | B2`.
For each candidate:

1. **Score the partition** (`_score_partition`):
   - Compute all cross-partition onsets: `{ onset(a, b) : a ∈ B1, b ∈ B2 }`
   - `cross_support = n_finite_cross / n_total_cross`
   - `cross_median = median of finite cross onsets`
   - Reject if `cross_support < min_cross_support` (default 0.5) or no finite cross onsets

2. **Check monotonicity** (`_is_monotone_feasible`):
   - Require `cross_median >= floor` (the parent's split time)
   - Require that both `B1` and `B2` can themselves be recursively split with
     all descendant split times ≥ `cross_median`
   - This prevents topology inversions where a child split appears earlier than
     its parent on the time axis
   - Result is memoized to avoid combinatorial blowup

3. **Score among feasible candidates** (priority order):
   1. Lower `internal_finite_count` — count of finite onsets *within* each child
      group. Lower = children are more internally similar = cleaner split.
   2. Higher `cross_support` — more evidence for the split
   3. Higher `cross_median` — later split = more informative

If no candidate passes → **unresolved composite** (dashed border in UI).

Recurse on each child, passing `cross_median` as the new `floor`.

---

## Step 5 — Pipeline (`build_emergence_timeline`)

Chains steps 1–4 into an `EmergenceTimeline`:
- `reference_validation` — coherence check result
- `scores` — per-class emergence times, sorted ascending (NaN last)
- `blocks` — emergence blocks, sorted ascending (NaN block last)
- `block_resolutions` — `block_id → ResolutionNode` tree per block

---

## Worked Example (PBX 5-class, reference = {inj_ctrl, wik_ab})

Onset matrix (p_sep=0.05, subsequent_frac=0.40):

```
                   1b+4   pbx4   pbx1b  ctrl   wik
pbx1b_pbx4          —      62     22     22     22
pbx4               62       —     82     22     22
pbx1b              22      82      —     26     30
inj_ctrl           22      22     26      —    NaN
wik_ab             22      22     30    NaN      —
```

**Step 1**: inj_ctrl vs wik_ab = NaN → status="valid", coherence=1.0

**Step 2** (emergence times relative to {inj_ctrl, wik_ab}):
- pbx1b_pbx4: median(22, 22) = 22
- pbx4:       median(22, 22) = 22
- pbx1b:      median(26, 30) = 28

**Step 3** (bin_width=4):
- bin 20 → {pbx1b_pbx4, pbx4}, emergence_time=22 (raw median)
- bin 28 → {pbx1b}, emergence_time=28

**Step 4** — resolve {pbx1b_pbx4, pbx4}:
- Only candidate: {pbx1b_pbx4} | {pbx4}
- Cross: onset(pbx1b_pbx4, pbx4) = 62 → support=1.0, cross_median=62
- Internal: no pairs within each singleton → internal_finite=0
- Monotone-feasible from floor=0 ✓
- Result: split at **62 hpf**

**Final tree**:
```
{inj_ctrl, wik_ab}      ← reference (composite, dashed)
        │
      22 hpf ── {pbx1b_pbx4, pbx4} emerge
                    └─ 62 hpf: pbx1b_pbx4 | pbx4
      28 hpf ── pbx1b (singleton)
```

---

## Known Limitations / Future Improvements

### 1. Cross_median discards within-split variance

A split `{A, B} | {C, D}` with cross onsets [20, 60, 22, NaN] gets labeled at
22 hpf. The 60 hpf outlier — which may mean A and D don't cleanly belong on
opposite sides — is invisible. A future improvement could report split
uncertainty (e.g. IQR of cross onsets) and display it as an error bar or node
width in the renderer.

### 2. Support threshold is binary

`min_cross_support = 0.5` is a hard cutoff. A split with 51% finite pairs is
treated identically to one with 100%. Graded confidence (e.g. a likelihood
score weighted by support) would give a more honest picture.

### 3. Bin width creates hard grouping boundaries

Two classes emerging at 23 and 25 hpf land in the same block (bin 20–24).
Two classes at 23 and 27 hpf land in different blocks. Whether they're compared
internally or placed on the trunk is entirely determined by this boundary.
A soft grouping (e.g. hierarchical clustering on emergence times before
block formation) would be more robust.

### 4. Block emergence time collapses to a single median

If block members emerge at 22 and 28 hpf, the block is placed at 25 hpf and
both members appear simultaneous. The spread is stored in `emergence_min` /
`emergence_max` but not rendered. Future: show as a vertical interval on the
branch.

### 5. Reference composite treated as a single point

If reference = {inj_ctrl, wik_ab} and a mutant separates from inj_ctrl at 22
but wik_ab at 30, emergence_time = 26 (median). The 8 hpf spread across
reference members is stored in `per_ref_onsets` but not shown.

### 6. Visualization lives in the explorer script, not in a library

The current renderer is ~300 lines of D3/JS inside `13_emergence_explorer.py`.
A matplotlib-based renderer for static figures (paper-quality output) should
live in `analyze/classification/viz/emergence.py`, following the pattern of
`viz/classification.py`, `viz/heatmaps.py`, etc.
