# zstack_motion_qc

Per-embryo Z-stack motion artifact detection for the morphseq pipeline.

## Problem

During Z-stack acquisition (15 slices, 50µm step, ~1-2s sweep), individual embryos
can move independently. This corrupts the LoG focus-stacked output and silently
propagates into downstream segmentation and morphology measurements. Detection must
be **per-embryo**, not per-frame, because well-behaved embryos dilute whole-frame signals.

## Design

Z slices are transient — they only exist inside `_focus_stack()`. The approach is:

1. **During focus stacking**: compute compact metric grids from the Z-stack while it
   is still in memory. Save one `.npz` per frame alongside the focus-stacked output.
2. **Post-hoc QC** (after segmentation produces embryo masks): load grids, intersect
   tiles with each embryo mask, derive per-embryo scalars and a PASS/WARN/FAIL flag.

## Module Layout

```
kernels.py     pure scalar/patch functions — no stacks, no files, no embryos
grids.py       per-stack grid builders     — (Z, Y, X) → (Z or Z-1, Ny, Nx)
summaries.py   stack-level reductions      — grids → scalars, no masks
io.py          .npz save/load             — one file per source frame
embryo_qc.py   mask-aware embryo QC       — grids + mask → scalars + flag
```

The ladder: `patch kernel → grid → stack summary → embryo summary`

## Chosen Metrics (from exploratory analysis in results/mcolon/20260421_motion_artifact_detection/)

| Metric | Signal | Grid shape |
|---|---|---|
| NCC between adjacent Z pairs | between-slice motion | (Z-1, Ny, Nx) |
| Shannon entropy per slice | within-slice blur / signal quality | (Z, Ny, Nx) |

**From the NCC grid you can derive** (in `embryo_qc.py`):
- `ncc_min` — worst tile across all pairs (primary motion flag)
- `bad_pair_frac` — fraction of pairs where mean tile NCC < threshold
- `local_ncc_std_mean` — spatial spread of NCC (non-uniform / partial motion)

**Relative entropy** (embryo entropy − background entropy) catches the tricky
within-slice blur case that NCC misses (e.g. B10 t=97: NCC clean, rel_entropy −0.51).

## QC Thresholds (from labeled examples, n=10)

```python
FAIL : ncc_min < 0.85  OR  bad_pair_frac > 0.10
WARN : rel_entropy_mean < threshold  (within-slice blur, tuned per-experiment)
PASS : everything else
```

## Usage

```python
from data_pipeline.quality_control.zstack_motion_qc import (
    compute_local_ncc_grid,
    compute_local_entropy_grid,
    save_grids,
    load_grids,
    embryo_ncc_summary,
    embryo_entropy_summary,
    embryo_qc_flag,
)

# Inside _focus_stack() — stack is (Z, Y, X) float32
ncc_grid     = compute_local_ncc_grid(stack_zyx, tile_size=128)
entropy_grid = compute_local_entropy_grid(stack_zyx, tile_size=128)
save_grids(out_path, ncc_grid, entropy_grid,
           tile_size=128, stride=128, stack_shape_yx=stack_zyx.shape[1:])

# Post-hoc, given embryo mask (H, W bool)
grids       = load_grids(out_path)
ncc_sum     = embryo_ncc_summary(grids["ncc_grid"], mask, tile_size=128, stride=128)
ent_sum     = embryo_entropy_summary(grids["entropy_grid"], bg_entropy_grid, mask, 128, 128)
flag        = embryo_qc_flag(ncc_sum, ent_sum)
```

## Next Steps

See `results/mcolon/20260421_motion_artifact_detection/NEXT_STEPS.md`.
