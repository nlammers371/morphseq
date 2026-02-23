# Design Proposal: Batch Processing Mode (vmap-based)

**Date**: 2026-02-13
**Status**: DESIGN PROPOSAL (Not Yet Implemented)
**Context**: Future optimization for processing 100s-1000s of OT problems with vmap

---

## Background

**Current implementation** (`02_run_batch_ot_export.py`):
- Sequential processing with frame caching
- OTT backend (GPU per-solve)
- ~1.76s/pair validated on 313 transitions
- ✅ Production-ready, variable-shape compatible

**Why consider vmap**:
- Processing 1000s of pairs (where 2-3× speedup matters)
- Shapes can be standardized via bucketing
- Amortize JIT compilation across batch dimension

**Expected speedup**: ~2-3× over sequential OTT (not 10×, since per-solve already fast)

---

## Prerequisites

Before implementing batch processing, the following infrastructure must be in place:

### 1. Shape Bucketing Implementation

**Status**: Spec exists, implementation placeholder

**Required changes in `pair_frame.py`**:

```python
def build_pair_frame_geometry(
    src_frame: UOTFrame,
    tgt_frame: UOTFrame,
    config: UOTConfig,
    crop_policy: Literal["union", "fixed", "bucketed"] = "union",
    bucket_shapes: Optional[list[tuple[int, int]]] = None,  # Work pixel units
) -> PairFrameGeometry:
    """
    Build geometry for OT pair frame.

    crop_policy:
        - "union": Tight bbox around union of masks (current default)
        - "fixed": Fixed canonical grid size (no cropping)
        - "bucketed": Round up to nearest bucket shape for vmap compatibility

    bucket_shapes: List of (H, W) shapes in work pixel units.
        Example: [(128, 256), (256, 512), (512, 1024)]
        Pair will be padded to smallest bucket that fits.
    """
    if crop_policy == "bucketed":
        if bucket_shapes is None:
            raise ValueError("bucket_shapes required when crop_policy='bucketed'")

        # Compute union bbox
        union_box = compute_union_bbox(src_mask, tgt_mask)
        union_h, union_w = union_box.height, union_box.width

        # Find smallest bucket that fits
        for bucket_h, bucket_w in sorted(bucket_shapes):
            if union_h <= bucket_h and union_w <= bucket_w:
                work_valid_box_yx = union_box
                work_pad_offsets_yx = (
                    (bucket_h - union_h) // 2,
                    (bucket_w - union_w) // 2,
                )
                work_shape_yx = (bucket_h, bucket_w)
                break
        else:
            raise ValueError(f"No bucket fits union shape ({union_h}, {union_w})")

        return PairFrameGeometry(
            work_valid_box_yx=work_valid_box_yx,
            work_pad_offsets_yx=work_pad_offsets_yx,
            work_shape_yx=work_shape_yx,
            # ... other fields
        )
```

### 2. Padding/Cropping Utilities

**Required in `pair_frame.py` or `uot_grid.py`**:

```python
def pad_to_bucket_shape(
    density: np.ndarray,
    mask: np.ndarray,
    geometry: PairFrameGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad density and mask to bucketed work shape.

    Returns:
        padded_density: (work_H, work_W) with zeros outside valid box
        padded_mask: (work_H, work_W) with False outside valid box
    """
    work_h, work_w = geometry.work_shape_yx
    pad_y, pad_x = geometry.work_pad_offsets_yx

    padded_density = np.zeros((work_h, work_w), dtype=density.dtype)
    padded_mask = np.zeros((work_h, work_w), dtype=bool)

    valid_h, valid_w = density.shape
    padded_density[pad_y:pad_y+valid_h, pad_x:pad_x+valid_w] = density
    padded_mask[pad_y:pad_y+valid_h, pad_x:pad_x+valid_w] = mask

    return padded_density, padded_mask


def unpad_from_bucket_shape(
    result_array: np.ndarray,
    geometry: PairFrameGeometry,
) -> np.ndarray:
    """
    Extract valid region from bucketed result.

    Args:
        result_array: (work_H, work_W, ...) on bucketed grid
        geometry: Contains work_valid_box_yx and work_pad_offsets_yx

    Returns:
        valid_array: (valid_H, valid_W, ...) cropped to original bbox
    """
    pad_y, pad_x = geometry.work_pad_offsets_yx
    valid_h, valid_w = geometry.work_valid_box_yx.height, geometry.work_valid_box_yx.width

    return result_array[pad_y:pad_y+valid_h, pad_x:pad_x+valid_w, ...]
```

---

## Option 1: Explicit Batch Backend (Recommended)

**Philosophy**: Separate backend class for vmap-based batch processing. Clear API, explicit opt-in.

### Implementation

**Create new backend**: `src/analyze/utils/optimal_transport/backends/ott_batch_backend.py`

```python
"""OTT Batch Backend with vmap for fixed-shape problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud

from ..config import UOTConfig
from ..result import UOTResult
from ..backend import UOTBackend


class OTTBatchBackend(UOTBackend):
    """
    OTT backend with vmap for batch processing.

    Requirements:
        - All problems in batch must have IDENTICAL tensor shapes
        - Use crop_policy="bucketed" in pair_frame.py
        - Batch size limited by GPU memory

    Expected speedup: ~2-3× over sequential OTTBackend for batches of 10-20.

    Usage:
        backend = OTTBatchBackend()
        results = backend.solve_batch(problems, config, batch_size=16)
    """

    def __init__(self):
        super().__init__()
        self._vmapped_solver = None  # Compiled once per shape
        self._compiled_shape = None

    def solve_batch(
        self,
        problems: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        config: UOTConfig,
        batch_size: int = 16,
    ) -> list[UOTResult]:
        """
        Solve batch of OT problems with vmap.

        Args:
            problems: List of (src_density, tgt_density, src_mask, tgt_mask).
                ALL must have IDENTICAL shapes.
            config: UOTConfig (same for all problems)
            batch_size: Number of problems to solve in parallel.
                Larger = more memory, but better GPU utilization.

        Returns:
            List of UOTResult objects (same order as input).

        Raises:
            ValueError: If problem shapes are not identical.
        """
        if not problems:
            return []

        # Validate identical shapes
        shapes = [p[0].shape for p in problems]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"All problems must have identical shapes for vmap. "
                f"Got shapes: {set(shapes)}. "
                f"Use crop_policy='bucketed' in pair_frame.py."
            )

        # Process in batches to avoid OOM
        results = []
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i+batch_size]
            batch_results = self._solve_batch_vmapped(batch, config)
            results.extend(batch_results)

        return results

    def _solve_batch_vmapped(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        config: UOTConfig,
    ) -> list[UOTResult]:
        """Solve batch with vmap (compiles once, reuses)."""

        # Stack into batch arrays
        src_densities = jnp.stack([jnp.array(p[0]) for p in batch])  # (B, H, W)
        tgt_densities = jnp.stack([jnp.array(p[1]) for p in batch])  # (B, H, W)
        src_masks = jnp.stack([jnp.array(p[2]) for p in batch])      # (B, H, W)
        tgt_masks = jnp.stack([jnp.array(p[3]) for p in batch])      # (B, H, W)

        # Compile vmapped solver if needed
        shape = src_densities.shape[1:]  # (H, W)
        if self._compiled_shape != shape or self._vmapped_solver is None:
            self._vmapped_solver = self._compile_vmapped_solver(config, shape)
            self._compiled_shape = shape

        # Solve batch (vmapped)
        batch_results = self._vmapped_solver(
            src_densities, tgt_densities, src_masks, tgt_masks
        )

        # Unpack into list of UOTResult objects
        results = []
        for i in range(len(batch)):
            result = UOTResult(
                cost=float(batch_results["cost"][i]),
                coupling_matrix=np.array(batch_results["coupling"][i]),
                velocity_px_per_frame_yx=np.array(batch_results["velocity"][i]),
                mass_created_px=np.array(batch_results["mass_created"][i]),
                mass_destroyed_px=np.array(batch_results["mass_destroyed"][i]),
                # ... other fields
            )
            results.append(result)

        return results

    def _compile_vmapped_solver(self, config: UOTConfig, shape: tuple[int, int]):
        """
        Compile vmapped Sinkhorn solver.

        This is called once per unique shape, then reused.
        """

        def single_solve(src_density, tgt_density, src_mask, tgt_mask):
            """Solve single OT problem (will be vmapped)."""

            # Flatten to point clouds (using mask)
            src_points = jnp.stack(jnp.where(src_mask), axis=-1)  # (N_src, 2)
            tgt_points = jnp.stack(jnp.where(tgt_mask), axis=-1)  # (N_tgt, 2)

            src_weights = src_density[src_mask]  # (N_src,)
            tgt_weights = tgt_density[tgt_mask]  # (N_tgt,)

            # Build geometry
            geom = pointcloud.PointCloud(
                src_points * config.coord_scale,
                tgt_points * config.coord_scale,
                epsilon=config.epsilon,
            )

            # Solve Sinkhorn
            out = sinkhorn.solve(
                geom,
                a=src_weights,
                b=tgt_weights,
                tau_a=config.marginal_relaxation,
                tau_b=config.marginal_relaxation,
            )

            # Extract outputs
            coupling = out.matrix
            cost = out.reg_ot_cost

            # Compute velocity field from coupling
            # (requires barycentric projection - see OTTBackend.solve)
            # ... implementation details ...

            return {
                "cost": cost,
                "coupling": coupling,
                "velocity": velocity_field,
                "mass_created": mass_created,
                "mass_destroyed": mass_destroyed,
            }

        # vmap over batch dimension
        vmapped_solve = jax.vmap(single_solve)

        # JIT compile
        vmapped_solve_jit = jax.jit(vmapped_solve)

        return vmapped_solve_jit
```

### Usage in Batch Export Script

**Modified `02_run_batch_ot_export.py`**:

```python
from analyze.utils.optimal_transport.backends.ott_batch_backend import OTTBatchBackend

def _build_backend(name: str, use_batch: bool = False):
    if name.lower() == "ott":
        if use_batch:
            return OTTBatchBackend(), "OTT_BATCH"
        else:
            return OTTBackend(), "OTT"
    # ...

def run_export(args: argparse.Namespace) -> None:
    # ...
    backend, backend_name = _build_backend(args.backend, use_batch=args.use_batch)

    if args.use_batch:
        # Group problems by shape bucket
        bucketed_problems = group_by_bucket_shape(transitions, frame_cache, cfg)

        for bucket_shape, bucket_problems in bucketed_problems.items():
            print(f"Processing bucket {bucket_shape}: {len(bucket_problems)} pairs")

            # Prepare batch inputs (all same shape)
            batch_inputs = [
                prepare_bucketed_pair(p, frame_cache, cfg)
                for p in bucket_problems
            ]

            # Solve batch
            batch_results = backend.solve_batch(
                batch_inputs,
                config=cfg,
                batch_size=args.batch_size,
            )

            # Unpack results (unpad from bucket shape)
            for problem, result in zip(bucket_problems, batch_results):
                result_unpacked = unpad_result(result, problem.geometry)
                # ... save to parquet ...
    else:
        # Sequential processing (current implementation)
        # ...
```

**CLI arguments**:

```python
parser.add_argument("--use-batch", action="store_true", help="Use vmap batch processing")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size for vmap")
parser.add_argument("--bucket-shapes", type=str, default="128x256,256x512,512x1024")
```

### Pros

✅ **Explicit opt-in**: Users choose batch mode via CLI flag
✅ **Clear separation**: Sequential and batch backends are distinct classes
✅ **Shape validation**: Fails fast if shapes don't match
✅ **Backward compatible**: Existing scripts unchanged
✅ **Testable**: Can compare batch vs sequential on same inputs
✅ **Memory control**: Batch size parameter prevents OOM

### Cons

⚠️ **API duplication**: Two backends with similar logic
⚠️ **Shape bucketing complexity**: Users must understand bucketing
⚠️ **Memory debugging**: vmap OOM errors harder to diagnose than sequential

---

## Option 2: Automatic Backend Selection

**Philosophy**: Single `OTTBackend` auto-detects when batching is beneficial and switches modes.

### Implementation Sketch

```python
class OTTBackend(UOTBackend):
    """OTT backend with automatic batch mode selection."""

    def solve_batch(self, problems, config):
        """
        Automatically decide: sequential or vmap?

        Criteria:
            - If all shapes identical AND batch_size >= 8: Use vmap
            - Otherwise: Use sequential
        """
        shapes = [self._get_problem_shape(p) for p in problems]

        if len(set(shapes)) == 1 and len(problems) >= 8:
            # All identical shapes, batch large enough
            return self._solve_batch_vmapped(problems, config)
        else:
            # Variable shapes or small batch
            return self._solve_batch_sequential(problems, config)
```

### Pros

✅ **User-friendly**: No flags, just works
✅ **Single backend class**: Less code duplication
✅ **Graceful fallback**: Auto-handles variable shapes

### Cons

⚠️ **Hidden behavior**: Users don't know which mode is used
⚠️ **Harder to debug**: Performance issues harder to diagnose
⚠️ **Less control**: Can't force batch mode for testing
⚠️ **Bucketing still required**: Shapes must be standardized upstream

---

## Option 3: Factory Function

**Philosophy**: Factory function returns appropriate backend based on context.

### Implementation Sketch

```python
def create_ott_backend(
    problems: Optional[list] = None,
    batch_mode: Literal["auto", "sequential", "vmap"] = "auto",
) -> UOTBackend:
    """
    Create OTT backend with optimal configuration.

    Args:
        problems: If provided, analyze shapes for auto mode
        batch_mode:
            - "auto": Choose based on problem shapes
            - "sequential": Always use sequential
            - "vmap": Always use vmap (requires identical shapes)

    Returns:
        OTTBackend (sequential) or OTTBatchBackend (vmap)
    """
    if batch_mode == "sequential":
        return OTTBackend()

    if batch_mode == "vmap":
        return OTTBatchBackend()

    if batch_mode == "auto":
        if problems is None:
            return OTTBackend()  # Default to sequential

        shapes = [get_problem_shape(p) for p in problems]
        if len(set(shapes)) == 1 and len(problems) >= 8:
            return OTTBatchBackend()
        else:
            return OTTBackend()

    raise ValueError(f"Unknown batch_mode: {batch_mode}")
```

### Pros

✅ **Flexible**: Supports auto, explicit sequential, explicit vmap
✅ **Clear API**: Factory pattern is common in ML codebases
✅ **Testable**: Can force modes for benchmarking

### Cons

⚠️ **Extra indirection**: Factory adds complexity
⚠️ **Shape analysis overhead**: Must scan problems twice

---

## Recommendation: Option 1 (Explicit Batch Backend)

**Rationale**:

1. **Clarity**: Users explicitly opt-in to batch mode (know what they're getting)
2. **Debuggability**: Easy to compare sequential vs batch on same data
3. **Safety**: Shape validation fails fast (no silent fallbacks)
4. **Control**: Batch size parameter prevents OOM surprises
5. **Backward compatibility**: Existing scripts unchanged

**Implementation priority**:

1. ✅ **Keep sequential as default** (production-validated, safe)
2. 🔶 **Implement bucketing in `pair_frame.py`** (prerequisite)
3. 🔶 **Create `OTTBatchBackend`** (new class, opt-in)
4. 🔶 **Add `--use-batch` flag** to batch export script
5. 🔶 **Validate on test dataset** (compare sequential vs batch outputs)
6. ✅ **Document batch mode** in skill + guide

**When to use batch mode**:

- ✅ Processing 100s-1000s of pairs (where 2-3× matters)
- ✅ Shapes can be bucketed (implementation complete)
- ✅ GPU memory sufficient for batch size (profiled)
- ❌ Pilot study (<100 pairs, sequential "fast enough")
- ❌ Variable shapes not standardized (bucketing not implemented)

---

## Performance Expectations

### Sequential OTT (Current)

| Metric | Value | Notes |
|--------|-------|-------|
| Per-pair runtime | ~1.76s | Validated on 313 pairs |
| Batch of 100 pairs | ~3 minutes | Linear scaling |
| Batch of 1000 pairs | ~30 minutes | Validated extrapolation |
| GPU utilization | ~60-80% | Per-solve, not batch |
| Memory footprint | O(unique frames) | Frame caching |

### vmap OTT Batch (Projected)

| Metric | Value | Notes |
|--------|-------|-------|
| Per-pair runtime | ~0.6-0.9s | 2-3× speedup |
| Batch of 100 pairs | ~1-1.5 minutes | Amortized compile |
| Batch of 1000 pairs | ~10-15 minutes | Assuming batch_size=16 |
| GPU utilization | ~85-95% | Batch parallelism |
| Memory footprint | O(batch_size × max_shape) | Higher than sequential |

**Assumptions**:
- Identical shapes per bucket (bucketing implemented)
- Batch size = 16 (fits in GPU memory)
- Compile overhead amortized (same shapes reused)

**Reality check**: Speedup is 2-3×, NOT 10×, because:
- Sequential per-solve already GPU-accelerated (OTT)
- vmap removes Python loop overhead + enables batch GPU ops
- Sinkhorn convergence still dominates (can't parallelize iterations)

---

## Implementation Checklist

**Before implementing vmap batch mode**:

- [ ] Validate use case: Processing 100s-1000s of pairs (not <100)
- [ ] Implement shape bucketing in `pair_frame.py`
- [ ] Test bucketing on real embryo data (ensure reasonable bucket sizes)
- [ ] Profile GPU memory for typical batch sizes (8, 16, 32)
- [ ] Benchmark sequential OTT on target dataset (establish baseline)

**Implementation steps**:

- [ ] Create `OTTBatchBackend` class with vmap-based `solve_batch`
- [ ] Implement padding/unpadding utilities for bucketed shapes
- [ ] Add `--use-batch` and `--batch-size` CLI flags to export script
- [ ] Validate correctness: Compare batch vs sequential on 100 pairs (should match)
- [ ] Benchmark performance: Measure actual speedup on GPU
- [ ] Document batch mode in skill + guide (when to use, how to configure)

**Success criteria**:

- ✅ Correctness: Batch results match sequential (within numerical tolerance)
- ✅ Performance: 2-3× speedup on batches of 10-20 pairs
- ✅ Memory: No OOM for batch_size=16 on typical embryo shapes
- ✅ Usability: CLI flags intuitive, errors informative

---

## Alternative: Hybrid Approach

**For very large datasets** (10,000+ pairs), consider:

1. **Group by shape bucket** (e.g., small/medium/large embryos)
2. **Sequential across buckets** (different shapes, can't vmap)
3. **vmap within buckets** (same shape, can vmap)

**Example**:

```python
# Group transitions by bucket shape
buckets = {
    (128, 256): [...],  # Small embryos
    (256, 512): [...],  # Medium embryos
    (512, 1024): [...], # Large embryos
}

# Sequential across buckets
for bucket_shape, bucket_problems in buckets.items():
    # vmap within bucket (all same shape)
    backend = OTTBatchBackend()
    results = backend.solve_batch(bucket_problems, config, batch_size=16)
```

**Speedup**: ~2-3× per bucket (best of both worlds)

---

## Related Documents

- `BATCH_PROCESSING_GUIDE.md`: Current sequential implementation (production-ready)
- `BATCH_PROCESSING_STATUS.md`: Investigation report (why sequential is optimal now)
- `~/.claude/skills/unbalanced-optimal-transport/SKILL.md`: UOT skill (GPU section)
- `src/analyze/optimal_transport_morphometrics/docs/ot_pair_frame_spec_v2_filled.md.txt`: Bucketing spec

---

## Questions?

**"Should I implement vmap batch mode now?"**

**For pilot study (4 mutants)**: No. Sequential is "fast enough" (~10-20 seconds total).

**For scaling to 1000+ pairs**: Maybe. Implement if:
- [ ] Profiling shows solver time dominates (not I/O, not feature extraction)
- [ ] 2-3× speedup would save significant time (30 min → 10-15 min)
- [ ] Shape bucketing is straightforward (reasonable bucket granularity)

**"Which option should I implement?"**

**Recommended**: Option 1 (Explicit Batch Backend)
- Clear API, explicit opt-in, backward compatible
- Easiest to test and debug
- Proven pattern in ML frameworks (e.g., TensorFlow Eager vs Graph mode)

**"What's the priority?"**

1. **HIGH**: Complete pilot study with sequential implementation
2. **MEDIUM**: Implement shape bucketing (prerequisite for vmap)
3. **LOW**: Create `OTTBatchBackend` (only if scaling to 1000s of pairs)

Sequential implementation is production-ready. Batch mode is an optimization, not a necessity.
