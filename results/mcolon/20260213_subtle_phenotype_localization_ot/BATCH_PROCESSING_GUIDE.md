# Batch Processing Guide for Subtle Phenotype Localization

**Date**: 2026-02-13
**Status**: READY TO USE

---

## Overview

For processing multiple mutant→WT comparisons at scale, use the **proven batch export implementation** from Stream D:

```
results/mcolon/20260213_stream_d_reference_embryo/pipeline/02_run_batch_ot_export.py
```

**Already validated**:
- 313 transitions processed successfully (0 failures)
- Mean runtime: ~1.76s per pair on GPU
- Production params: epsilon=1e-4, reg_m=10.0, canonical grid with yolk alignment

---

## Why This Implementation is Optimal

### Sequential Processing (Not vmap)

**This is by design, not a limitation**:

1. **Variable embryo shapes** → Different bbox sizes after cropping
2. **Shape bucketing not implemented** → vmap requires identical tensor shapes
3. **Recompilation storms avoided** → Sequential prevents JAX from recompiling for each unique shape
4. **Smart frame caching** → Reuses loaded masks across pairs (huge I/O savings)

### JAX/OTT Strengths Leveraged

✅ **OTT backend on GPU**: ~2-5× faster than POT/CPU per solve
✅ **JIT compilation**: OTT solver is JIT-compiled (amortized over sequential calls)
✅ **Numerical stability**: OTT handles epsilon=1e-4 better than POT on canonical grid
✅ **Production-ready**: 313 pairs processed with 0 failures

**Key insight**: JAX speedup comes from JIT compilation + GPU execution **per solve**, not from batching. Sequential processing with cached frames is the right architecture for variable-shape OT problems.

---

## Quick Start: Adapt for Pilot Study

### 1. Create Transition Manifest

Create a CSV with columns:
- `pair_id` (unique identifier)
- `embryo_id` (for consecutive frames) OR `src_embryo_id`, `tgt_embryo_id` (for WT→mutant)
- `frame_src`, `frame_tgt`
- `genotype`, `set_type` (metadata)
- `bin_src_hpf`, `bin_tgt_hpf` (for grouping)
- `analysis_use` (bool, filter flag)
- `is_control_pair` (bool, mark identity/cross-embryo controls)

**Example for WT→mutant comparisons** (48 hpf):

```csv
pair_id,src_embryo_id,tgt_embryo_id,frame_src,frame_tgt,genotype,set_type,bin_src_hpf,bin_tgt_hpf,analysis_use,is_control_pair
wt_mut_001,20251113_A05_e01,20251212_mutant_001,14,14,cep290_homozygous,mutant,48,48,True,False
wt_mut_002,20251113_A05_e01,20251212_mutant_002,14,14,cep290_homozygous,mutant,48,48,True,False
wt_identity,20251113_A05_e01,20251113_A05_e01,14,14,cep290_wildtype,control,48,48,True,True
```

**Note**: Use `is_control_pair=True` for identity checks (WT→same WT frame).

### 2. Run Batch Export

```bash
cd /home/user/morphseq

python results/mcolon/20260213_stream_d_reference_embryo/pipeline/02_run_batch_ot_export.py \
  --csv results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv \
  --transitions results/mcolon/20260213_subtle_phenotype_localization_ot/pilot_manifest.csv \
  --output-root results/mcolon/20260213_subtle_phenotype_localization_ot/ot_exports \
  --run-id pilot_48hpf_wt_mut_v1 \
  --backend ott \
  --epsilon 1e-4 \
  --reg-m 10.0 \
  --max-support-points 5000 \
  --data-root morphseq_playground \
  --resume \
  --include-control
```

**Parameters**:
- `--csv`: Source CSV with embryo masks (RLE format)
- `--transitions`: Your manifest CSV defining WT→mutant pairs
- `--output-root`: Where to save results
- `--run-id`: Unique identifier for this batch run
- `--backend`: `ott` (GPU, recommended) or `pot` (CPU)
- `--epsilon`: 1e-4 (validated for canonical grid + OTT concordance)
- `--reg-m`: 10.0 (marginal relaxation)
- `--max-support-points`: 5000 (adequate for 48 hpf embryos)
- `--data-root`: Path to morphseq_playground (for yolk masks)
- `--resume`: Skip already-processed pairs (safe to re-run)
- `--include-control`: Process control pairs for QC

### 3. Output Structure

**Parquet files** (structured data):
- `ot_pair_metrics.parquet`: Summary metrics per pair (cost, runtime, convergence, metadata)
- `ot_feature_matrix.parquet`: DCT-compressed feature vectors per pair

**Artifacts** (raw fields):
- `pair_artifacts/{pair_id}/fields.npz`: Mass maps + velocity fields (canonical grid)
- `pair_artifacts/{pair_id}/coupling.npz`: Transport plan (if `store_coupling=True`)

**Log**:
- `run_log_{run_id}.csv`: Per-pair status (ok/failed/skipped)

### 4. Load Results for Visualization

```python
import pandas as pd
import numpy as np
from pathlib import Path

output_root = Path("results/mcolon/20260213_subtle_phenotype_localization_ot/ot_exports")
run_id = "pilot_48hpf_wt_mut_v1"

# Load metrics
metrics = pd.read_parquet(output_root / "ot_pair_metrics.parquet")
metrics = metrics[metrics["run_id"] == run_id].copy()

# Load fields for a specific pair
pair_id = "wt_mut_001"
fields = np.load(output_root / "pair_artifacts" / pair_id / "fields.npz")
cost_src_px = fields["cost_src_px"]  # Cost on WT grid
velocity_yx = fields["velocity_px_per_frame_yx"]  # Velocity on WT grid
mass_created = fields["mass_created_um2"]  # μm²
mass_destroyed = fields["mass_destroyed_um2"]  # μm²
```

---

## Advanced: Frame Caching Strategy

**Why frame caching matters**:
- Loading a mask from RLE + yolk from disk: ~50-100ms
- If you have 4 mutants × 1 WT reference = 4 pairs, naïve approach loads WT mask 4 times
- Frame caching: Load WT once, reuse across all 4 pairs

**How it works** (automatic in `02_run_batch_ot_export.py`):
1. Build set of all unique `(embryo_id, frame_index)` needed for all transitions
2. Load all frames into `frame_cache` dict
3. For each pair, lookup `src` and `tgt` from cache (no reload)

**Memory footprint**: O(unique frames), not O(pairs). For 20 mutants × 1 WT = 21 frames cached, not 20 pairs loaded.

---

## Production Parameters (Validated)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `epsilon` | `1e-4` | POT-OTT concordance on canonical grid (0.21% cost diff) |
| `reg_m` | `10.0` | Standard marginal relaxation (0.5-1% creation/destruction) |
| `max_support_points` | `3000`-`5000` | 48 hpf embryos fit comfortably; 5000 for safety |
| `coord_scale` | `1/576` | Auto-computed from canonical grid shape (256×576) |
| `metric` | `sqeuclidean` | Standard for morphology OT |
| `canonical_grid_align_mode` | `yolk` | Biological alignment (falls back to centroid if yolk missing) |
| `store_coupling` | `True` | Needed for barycentric projection (velocity field) |

---

## Resume Support

**Safe to interrupt and re-run**:

```bash
# Run interrupted after 50 pairs
python 02_run_batch_ot_export.py ... --resume

# Will skip first 50 pairs, continue from pair 51
```

**How it works**:
- Reads existing `ot_pair_metrics.parquet`
- Builds set of `(run_id, pair_id)` already completed
- Skips those pairs in loop

**Important**: If you change parameters (epsilon, reg_m), use a **new run_id** to avoid mixing results.

---

## Backend Selection

### OTT (Recommended)

```bash
--backend ott
```

**Pros**:
- GPU-enabled (~2-5× faster per solve than POT)
- Better numerical stability at epsilon=1e-4 on canonical grid
- JIT-compiled Sinkhorn (fast after first compile)

**Cons**:
- Requires ott-jax + jax installed
- GPU memory limits (but 48 hpf embryos fit easily)

### POT (Fallback)

```bash
--backend pot
```

**Pros**:
- CPU-only, no GPU required
- Widely used reference implementation

**Cons**:
- Slower (~2-5× slower than OTT/GPU)
- Numerical issues at epsilon=1e-5 on canonical grid (use epsilon=1e-4)

---

## Smoke Test

**Before processing 100s of pairs, run a smoke test**:

```bash
# Create manifest with 3 pairs (2 mutants + 1 identity control)
python 02_run_batch_ot_export.py \
  ... \
  --limit 3 \
  --run-id smoke_test_001
```

**Check**:
- All 3 pairs succeed (status=ok in log)
- Runtime ~1-3s per pair (GPU)
- Fields artifacts exist in `pair_artifacts/`
- Metrics parquet has 3 rows

---

## Future: vmap-Based Batching

**See**: `BATCH_PROCESSING_DESIGN.md` for complete design proposal with three implementation options.

**When vmap would help**:
- Identical tensor shapes (requires shape bucketing or padding)
- Many pairs processed in tight loop (amortize compile cost)
- GPU memory sufficient for batch size

**What's needed** (not currently implemented):
1. Implement bucketing in `pair_frame.py` (infrastructure exists, not activated)
2. Pad/crop all pairs to fixed work grid size
3. Replace sequential loop with `jax.vmap` over batch dimension
4. JIT-compile vmapped solver

**Expected speedup**: ~2-3× over current sequential OTT (not 10×, since per-solve is already fast)

**Recommended approach**: Option 1 (Explicit Batch Backend)
- New `OTTBatchBackend` class with vmap-based `solve_batch`
- CLI flag: `--use-batch` to opt-in
- Shape validation fails fast if shapes don't match
- Backward compatible (existing scripts unchanged)

**Why not now**:
- Variable embryo shapes make bucketing complex
- Sequential is "fast enough" (~1.76s/pair for 313 pairs = 9 minutes total)
- Adds complexity for marginal gain in pilot study

---

## Checklist for Pilot Study

- [ ] Create manifest CSV with WT→mutant pairs (4 mutants, 1 WT reference, 48 hpf)
- [ ] Add identity control pair (WT→same WT) for QC
- [ ] Run smoke test with `--limit 5`
- [ ] Verify smoke test succeeds (check log, metrics parquet)
- [ ] Run full batch with `--resume` (safe to interrupt)
- [ ] Load results into Python for visualization (see `utils/ot_features.py`)
- [ ] Plot cost density, velocity, mass delta on WT reference grid
- [ ] Compute S-bin profiles along WT centerline
- [ ] Compare mutants to identity control (sanity check)

---

## Support

**Example workflows**:
- Stream D reference embryo: `results/mcolon/20260213_stream_d_reference_embryo/`
- Canonical grid spike: `src/analyze/optimal_transport_morphometrics/docs/phase2_implemnetation_tracking/stream_a_ott_backend/spike_test_results/`

**Documentation**:
- Batch export source: `02_run_batch_ot_export.py` (self-documenting CLI)
- Design proposal: `BATCH_PROCESSING_DESIGN.md` (future vmap implementation options)
- Status report: `BATCH_PROCESSING_STATUS.md` (investigation findings)
- UOT skill: `~/.claude/skills/unbalanced-optimal-transport/SKILL.md`
- Plotting convention: `PLOTTING_CONVENTION.md`

**Questions?** Read the DECISIONS.md files in Stream A (OTT backend) and Stream D (reference embryo).
