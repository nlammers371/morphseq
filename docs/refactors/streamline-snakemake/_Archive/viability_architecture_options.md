# Viability/Death Detection Architecture Options

**Context:** The UNet viability mask is used to compute `fraction_alive`, which then feeds into death persistence validation. Currently this is split across multiple modules and creates confusing data flow.

---

## Current State Analysis

### Data Flow:
```
UNet viability mask (Build02B)
    ↓
compute_fraction_alive() [qc_utils.py]
    ↓
fraction_alive column added to tracking data (Build03A)
    ↓
detect_persistent_death_inflection() [death_detection.py]
    ↓
dead_flag2 + dead_inflection_time_int (Build04)
```

### Issues:
1. **Split computation:** UNet mask → fraction_alive (Build03) → dead_flag2 (Build04)
2. **Confusing naming:**
   - `fraction_alive` = 1 - (dead_pixels / total_embryo_pixels)
   - Actually measuring "fraction NOT necrotic" (biological term more accurate)
   - `dead_flag` (simple threshold) vs `dead_flag2` (persistence validated)
3. **Multiple outputs:**
   - Build02B: viability masks
   - Build03A: fraction_alive column
   - Build04: dead_flag2, dead_inflection_time_int
4. **Rule dependencies unclear:** Which rules need viability masks vs fraction_alive vs death flags?

---

## Option 1: Unified Death Detection Module (RECOMMENDED - APPROVED)

### Structure:
```python
# Single module: quality_control/embryo_viability_qc.py

def compute_fraction_alive(emb_mask: np.ndarray,
                           via_mask: np.ndarray) -> float:
    """
    Compute fraction of embryo pixels that are alive (not necrotic).

    Args:
        emb_mask: Binary embryo mask (from SAM2)
        via_mask: Binary viability mask (from UNet) where 1 = necrotic/dead tissue

    Returns:
        Fraction in [0, 1] representing viable tissue

    Note:
        fraction_alive = 1 - (necrotic_pixels / total_embryo_pixels)
        Biologically valid: measures cellular viability
    """
    total = int(emb_mask.sum())
    if total == 0:
        return np.nan
    dead = int((emb_mask & via_mask).sum())
    alive = total - dead
    return max(0.0, min(1.0, alive / total))


def compute_viability_qc(tracking_df: pd.DataFrame,
                        sam2_masks_dir: Path,
                        unet_via_masks_dir: Path,
                        images_dir: Path,
                        persistence_threshold: float = 0.25,
                        min_decline_rate: float = 0.05,
                        buffer_hours: float = 2.0) -> pd.DataFrame:
    """
    Unified viability QC: compute fraction_alive + death detection with persistence.

    Algorithm:
        1. For each snip_id, load SAM2 embryo mask + UNet viability mask
        2. Compute fraction_alive = 1 - (dead_pixels / embryo_pixels)
        3. For each embryo_id, detect fraction_alive decline points
        4. Validate persistence: ≥25% of post-inflection points must have simple dead threshold
        5. If persistent, mark death inflection time
        6. Apply 2-hour buffer using predicted_stage_hpf

    Returns DataFrame with columns:
        - snip_id
        - embryo_id
        - time_int
        - fraction_alive              # Raw viability metric (0-1)
        - dead_flag                   # Unified death flag (persistence validated)
        - dead_inflection_time_int    # Precise death timepoint

    Note: Consolidates legacy dead_flag + dead_flag2 into single robust flag
    """
    pass
```

### Data Output:
```
quality_control_flags/{experiment_id}/embryo_viability.csv
Columns:
    - snip_id
    - embryo_id
    - time_int
    - fraction_alive             # Raw metric: 1 - (necrotic/total)
    - dead_flag                  # Unified death flag (persistence validated, 2hr buffer)
    - dead_inflection_time_int   # Death timepoint (same for all snips of embryo)
```

### Snakemake Rule:
```python
rule qc_viability:
    input:
        tracking_csv=segmentation/embryo_tracking/{experiment_id}/tracking_table.csv,
        sam2_masks=segmentation/embryo_tracking/{experiment_id}/masks/,
        viability_masks=segmentation/auxiliary_masks/{experiment_id}/viability/,
        images=processed_images/stitched_FF/{experiment_id}/
    output:
        quality_control_flags/{experiment_id}/embryo_viability.csv
    run:
        from data_pipeline.quality_control.embryo_viability_qc import detect_persistent_death
```

### Pros:
- ✅ Single source of truth for death detection
- ✅ Keeps familiar `fraction_alive` naming (biologically valid)
- ✅ One robust `dead_flag` (no confusion between v1/v2)
- ✅ All death-related logic in one module
- ✅ Easier to test, debug, maintain
- ✅ Minimal breaking changes (same column names, just consolidated logic)

### Cons:
- ⚠️ Need to update code expecting separate `dead_flag2` column (merge to `dead_flag`)
- ⚠️ Moderate refactor effort (but cleaner result)

---

## Option 2: Minimal Change - Move fraction_alive to Build02B Output

### Structure:
Keep existing modules but reorganize data flow:

```python
# segmentation/unet/inference.py (Build02B equivalent)
def compute_auxiliary_metrics(emb_mask, via_mask, yolk_mask, focus_mask, bubble_mask):
    """
    Compute metrics directly from UNet auxiliary masks.

    Returns dict with:
        - fraction_alive
        - (no other QC flags yet - spatial analysis happens later)
    """
    pass

# quality_control/viability_tracking_qc.py (unchanged)
def compute_dead_flag2_persistence(df):
    """Use fraction_alive to compute dead_flag2"""
    pass
```

### Data Output:
```
# After Build02B (UNet segmentation)
segmentation/auxiliary_masks/{experiment_id}/viability_metrics.csv
Columns:
    - image_id
    - embryo_id (if available from tracking)
    - fraction_alive

# After Build04 (QC)
quality_control_flags/{experiment_id}/viability_tracking.csv
Columns:
    - snip_id
    - embryo_id
    - dead_flag2
    - dead_inflection_time_int
```

### Snakemake Rules:
```python
rule unet_segment:
    input: processed_images/stitched_FF/{experiment_id}/
    output:
        masks=segmentation/auxiliary_masks/{experiment_id}/{embryo,viability,yolk,focus,bubbles}/,
        metrics=segmentation/auxiliary_masks/{experiment_id}/viability_metrics.csv  # NEW
    run: from data_pipeline.segmentation.unet.inference import run_all_models

rule qc_viability:
    input:
        tracking_csv=segmentation/embryo_tracking/{experiment_id}/tracking_table.csv,
        viability_metrics=segmentation/auxiliary_masks/{experiment_id}/viability_metrics.csv
    output: quality_control_flags/{experiment_id}/viability_tracking.csv
    run: from data_pipeline.quality_control.viability_tracking_qc import compute_dead_flag2_persistence
```

### Pros:
- ✅ Minimal code changes
- ✅ Preserves existing naming/interfaces
- ✅ Faster to implement

### Cons:
- ❌ Still split across multiple steps
- ❌ Confusing naming persists (`fraction_alive` vs biological reality)
- ❌ `dead_flag` vs `dead_flag2` confusion remains
- ❌ Harder to understand data flow

---

## Option 3: Hybrid - Compute in Build02B, Consolidate Flags in QC

### Structure:
```python
# segmentation/unet/inference.py
def compute_fraction_not_necrotic(emb_mask, via_mask):
    """Compute raw viability metric from masks"""
    pass

# quality_control/embryo_viability_qc.py
def compute_viability_qc(df):
    """
    Takes fraction_not_necrotic + tracking data, returns:
        - dead_flag (unified, persistence validated)
        - dead_inflection_time_int

    Note: Eliminates dead_flag vs dead_flag2 distinction
    """
    pass
```

### Data Output:
```
# After Build02B
segmentation/auxiliary_masks/{experiment_id}/auxiliary_metrics.csv
Columns:
    - image_id
    - embryo_id
    - fraction_not_necrotic  # Renamed from fraction_alive

# After QC
quality_control_flags/{experiment_id}/embryo_viability.csv
Columns:
    - snip_id
    - embryo_id
    - dead_flag                  # Single unified flag (no v2)
    - dead_inflection_time_int
```

### Snakemake Rules:
```python
rule unet_segment:
    input: processed_images/stitched_FF/{experiment_id}/
    output:
        masks=segmentation/auxiliary_masks/{experiment_id}/{type}/,
        metrics=segmentation/auxiliary_masks/{experiment_id}/auxiliary_metrics.csv
    run: from data_pipeline.segmentation.unet.inference import run_all_models

rule qc_viability:
    input:
        tracking_csv=segmentation/embryo_tracking/{experiment_id}/tracking_table.csv,
        auxiliary_metrics=segmentation/auxiliary_masks/{experiment_id}/auxiliary_metrics.csv
    output: quality_control_flags/{experiment_id}/embryo_viability.csv
    run: from data_pipeline.quality_control.embryo_viability_qc import compute_viability_qc
```

### Pros:
- ✅ Clear separation: raw metrics (Build02B) → QC flags (Build04)
- ✅ Better naming (`fraction_not_necrotic`, single `dead_flag`)
- ✅ Moderate refactor effort
- ✅ Clear which data is "feature" vs "QC flag"

### Cons:
- ⚠️ Still some code to update for renamed columns
- ⚠️ Intermediate complexity

---

## Option 4: Two-Stage Death Detection (Research-Friendly)

### Structure:
Keep intermediate outputs for research/debugging:

```python
# quality_control/embryo_viability_qc.py
def compute_raw_viability_features(df):
    """Stage 1: Compute fraction_not_necrotic from masks"""
    return df with ['fraction_not_necrotic']

def compute_death_flags(df):
    """Stage 2: Apply persistence validation"""
    return df with ['dead_flag', 'dead_inflection_time_int']
```

### Data Output:
```
computed_features/{experiment_id}/viability.csv
Columns:
    - snip_id
    - fraction_not_necrotic  # Raw feature

quality_control_flags/{experiment_id}/embryo_viability.csv
Columns:
    - snip_id
    - dead_flag
    - dead_inflection_time_int
```

### Snakemake Rules:
```python
rule compute_viability_features:
    input:
        sam2_masks=...,
        unet_via_masks=...
    output: computed_features/{experiment_id}/viability.csv
    run: from data_pipeline.snip_processing.embryo_features.viability import compute_raw_viability_features

rule qc_viability:
    input:
        viability_features=computed_features/{experiment_id}/viability.csv,
        tracking_csv=...
    output: quality_control_flags/{experiment_id}/embryo_viability.csv
    run: from data_pipeline.quality_control.embryo_viability_qc import compute_death_flags
```

### Pros:
- ✅ Explicit feature vs QC flag separation
- ✅ Intermediate outputs useful for research
- ✅ Matches document's "data vs metadata" distinction

### Cons:
- ❌ More output files
- ❌ Slightly more complex DAG
- ❌ Potential confusion about which file to use

---

## Recommendation: Option 1 (Unified Module)

### Rationale:
1. **Death detection is fundamentally QC, not feature extraction**
   - `fraction_not_necrotic` is just an intermediate metric
   - The actionable output is `dead_flag` (pass/fail decision)
   - No research value in keeping them separate

2. **Biological accuracy matters**
   - UNet viability mask detects **necrotic tissue** (cell death)
   - `fraction_alive` is misleading (alive ≠ not necrotic)
   - `fraction_not_necrotic` is scientifically accurate

3. **Single robust flag eliminates confusion**
   - No more `dead_flag` vs `dead_flag2`
   - Persistence validation is the RIGHT way, make it default
   - Simpler for users: "Is embryo dead? Check `dead_flag`"

4. **Cleaner code architecture**
   - All death detection logic in one file
   - Easier to test (one module, one test suite)
   - Easier to improve algorithm later

### Implementation Strategy:

1. **Create new unified module:**
   ```
   src/data_pipeline/quality_control/embryo_viability_qc.py
   ```

2. **Keep old modules during transition:**
   ```
   src/build/qc_utils.py (mark deprecated)
   src/data_pipeline/quality_control/death_detection.py (mark deprecated)
   ```

3. **Update incrementally:**
   - Week 1: Create new module, test alongside old
   - Week 2: Update Build03A/04 to use new module
   - Week 3: Deprecate old modules, update docs

4. **Backward compatibility shim:**
   ```python
   # In qc_utils.py
   def compute_fraction_alive(*args, **kwargs):
       warnings.warn("Use embryo_viability_qc.compute_fraction_not_necrotic", DeprecationWarning)
       return compute_fraction_not_necrotic(*args, **kwargs)
   ```

---

## Decision Matrix

| Criterion | Option 1 | Option 2 | Option 3 | Option 4 |
|-----------|----------|----------|----------|----------|
| Code clarity | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Biological accuracy | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Ease of testing | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Implementation speed | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Research flexibility | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Maintenance burden | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**Winner: Option 1** - Best long-term architecture for "boring, predictable code that works"

---

## Next Steps

1. Review this analysis
2. Choose preferred option (recommend Option 1)
3. Create detailed implementation plan
4. Update pipeline rules document with chosen architecture
