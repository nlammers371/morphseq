# Final Analysis Table: Unified Features, QC, and Embeddings

**Date:** 2025-10-09
**Concept:** Progressive consolidation → single analysis-ready table per experiment

---

## The Idea: Three-Stage Consolidation

```
STAGE 1: Snip Features
    tracking_table.csv + spatial + shape + stage
        ↓
    consolidated_snip_features.csv

STAGE 2: Features + QC
    consolidated_snip_features.csv + ALL QC flags + use_embryo
        ↓
    snip_features_qc.csv

STAGE 3: Features + QC + Embeddings
    snip_features_qc.csv + embeddings (where use_embryo == True)
        ↓
    analysis_table.csv ⭐ (FINAL OUTPUT for analysis)
```

---

## Why This Design is Excellent

### ✅ Clear Progressive Flow
```
Raw Data → Features → QC → Embeddings → Analysis
```
Each stage adds information, never removes it

### ✅ Easy Filtering
```python
# All snips (including filtered out)
df = pd.read_csv('snip_features_qc.csv')

# Only QC-passed snips
df = df[df['use_embryo'] == True]

# Only snips with embeddings computed
df = df[df['embedding_computed'] == True]

# Analysis-ready (features + QC + embeddings)
df = pd.read_csv('analysis_table.csv')  # Pre-filtered!
```

### ✅ Embedding Concatenation is Simple
```python
# snip_features_qc.csv already has snip_id + features + QC
# Just add embedding columns: z0, z1, ..., z{dim-1}
# Add flag: embedding_computed = True for computed rows

# For snips with use_embryo == False:
#   - embedding_computed = False
#   - z0, z1, ... = NaN
```

### ✅ Single Analysis File
- Researchers get ONE file: `analysis_table.csv`
- Everything needed for analysis in one place
- Clear separation: intermediate files vs final output

---

## Proposed File Structure

```
computed_features/{experiment_id}/
    ├── spatial.csv
    ├── shape.csv
    ├── developmental_stage.csv
    ├── consolidated_snip_features.csv        # STAGE 1: Snip features only
    └── snip_features_qc.csv                  # STAGE 2: Features + QC flags

analysis_tables/{experiment_id}/              # NEW: Final output folder
    └── analysis_table.csv                    # STAGE 3: Features + QC + Embeddings ⭐
```

---

## Stage 1: consolidated_snip_features.csv

**Purpose:** All snip-level features merged

**Columns:**
```
From tracking_table.csv (43 columns):
    - snip_id, image_id, embryo_id, frame_index
    - bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
    - area_px
    - time_int
    - All raw metadata (microscope, well info, etc.)

From spatial.csv:
    - centroid_x, centroid_y, centroid_x_um, centroid_y_um
    - orientation_angle

From shape.csv:
    - area_um2, perimeter_px, perimeter_um
    - circularity, aspect_ratio

From developmental_stage.csv:
    - predicted_stage_hpf
    - stage_confidence
```

**Row count:** ALL snips (every embryo × frame)

**Use case:** Feature computation, intermediate analysis

---

## Stage 2: snip_features_qc.csv

**Purpose:** Features + QC flags + use_embryo decision

**Columns:**
```
From consolidated_snip_features.csv:
    - [All columns from Stage 1]

From consolidated_qc.csv:
    # Imaging quality (4 flags)
    - frame_flag
    - no_yolk_flag
    - focus_flag
    - bubble_flag

    # Viability (3 columns)
    - fraction_alive
    - dead_flag
    - dead_inflection_time_int

    # Segmentation quality (7 flags)
    - HIGH_SEGMENTATION_VAR_SNIP
    - MASK_ON_EDGE
    - DETECTION_FAILURE
    - OVERLAPPING_MASKS
    - LARGE_MASK
    - SMALL_MASK
    - DISCONTINUOUS_MASK

    # Tracking quality (3 columns)
    - speed_um_per_s
    - trajectory_smoothness_score
    - tracking_error_flag

    # Size validation (1 flag)
    - sa_outlier_flag

From use_embryo_flags.csv:
    - use_embryo                    # Boolean: pass all filters?
    - exclusion_reasons             # Comma-separated failed flags

NEW metadata columns:
    - embedding_computed            # Boolean: has embedding been computed?
```

**Row count:** ALL snips (every embryo × frame)

**Use case:**
- QC analysis (which snips failed which checks?)
- Filter decisions (why were snips excluded?)
- Intermediate for embedding generation

---

## Stage 3: analysis_table.csv ⭐

**Purpose:** Final analysis-ready table with embeddings

**Columns:**
```
From snip_features_qc.csv:
    - [All columns from Stage 2]

From embeddings:
    - z0, z1, z2, ..., z{dim-1}     # VAE latent dimensions
    - embedding_model_name           # Which model was used
    - embedding_computed = True      # Flag set to True for all rows here
```

**Row count:** ONLY `use_embryo == True` snips (filtered!)

**Location:** `analysis_tables/{experiment_id}/analysis_table.csv`

**Use case:**
- Primary file for all downstream analysis
- Phenotypic analysis, clustering, visualization
- Has everything: features + QC + embeddings

---

## Snakemake Rules

### Rule 1: Consolidate snip features
```python
rule consolidate_snip_features:
    input:
        tracking=segmentation/embryo_tracking/{experiment_id}/tracking_table.csv,
        spatial=computed_features/{experiment_id}/spatial.csv,
        shape=computed_features/{experiment_id}/shape.csv,
        stage=computed_features/{experiment_id}/developmental_stage.csv
    output:
        computed_features/{experiment_id}/consolidated_snip_features.csv
    run:
        from data_pipeline.feature_extraction.consolidation import consolidate_snip_features
```

### Rule 2: Merge features + QC
```python
rule merge_features_qc:
    input:
        features=computed_features/{experiment_id}/consolidated_snip_features.csv,
        qc_consolidated=quality_control_flags/{experiment_id}/consolidated_qc.csv,
        use_embryo=quality_control_flags/{experiment_id}/use_embryo_flags.csv
    output:
        computed_features/{experiment_id}/snip_features_qc.csv
    run:
        from data_pipeline.integration.merge_features_qc import merge_features_qc
```

### Rule 3: Generate embeddings (uses snip_features_qc.csv)
```python
rule generate_embeddings:
    input:
        snips=extracted_snips/{experiment_id}/,
        manifest=extracted_snips/{experiment_id}/snip_manifest.csv,
        features_qc=computed_features/{experiment_id}/snip_features_qc.csv  # Uses this for filtering!
    output:
        latent_embeddings/{model_name}/{experiment_id}_latents.csv
    run:
        from data_pipeline.embeddings.inference import ensure_embeddings
        # Internally filters to use_embryo == True
```

### Rule 4: Create final analysis table
```python
rule create_analysis_table:
    input:
        features_qc=computed_features/{experiment_id}/snip_features_qc.csv,
        embeddings=latent_embeddings/{model_name}/{experiment_id}_latents.csv
    output:
        analysis_tables/{experiment_id}/analysis_table.csv
    run:
        from data_pipeline.integration.create_analysis_table import create_analysis_table
```

---

## Implementation Details

### merge_features_qc.py
```python
def merge_features_qc(
    features_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    use_embryo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge features with QC flags and use_embryo decisions.

    Returns DataFrame with:
        - All features
        - All QC flags
        - use_embryo, exclusion_reasons
        - embedding_computed = False (placeholder for Stage 3)
    """
    # Merge on snip_id
    merged = features_df.merge(qc_df, on='snip_id', how='left')
    merged = merged.merge(use_embryo_df, on='snip_id', how='left')

    # Add placeholder for embedding status
    merged['embedding_computed'] = False
    merged['embedding_model_name'] = ''

    return merged
```

### create_analysis_table.py
```python
def create_analysis_table(
    features_qc_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    model_name: str
) -> pd.DataFrame:
    """
    Create final analysis table by merging features+QC with embeddings.

    Key steps:
        1. Filter features_qc_df to use_embryo == True
        2. Merge with embeddings on snip_id
        3. Set embedding_computed = True
        4. Set embedding_model_name
        5. Verify all expected embeddings are present

    Returns analysis-ready DataFrame.
    """
    # Filter to QC-passed snips
    analysis_df = features_qc_df[features_qc_df['use_embryo'] == True].copy()

    # Merge embeddings
    analysis_df = analysis_df.merge(embeddings_df, on='snip_id', how='left')

    # Update metadata
    analysis_df['embedding_computed'] = True
    analysis_df['embedding_model_name'] = model_name

    # Verify no missing embeddings
    missing_embeddings = analysis_df['z0'].isna().sum()
    if missing_embeddings > 0:
        raise ValueError(f"{missing_embeddings} snips missing embeddings!")

    return analysis_df
```

---

## Example Usage for Researchers

### Load final analysis table:
```python
import pandas as pd

# Single file has everything!
df = pd.read_csv('analysis_tables/20240915/analysis_table.csv')

# Ready for analysis
print(f"Rows: {len(df)}")  # Only use_embryo == True snips
print(f"Columns: {len(df.columns)}")  # ~100+ columns (features + QC + embeddings)

# All snips have embeddings
assert df['embedding_computed'].all()

# Can directly analyze
import umap
reducer = umap.UMAP()
embedding_cols = [c for c in df.columns if c.startswith('z')]
embedding = reducer.fit_transform(df[embedding_cols])
```

### Compare with intermediate files:
```python
# If you need ALL snips (including filtered out):
df_all = pd.read_csv('computed_features/20240915/snip_features_qc.csv')

# See which snips were filtered
df_filtered = df_all[df_all['use_embryo'] == False]
print(df_filtered['exclusion_reasons'].value_counts())

# Embeddings were only computed for use_embryo == True
df_with_embeddings = df_all[df_all['embedding_computed'] == True]
```

---

## Benefits of This Design

### ✅ Progressive Consolidation
- Each stage builds on previous
- Clear data provenance
- Easy to debug (check intermediate files)

### ✅ Clear Separation
```
computed_features/     → Working directory (intermediate files)
analysis_tables/       → Final output (analysis-ready)
```

### ✅ Flexible Analysis
```python
# Want all snips? Use snip_features_qc.csv
# Want analysis-ready? Use analysis_table.csv
# Want just embeddings? Use latent_embeddings/
```

### ✅ Easy Embedding Updates
```python
# Re-run embeddings with new model:
# 1. Generate new embeddings (from snip_features_qc.csv)
# 2. Create new analysis_table.csv with new embedding columns
# 3. Old analysis_table.csv preserved (versioned by model_name in path)
```

### ✅ Single Source for Analysis
- Researchers get ONE file per experiment
- Everything in one place
- No complex joins needed
- Clear documentation of what's included

---

## Alternative: Keep snip_features_qc.csv and update in place?

### Option A: Create separate analysis_table.csv (RECOMMENDED)
```
snip_features_qc.csv        # ALL snips, embedding_computed = False for most
analysis_table.csv          # ONLY use_embryo == True, all have embeddings
```
**Pros:**
- Clear separation (intermediate vs final)
- analysis_table.csv is always clean (no filtered snips)
- Can update embeddings without touching snip_features_qc.csv

**Cons:**
- One extra file

### Option B: Update snip_features_qc.csv with embeddings
```
snip_features_qc.csv        # ALL snips, some have embeddings
    - use_embryo == True → embeddings present
    - use_embryo == False → embeddings NaN
```
**Pros:**
- Single file
- Can see filtered snips alongside kept snips

**Cons:**
- Large file with many NaN columns
- Harder to use for analysis (must filter use_embryo)
- Mixing intermediate and final data

---

## Recommendation: Option A (Separate analysis_table.csv)

### File Structure:
```
computed_features/{experiment_id}/
    ├── consolidated_snip_features.csv    # Stage 1: Features only
    └── snip_features_qc.csv              # Stage 2: Features + QC (ALL snips)

analysis_tables/{experiment_id}/          # Final outputs
    └── analysis_table.csv                # Stage 3: Features + QC + Embeddings (filtered)
```

### Why?
1. **Clear separation** of intermediate vs final
2. **analysis_table.csv is clean** - only analysis-ready snips
3. **Easy to version** - different embedding models → different analysis_table.csv files
4. **Preserves full history** - snip_features_qc.csv shows ALL snips + why filtered

---

## Summary

### ✅ Three-Stage Consolidation:
1. **consolidated_snip_features.csv** - All features merged
2. **snip_features_qc.csv** - Features + QC flags (ALL snips)
3. **analysis_table.csv** - Features + QC + Embeddings (filtered) ⭐

### ✅ Benefits:
- Progressive data enrichment
- Clear separation (intermediate → final)
- Single file for analysis
- Easy filtering with `embedding_computed` flag
- Preserved provenance (can trace back through stages)

### ✅ Analysis Workflow:
```python
# Researchers use analysis_table.csv - that's it!
df = pd.read_csv('analysis_tables/20240915/analysis_table.csv')

# Everything needed for analysis is there:
# - Features (morphology, spatial, stage)
# - QC flags (for stratification)
# - Embeddings (for clustering, UMAP, etc.)
# - Metadata (experiment info, well info)
```

This design gives you a clear data flow with progressive consolidation and a single analysis-ready output file per experiment!
