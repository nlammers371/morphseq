# UOT MVP Implementation

This directory contains the Minimum Viable Product (MVP) implementation for Unbalanced Optimal Transport (UOT) analysis on real embryo mask data from the CEP290 dataset.

## Overview

The MVP consists of two Python scripts that validate the UOT pipeline on real data:

1. **`run_consecutive_frames.py`**: Tests UOT on consecutive frames within single embryos
2. **`run_cross_embryo_comparison.py`**: Compares different embryos at similar developmental stages

## Quick Start

```bash
# From the morphseq root directory
cd results/mcolon/20260121_uot-mvp

# Run consecutive frames analysis
python run_consecutive_frames.py

# Run cross-embryo comparison
python run_cross_embryo_comparison.py
```

## Scripts

### 1. Consecutive Frames Analysis (`run_consecutive_frames.py`)

**Purpose**: Validate that the UOT pipeline works correctly on consecutive frames within a single embryo.

**What it does**:
- Loads mask data from the CEP290 dataset CSV
- Tests multiple embryos (20251113_A05_e01, 20251113_E04_e01)
- Analyzes frame pairs at different time intervals:
  - Consecutive frames (1 frame = ~0.32 hours)
  - ~1 hour apart (3 frames = ~0.96 hours)
  - ~2 hours apart (6 frames = ~1.92 hours)
- Tests multiple starting frames (80, 100, 120) across development
- Generates visualizations and metrics for each pair

**Expected results**:
- Cost should increase with larger time gaps
- Consecutive frames should have low cost
- Created/destroyed mass should be small (< 10% of total)
- Velocity fields should be smooth and coherent

**Output structure**:
```
consecutive_frames/
├── 20251113_A05_e01/
│   ├── frames_80_to_81/
│   │   ├── creation_destruction.png
│   │   ├── velocity_field.png
│   │   ├── transport_spectrum.png
│   │   └── side_by_side_masks.png
│   ├── frames_80_to_83/
│   │   └── ...
│   ├── cost_over_intervals.png
│   ├── metrics_summary.csv
│   └── metrics_summary_stats.csv
└── 20251113_E04_e01/
    └── ...
```

### 2. Cross-Embryo Comparison (`run_cross_embryo_comparison.py`)

**Purpose**: Test UOT on morphological differences between different embryos at similar developmental stages.

**What it does**:
- Identifies frames near 48 hpf for embryos 20251113_A05 and 20251113_E04
- Selects the frame closest to 48.0 hpf for each embryo
- Runs UOT to compute "morphing" path between the two embryos
- Generates comparison visualizations highlighting differences

**Expected results**:
- Cost should be higher than consecutive frames of same embryo
- Creation/destruction highlights morphological differences
- Velocity field shows regional biases (e.g., head vs tail)
- Overlay shows spatial alignment and mismatch regions

**Output structure**:
```
cross_embryo/
└── A05_vs_E04_48hpf/
    ├── creation_destruction.png
    ├── velocity_field.png
    ├── transport_spectrum.png
    ├── side_by_side_comparison.png
    ├── overlay_with_difference.png
    ├── metrics_summary.txt
    └── metrics.csv
```

## UOT Configuration

Both scripts use the same standard configuration:

```python
UOTConfig(
    epsilon=1e-2,              # Entropy regularization
    marginal_relaxation=10.0,  # Mass flexibility (KL penalty)
    downsample_factor=4,       # Reduces 2189x1152 → ~547x288
    mass_mode="uniform",       # Uniform mass distribution
    max_support_points=5000,   # Maximum points for support sampling
    store_coupling=True,       # Save full coupling matrix
    random_seed=42,            # Reproducibility
)
```

**Key parameters**:
- **epsilon**: Controls entropy regularization (lower = more precise, slower)
- **marginal_relaxation**: Controls how much mass can be created/destroyed (lower = stricter balance)
- **downsample_factor**: Reduces computational cost while preserving structure
- **max_support_points**: Limits number of support points for tractability

## Visualizations

Each analysis generates 5 core visualizations:

### 1. Creation/Destruction Heatmaps
Shows where mass is created (appears) and destroyed (disappears) in the target frame.
- **Interpretation**: High values at boundaries indicate growth/shrinkage
- **Expected**: Low values for consecutive frames, higher for cross-embryo

### 2. Velocity Field Overlay
Quiver plot showing transport direction and magnitude overlaid on source mask.
- **Interpretation**: Arrows show how mass moves from source to target
- **Expected**: Smooth, coherent patterns (not random noise)

### 3. Transport Spectrum
Histogram of transport distances (how far mass travels).
- **Interpretation**: Peak near 0 = local deformation, long tail = global rearrangement
- **Expected**: Narrow for consecutive frames, broader for cross-embryo

### 4. Side-by-Side Masks
Direct comparison of source and target masks.
- **Interpretation**: Visual check that masks are properly aligned
- **Expected**: Similar morphology for consecutive frames

### 5. Overlay with Difference (cross-embryo only)
Color-coded overlay showing spatial overlap and differences.
- **Red**: Only in embryo A
- **Green**: Only in embryo B
- **Yellow**: Overlap
- **Interpretation**: Highlights morphological differences

## Metrics

The following metrics are computed for each pair:

- **cost**: Total UOT cost (lower = more similar)
- **transported_mass**: Mass moved from source to target
- **created_mass**: Mass that appears in target
- **destroyed_mass**: Mass that disappears from source
- **mean_transport_distance**: Average distance mass travels (pixels)
- **max_transport_distance**: Maximum transport distance
- **transport_mass_fraction**: Fraction of mass that is transported (vs created/destroyed)

**Mass conservation**:
```
transported_mass + created_mass ≈ total_target_mass
```

## Data Source

**Input CSV**: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`

**Required columns**:
- `embryo_id`: Unique embryo identifier
- `frame_index`: Frame number within embryo timeseries
- `mask_rle`: RLE-encoded binary mask
- `mask_height_px`, `mask_width_px`: Mask dimensions
- `predicted_stage_hpf`: Predicted developmental stage (hours post-fertilization)

## Validation Criteria

### Consecutive Frames Script

✅ Successfully loads masks from CSV for multiple embryos
✅ Computes UOT for frame pairs at 1-2 hour intervals
✅ Generates all 5 visualization types without errors
✅ Shows expected pattern: cost increases with time gap
✅ Velocity fields are smooth and coherent (not noisy)
✅ Created/destroyed mass is small for consecutive frames (< 10% of total)

### Cross-Embryo Script

✅ Successfully identifies embryos at ~48hpf
✅ Computes UOT between different embryos
✅ Cost is higher than consecutive frames of same embryo
✅ Visualizations highlight morphological differences
✅ Metrics are interpretable and biologically meaningful

## Parameter Tuning

If results look noisy or unstable, try adjusting:

**For faster but coarser results**:
- Increase `downsample_factor` to 8
- Increase `epsilon` to 1e-1
- Decrease `max_support_points` to 2000

**For stricter mass balance**:
- Decrease `marginal_relaxation` to 5.0 or 2.0

**For finer resolution**:
- Decrease `downsample_factor` to 2 (slower, more memory)
- Decrease `epsilon` to 1e-3 (more precise, slower convergence)

## Troubleshooting

### Memory errors
- Increase `downsample_factor`
- Decrease `max_support_points`

### Noisy velocity fields
- Increase `epsilon` (more smoothing)
- Check that masks are properly aligned
- Verify mask quality (no artifacts)

### High created/destroyed mass
- Expected for cross-embryo comparisons
- For consecutive frames, may indicate:
  - Large time gaps
  - Segmentation inconsistencies
  - Actual growth/regression

### NaN or Inf values
- Check mask quality (ensure non-empty)
- Verify RLE decoding is correct
- Check for zero-mass regions

## Future Extensions

After MVP validation, consider:

1. **Forward projection stability**: Test if transport maps can predict future frames
2. **Regional analysis**: Separate head/trunk/tail dynamics
3. **Parameter sweeps**: Systematically test epsilon and marginal_relaxation
4. **Batch processing**: Scale to all embryos in dataset
5. **Phenotype correlation**: Link transport metrics to genetic labels

## References

- **UOT backend**: `src/analyze/utils/optimal_transport/`
- **Visualization functions**: `src/analyze/optimal_transport_morphometrics/uot_masks/viz.py`
- **I/O utilities**: `src/analyze/optimal_transport_morphometrics/uot_masks/frame_mask_io.py`
- **Main UOT entry point**: `src/analyze/optimal_transport_morphometrics/uot_masks/run_transport.py`

## Contact

For questions or issues, contact the morphseq team.
