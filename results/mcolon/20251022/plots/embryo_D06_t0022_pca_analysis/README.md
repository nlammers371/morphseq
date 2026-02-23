# Embryo D06 t=0022 - PCA-based Curvature Analysis

**Date**: October 24, 2025  
**Embryo ID**: 20251017_part2_D06  
**Timepoint**: t=0022  
**Method**: PCA-based slicing with Gaussian smoothing

## Overview

This folder contains the curvature analysis results for a single embryo timepoint using the PCA-based centerline extraction method with Gaussian smoothing applied to the segmentation mask.

## Files

### Visualization
- **`embryo_D06_t0022_pca_smoothed_analysis.png`**: Main comprehensive figure showing:
  - Original image, smoothed mask overlay
  - Centerline extraction with principal axis
  - Smoothed spline representation
  - Full curvature profile along arc length

### Masks
- **`embryo_D06_t0022_mask_original.png`**: Original binary segmentation mask (113,639 pixels)
- **`embryo_D06_t0022_mask_smoothed.png`**: Gaussian-smoothed mask (σ=10, 113,316 pixels)

### Data Files
- **`embryo_D06_t0022_centerline_pca.npy`**: Raw centerline points (100 points, NumPy array)
- **`embryo_D06_t0022_spline_points.npy`**: Smoothed spline points (200 points, NumPy array)
- **`embryo_D06_t0022_curvature_profile.csv`**: Curvature data with columns:
  - `arc_length_pixels`: Position along centerline in pixels
  - `curvature_per_pixel`: Curvature value (1/pixels)
  
### Metadata
- **`embryo_D06_t0022_metadata.json`**: Complete analysis parameters and summary statistics

## Method Details

### PCA-based Slicing
The PCA-based slicing method is the most robust approach for elongated embryos:
1. Computes principal component analysis on mask coordinates
2. Projects points onto principal axis
3. Creates perpendicular slices along the axis
4. Computes centroid of each slice to get centerline points
5. Fits parametric spline for smooth curvature calculation

### Gaussian Smoothing
- **Sigma**: 10.0 pixels
- Applied to mask before centerline extraction
- Reduces noise and produces smoother curvature profiles

### Spline Fitting
- Smoothing parameter: 0.01
- Cubic spline (k=3)
- 200 evaluation points for curvature calculation

## Key Results

- **Total arc length**: 1,018.5 pixels
- **Mean curvature**: 0.005017 (1/pixels)
- **Standard deviation**: 0.007265 (1/pixels)
- **Max curvature**: 0.052671 (1/pixels)
- **Min curvature**: 0.000074 (1/pixels)

### Principal Axis
- Direction: [0.212, 0.977]
- Angle from horizontal: ~77.7 degrees

### Center of Mass
- Position: [529.6, 1147.7] pixels

## Source Data

- **Image**: `morphseq_playground/sam2_pipeline_files/raw_data_organized/20251017_part2/images/20251017_part2_D06/20251017_part2_D06_ch00_t0022.jpg`
- **Mask**: `morphseq_playground/sam2_pipeline_files/exported_masks/20251017_part2/masks/20251017_part2_D06_ch00_t0022_masks_emnum_1.png`

## Script

Analysis performed with:
```bash
conda run -n segmentation_grounded_sam python results/mcolon/20251022/extract_embryo_D06_t0022_pca_smoothed.py
```

## Notes

This embryo was selected for its nice curved shape, ideal for demonstrating the curvature analysis pipeline.
