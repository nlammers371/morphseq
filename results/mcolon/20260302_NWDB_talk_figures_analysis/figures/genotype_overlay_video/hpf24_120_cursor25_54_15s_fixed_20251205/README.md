# Genotype Overlay Videos (Fixed Featured Embryos)

This folder contains NWDB talk genotype overlay animations generated with
`results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_curvature_animation.py`.

## Intent

- Keep the **curvature trace context** on a long x-axis: **24–120 HPF** (`--t-min/--t-max`).
- Animate only an early “talk window” using the cursor and embryo snips:
  - **1–30 hours after 24 HPF** → **25–54 HPF** (`--cursor-min/--cursor-max`).
- Use the **same featured embryos** as the earlier reference outputs in:
  - `results/mcolon/20260302_NWDB_talk_figures_analysis/figures/genotype_overlays/`

## Featured Embryos (locked)

- Wildtype: `20251205_F11_e01`
- Heterozygous: `20251205_H07_e01`
- Homozygous: `20251205_A03_e01`

## Command (15 seconds)

Rendered at 20 fps for 15 seconds:

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python

"$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_curvature_animation.py \
  --panel-by genotype \
  --genotypes heterozygous,homozygous \
  --featured-embryo-ids 20251205_H07_e01,20251205_A03_e01 \
  --t-min 24 --t-max 120 \
  --cursor-min 25 --cursor-max 54 \
  --fps 20 --n-frames-out 300 \
  --figures-subdir genotype_overlay_video/hpf24_120_cursor25_54_15s_fixed_20251205

"$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_curvature_animation.py \
  --panel-by genotype \
  --genotypes wildtype \
  --featured-embryo-ids 20251205_F11_e01 \
  --t-min 24 --t-max 120 \
  --cursor-min 25 --cursor-max 54 \
  --fps 20 --n-frames-out 300 \
  --figures-subdir genotype_overlay_video/hpf24_120_cursor25_54_15s_fixed_20251205
```

## Outputs

Each genotype produces two MP4s:

- `curvature_animation_<genotype>_<embryo_id>.mp4` (trace + moving cursor)
- `embryo_animation_<genotype>_<embryo_id>.mp4` (snip frames synced to cursor HPF)

The script also writes static background PNGs:

- `background_unfaded_<genotype>_<embryo_id>.png`
- `background_unfaded_legend_outside_<genotype>_<embryo_id>.png`
