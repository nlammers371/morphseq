# Genotype Overlay Videos (Trace Only, Fixed Featured Embryos)

This folder contains trace-only NWDB talk genotype overlay animations generated with
`results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_curvature_animation.py`.

## Intent

- Keep the same **24-120 HPF** data window as the background-overlay version.
- Keep the standard curvature y-axis fixed at **0-1**.
- Animate only the **25-54 HPF** talk window with the same featured embryos.
- Remove all population background traces so the movie shows only the selected embryo trace and moving cursor.
- Let the trace-only plot x-axis auto-resolve from the rendered reference figure instead of forcing exact limits.
- Export a matching static single-trace PNG for each embryo, without a trace-end dot.

## Featured Embryos (locked)

- Wildtype: `20251205_F11_e01`
- Heterozygous: `20251205_H07_e01`
- Homozygous: `20251205_A03_e01`

## Command (20 seconds)

Rendered at 20 fps for 20 seconds:

```bash
CONDA_SOLVER=classic /net/trapnell/vol1/home/mdcolon/software/miniconda3/condabin/conda run -n segmentation_grounded_sam --no-capture-output python \
  results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_curvature_animation.py \
  --panel-by genotype \
  --genotypes wildtype,heterozygous,homozygous \
  --featured-embryo-ids 20251205_F11_e01,20251205_H07_e01,20251205_A03_e01 \
  --t-min 24 --t-max 120 \
  --cursor-min 25 --cursor-max 54 \
  --fps 20 --n-frames-out 400 \
  --plot-style trace_only \
  --figures-subdir genotype_overlay_video/hpf24_120_cursor25_54_20s_trace_only_fixed_20251205
```

## Outputs

Each genotype produces:

- `curvature_animation_<genotype>_<embryo_id>.mp4` (trace-only animation with moving cursor)
- `embryo_animation_<genotype>_<embryo_id>.mp4` (embryo snip movie synced to the same HPF cursor)
- `trace_only_static_<genotype>_<embryo_id>.png` (static single-trace plot)
