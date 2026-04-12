# Genotype Overlay Videos (4 HPF Bins, Poster Size)

This folder contains poster-oriented genotype overlay renders generated with
`results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_curvature_animation.py`.

## Rendering contract

- Native script figure size (`500x450` px; no custom resize override)
- Fixed embryos:
  - Wildtype: `20251205_F11_e01`
  - Heterozygous: `20251205_H07_e01`
  - Homozygous: `20251205_A03_e01`
- Background plots use **4 HPF** trend bins
- Poster-ready legend placement is the outside-legend PNG:
  - `background_unfaded_legend_outside_<genotype>_<embryo_id>.png`

## Command (15 seconds)

```bash
CONDA_SOLVER=classic /net/trapnell/vol1/home/mdcolon/software/miniconda3/condabin/conda run -n segmentation_grounded_sam --no-capture-output python \
  results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_curvature_animation.py \
  --panel-by genotype \
  --genotypes wildtype,heterozygous,homozygous \
  --featured-embryo-ids 20251205_F11_e01,20251205_H07_e01,20251205_A03_e01 \
  --t-min 24 --t-max 120 \
  --cursor-min 25 --cursor-max 54 \
  --fps 20 --n-frames-out 300 \
  --bin-width 4 \
  --plot-style background \
  --skip-embryo-video \
  --figures-subdir genotype_overlay_video/hpf24_120_cursor25_54_15s_fixed_20251205_4hpf_bins
```
