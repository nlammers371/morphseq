## Phenotype transition homozygous assets

Organized outputs for the homozygous-background phenotype transition sequence.

### Featured embryos
- `Homozygous_Reference`: `20251205_A03_e01`
- `Low_to_High`: `20251106_H04_e01`
- `High_to_Low`: `20251113_A02_e01`

### Layout
- `20s/`: four curvature MP4s spanning `24` to `120` HPF over `20` seconds
- `15s/`: the same four MP4s over `15` seconds
- `static_plots/`: canonical presentation PNGs using the legacy phenotype-overlay static export style

### Source of truth
- Presentation stills come from `static_plots/`
- `background_unfaded_*.png` files inside `15s/` and `20s/` are video helper/debug exports, not the reference slide style

### Numbering
- `01`: homozygous background + `Low_to_High`
- `02`: homozygous background + `High_to_Low`
- `03`: homozygous background + `Low_to_High`, then `High_to_Low`
- `04`: homozygous background + `High_to_Low`, then `Low_to_High`

### Generation commands
```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_phenotype_genotype_transition.py
"$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_phenotype_genotype_transition.py --figures-subdir phenotype_transition_homozygous/15s --n-frames-out 300
"$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/make_nwdb_phenotype_transition_static_plots.py
```
