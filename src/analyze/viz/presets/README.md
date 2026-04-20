# `analyze.viz.presets`

This directory contains project-specific color preset definitions built on top of
the generic `analyze.viz` styling API.

## What lives here

- reusable `ColorPreset` objects for MorphSeq analyses
- preset modules such as `morphseq.py`
- no plotting engine code

## What this is not

- not the generic visual styling engine
- not a hidden registry of names
- not a place for plot-specific defaults

## How to use it

Preset modules import the main package and export explicit preset objects:

```python
from analyze.viz.presets.morphseq import PBX_TALK
from analyze.viz.presets.nwdb import NWDB_PHENOTYPE_TALK
from analyze.viz.plotting import plot_feature_over_time

fig = plot_feature_over_time(
    df,
    features="curvature",
    color_by="genotype",
    color_preset=PBX_TALK,
)
```

The object passed at the call site is the provenance checkpoint. The figure
should be explainable from the script alone.

NWDB phenotype transition plots use:

```python
from analyze.viz.presets.nwdb import NWDB_PHENOTYPE_TALK
```

with colors:

- `High_to_Low` -> `#E76FA2`
- `Low_to_High` -> `#2FB7B0`
- `Not Penetrant` -> `#3A3A3A`
