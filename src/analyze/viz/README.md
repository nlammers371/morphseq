# Generic Visualization Utilities

**Location:** `src/analyze/viz/`

## Overview

Domain-agnostic visualization tools for time series data. These plotting functions have **no dependencies** on trajectory analysis code, making them reusable for any time series visualization needs.

## Modules

### `plotting/feature_over_time.py` - Generic Time Series Plotting
- `plot_feature_over_time()`: Plot time series grouped by any categorical variable
- Supports multiple backends: Plotly (interactive) and Matplotlib (static)
- Flexible grouping and faceting options
- Error bands (SD, SE, IQR, MAD)
- Customizable colors and styling

## Key Features

### Backend Flexibility
- **Plotly**: Interactive HTML plots with hover tooltips
- **Matplotlib**: High-quality static PNG/PDF exports
- **Both**: Generate both formats simultaneously

### Grouping Options
- Group by any categorical column
- Multiple facets (rows and columns)
- Color by different grouping variable
- Automatic legend generation

### Statistical Overlays
- Central tendency: mean or median
- Error bands: SD, SE, IQR, MAD
- Individual traces with transparency
- Smooth trend lines

## Usage Examples

### Basic Time Series Plot
```python
from src.analyze.viz.plotting import plot_feature_over_time
import pandas as pd

# Simple grouped plot
fig = plot_feature_over_time(
    df,
    features='measurement',
    time_col='time',
    color_by='condition',
    backend='plotly',
    output_path='plot.html'
)
```

### Faceted Plot with Error Bands
```python
# Multiple facets with error bands
fig = plot_feature_over_time(
    df,
    features='metric_value',
    time_col='time_hpf',
    color_by='genotype',
    row_by='experiment',
    col_by='treatment',
    show_error_band=True,
    error_type='iqr',
    trend_statistic='median',
    backend='matplotlib',
    output_path='faceted_plot.png'
)
```

### Style Presets
```python
from src.analyze.viz.plotting import (
    default_style,
    paper_style,
    presentation_style,
)

base = default_style()
paper = paper_style()
talk = presentation_style()
```

Use these presets when you want a standard layout without spelling out every
trace and spacing knob. If you need to tune one thing, pass `style=` and then
override the field you care about.

### Custom Colors
```python
# Custom color palette
fig = plot_feature_over_time(
    df,
    features='value',
    time_col='time',
    color_by='category',
    color_palette=['#FF5733', '#33FF57', '#3357FF'],
    label_map={'a': 'A', 'b': 'B'},
    show_individual=True,
    backend='both',  # Generate both HTML and PNG
    output_path='custom_colors.html'
)
```

## Design Principles

1. **Domain Agnostic**: Works with any time series data, not specific to embryos or morphology
2. **No Trajectory Dependencies**: No imports from `trajectory_analysis/`
3. **Flexible**: Highly configurable grouping, coloring, and faceting
4. **Consistent API**: Similar interface to domain-specific plotting functions

## Architecture

```
viz/
├── README.md (this file)
├── __init__.py
└── plotting/
    ├── __init__.py
    └── feature_over_time.py  # Generic time series plotting
```

## Relationship to Other Modules

### Generic vs. Domain-Specific

**This Module (`viz/`)**: Generic, reusable visualization
- No domain knowledge
- Works with any DataFrame
- Basic statistical overlays
- Flexible grouping

**Domain-Specific (`trajectory_analysis/viz/`)**: Morphology-aware plotting
- Understands embryo trajectories
- Uses pair analysis logic
- Custom genotype coloring
- Morphology-specific faceting

### When to Use Each

Use `viz.plotting`:
- General-purpose time series visualization
- No trajectory-specific requirements
- Want clean, dependency-free plots

Use `trajectory_analysis.viz.plotting`:
- Embryo trajectory analysis
- Need pair-specific logic
- Want genotype-aware coloring
- Morphology-specific features

## Migration Notes

During the deep restructuring (Phases 1-2, commit d1f411ee), generic plotting functions were extracted from `trajectory_analysis/viz/plotting/` to this module.

### Before
```python
# Old location (domain-specific)
from src.analyze.trajectory_analysis.viz.plotting import plot_time_series
```

### After
```python
# Generic plotting
from src.analyze.viz.plotting import plot_feature_over_time

# Domain-specific (multi-metric trajectories)
from src.analyze.viz.plotting import plot_feature_over_time
```

### Example API
```python
from src.analyze.viz.plotting import plot_feature_over_time, presentation_style

fig = plot_feature_over_time(
    df,
    features="curvature",
    time_col="predicted_stage_hpf",
    color_by="genotype",
    color_lookup={
        "cep290_homozygous": "#B2182B",
        "cep290_wildtype": "#888888",
    },
    label_map={
        "cep290_homozygous": "Homozygous",
        "cep290_wildtype": "Wildtype",
    },
    style=presentation_style(),
    backend="matplotlib",
    ylim=(0, 1),
)
```

## Related Modules

- **`utils/timeseries/`**: Generic time series utilities (DTW, DBA, interpolation)
- **`trajectory_analysis/viz/`**: Domain-specific trajectory visualization
