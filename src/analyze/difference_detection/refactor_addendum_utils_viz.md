# Refactor Addendum: Utils + Visualization

This addendum captures the revised organization for shared utilities and plotting.

## Utilities (src/analyze/utils/)

Goal: Move shared data preparation out of analysis folders to prevent circular
imports.

### File Structure

```
src/analyze/utils/
│
├── binning.py                         # Data prep (new)
│   ├── add_time_bins()                # Label rows (no aggregation)
│   ├── bin_embryos_by_time()          # Aggregate (mean per bin)
│   └── filter_binned_data()
│
└── splitting.py                       # Existing logic (keep)
```

### Action Items

- Extract binning logic from penetrance and trajectory_analysis into
  `binning.py`.
- Ensure `difference_detection` imports from `utils.binning`, not vice versa.

## Visualization (src/analyze/viz/ and trajectory_analysis/)

Goal: Organize plots by topic (generic vs trajectory) and topology (single vs
faceted).

### File Structure

```
src/analyze/
│
├── viz/plotting/
│   └── scatter_3d.py                  # Generic 3D scatter (PCA/UMAP)
│                                      # (moved from trajectory_analysis)
│
└── trajectory_analysis/viz/plotting/
    │
    ├── time_series.py                 # Single-panel trajectory plots
    │                                  # (moved from src/analyze/utils/plotting.py)
    │
    └── faceted/                       # Folder (renamed from faceted.py)
        ├── time_series.py             # Grid of trajectories
        │                              # (moved from utils/plotting_faceted.py)
        ├── proportions.py             # Grid of bar charts
        │                              # (extracted from old faceted.py)
        ├── shared.py                  # Shared helpers (colors, layout)
        └── __init__.py                # Exposes functions
```

### Action Items

- Rename folder: create `faceted/` directory inside `plotting/`.
- Split file: break the old `faceted.py` into `proportions.py` (bars) and
  `time_series.py` (lines).
- Move generic: move `plotting_3d.py` to `src/analyze/viz/plotting/scatter_3d.py`
  so it can be used without trajectory dependencies.
- Clean utils: empty out `src/analyze/utils/plotting.py` and move contents to
  `trajectory_analysis/viz/plotting/time_series.py`.
