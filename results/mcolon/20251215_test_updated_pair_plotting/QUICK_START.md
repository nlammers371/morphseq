# Quick Start - Refactored Pair Analysis Plotting

## TL;DR

**Old way:** 611 lines + 1255 lines of duplicated code
**New way:** 3 lines that work for ANY gene

---

## Before & After

### BEFORE (Old Code - 611 lines)
```python
# results/mcolon/20251113_curvature_pair_analysis/analyze_pairs.py
# Hardcoded for CEP290, matplotlib only, 611 lines

GENOTYPE_COLORS = {
    'cep290_wildtype': '#2E7D32',
    'cep290_heterozygous': '#FFA500',
    'cep290_homozygous': '#D32F2F',
}

fig = plot_all_pairs_overview(df, pairs, output_path=...)
# ... 600 more lines of matplotlib boilerplate
```

### AFTER (New Code - 3 lines)
```python
from src.analyze.trajectory_analysis import plot_pairs_overview

# Works for ANY gene (cep290, b9d2, tmem67, etc.)
fig = plot_pairs_overview(df, backend='both', output_path='output')
# Generates: PNG + interactive HTML
```

---

## Usage Examples

### 1. Overview Grid (All Pairs × Genotypes)
```python
from src.analyze.trajectory_analysis import plot_pairs_overview

fig = plot_pairs_overview(
    df[df['gene'] == 'cep290'],
    backend='plotly'  # Interactive HTML
)
fig.show()  # In Jupyter
```

### 2. Genotypes Overlaid by Pair
```python
from src.analyze.trajectory_analysis import plot_genotypes_by_pair

fig = plot_genotypes_by_pair(
    df[df['gene'] == 'b9d2'],
    backend='matplotlib'  # Static PNG
)
```

### 3. Single Genotype Across All Pairs
```python
from src.analyze.trajectory_analysis import plot_single_genotype_across_pairs

fig = plot_single_genotype_across_pairs(
    df,
    genotype='cep290_homozygous',
    backend='both'  # Both PNG and HTML
)
```

### 4. Generic Faceting (Any Columns)
```python
from src.analyze.trajectory_analysis import plot_trajectories_faceted

# Group by experiment × treatment
fig = plot_trajectories_faceted(
    df,
    row_by='experiment_id',
    col_by='treatment',
    overlay='genotype',
    color_by='genotype',
    backend='plotly'
)
```

---

## Key Features

✅ **Automatic Coloring** - Colors by genotype suffix (independent of gene name)
  - wildtype → green (#2E7D32)
  - heterozygous → orange (#FFA500)
  - homozygous → red (#D32F2F)

✅ **Dual Backends** - Same API, different outputs
  - `backend='plotly'` → Interactive HTML with hover
  - `backend='matplotlib'` → Static PNG for papers
  - `backend='both'` → Both at once

✅ **Auto-Detect** - No need to specify pairs/genotypes
  ```python
  plot_pairs_overview(df)  # Finds all pairs and genotypes automatically
  ```

✅ **Missing Pair Column** - Auto-creates fallback
  ```python
  # If 'pair' column missing, creates '{genotype}_unknown_pair'
  plot_pairs_overview(df)  # Still works!
  ```

---

## Test Status

✅ All 9 core tests pass

Run tests:
```bash
python results/mcolon/test_updated_pair_plotting/test_refactored_plotting.py
```

Test coverage:
- ✅ Suffix extraction (case-insensitive)
- ✅ Color mapping
- ✅ Matplotlib faceting
- ✅ Plotly faceting with hover
- ✅ Dual backend output
- ✅ All 3 pair-specific functions
- ✅ Unknown pair fallback

---

## Files to Know

| File | Purpose |
|------|---------|
| `src/analyze/trajectory_analysis/faceted_plotting.py` | Level 1 generic plotting (1000+ lines) |
| `src/analyze/trajectory_analysis/pair_analysis/plotting.py` | Level 2 pair-specific wrappers |
| `src/analyze/trajectory_analysis/genotype_styling.py` | Suffix-based color mapping |
| `src/analyze/trajectory_analysis/PLOTTING_README.md` | Complete documentation (400+ lines, 70+ examples) |
| `test_refactored_plotting.py` | Test suite (all pass ✅) |
| `HANDOFF.md` | Detailed handoff document |

---

## Next Steps

1. ✅ **Core implementation done** - All code written and tested
2. ⏳ **Migrate existing analyses** - Update `analyze_pairs.py` and `analyze_b9d2_pairs_unified.py`
3. ⏳ **Test against real data** - Verify CEP290, B9D2 outputs match original
4. ⏳ **Update documentation** - Add to main README, create tutorial notebook

See `HANDOFF.md` for detailed follow-up checklist.

---

## Architecture

```
Level 2 (Pair-Specific)
  plot_pairs_overview()
  plot_genotypes_by_pair()
  plot_single_genotype_across_pairs()
         ↓
Level 1 (Generic)
  plot_trajectories_faceted(row_by, col_by, overlay)
         ↓
Shared Components
  genotype_styling (colors, ordering)
  _prepare_facet_grid_data (data prep)
  data_utils (trajectories, binning)
```

---

## Common Mistakes to Avoid

❌ **Don't:** Hardcode genotype names
```python
# BAD
colors = {'cep290_wildtype': 'green', 'b9d2_wildtype': 'green'}

# GOOD
from src.analyze.trajectory_analysis import get_color_for_genotype
color = get_color_for_genotype('any_gene_wildtype')  # Always green
```

❌ **Don't:** Create separate functions per gene
```python
# BAD
def plot_cep290(df): ...
def plot_b9d2(df): ...
def plot_tmem67(df): ...

# GOOD
plot_pairs_overview(df)  # Works for all genes!
```

❌ **Don't:** Forget to specify backend
```python
# BAD - defaults to plotly, may be slow in terminal
fig = plot_pairs_overview(df)

# GOOD - explicit backend
fig = plot_pairs_overview(df, backend='matplotlib')  # Fast static output
```

---

## Quick Links

- **Full Documentation:** `src/analyze/trajectory_analysis/PLOTTING_README.md`
- **Test Suite:** `results/mcolon/test_updated_pair_plotting/test_refactored_plotting.py`
- **Handoff Details:** `results/mcolon/test_updated_pair_plotting/HANDOFF.md`
- **Implementation Plan:** `/net/trapnell/vol1/home/mdcolon/.claude/plans/tidy-weaving-quasar.md`

---

**Status:** ✅ Implementation Complete, Ready for Integration Testing
