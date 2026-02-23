# B9D2 Analysis - Handoff Part 2

**Date:** 2026-01-05
**Status:** Phase 1 (Data Generation) Complete - Ready for Phase 2 (Plotting) + Additional Control

---

## 🎯 What's Been Completed

### Phase 1: Data Generation Scripts (✅ COMPLETE)

Three analysis scripts have been created that generate classification data:

1. **`b9d2_hta_analysis.py`** - HTA phenotype analysis
2. **`b9d2_ce_analysis.py`** - CE phenotype analysis
3. **`control_controls_analysis.py`** - Control validation analysis

**All scripts follow the same pattern:**
- Load b9d2 data from `results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv`
- Map cluster_categories: `unlabeled` → `Not_Penetrant`, `wildtype` → `Not_Penetrant`
- Run 3 comparisons × 3 feature types = 9 classifications per script
- Features: curvature, length, embedding (z_mu_b)
- Output: CSV files organized by comparison/feature + summary.csv

### Key Implementation Details

**Critical Data Preprocessing:**
```python
# Map cluster categories
df['cluster_categories'] = df['cluster_categories'].replace({
    'unlabeled': 'Not_Penetrant',
    'wildtype': 'Not_Penetrant'
})

# Non-penetrant hets = heterozygous + Not_Penetrant
nonpen_het_ids = df[
    (df['genotype'] == 'b9d2_heterozygous') &
    (df['cluster_categories'] == 'Not_Penetrant')
]['embryo_id'].unique().tolist()
```

**Important:** `experiment_id` is an **integer** (20251125), not a string ('20251125')

---

## 📂 Current File Structure

```
results/mcolon/20260105_refined_embedding_and_metric_classification/
├── b9d2_hta_analysis.py              ✅ Data generation (9 classifications)
├── b9d2_ce_analysis.py                ✅ Data generation (9 classifications)
├── control_controls_analysis.py       ✅ Data generation (9 classifications)
├── HANDOFF_PT2.md                     📄 This file
├── output/
│   ├── b9d2_hta/                      (pending - not yet run)
│   ├── b9d2_ce/                       (pending - not yet run)
│   └── control_controls/              (pending - not yet run)
└── utils/                             ✅ Plotting utilities (already exist)
    ├── preprocessing.py
    ├── plotting_functions.py
    └── plotting_layouts.py
```

---

## 🎬 Next Steps (In Order)

### Step 1: Add Another Control to `control_controls_analysis.py`

**User Request:** "I'm going to add another control to this control controls"

**What to do:**
1. Ask the user which control comparison they want to add
2. Update the `define_groups()` function to include the new comparison
3. Update the `comparisons` dictionary with the new group IDs
4. The rest of the script will automatically handle it (loop over comparisons)

**Likely candidates for additional controls:**
- Pair-specific comparisons (e.g., pair_2 het vs pair_8 het)
- Additional negative controls (e.g., split WTs within same pair)
- Cross-experiment validation (20251121 vs 20251125)

**Pattern to follow:**
```python
# In define_groups() function, add new group definitions
new_group_ids = df[
    (df['pair'] == 'b9d2_pair_X') &
    (df['genotype'] == 'b9d2_Y') &
    ...
]['embryo_id'].unique().tolist()

# Then add to comparisons dict
comparisons = {
    'pair2_Het_vs_WT': (pair2_het_nonpen, pair2_wt, 'pair2_Het', 'pair2_WT'),
    'pair8_Het_vs_WT': (pair8_het_nonpen, pair8_wt, 'pair8_Het', 'pair8_WT'),
    'pair2_WT_vs_pair8_WT': (pair2_wt, pair8_wt, 'pair2_WT', 'pair8_WT'),
    'NEW_COMPARISON': (group1_ids, group2_ids, 'Group1Label', 'Group2Label'),  # ADD THIS
}
```

### Step 2: Run Data Generation Scripts

**Order of execution:**
```bash
cd results/mcolon/20260105_refined_embedding_and_metric_classification

# Run control validation first (fastest, most critical)
python control_controls_analysis.py  # ~15-20 min

# Then run phenotype analyses
python b9d2_hta_analysis.py          # ~15-20 min
python b9d2_ce_analysis.py           # ~15-20 min
```

**Expected outputs:**
- `output/control_controls/` - 9 CSV files (or more if additional control added)
- `output/b9d2_hta/` - 9 CSV files
- `output/b9d2_ce/` - 9 CSV files
- Each directory includes `summary.csv` with aggregate statistics

### Step 3: Create Phase 2 Plotting Scripts

**Three scripts to create:**

1. **`control_controls_plotting.py`** - Visualize control validation
2. **`b9d2_hta_plotting.py`** - Visualize HTA comparisons
3. **`b9d2_ce_plotting.py`** - Visualize CE comparisons

**Pattern to follow (see `cep290_genotype_comparison_analysis.py` for reference):**
```python
import pandas as pd
from pathlib import Path
from utils.preprocessing import prepare_auroc_data
from utils.plotting_layouts import create_feature_comparison_panels

# Load saved CSV files
OUTPUT_DIR = Path(__file__).parent / "output" / "control_controls"
curv_csv = OUTPUT_DIR / "pair2_het_vs_wt" / "classification_curvature.csv"
df_curv = pd.read_csv(curv_csv)

# Prepare for plotting
auroc_data = prepare_auroc_data(df_curv)

# Create plots using existing utilities
# See utils/plotting_layouts.py for available functions
```

**Plotting options available:**
- `create_feature_comparison_panels()` - 1x3 panel (curvature | length | embedding)
- `plot_multiple_aurocs()` - Overlay multiple comparisons
- `create_three_panel_comparison()` - Full 3-panel figure (AUROC + divergence + trajectories)

### Step 4: Initial Plot Set (Start Simple)

**For each phenotype, create 2 plots:**

1. **Feature comparison panel** - Shows all 3 features side-by-side
   - Example: `control_controls_feature_comparison.png`
   - Uses: `create_feature_comparison_panels()`

2. **Comparison overlay** - Shows all 3 comparisons overlaid for embedding
   - Example: `control_controls_embedding_overlay.png`
   - Uses: `plot_multiple_aurocs()`

**After initial plots are working, iterate to add:**
- Divergence panels
- Individual trajectory plots
- Custom layouts per user request

---

## 📋 Reference Files

### Templates to Copy From:
- **`cep290_genotype_comparison_analysis.py`** - Best example of complete analysis
- **Line 167-196**: Shows how to use `run_comparison_with_features()`
- **Line 227-254**: Shows how to create feature comparison panels

### Key Functions Available:

**From `utils/preprocessing.py`:**
- `prepare_auroc_data()` - Adds significance flags
- `smooth_divergence()` - Gaussian smoothing for divergence
- `smooth_trajectories()` - Per-embryo smoothing

**From `utils/plotting_functions.py`:**
- `plot_auroc_with_null()` - Single AUROC with null distribution
- `plot_multiple_aurocs()` - Multiple AUROCs overlaid
- `plot_divergence_timecourse()` - Metric divergence over time

**From `utils/plotting_layouts.py`:**
- `create_auroc_only_figure()` - Simple AUROC plot wrapper
- `create_feature_comparison_panels()` - 1x3 feature comparison
- `create_three_panel_comparison()` - Full 3-panel figure

---

## 🔍 Data Schema

### CSV Output Format (from compare_groups):

**classification_*.csv columns:**
- `time_bin` - Time bin start (hpf)
- `time_bin_center` - Bin midpoint (preferred for plotting)
- `auroc_observed` - Observed AUROC value
- `auroc_null_mean` - Null distribution mean (permutation test)
- `auroc_null_std` - Null distribution standard deviation
- `pval` - P-value from permutation test
- `n_positive` - Sample size in group 1
- `n_negative` - Sample size in group 2
- `positive_class` - Group 1 label
- `negative_class` - Group 2 label

**After `prepare_auroc_data()`:**
- `is_significant_05` - Boolean flag for p < 0.05
- `is_significant_01` - Boolean flag for p < 0.01

---

## ⚠️ Common Issues & Solutions

### Issue 1: "No module named 'analyze.difference_detection'"
**Solution:** Scripts include path setup at top:
```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

### Issue 2: experiment_id filter returns 0 rows
**Solution:** experiment_id is **integer**, not string
```python
df[df['experiment_id'] == 20251125]  # ✅ Correct
df[df['experiment_id'] == '20251125']  # ❌ Wrong
```

### Issue 3: "Not enough embryos" warnings
**Solution:** Check group definitions and phenotype exclusions
```python
print(f"Group size: {len(group_ids)}")  # Must be >= 3
```

### Issue 4: Plotting scripts can't find CSV files
**Solution:** Ensure data generation scripts completed successfully
```bash
ls -la output/control_controls/pair2_het_vs_wt/
# Should see: classification_curvature.csv, classification_length.csv, classification_embedding.csv
```

---

## 🎨 Plotting Style Guidelines

**Color scheme for comparisons:**
```python
comparison_colors = {
    'Phenotype_vs_NonPenHet': '#D32F2F',  # Red (primary comparison)
    'Phenotype_vs_WT': '#9467BD',         # Purple (secondary)
    'NonPenHet_vs_WT': '#888888',         # Gray (baseline)
}
```

**Line styles:**
```python
comparison_styles = {
    'Primary': '-',      # Solid
    'Secondary': '-',    # Solid
    'Baseline': '--',    # Dashed (weaker signal expected)
}
```

**Significance markers:**
- Green filled circles (●) = p < 0.05
- Gray open circles (○) = p >= 0.05
- Reference line at AUROC = 0.5 (chance)

---

## 📊 Expected Results (For Validation)

### control_controls_analysis.py:

| Comparison | Expected AUROC | Expected p-value | Interpretation |
|------------|----------------|------------------|----------------|
| pair2_Het_vs_WT | > 0.6 | < 0.05 | Het phenotype detected |
| pair8_Het_vs_WT | > 0.6 | < 0.05 | Het phenotype detected |
| pair2_WT_vs_pair8_WT | ~ 0.5 | > 0.05 | PASS (no difference) |

**Validation logic:**
- If hets differ from WT **AND** WTs don't differ from each other → het signal is REAL
- If WTs differ from each other → potential batch effects

### b9d2_hta_analysis.py:

| Comparison | Expected AUROC | Notes |
|------------|----------------|-------|
| HTA_vs_NonPenHets | > 0.7 | Late-onset (>60 hpf) |
| HTA_vs_WT | > 0.7 | Similar to NonPenHets |
| NonPenHets_vs_WT | ~ 0.5-0.6 | May show weak signal |

### b9d2_ce_analysis.py:

| Comparison | Expected AUROC | Notes |
|------------|----------------|-------|
| CE_vs_NonPenHets | > 0.8 | Strong signal, early onset |
| CE_vs_WT | > 0.9 | Very strong signal |
| NonPenHets_vs_WT | ~ 0.5-0.6 | May show weak cryptic signal |

---

## 🚀 Quick Start for Next Agent

**To continue this work:**

1. **Ask user about additional control:**
   - "Which control comparison would you like to add to control_controls_analysis.py?"
   - Update the script based on their response

2. **Run data generation:**
   ```bash
   python control_controls_analysis.py
   python b9d2_hta_analysis.py
   python b9d2_ce_analysis.py
   ```

3. **Create plotting scripts** (one at a time):
   - Start with `control_controls_plotting.py`
   - Use `cep290_genotype_comparison_analysis.py` as template
   - Load CSVs from `output/` directories
   - Use utilities from `utils/` (preprocessing, plotting_functions, plotting_layouts)

4. **Iterate on plots:**
   - Begin with 2 basic plots per phenotype
   - Add more plot types as user requests
   - User can review and request modifications

---

## 📝 Plan File Location

**Full implementation plan:** `/net/trapnell/vol1/home/mdcolon/.claude/plans/reactive-wobbling-quiche.md`

Contains detailed task descriptions, file structure, and execution order.

---

## ✅ Todo List Status

1. ✅ Create b9d2_hta_analysis.py script
2. ✅ Create b9d2_ce_analysis.py script
3. ✅ Create control_controls_analysis.py script
4. ⏳ Add additional control to control_controls_analysis.py (IN PROGRESS)
5. ⏸️ Test control_controls_analysis.py (PENDING)
6. ⏸️ Create control_controls_plotting.py (Phase 2)
7. ⏸️ Create b9d2_hta_plotting.py (Phase 2)
8. ⏸️ Create b9d2_ce_plotting.py (Phase 2)

---

## 🎓 Key Concepts to Remember

### Non-Penetrant Hets
- Genotype: `b9d2_heterozygous`
- Phenotype: `Not_Penetrant` (after mapping from 'unlabeled')
- NOT in any phenotype list (CE, HTA, BA_rescue)
- These are the **critical control group** for validation

### Why Three Comparisons Per Phenotype?
1. **Phenotype vs NonPenHets** - Primary comparison (appropriate control)
2. **Phenotype vs WT** - Secondary comparison (strictest control)
3. **NonPenHets vs WT** - Baseline validation (tests if controls are equivalent)

### Why Three Feature Types?
1. **Curvature** - Overt morphological phenotype
2. **Length** - Gross morphology change
3. **Embedding** - May detect cryptic phenotypes before metrics

### Cryptic Phenotype Detection
- If embedding AUROC > metric AUROC at early timepoints → cryptic window
- Indicates VAE detects subtle shape changes before measurable morphology

---

## 🆘 If Something Breaks

**Check these in order:**

1. **Path issues?**
   ```python
   PROJECT_ROOT = Path(__file__).resolve().parents[3]
   print(f"PROJECT_ROOT: {PROJECT_ROOT}")
   ```

2. **Data not loading?**
   ```python
   print(f"Data path exists: {DATA_PATH.exists()}")
   print(f"Experiment IDs: {df['experiment_id'].unique()}")
   ```

3. **No groups found?**
   ```python
   print(f"Available pairs: {df['pair'].unique()}")
   print(f"Available genotypes: {df['genotype'].unique()}")
   ```

4. **Import errors?**
   ```bash
   cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
   python -c "from src.analyze.difference_detection.comparison import compare_groups"
   ```

---

**Good luck! The infrastructure is solid - just need to add the control and create plotting scripts. 🚀**
