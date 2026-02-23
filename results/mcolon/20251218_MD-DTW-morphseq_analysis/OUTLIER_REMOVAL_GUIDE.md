# Outlier Removal Guide for MD-DTW Analysis

## Problem

Singleton outliers inflate k in hierarchical clustering:
```
k=2: {0: 93, 1: 1}  <- 1 singleton
k=3: {0: 91, 1: 2, 2: 1}  <- 2 + 1 singleton
k=4: {0: 91, 1: 1, 2: 1, 3: 1}  <- 3 singletons
k=5: {0: 10, 1: 81, 2: 1, 3: 1, 4: 1}  <- Main cluster finally splits at k=5
```

These 3 outlier embryos prevent you from seeing the true biological clusters (HTA vs CE) until k=5.

---

## Solution: Systematic Outlier Removal

### New Functions Added to `md_dtw_prototype.py`

#### 1. `identify_outliers()`
Identifies outlier embryos based on distance matrix.

**Methods:**
- **`percentile`** (recommended): Flag embryos with median distance > Nth percentile
  - Example: 95th percentile → flag top 5% most distant embryos
  - Adaptive to your data distribution

- **`mad`** (robust): Median Absolute Deviation
  - Flag embryos > median + 3×MAD
  - Most robust to extreme outliers

- **`median_distance`**: Manual threshold
  - Use when you know the expected distance scale

**Usage:**
```python
from md_dtw_prototype import identify_outliers

outlier_ids, inlier_ids, info = identify_outliers(
    D,                      # Distance matrix
    embryo_ids,             # Embryo ID list
    method='percentile',    # Detection method
    percentile=95,          # 95th percentile cutoff
    verbose=True
)

print(f"Outliers: {outlier_ids}")  # e.g., ['embryo_42', 'embryo_88', 'embryo_91']
```

#### 2. `remove_outliers_from_distance_matrix()`
Convenience wrapper that removes outliers and returns clean distance matrix.

**Usage:**
```python
from md_dtw_prototype import remove_outliers_from_distance_matrix

D_clean, embryo_ids_clean, info = remove_outliers_from_distance_matrix(
    D,
    embryo_ids,
    outlier_detection_method='percentile',
    outlier_percentile=95,
    verbose=True
)

# Now cluster on D_clean instead of D
```

---

## Testing Workflow

### Step 1: Test Outlier Detection Parameters

Run the test script to compare different methods:

```bash
cd results/mcolon/20251218_MD-DTW-morphseq_analysis

python test_outlier_detection.py --experiment 20251121
```

This will:
1. **Test multiple methods** (percentile 90, 95, 98, and MAD)
2. **Generate comparison plots** showing which embryos are flagged
3. **Compare clustering** before/after outlier removal
4. **Save results** to `output/outlier_testing/`

**Output files:**
- `outlier_detection_percentile_90.png` - Shows distribution + threshold for 90th %ile
- `outlier_detection_percentile_95.png` - Shows distribution + threshold for 95th %ile
- `outlier_detection_percentile_98.png` - Shows distribution + threshold for 98th %ile
- `outlier_detection_mad_(3x).png` - Shows distribution + threshold for MAD method
- `dendrogram_with_outliers.png` - Dendrogram showing singleton problem
- `dendrogram_without_outliers.png` - Dendrogram after outlier removal (clean clusters!)
- `clustering_comparison.png` - Side-by-side comparison of cluster sizes

### Step 2: Review Results

Inspect the plots to determine:
1. **Which outliers are real** (check their trajectories in multimetric plots)
2. **Optimal threshold** (95th percentile is usually good, but you can adjust)
3. **Cluster improvement** (does removing outliers reveal HTA vs CE split?)

### Step 3: Update Main Analysis

Once you've chosen your outlier removal parameters, update `run_analysis.py`:

```python
# In run_md_dtw_analysis() function, after computing distance matrix:

# Add outlier removal step
from md_dtw_prototype import remove_outliers_from_distance_matrix

print("\n" + "=" * 70)
print("Step 2.5: Removing Outliers")
print("=" * 70)

D_clean, embryo_ids_clean, outlier_info = remove_outliers_from_distance_matrix(
    D,
    embryo_ids,
    outlier_detection_method='percentile',
    outlier_percentile=95,  # Adjust based on test results
    verbose=verbose,
)

# Save outlier info
results['outliers'] = {
    'outlier_ids': [embryo_ids[i] for i in outlier_info['outlier_indices']],
    'inlier_ids': embryo_ids_clean,
    'method': outlier_info['method'],
    'threshold': outlier_info['threshold'],
}

# Replace D and embryo_ids with cleaned versions
results['D'] = D_clean
results['embryo_ids'] = embryo_ids_clean

# Continue with clustering on D_clean...
```

---

## Expected Outcome

**Before outlier removal:**
```
k=2: {0: 93, 1: 1}
k=3: {0: 91, 1: 2, 2: 1}
k=5: {0: 10, 1: 81, 2: 1, 3: 1, 4: 1}  <- Need k=5 to split main cluster
```

**After outlier removal (removing 3 outliers):**
```
k=2: {0: 45, 1: 46}  <- Clean split! HTA vs CE?
k=3: {0: 45, 1: 25, 2: 21}  <- Further sub-clustering if biologically relevant
```

Now you can visually inspect the multimetric plots for k=2 and k=3 to see if they correspond to HTA (curvy + normal length) vs CE (curvy → shortened).

---

## Rationale: Why This Works

### The Outlier Problem
Hierarchical clustering with average linkage will always separate the most distant points first. If you have 3 embryos that are extremely weird (high median distance to all others), the algorithm will:
1. Split them off as singletons at k=2, 3, 4
2. Only split the main cluster at k=5+

This masks the true biological structure.

### The Solution
By removing outliers BEFORE clustering:
1. **Cleans the distance matrix** to only include "normal" variation
2. **Clustering focuses on biological signal** (HTA vs CE) instead of outliers
3. **Lower k values** reveal true phenotypic clusters

### Choosing the Right Threshold
- **Too aggressive** (e.g., 90th percentile): Risk removing true biological variation
- **Too conservative** (e.g., 98th percentile): Risk keeping singleton outliers
- **Sweet spot**: 95th percentile usually works well

**Pro tip:** Use the test script to visualize the median distance distribution and see where the natural "break" is between inliers and outliers.

---

## Next Steps

1. **Run test script**: `python test_outlier_detection.py`
2. **Review plots** in `output/outlier_testing/`
3. **Identify the 3 outlier embryos** and check their trajectories
4. **Choose optimal threshold** (likely 95th percentile)
5. **Update `run_analysis.py`** to include outlier removal
6. **Re-run full analysis** and inspect multimetric plots for k=2, k=3
7. **Validate** that clusters match HTA vs CE phenotypes
