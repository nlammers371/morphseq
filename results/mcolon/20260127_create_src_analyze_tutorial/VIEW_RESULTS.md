# Quick Start: Viewing Tutorial Results

## 📊 Interactive 3D Visualizations (Script 5)

**Location**: `output/figures/05/`

### Open in Browser

```bash
# On local machine (after copying files):
open output/figures/05/05_Not_Penetrant_3d_spline.html
open output/figures/05/05_Low_to_High_3d_spline.html
open output/figures/05/05_High_to_Low_3d_spline.html
open output/figures/05/05_Intermediate_3d_spline.html
```

### What You'll See

Each HTML file contains an **interactive 3D Plotly plot** showing:
- **Raw data points**: Colored by experiment ID (20260122 vs 20260124)
- **Fitted spline**: Red trajectory line through the cluster
- **3D axes**:
  - X: `baseline_deviation_normalized` (body curvature)
  - Y: `total_length_um` (body length)
  - Z: `predicted_stage_hpf` (developmental time)

### How to Interact
- **Rotate**: Click and drag
- **Zoom**: Scroll
- **Pan**: Right-click and drag
- **Reset**: Double-click
- **Hover**: See data point details

---

## 📈 Classification Test Results (Script 6)

**Location**: `output/results/`

### Load in Python

```python
import pandas as pd

# One-vs-rest cluster validation
df_cluster_20260122 = pd.read_csv("output/results/20260122_clusterlabel_ovr.csv")
df_cluster_20260124 = pd.read_csv("output/results/20260124_clusterlabel_ovr.csv")

# Genotype comparisons
df_geno_20260122 = pd.read_csv("output/results/20260122_geno_crispant_vs_ab.csv")
df_geno_20260124 = pd.read_csv("output/results/20260124_geno_crispant_vs_ab.csv")

# Not Penetrant validation
df_not_pen = pd.read_csv("output/results/20260122_not_penetrant_crispant_vs_ab.csv")
```

### Key Columns

- `group`: Cluster or genotype being tested
- `reference`: What it's compared against ("rest" or other genotype)
- `bin_start_hpf`, `bin_end_hpf`: Time window
- `auroc`: Discriminability metric (0.5 = random, 1.0 = perfect)
- `p_value`: Statistical significance from permutation test
- `n_group`, `n_reference`: Sample sizes

### Quick Visualization

```python
import matplotlib.pyplot as plt

# Plot AUROC over time for Intermediate cluster
df_int = df_cluster_20260124[df_cluster_20260124['group'] == 'Intermediate']

plt.figure(figsize=(10, 5))
plt.plot(df_int['bin_start_hpf'], df_int['auroc'], 'o-', label='Intermediate vs Rest')
plt.axhline(0.5, color='gray', linestyle='--', label='Random')
plt.xlabel('Time (hpf)')
plt.ylabel('AUROC')
plt.title('Cluster Separability Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📉 Spline Data (Script 5)

### Load Spline Coordinates

```python
import pandas as pd

# CSV format (easy to use)
splines = pd.read_csv("output/results/05_projection_splines_by_cluster.csv")

# View structure
print(splines.head())
print(f"Clusters: {splines['cluster_label'].unique()}")
```

### Spline DataFrame Structure

```
Columns:
- baseline_deviation_normalized: X coordinate
- total_length_um: Y coordinate
- baseline_deviation_normalized_se: X standard error
- total_length_um_se: Y standard error
- spline_point_index: Point number (0-199)
- cluster_label: Cluster name
```

### Load Pickle (Full Objects)

```python
import pickle

with open("output/results/05_projection_splines_by_cluster.pkl", "rb") as f:
    splines_full = pickle.load(f)
```

---

## 📁 Full Output Directory

```
output/
├── figures/
│   └── 05/
│       ├── 05_High_to_Low_3d_spline.html          ← Open in browser
│       ├── 05_Intermediate_3d_spline.html         ← Open in browser
│       ├── 05_Low_to_High_3d_spline.html          ← Open in browser
│       ├── 05_Not_Penetrant_3d_spline.html        ← Open in browser
│       └── 05_projection_splines_by_cluster.png   ← 2D overview
└── results/
    ├── 05_projection_splines_by_cluster.csv       ← Spline data
    ├── 05_projection_splines_by_cluster.pkl       ← Spline objects
    ├── 20260122_clusterlabel_ovr.csv              ← Cluster validation
    ├── 20260122_geno_crispant_vs_ab.csv           ← Genotype comparison
    ├── 20260122_not_penetrant_crispant_vs_ab.csv  ← Penetrance test
    ├── 20260124_clusterlabel_ovr.csv              ← Cluster validation
    └── 20260124_geno_crispant_vs_ab.csv           ← Genotype comparison
```

---

## 🔍 Key Findings at a Glance

### Script 5: Trajectory Shapes

**View the 3D plots to see**:
- **Not Penetrant**: Remains close to baseline (flat trajectory)
- **Low_to_High**: Starts low, increases over time
- **High_to_Low**: Starts high, decreases over time
- **Intermediate**: Mixed pattern between extremes

### Script 6: Statistical Validation

**Cluster Separability (AUROC)**:
- **Intermediate** (20260124): AUROC = 0.67-0.76 at late timepoints (significant)
- **Low_to_High** (20260124): AUROC = 0.63 at early timepoints (significant)
- **High_to_Low**: Variable, peaks at 32-36 hpf
- **Not Penetrant**: Modest separation (AUROC ~ 0.6-0.7)

**Penetrance**:
- Not all cep290_crispants show phenotype (incomplete penetrance)
- "Not Penetrant" crispants are indistinguishable from wildtype (AUROC ~ 0.5)

---

## 📝 Next Steps

1. **Explore 3D plots** → Understand phenotype trajectories
2. **Analyze AUROC trends** → Identify critical developmental windows
3. **Compare experiments** → Assess reproducibility across batches
4. **Biological interpretation** → Link morphology to ciliopathy pathology

---

## 📚 Documentation

See `TUTORIAL_EXECUTION_SUMMARY.md` for:
- Complete execution details
- Technical implementation
- Issues resolved
- Scientific interpretation
