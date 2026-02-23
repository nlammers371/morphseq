# New Trend Plots for DTW Clustering Analysis

## Overview

Two new comprehensive trend visualization plots have been added to **07b_dtw_trajectory_clustering.py** to provide clear visual understanding of how different trajectory groups progress through development.

---

## Plot 31: General Temporal Trends by Cluster

### Purpose
Shows the detailed temporal progression **within each cluster**, including individual trajectories, mean trends, variability, and statistical annotations.

### Visual Elements

**Layout:** Subplots (one panel per cluster, side-by-side)
- If k=2: Two panels
- If k=3: Three panels
- If k=4: Four panels

**Per-Panel Components:**

1. **Individual Trajectories** (Light gray lines, α=0.15)
   - Every embryo in the cluster plotted separately
   - Shows the "cloud" of similar trajectories
   - Helps visualize within-cluster variability

2. **Mean Trajectory** (Bold black line with markers)
   - Central tendency of the cluster
   - Thick line (3.5 pt) with circle markers for visibility
   - Easy to see the dominant pattern

3. **IQR Band** (Black shaded region, α=0.25)
   - Interquartile range (25th to 75th percentile)
   - More robust to outliers than ±1 SD
   - Shows where the "bulk" of embryos fall

4. **±1 SD Band** (Blue shaded region, α=0.15)
   - Standard deviation confidence region
   - Shows full spread of normal variation
   - Broader than IQR

5. **Early Window** (Cyan vertical band, α=0.15)
   - Marks the 44-50 hpf window
   - Where we measure "early" curvature for correlation test

6. **Late Window** (Red vertical band, α=0.15)
   - Marks the 80-100 hpf window
   - Where we measure "late" curvature for correlation test

7. **Title Annotations:**
   ```
   Cluster 0
   (42 embryos)
   WT=15, Het=18, Homo=9
   r=-0.84 (p=0.002)
   Anti-correlated
   ```
   - Cluster ID
   - Number of embryos
   - Genotype breakdown (count)
   - Pearson correlation coefficient
   - Permutation p-value
   - Interpretation (Anti-correlated/Correlated/Uncorrelated)

### Interpretation Guide

**What to look for:**

- **Steep positive slope in early window + steep negative in late window** = Anti-correlated (flip-flop)
- **Shallow slope throughout** = Uncorrelated (stable progression)
- **Consistent positive/negative slope** = Correlated (monotonic change)
- **Large IQR/SD bands** = High variability within cluster (may need more k)
- **Small IQR/SD bands** = Tight clustering (consistent progression pattern)

### Example Patterns

```
Cluster A (Anti-correlated):        Cluster B (Uncorrelated):
Early: ↑↑↑ (high)                   Across time: ➡ (stable)
Late:  ↓↓↓ (low)
r = -0.85                            r = +0.10
```

---

## Plot 32: Cluster Trajectories Overlay

### Purpose
Direct **side-by-side comparison** of how all clusters progress, allowing immediate visual assessment of differences between groups.

### Visual Elements

**Layout:** Single plot with all clusters overlaid

**Key Components:**

1. **Multiple Colored Lines** (One per cluster)
   - Cluster 0: Red
   - Cluster 1: Blue
   - Cluster 2: Green
   - Cluster 3: Purple
   - Each line is the **mean trajectory** of that cluster

2. **Confidence Bands** (±1 SD per cluster)
   - Same color as cluster line, lighter shade
   - Shows uncertainty in mean trajectory
   - Allows assessment of how much overlap exists

3. **Early Window** (Cyan vertical band, α=0.1)
   - Clear marker for measurement window
   - Easy to see where clusters diverge

4. **Late Window** (Red vertical band, α=0.1)
   - Clear marker for measurement window
   - Highlights how groups differ at endpoint

5. **Legend**
   - Shows cluster IDs with colors
   - Shows window labels

### Interpretation Guide

**What to look for:**

- **Trajectories crossing over** = Flip-flop pattern (early-high↔late-low exchange)
- **Parallel trajectories** = Different severity levels (not truly anti-correlated)
- **Large separation in early window** = Groups diverge early
- **Convergence in late window** = Groups merge later (suggests compensation)
- **Non-overlapping confidence bands** = Statistically distinct groups

### Use Cases

1. **Quickly assess if k=2 vs k=3 is adequate:**
   - k=2 with clear crossing pattern? → Flip-flop hypothesis confirmed
   - k=3 with one intermediate group? → More complex dynamics

2. **Identify functional categories:**
   - Rapid changers vs slow changers
   - High baseline vs low baseline
   - Early responders vs late responders

3. **Publication-ready visualization:**
   - Clean, interpretable even to non-specialists
   - Works well in main figure or supplement
   - No overlapping text annotations

---

## How These Plots Work Together

### Plot 22 (Scatter) → Plot 31 (Trends) → Plot 32 (Overlay)

| Aspect | Plot 22 | Plot 31 | Plot 32 |
|--------|---------|---------|---------|
| **View** | Early vs Late (2D) | Time series evolution | Time series comparison |
| **Scale** | All clusters in one space | Individual clusters | All clusters overlay |
| **Detail** | Correlation r, p-values | Individual trajectories visible | Mean trends only |
| **Best for** | Statistical evidence | Within-group variability | Between-group comparison |

---

## Configuration Parameters

These plots respect the DTW analysis configuration:

```python
EARLY_WINDOW = (44, 50)   # hpf
LATE_WINDOW = (80, 100)   # hpf
METRIC_NAME = 'normalized_baseline_deviation'
```

You can modify these in the script to focus on different developmental stages.

---

## Output Files

```
plots_07b/
├── plot_31_temporal_trends.png      # Side-by-side per-cluster trends
└── plot_32_trajectory_overlay.png   # All clusters overlaid for direct comparison
```

Both are high-resolution (150 dpi) PNG files suitable for publication.

---

## Example Output Scenarios

### Scenario 1: Clear Flip-Flop (k=2)

**Plot 31:**
- Cluster 0: High early (IQR ≈ 0.4), Low late (IQR ≈ 0.15)
- Cluster 1: Low early (IQR ≈ 0.15), High late (IQR ≈ 0.4)
- Both show crossing pattern between early and late windows

**Plot 32:**
- Two lines crossing clearly
- Minimal overlap in early/late windows
- Clear X-pattern

**Interpretation:** Flip-flop hypothesis supported ✓

---

### Scenario 2: Severity Gradation (k=3)

**Plot 31:**
- Cluster 0: Low throughout (stable, r ≈ 0.0)
- Cluster 1: Medium throughout (stable, r ≈ 0.1)
- Cluster 2: High throughout (stable, r ≈ 0.2)

**Plot 32:**
- Three parallel lines, no crossing
- Constant vertical separation
- Similar slopes

**Interpretation:** Not flip-flop, but severity gradient

---

### Scenario 3: Complex Dynamics (k=4)

**Plot 31:**
- Cluster 0: Anti-correlated (r = -0.8)
- Cluster 1: Anti-correlated opposite (r = -0.75)
- Cluster 2: Uncorrelated (r = 0.1)
- Cluster 3: Correlated (r = 0.6)

**Plot 32:**
- Two major crossing groups (0 & 1)
- One stable group (2)
- One consistently high group (3)

**Interpretation:** Multiple compensation strategies + some stable embryos

---

## Tips for Interpretation

1. **Check early window variability first**
   - Small IQR = tight progression
   - Large IQR = heterogeneous starting conditions

2. **Look at the crossing point**
   - Early (before 70 hpf) = early divergence
   - Late (after 90 hpf) = late divergence
   - Gradual = continuous compensation

3. **Compare genotype distribution** (in Plot 31 title)
   - Cluster with more Homo = likely true disease phenotype
   - Even mix = confounding variable or multiple mechanisms

4. **Use confidence bands to assess statistical power**
   - Narrow bands = high confidence
   - Wide bands = need more embryos for robust conclusions

5. **Cross-reference with Table 5**
   - Plot 31 shows visual pattern
   - Table 5 shows statistical significance
   - Both must agree for strong interpretation

---

## When to Use Which Plot

| Question | Plot |
|----------|------|
| Is the group truly anti-correlated? | 31 (see crossing + 22) |
| How different are the groups? | 32 (see separation) |
| How much variability is in group? | 31 (see IQR/SD bands) |
| When do groups diverge? | 32 (see x-axis alignment) |
| Are there hidden subgroups? | 31 (look for bimodal IQR) |
| Which group is the mutant phenotype? | Compare genotype breakdown in 31 |

---

## Summary

**Plot 31** = "What's happening **within** each group?"
**Plot 32** = "How do groups differ **from each other**?"

Together, they provide complete visualization of DTW-discovered trajectory dynamics.
