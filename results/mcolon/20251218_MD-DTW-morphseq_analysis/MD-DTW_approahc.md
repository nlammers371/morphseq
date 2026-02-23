# MD-DTW Strategy for b9d2 Phenotype Analysis

## 1. The Problem: Coupled Phenotypic Divergence
The `b9d2` mutants exhibit two distinct phenotypic endpoints that share a common early developmental trajectory but diverge later in time. 

* **Phenotype A (BA - Body Axis):** Characterized by curvature deviations *without* severe shortening. (e.g., `b9d2_pair_5` line).
* **Phenotype B (CE - Convergent Extension):** Characterized by early curvature deviations that evolve into severe shortening (Convergent Extension defects) after ~32hpf. (e.g., `b9d2_pair_4` line).
Note there is a WT phenotype which is no phenotype (wildtypes shoudl make the msajjority of the population, at least incomparison to ohte r clusterts, should have minimal Wts. This is in part a study of penetrance. )


**Why MD-DTW?**
Univariate analysis is insufficient because:
1.  **Shared Features:** Both phenotypes share high curvature initially. Analyzing curvature alone might group them together incorrectly for too long.
2.  **Coupled Signal:** The distinguishing factor is the *relationship* between Length and Curvature over time. A "short" embryo is only a "CE mutant" if it follows a specific curvature history.
3.  **Heterochrony:** The onset of the CE defect (shortening) may vary slightly between embryos (e.g., 30hpf vs 34hpf). Standard Euclidean distance would penalize this temporal shift heavily. **Multivariate Dynamic Time Warping (MD-DTW)** allows us to align these physiological events elastically while considering both Length and Curvature simultaneously.

## 2. Expected Phenotypes & Hypothesis
We expect MD-DTW to separate the population into clusters based on the *shape* of their multivariate evolution:

* **Cluster 1 (WT-like):** Low Curvature, High Length growth.
* **Cluster 2 (BA / Curvy-Only):** High Curvature, Normal/High Length growth.
* **Cluster 3 (CE / Short-Curvy):** High Curvature $\rightarrow$ Transition $\rightarrow$ Stalled Length growth.

**Hypothesis:** Using MD-DTW on the joint vector $\vec{v}_t = [Z_{curve}, Z_{length}]$, we will detect the split between Cluster 2 and Cluster 3 earlier and more robustly than by thresholding single metrics at fixed time points.

## 3. The Goal: Clustering & Visualization
The immediate objective is to perform unsupervised clustering to recover the biological groups ("BA" vs "CE") without a priori labels.

### A. Selection Strategy
1.  Compute the $N \times N$ distance matrix using MD-DTW.
2.  Generate a **Dendrogram** (Hierarchical Clustering, likely Ward's linkage).
3.  **Manual K-Selection:** Visually inspect the dendrogram cut height to identify stable clusters. We look for the branch split that separates `b9d2_pair_5` (BA) from `b9d2_pair_4` (CE).

### B. Visualization of Trajectories
Once clusters are assigned, we validate them visually. Since we have 3 dimensions (Time, Length, Curvature), we cannot plot a simple 2D line.
* **Method 1: Parallel Coordinate Time-Series:** * Plot 1: Curvature vs. Time (Colored by Cluster ID).
    * Plot 2: Length vs. Time (Colored by Cluster ID).
    * *Look for:* Does the Red cluster (CE) consistently drop in Plot 2 while staying high in Plot 1?
* **Method 2: 2D Phase Plane Trajectories:**
    * X-axis: Length, Y-axis: Curvature.
    * Plot the path of each embryo as a line $\vec{p}(t)$.
    * *Look for:* Distinct "loops" or path diversions in the phase space.

## 4. Implementation & Tools
We will integrate this into the existing `morphseq` infrastructure, specifically modifying `src/analyze/trajectory_analysis/TRAJECTORY_ANALYSIS_README.md` to include multivariate capabilities.

**Tools:**
* **`tslearn`:** For calculating the Multivariate DTW soft-DTW or path metrics.
* **`scipy.cluster.hierarchy`:** For `linkage`, `dendrogram`, and `fcluster`.
* **`scikit-learn`:** For `StandardScaler` (Vital: Curvature and Length must be Z-scored to have equal weight in the DTW cost function).

**Infrastructure Integration:**
* **Input Data:** Leverage existing `pandas` dataframes of time-series. Convert to `numpy` array of shape `(N_embryos, T_timesteps, 2_features)`.
* **Output:** Save cluster labels to a `.csv` that can be fed into the `difference_detection` module for statistical testing.