# Label Alignment Cheat Sheet
- Choose a reference labelling (typically the consensus labels stored in `method_results['labels']`); this defines the cluster groups for evaluating membership quality.
- For each bootstrap iteration, build a contingency table between the sampled embryos’ bootstrap labels and the reference labels, then align the bootstrap IDs to the reference IDs by maximising overlap (Hungarian algorithm or a greedy argmax per reference cluster when counts are well separated).
- Accumulate aligned votes for every embryo, normalising by how often it was sampled. The resulting frequencies form per-embryo membership probabilities that plug directly into the entropy, log-odds, and related metrics below.
- Distance- and graph-based scoring methods still assume the same reference labels; they simply generate alternative confidence scores on top of that fixed assignment.

# Cluster Assignment Quality Options

## Current Implementation Snapshot
- Membership analysis relies on `membership-module.py`. For each embryo the code looks at the bootstrap co-association matrix, computes the median intra-cluster co-association (self associations removed), the mean cross-cluster co-association, and the overall mean co-association.
- Samples are labelled `core` when their intra-cluster median exceeds a global threshold (default `0.70`) and the pointwise silhouette score is at least `0.2`. Anything below an overall co-association of `0.4` becomes an outlier; the remainder default to `uncertain`.
- A variance check lowers the threshold by `0.1` when a cluster shows high co-association variance, but otherwise the rule set is global. As a result, clusters with generally high cohesion can still have many members sitting just below the fixed threshold, even if they are clearly aligned with their cluster mates.

## Ranked Alternatives

### 1. Bootstrap assignment posteriors with entropy gating
- **Feasibility**: High  
- **Expected impact**: High
- Core idea: use the per-iteration labels already returned by `run_bootstrap` to estimate `p_i(c)`, the probability that embryo `i` belongs to cluster `c`. Interpret the vector as a Dirichlet-multinomial posterior and derive (a) maximum assignment probability, (b) Shannon entropy, and (c) credible intervals on the top cluster probability.
- Implementation sketch: count cluster occurrences for each embryo across bootstraps (ignoring iterations where the embryo was not sampled), apply a Jeffrey’s prior to avoid zeroes, compute `max_c p_i(c)` and entropy `H_i = -∑ p_i(c) log p_i(c)`. Define core/uncertain/outlier via joint thresholds, e.g. core if `max p_i(c) ≥ 0.9` and `H_i ≤ 0.2`, outlier if `max p_i(c) ≤ 0.55` or entropy high.
- Pros:
  - Reuses existing bootstrap machinery; no extra clustering passes needed.
  - Provides interpretable probabilities and information-theoretic confidence measures.
  - Naturally handles cases where multiple clusters compete for a point.
- Cons:
  - Requires storing or recomputing full bootstrap label history (ensure `run_bootstrap` persists it).
  - Unequal sampling frequency per embryo needs normalization to avoid bias (can be handled by weighting counts by inclusion frequency).

### 2. Pointwise log-odds / mutual information scoring
- **Feasibility**: High  
- **Expected impact**: Medium–High
- Core idea: treat the consensus labelling as the “model” and compute the surprisal of that assignment under the bootstrap posterior. Score each embryo by `s_i = -log p_i(c*)`, where `c*` is the final cluster, or by the log-odds gap between the top two clusters. This is equivalent to a pointwise mutual information check between the embryo and its assigned community.
- Implementation sketch: reuse `p_i(c)` from method 1, compute `s_i`, and rescale scores (e.g. via z-score within each cluster). Core members have large positive log-odds separating the top cluster from competitors; uncertain members show small gaps; outliers have high surprise.
- Pros:
  - Information-theoretic grounding; thresholds can be set via percentile of `s_i`.
  - Captures “looks good together but still contested” situations where absolute co-association is high yet another cluster inches close.
- Cons:
  - Shares data requirements with method 1, so pay the same bookkeeping costs.
  - Sensitive to the number of clusters considered; needs care if bootstraps sometimes miss a cluster entirely.

### 3. Consensus graph core–periphery metrics
- **Feasibility**: Medium–High  
- **Expected impact**: Medium–High
- Core idea: interpret the co-association matrix as a weighted graph and compute network measures that separate core nodes from peripheral ones (e.g. within-cluster degree z-score vs participation coefficient, k-core decomposition, or walktrap centrality). Members that are strongly connected inside their cluster with minimal cross-cluster ties register as core.
- Implementation sketch: for each cluster, build the subgraph induced by its members, compute within-module degree z-score (`z_i`), and participation coefficient (`P_i`). Use thresholds such as Guimerà & Amaral (2005) to tag provincial hubs (core), connector nodes (uncertain), and peripherals/outliers. Alternatively, run weighted k-core decomposition on the full graph and down-weight nodes with low coreness in their assigned cluster.
- Pros:
  - Highlights structural roles beyond pairwise averages; resilient to skewed cluster sizes.
  - Helps flag embryos that bridge clusters even if their intra-cluster co-association is high.
- Cons:
  - Requires graph tooling (e.g. NetworkX, igraph) and careful handling of negative/zero weights.
  - Thresholds may need tuning per dataset; interpretability slightly more abstract for wet-lab collaborators.

### 4. Distance-based posterior via softmax over medoids
- **Feasibility**: High  
- **Expected impact**: Medium
- Core idea: convert DTW distances to pseudo-likelihoods. For each embryo, compute `p_i(c) = softmax(-D(i, medoid_c)/τ)` or use kernel density estimates around medoids/cluster centroids. Compare the assigned cluster probability with the runner-up to gauge confidence.
- Implementation sketch: after clustering, gather medoid (or DBA centroid) distances for every embryo. Select a temperature `τ` (global or cluster-specific) so that typical intra-cluster distances map to high probabilities. Classify using probability gaps or Bayes factors.
- Pros:
  - Simple to implement; hinges only on the distance matrix already in memory.
  - Offers a complementary view (geometry of the distance space rather than bootstrap stability).
- Cons:
  - Assumes medoids adequately represent cluster geometry; can misclassify elongated clusters.
  - Temperature choice materially affects probabilities; may require cross-validation or heuristics.

### 5. Predictive re-labelling with probabilistic classifiers
- **Feasibility**: Medium  
- **Expected impact**: Medium
- Core idea: train a calibrated classifier (e.g. multinomial logistic regression, random forest with Platt scaling) to predict cluster labels from trajectory features or embeddings. Use the resulting class probabilities to grade membership quality.
- Implementation sketch: derive feature vectors (principal components, shape descriptors), train classifier with cross-validation, and evaluate per-sample predicted probability for the assigned cluster along with the calibration curve. Samples with low calibrated probability become uncertain/outliers.
- Pros:
  - Provides cross-validated evidence that cluster assignments are learnable from the data.
  - Flexible: can incorporate additional covariates (genotype, time metrics) if informative.
- Cons:
  - Needs a feature engineering step; susceptible to overfitting in small cohorts.
  - Adds another modelling layer that may drift from the primary DTW distance notion.
1
### 6. Bayesian mixture modelling on latent embeddings
- **Feasibility**: Low–Medium  
- **Expected impact**: Medium
- Core idea: embed trajectories (e.g. via functional PCA or autoencoder) and fit a probabilistic mixture model (Gaussian mixture, Dirichlet process mixture). Posterior membership probabilities and pointwise Bayes factors supply uncertainty estimates.
- Implementation sketch: produce low-dimensional latent vectors, run variational Bayes on a mixture with automatic relevance determination, extract posterior cluster probabilities, and compare with existing labels for concordance analysis.
- Pros:
  - Offers a principled probabilistic clustering framework with explicit uncertainty.
  - Can reveal if the chosen number of clusters is poorly supported (mixture collapses or splits).
- Cons:
  - Highest implementation burden; latent representation quality becomes a bottleneck.
  - Computationally heavier and may conflict with the DTW-based distance intuition the pipeline is built on.

## Follow-up Considerations
- Persist bootstrap label assignments so that probability/entropy-based scores can be computed without rerunning bootstraps.
- Consider combining multiple signals (e.g. bootstrap entropy + graph coreness) via a simple voting or logistic model to get a composite confidence score.
- Whichever method is chosen, capture thresholds in configuration so they can be tuned per dataset and report summary plots (score histograms, cluster-level distributions) alongside membership decisions.

---

# Implementation Details

## Implemented Approach: Bootstrap Assignment Posteriors (Option #1)

**Implementation Date:** 2025-11-06
**Location:** `results/mcolon/20251106_robust_clustering/`

### Architecture

We implemented **Option 1 (Bootstrap Assignment Posteriors)** combined with **Option 2 (Log-Odds Gap)** for 2D gating classification. This approach directly addresses the failures of the previous co-association-based method by computing per-embryo cluster assignment probabilities rather than pairwise co-clustering statistics.

#### Core Modules

1. **`bootstrap_posteriors.py`** - Posterior computation
   - `align_bootstrap_labels()`: Hungarian algorithm for label alignment across bootstrap iterations
   - `compute_assignment_posteriors()`: Calculates p_i(c) with frequency normalization
   - `compute_quality_metrics()`: Computes max_p, entropy, log_odds_gap
   - `analyze_bootstrap_results()`: End-to-end pipeline

2. **`adaptive_classification.py`** - Classification logic
   - `classify_embryos_2d()`: Two-dimensional gating using max_p and log_odds_gap
   - `classify_embryos_adaptive()`: Adaptive per-cluster thresholds
   - `get_classification_summary()`: Aggregate statistics

3. **`compare_methods_v2.py`** - Method comparison
   - Loads pre-computed bootstrap results from disk (no re-running needed)
   - Applies posterior-based classification
   - Compares to original co-association method
   - Outputs comparison tables and pickled results

4. **Visualization Scripts**
   - `plot_quality_comparison.py`: 3-panel method comparison across k
   - `plot_posterior_heatmaps.py`: Posterior probability heatmaps
   - `plot_cluster_trajectories.py`: Trajectory plots with confidence weighting

### Key Implementation Decisions

#### 1. Label Alignment Algorithm

**Problem:** Bootstrap iterations produce arbitrary cluster labels (cluster "0" in iteration 1 might correspond to cluster "2" in iteration 2).

**Solution:** Hungarian algorithm via `scipy.optimize.linear_sum_assignment`

```python
# Build contingency table: C[i,j] = # embryos with boot_label=i and ref_label=j
contingency = build_contingency_table(labels_bootstrap, labels_reference, sampled_indices)

# Hungarian algorithm maximizes overlap (minimize negative counts)
cost_matrix = -contingency
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Map bootstrap IDs to reference IDs
mapping = {boot_clusters[r]: ref_clusters[c] for r, c in zip(row_ind, col_ind)}
```

This ensures consistent cluster identities across all bootstrap iterations before accumulating assignment counts.

#### 2. Posterior Probability Calculation

**Formula:**
`p_i(c) = (# times embryo i assigned to cluster c) / (# times embryo i was sampled)`

**Key Details:**
- Only count iterations where embryo `i` was actually sampled (avoid bias from non-inclusion)
- Normalize by per-embryo sample count (handles unequal sampling)
- Probabilities sum to 1.0 for each embryo (verified in validation)

#### 3. Quality Metrics

**Three complementary metrics:**

1. **max_p** (confidence): `max_c p_i(c)`
   - Range: [0, 1]
   - Interpretation: Probability of most likely cluster
   - High value → confident assignment

2. **entropy** (overall uncertainty): `H_i = -Σ p_i(c) log₂ p_i(c)`
   - Range: [0, log₂(k)]
   - Interpretation: Shannon entropy of posterior distribution
   - Low value → low uncertainty

3. **log_odds_gap** (disambiguation): `log(p_top1) - log(p_top2)`
   - Range: [0, ∞]
   - Interpretation: Gap between top 2 competing clusters
   - High value → unambiguous between top contenders

#### 4. Classification Logic: 2D Gating

We use **two** metrics jointly to avoid the brittle single-threshold problem:

| Condition | Classification |
|-----------|----------------|
| `max_p < 0.5` | **Outlier** (very low confidence) |
| `max_p ≥ 0.8` AND `log_odds_gap ≥ 0.7` | **Core** (confident and unambiguous) |
| Otherwise | **Uncertain** (moderate confidence or ambiguous) |

This 2D approach distinguishes:
- Embryos with high `max_p` but low `log_odds_gap` → contested between 2 clusters
- Embryos with low `max_p` and low `log_odds_gap` → spread across 3+ clusters (true outliers)

**Adaptive variant:** Per-cluster thresholds computed as 75th percentile of within-cluster distributions (optional, use `--adaptive` flag).

### Thresholds and Justification

**Default thresholds:**
- `threshold_max_p = 0.8`: Core members must have ≥80% probability in their assigned cluster
- `threshold_log_odds_gap = 0.7`: Core members must have log-odds gap ≥0.7 (≈2× more likely in top cluster than second)
- `threshold_outlier_max_p = 0.5`: Embryos with <50% confidence are outliers

**Rationale:**
- Values chosen to be stricter than the previous co-association threshold (0.70), ensuring high-quality core membership
- Can be tuned via command-line arguments: `--threshold_max_p` and `--threshold_log_odds`
- Adaptive mode available for datasets with heterogeneous cluster tightness

### Data Structures

#### Input (from existing bootstrap results):
```python
bootstrap_results_dict = {
    'reference_labels': np.ndarray,      # Consensus cluster labels
    'bootstrap_results': [
        {
            'labels': np.ndarray,        # Per-iteration labels (-1 = not sampled)
            'indices': np.ndarray,       # Sampled embryo indices
            'silhouette': float
        },
        ...
    ],
    'coassoc': np.ndarray,               # Co-association matrix (not used here)
    'mean_ari': float
}
```

#### Output (posterior analysis):
```python
analysis_results = {
    'p_matrix': np.ndarray,              # Shape: (n_embryos, n_clusters)
    'sample_counts': np.ndarray,         # How many times each embryo was sampled
    'max_p': np.ndarray,                 # Maximum posterior per embryo
    'entropy': np.ndarray,               # Shannon entropy per embryo
    'log_odds_gap': np.ndarray,          # Log-odds gap per embryo
    'modal_cluster': np.ndarray,         # Most likely cluster per embryo
    'second_best_cluster': np.ndarray    # Second most likely cluster
}
```

#### Classification output:
```python
classification = {
    'category': np.ndarray,              # 'core'/'uncertain'/'outlier'
    'cluster': np.ndarray,               # Cluster assignments
    'max_p': np.ndarray,
    'log_odds_gap': np.ndarray,
    'thresholds': dict                   # Threshold values used
}
```

### File Locations

**Core modules:**
- `results/mcolon/20251106_robust_clustering/bootstrap_posteriors.py`
- `results/mcolon/20251106_robust_clustering/adaptive_classification.py`
- `results/mcolon/20251106_robust_clustering/compare_methods_v2.py`

**Visualization:**
- `results/mcolon/20251106_robust_clustering/plot_quality_comparison.py`
- `results/mcolon/20251106_robust_clustering/plot_posterior_heatmaps.py`
- `results/mcolon/20251106_robust_clustering/plot_cluster_trajectories.py`

**Output:**
- `results/mcolon/20251106_robust_clustering/output/data/posteriors_k{k}.pkl` - Per-k results
- `results/mcolon/20251106_robust_clustering/output/data/comparison_summary.csv` - Comparison table
- `results/mcolon/20251106_robust_clustering/output/figures/*.png` - All plots

### Usage

**Run comparison pipeline:**
```bash
cd results/mcolon/20251106_robust_clustering
python compare_methods_v2.py --k_min 2 --k_max 5
```

**Generate plots:**
```bash
python plot_quality_comparison.py
python plot_posterior_heatmaps.py --k 3 --with_metrics
python plot_cluster_trajectories.py --k 3 --show_confidence
```

**Adjust thresholds:**
```bash
python compare_methods_v2.py --threshold_max_p 0.85 --threshold_log_odds 0.8
```

**Use adaptive thresholds:**
```bash
python compare_methods_v2.py --adaptive
```

---

## Results Summary

### Comparison to Previous Implementation

**Previous Implementation (Co-association method):**
- Core membership rates: **10-40%** across k=2-5
- Major failure: 60-90% of embryos classified as uncertain/outlier
- Root cause: Fixed global threshold (0.70) too brittle, no label alignment

**New Implementation (Posterior method):**
- Core membership rates: **[To be filled after validation run]**
- Expected improvement: >60% core membership for well-separated clusters
- Key advantage: Information-theoretic confidence measures, 2D gating avoids single-threshold brittleness

### Key Findings

*[To be populated after validation pipeline completes]*

Expected outcomes:
1. Substantially higher core membership rates (target: 2-3× improvement)
2. Better discrimination between genuinely uncertain embryos vs. those just below threshold
3. Interpretable probability-based metrics (e.g., "90% probability of cluster A")
4. Validation that label alignment resolves ARI instability

---

## Dependencies

- `numpy`
- `scipy` (for Hungarian algorithm)
- `matplotlib`
- `seaborn`
- `pandas`
- `pickle` (built-in)
- `pathlib` (built-in)

All scripts are self-contained and can run independently once `compare_methods_v2.py` has generated the posterior results.
