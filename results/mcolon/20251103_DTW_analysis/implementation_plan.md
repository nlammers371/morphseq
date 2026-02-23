# TRAJECTORY CLUSTERING - EXPLORATION STRUCTURE

## DIRECTORY LAYOUT
```
code/
├── config.py              # Paths, parameters
├── io.py                  # save/load helpers
├── 0_dtw.py               # Distance matrix
├── 1_cluster.py           # K-medoids + bootstrap
├── 2_select_k.py          # Metrics to pick K
├── 3_membership.py        # Core/uncertain/outlier
├── 4_fit_models.py        # Mixed-effects per cluster
└── explore.py             # Jupyter-style experiments

output/
├── 0_dtw/
│   ├── data/
│   └── plots/
├── 1_cluster/
│   ├── data/
│   └── plots/
└── ...
```

## BUILD ORDER

1. **config.py** - Paths, random seed
2. **io.py** - `save_data(step, name, obj)`, `load_data(step, name)`, `save_plot(step, name, fig)`
3. **0_dtw.py** - `precompute_dtw(ts, dtw_func)`, `load_dtw()`
4. **1_cluster.py** - `baseline_k_medoids(D, K)`, `bootstrap_stability(D, labels, K)`
5. **2_select_k.py** - `evaluate_k(D, k_range)` (stability + silhouette)
6. **3_membership.py** - `classify_core_uncertain(D, labels, co_assoc)`
7. **4_fit_models.py** - `fit_mixed_effects(ts, t, core_mask)` per cluster
8. **explore.py** - Run steps ad-hoc, inspect, plot

## KEY CONSTRAINT
No pipelines/automation. Each script is standalone.
Use explore.py to call functions in sequence, inspect results, iterate.

Example:
```python
# explore.py
from code import io, dtw, cluster, select_k, membership, fit_models

# Step 0
D, meta = dtw.precompute_dtw(ts, my_dtw_func)
io.save_data(0, 'distances', D)

# Step 1
labels_k3 = cluster.baseline_k_medoids(D, 3)
boot = cluster.bootstrap_stability(D, labels_k3, 3)
io.save_data(1, 'labels_k3', labels_k3)
io.save_plot(1, 'ari_scores', plot_ari(boot))

# ... inspect, iterate, etc.
```