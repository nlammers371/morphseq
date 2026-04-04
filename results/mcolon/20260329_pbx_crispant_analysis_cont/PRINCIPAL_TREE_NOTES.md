# Principal Tree Fitting — Design Notes

## What we're building

An elastic principal tree fitted to the embryo-level 3D space-time cloud from
cosmological condensation, with embryo-level branch significance testing.

The core idea: rather than averaging per condition before graph fitting (which
loses embryo heterogeneity, incomplete penetrance, rescue structure), we fit
the tree to all individual embryo-timepoint observations directly.

---

## The space

For each observed (embryo e, time bin t):

    z_{e,t} = (x_{e,t}, y_{e,t}, α·t_hpf)

where α = t_weight scales the temporal axis relative to the spatial axes.
This 3D cloud is what ElPiGraph fits the principal tree to.

---

## Statistical design

The inferential unit is the **embryo**, not the observation.

1. Fit principal tree to the full 3D cloud (label-agnostic)
2. Identify branch nodes (degree ≥ 3)
3. BFS from each branch node → connected arms
4. Assign each embryo to an arm by majority vote across its projected observations
5. Build genotype × arm contingency table
6. Permute embryo genotype labels (NOT observation labels) → null distribution
7. Chi-square statistic; effect size = Cramér's V

Permuting observations instead of embryos would break the within-embryo
dependence structure and inflate significance.

---

## ElPiGraph parameters

| Parameter | Role | Effect of increasing |
|---|---|---|
| `n_nodes` | Tree complexity — how many nodes | More nodes → finer resolution, arms can track further into tails |
| `Lambda` | Elasticity — pulls nodes toward assigned data mean | Higher → nodes cluster near dense center, arms get pulled back in |
| `Mu` | Bending/branching penalty | Higher → straighter tree, fewer branches, arms less curvy |

### What we learned from the bifurcating trunk sweep

- **Too high Mu (0.1):** arms don't curve enough to follow the data
- **Too low Lambda (0.001):** nodes spread too freely, spurious cross-branch
  connections appear — branches that shouldn't be linked get joined
- **Too low Mu (0.01) alone:** better curvature but still some structural issues
- **Winner: n_nodes=25, Lambda=0.01, Mu=0.05** — clean trunk, arms reach
  the data endpoints, no spurious connections, single significant branch at
  the correct location (t≈7–8, label 0 vs label 1 on separate arms)

### Validated on synthetic bifurcating trunk (E_all_on)

Data: 80 embryos × 13 time bins, 2 labels, Y-shape with known fork at t≈6
Result: 1 significant branch (p=0.001, V≈0.56), labels cleanly separated

Results folder: `results/bifurcating_tree_testing/n25_lam001_mu005`

---

## Files

| File | Description |
|---|---|
| `trajectory_cosmology/principal_tree.py` | Core pipeline: cloud building, ElPiGraph fitting, projection, arm assignment, permutation test |
| `trajectory_cosmology/principal_tree_viz.py` | 2D schematics (t,x) and (t,y), branch bars, 3D static + rotating GIF |
| `10_principal_tree.py` | Driver script |
| `trajectory_cosmology/principal_graph.py` | Earlier MST-on-condition-means approach — kept as fast summary scaffold |

### Key CLI args for `10_principal_tree.py`

```bash
conda run -n segmentation_grounded_sam --no-capture-output python \
  results/mcolon/20260329_pbx_crispant_analysis_cont/10_principal_tree.py \
  --positions-npz <path>/condensed_positions.npz \
  --output-dir <out_dir> \
  --t-weight 3.0 \
  --n-nodes 25 \
  --lambda-elpi 0.01 \
  --mu-elpi 0.05 \
  --n-perm 1000
```

---

## Input data

| Dataset | Path | Shape |
|---|---|---|
| Synthetic Y (bifurcating trunk v5) | `results/bifurcating_trunk_v5/E_all_on/condensed_positions.npz` | 80 embryos × 13 bins, 2 labels |
| Real PBX (aligned UMAP bin4 perm500) | `results/pairwise_shrunk_condensation_aligned_umap_bin4_perm500/condensed_positions.npz` | 235 embryos × 27 bins, 5 conditions |

---

## Parameter sweep history (bifurcating trunk)

| Folder | n_nodes | Lambda | Mu | Verdict |
|---|---|---|---|---|
| `n10_lam001_mu010_v2` | 10 | 0.01 | 0.10 | Arms don't reach endpoints, overshoots in y |
| `n15_lam001_mu005` | 15 | 0.01 | 0.05 | Better but arms still short |
| `n15_lam001_mu001` | 15 | 0.01 | 0.01 | Good curvature, arms reach further |
| `n25_lam001_mu005` | 25 | 0.01 | 0.05 | **Best** — clean Y, arms track to endpoints |
| `n15_lam0005_mu001` | 15 | 0.005 | 0.01 | Spurious cross-branch connections |
| `n15_lam0001_mu001` | 15 | 0.001 | 0.01 | Worse — branches connect incorrectly |
