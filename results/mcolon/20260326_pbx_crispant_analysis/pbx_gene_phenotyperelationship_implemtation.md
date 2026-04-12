Here's a draft:

---

# Phenotypic Positioning via Pairwise Classification Graphs

## Biological Goal

Demonstrate that pbx1b/pbx4 double mutants occupy a distinct phenotypic space from wild-type, and that single mutants arrange along a severity gradient: pbx1b closest to WT, pbx4 intermediate, doubles furthest. Show this emerges from classification discriminability rather than any single morphological feature, and that the ordering is dynamic — potentially converging or diverging over developmental time.

## Computational Goal

Construct a per-embryo embedding in pairwise classification probability space. Each embryo's position is determined by how all binary classifiers see it across all genotype comparisons. Cluster and visualize embryos to reveal genotype-level structure, within-genotype heterogeneity, and temporal trajectory divergence.

## Genotypes

- wik-ab (wild-type)
- inj-ctrl
- pbx1b
- pbx4
- pbx1b_pbx4 (double)

## Method

### Step 1: Pairwise classification

Run `all_pairs` mode via MorphSeq classification API. For G=5 genotypes, produces K=10 pairwise binary classifiers per timepoint. For each embryo $i$ at timepoint $t$, collect predicted probability from each classifier $k$:

$$\mathbf{v}^{(i,t)} = [p_1^{(i,t)}, \ldots, p_{10}^{(i,t)}] \in \mathbb{R}^{10}$$

### Step 2: Snapshot clustering (single timepoint)

Pick one early and one late timepoint. Compute pairwise distance matrix $D^{(t)}_{ij} = d(\mathbf{v}^{(i,t)}, \mathbf{v}^{(j,t)})$. Embed via UMAP (2D). Color by genotype. Assess:

- Do genotypes form distinct clusters?
- Is the spatial ordering consistent with severity gradient?
- Which embryos are outliers relative to their genotype?

### Step 3: Joint 3D UMAP over all timepoints

Pool all embryo-timepoint pairs into a single point cloud in $\mathbb{R}^{10}$. Embed jointly in 3D UMAP. Each embryo becomes a trajectory (sequence of points connected over time). Visualize with:

- Color by genotype
- Line traces per embryo across timepoints
- Opacity or size gradient encoding time

### Step 4: Temporal alignment in 3D UMAP

Use the third UMAP dimension (or an explicit time axis) to align trajectories temporally. Assess:

- Do WT trajectories form a coherent bundle?
- When do double mutant trajectories diverge from WT?
- Does pbx1b stay near WT throughout, or does it diverge late?
- Is there a developmental window where discriminability spikes?

## Expected Outcome

A visualization showing embryos positioned by how classifiers perceive them, with genotype clusters ordered along a phenotypic severity axis that evolves over developmental time. Individual embryos that defy their genotype's typical trajectory become immediately visible.

## Open Questions

- Distance metric: cosine vs euclidean on probability vectors
- Whether to include on-diagonal (same-genotype) classifier scores or restrict to off-diagonal
- UMAP hyperparameters: n_neighbors, min_dist tuning for this data geometry
- Whether to weight timepoints equally or emphasize periods of maximal divergence

---

Want to refine anything or start on the implementation?