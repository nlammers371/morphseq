## Problem Statement

I want to infer and visualize the **time-varying phenotypic relationship structure** among a set of perturbation conditions (single genes and gene combinations) from morphology-based classification results.

Each condition is a node. The goal is **not necessarily to force a tree**, because the phenotypic relationships may allow:

- branching
- reconvergence
- transient phenotype emergence
- transient phenotype disappearance
- non-tree-like local structure

Instead, I want a representation that captures the **main evolving structure of the phenotype landscape over developmental time**.

A key requirement is that **time is ordered**, and the same condition should be tracked across time.

---

## Current Setting

I have multiple perturbation conditions, for example:

- wild type
- single perturbations
- double perturbations
- other gene combinations

For each developmental time point or time window, I can evaluate how distinguishable one condition is from another using classification-based metrics.

The main downstream goal is to build a **dynamic condition-level structure** over time, where:

- each node is a condition
- edge weights reflect phenotypic similarity or separability
- the graph evolves smoothly over time
- reconvergence is allowed
- the representation should preserve the major relational structure in a way that is interpretable

---

## Current Primitives

### 1. Condition set
Let there be \(N\) perturbation conditions.

These are the entities I want as the nodes of the dynamic graph.

---

### 2. Time axis
Let there be \(T\) developmental time points or time windows.

The output should respect temporal ordering.

---

### 3. Pairwise classification primitive
For each pair of conditions \(i,j\) and each time \(t\), I can compute a **pairwise distinguishability / separability score** from a one-vs-one classifier.

Examples include:

- AUROC
- balanced accuracy
- misclassification/confusion rate
- some transformed distance-from-chance measure

This gives a tensor of pairwise relationships over time:

\[
D_{ij}(t), \quad i,j \in \{1,\dots,N\}, \; t \in \{1,\dots,T\}
\]

Interpretation:
- low distinguishability = conditions are phenotypically similar at that time
- high distinguishability = conditions are phenotypically distinct at that time

This pairwise tensor is the **main evidence layer**.

---

### 4. Dynamic condition-level graph
At each time \(t\), I want a graph:

\[
G_t = (V, E_t)
\]

where:
- \(V\) = perturbation conditions
- edge weights are derived from the pairwise tensor

For example, edge weights may be:
- direct similarity / affinity
- direct separability
- some transformed distance

So the node is **not** a comparison.  
The node is a **condition**, and comparisons live on the edges.

---

## Main Conceptual Challenge

The main challenge is:

### How do I turn a time series of pairwise condition-condition relationships into a **global, temporally coherent structure**?

I want a representation that:

- summarizes the phenotype landscape at each time
- can be stitched across time
- does not arbitrarily rearrange the same nodes from frame to frame
- allows branching and reconvergence
- does not require hand-defining branch events for every pair

---

## Important Constraints

### 1. Tree structure may be too restrictive
A simple lineage tree may fail because:

- phenotypes may emerge and later disappear
- two conditions may diverge and later reconverge
- combinatorial perturbations may create transient or non-monotonic effects
- the phenotype landscape may be more graph-like than tree-like

So I do **not** want to assume a tree is the correct primitive.

---

### 2. I still want a low-dimensional interpretable structure
Even if the underlying object is a dynamic graph, I would like some kind of interpretable structure such as:

- a graph layout over time
- a low-dimensional embedding over time
- a smoothed 3D representation
- or another dynamic relational visualization

This representation does **not** have to be uniquely defined, but it should preserve the major evolving relationships.

---

### 3. Temporal coherence matters
If I embed or lay out the graph at each time point independently, the same condition may jump around arbitrarily between frames.

So any dynamic visualization method needs to enforce or encourage:

- smooth node trajectories over time
- preservation of relative neighborhood structure
- “mental map” stability across time

---

## Current Open Questions

### A. What is the best edge definition?
At each time point, should edges be defined from:

- direct one-vs-one pairwise similarity/separability
- a transformed confusion-based measure
- a smoothed kernel over pairwise distances
- a second-order similarity such as cosine similarity of relationship profiles

---

### B. What is the best global representation at each time?
Given an \(N \times N\) condition-condition relationship matrix at time \(t\), how should I represent it globally?

Options I have been considering:
- heat map / matrix view
- weighted graph
- 2D or 3D layout
- dynamic embedding
- kernel or diffusion-like representation

---

### C. How should the dynamic structure be stitched through time?
I want a method that takes the sequence of per-time relationship matrices/graphs and produces a coherent evolving structure.

The method should:
- track the same nodes over time
- allow smooth motion
- allow reconvergence
- avoid overcalling branches from transient events

---

### D. Should I define explicit branches at all?
One possibility is to avoid explicitly labeling branches at first, and instead represent the evolving structure directly as:

- a dynamic heat map
- a dynamic graph
- a dynamic low-dimensional embedding

Then, only later, if the structure supports it, derive branch-like summaries.

---

## Current Intuition / Likely Direction

My current intuition is:

1. The primary evidence object should be the **pairwise one-vs-one relationship tensor over time**.
2. The next primitive should be a **dynamic graph with one node per perturbation condition**.
3. A **time-varying heat map** of the \(N \times N\) condition matrix may be one of the cleanest first visualizations.
4. A useful next step may be a **temporally coherent graph or embedding**, where the same node is tracked over time and allowed to move smoothly.
5. The final goal is to capture the **main phenotypic relationship structure over time**, not necessarily to force a strict tree.

---

## What I Want Researched

I want methods from any field that address problems like this:

- dynamic weighted graphs with fixed node identity over time
- temporally coherent graph layouts / embeddings
- methods that preserve relative relationships while allowing time evolution
- approaches that allow branching and reconvergence
- ways to visualize or summarize evolving pairwise relationship matrices
- methods for dynamic manifold / graph structure when the node set is fixed but edges change over time
- any approaches from systems biology, network science, single-cell trajectory analysis, social networks, phylogenetics, time-varying graphs, or manifold learning that could be adapted here

I am especially interested in methods that can start from a **time series of condition-condition similarity or distance matrices** and produce:

- an interpretable dynamic graph
- a smooth 2D/3D structure
- or another principled visualization of evolving phenotypic relationships