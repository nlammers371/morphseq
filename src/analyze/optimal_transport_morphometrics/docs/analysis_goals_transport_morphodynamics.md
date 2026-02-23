# Overarching Analysis Goals: Transport-Based Morphological Dynamics

This document defines the conceptual/biological end goals for transport-based morphodynamics analysis.
For the concrete MVP implementation plan and naming conventions, see `src/analyze/optimal_transport_morphometrics/docs/mvp_masterplan_uot.md`.

## Introduction: From Scalar Tracking to Transport Dynamics

The MorphSeq pipeline has achieved significant success by tracking morphological measurements over time, enabling us to capture developmental dynamics rather than just endpoint phenotypes. This temporal approach has proven powerful for detecting trajectory divergence and classifying mutant severity.

However, scalar morphometrics—even when tracked over time—have limitations. They struggle with subtle distributional changes (e.g., melanocyte migration patterns) and cannot tell us *where* a difference arises, only *that* it exists.

To address this, we propose a complementary paradigm: **Transport-Based Morphological Dynamics.**

*What if we treated development not as a series of snapshots, but as a mass transport problem?*

At its heart, this framework provides a **universal, pixel-based, interpretable metric** that gives us both **morphological distance** and **dynamics**. Whether you are looking at a jaw, a discrete eye, or a scattered cloud of melanocytes, optimal transport gives us a rigorous way to measure the difference between any two states at the pixel level.

Because this metric abstracts to anything you can segment, it unlocks two critical capabilities that standard morphometrics lack:

1. **Dynamics, not just Distance:** We don't just calculate how different two states are; we model the flow required to transform A into B. This captures the velocity and direction of morphological change, allowing us to cluster embryos based on their developmental trajectories.

2. **Per-Pixel Interpretability:** The metric is spatially resolved. It generates a pixel-level map showing exactly where mass is moving (or failing to move), allowing us to attribute specific phenotypic defects to precise anatomical regions.

---

## The Framework: Transport-Based Morphological Dynamics

This framework extends our current scalar tracking by treating development as a mass transport problem. 

At its core, optimal transport provides a **dual readout** for any structure you can segment:
1. **Morphological distance** via transport cost — a pixel-level measure of how different two states are
2. **Dynamics** via the transport path/flow field — how mass physically moves to get from A to B

This abstraction applies universally—whether you're analyzing a discrete organ like the eye, a contiguous region like the jaw, or a scattered cell population like melanocytes. The same metric gives you both *where* differences arise and *how* structures change over time.

### 1. Per-Structure Dynamics (The Raw Measurements)

For any morphological unit you can segment, optimal transport provides a rigorous metric for how that structure physically transforms between timepoints.

**Tail Curvature (Obvious Phenotype)**

*What it measures:* The cost to straighten or bend the tail segment between frames.

*Readout:* High transport cost localized specifically to the tail region, quantifying the mechanical "work" of the deformation.

**Melanocyte Migration (Subtle Phenotype)**

*What it measures:* The flow vectors required to map the distribution at $t$ to $t+1$.

*Readout:* Vector fields showing directional bias (or lack thereof). We can detect failed migration (disordered vectors) even if the total cell count is normal.

**Head/Jaw Growth (Mass Change)**

*What it measures:* The creation of new "mass" in the anterior region.

*Readout:* Explicit separation of "growth" (mass creation term) from "reshaping" (transport term).

### 2. Temporal Dynamics Divergence (The Rate of Change)

By tracking transport cost over time, we measure the *derivative* of development. This allows us to detect when the rate of morphological change in mutants uncouples from wild-type.

Crucially, we can interrogate this divergence at three levels of abstraction:

**Level 1: Mass Dynamics (Growth & Death)**

*The Question:* Is the mutant growing or losing tissue at a different rate?

*Readout:* A plot of the creation/destruction term over time. This detects onset of growth retardation or tissue necrosis before it affects overall shape.

**Level 2: Transport Dynamics (The Vector Field)**

*The Question:* Is the tissue moving or reshaping differently?

*What it captures:* The full vector field showing where mass moves—direction and magnitude at each pixel.

*Readout:* We can compare how this vector field changes over time, or compute the delta between mutant and wild-type flow fields. This reveals spatial patterns of abnormal movement (e.g., disordered migration, failure of directed extension) that a scalar summary would miss.

**Level 3: Trajectory Embeddings (Compressing Morphological History)**

*The Question:* Can we summarize an embryo's entire developmental history into a single vector for clustering?

*The Goal:* Compress the full sequence of mass changes and transport dynamics over time into a representation that captures the "shape" of development—not just where an embryo ends up, but how it got there.

*Possible Methods:* We propose compressing the high-dimensional per-timepoint vector fields using approaches like Geometric PCA or Autoencoders to yield a lower-dimensional "flow signature" for each frame. These compressed sequences can then be aligned using Dynamic Time Warping (DTW) to calculate a global "trajectory distance" between embryos.

*Readout:* A per-embryo embedding that enables trajectory-based clustering—grouping embryos by *how* they developed, not just their endpoint phenotype. This could distinguish a mutant that follows a fundamentally different path from one that is simply delayed.

### 3. Decomposing Embryos into Morphological Unit Dynamics

This framework provides a natural way to represent each embryo as the *composition* of individual morphological unit dynamics over time—eye, head, tail, yolk, melanocytes, each contributing its own trajectory.

Previously, disentangling these contributions was difficult. Scalar morphometrics blend everything into summary statistics; deep learning embeddings don't tell you which structure drives the difference. 

With transport-based analysis, each segmentable structure gets its own portrait:
- How did the eye change over time in this embryo?
- How did the tail change?
- How did melanocyte distribution evolve?

By comparing these per-structure portraits across genotypes, we can automatically identify which morphological units are driving phenotypic differences—and which are developing normally. The pixel-level interpretability means we don't just know *that* the head is abnormal, we know *where* in the head and *how* its dynamics differ.

This turns phenotype analysis from "this embryo is different" into "this embryo differs specifically in head growth dynamics while tail and eye develop normally."

---

## Visualizing Morphological Flux

A transport plan is a high-dimensional object (2D space × time × magnitude). To make this interpretable for biologists, we generate three complementary visualizations:

### Transport Vector Fields

We visualize the transport plan as a vector field overlaid on the embryo image. Arrow length indicates the magnitude of mass movement; color indicates the "effort" (transport cost) required. This directly shows *where* structures are moving and *how far*.

### Difference Heatmaps

For comparing mutant vs. wild-type, we plot the Sinkhorn divergence spatially. This creates a "hotspot" map showing exactly which pixels contribute most to the statistical difference between groups. A curved tail lights up the tail; disrupted NC migration lights up the relevant pigment regions.

### The "Morphological Movie"

By interpolating between timepoints using the optimal transport map (displacement interpolation), we reconstruct a smooth video of development. This highlights exactly when and where the mutant trajectory diverges from wild-type—making the "moment of divergence" visually apparent, not just a number in a table.

---

## How It Works: The Transport-Based Pipeline

The core idea: model development as the physical transport of mass. By calculating the optimal "work" required to transform one embryo stage into the next, we generate interpretable, dynamic representations.

### Pipeline Overview

1. **Segment masks** for each structure of interest (melanocytes, eye, head/jaw, etc.) across timepoints.

2. **Canonicalize coordinates** to make embryos comparable despite growth and pose variation. Two strategies:
   - *Yolk-based:* Use the yolk centroid, orientation, and size to define a common coordinate frame. Best for comparing structures in their anatomical context.
   - *Mask-centered:* Use the centroid of the structure itself (e.g., center the eye mask on its own centroid). This may be more robust when analyzing severely affected embryos where gross morphology is distorted—it focuses on how the structure changes relative to itself, independent of body-wide deformation.

3. **Convert masks to probability densities** on a canonical grid—one density per structure per timepoint.

4. **Compute optimal transport** between consecutive timepoints. This yields:
   - Transport cost (how much "work" to move mass)
   - Transport plan (which pixels map to which)
   - Mass change (appearance/disappearance of structures)

5. **Extract transition features** from the transport plan:
   - Total transport cost
   - Directional bias (AP, DV movement)
   - Mass creation/destruction
   - Regional breakdown (head vs. trunk)

6. **Cluster on dynamics** using:
   - Per-transition features (what's happening at each timepoint)
   - Embryo-averaged features (overall developmental signature)
   - Full trajectories via DTW or sequence models (temporal pattern matching)

### Why Optimal Transport?

Optimal transport is the natural mathematical framework for this problem because it provides a **universal, pixel-based metric** with the properties we need:

- **Mass conservation (or violation):** Unbalanced OT explicitly quantifies how much structure appears or disappears—directly measuring growth, death, or migration.
- **Spatial correspondence:** The transport plan tells us which regions at time $t$ map to which regions at time $t+1$—this is where per-pixel interpretability comes from.
- **Interpretable cost:** Transport cost has a physical interpretation (work = mass × distance), unlike arbitrary distance metrics. This gives us a principled measure of morphological distance.
- **Abstracts to any structure:** The same metric applies whether you're comparing jaws, eyes, or scattered melanocytes—anything you can segment becomes comparable.

### Technical Detail: Unbalanced OT

For structures like melanocytes where cells appear, disappear, and move, we use unbalanced optimal transport:

$$\pi = \arg\min_{\pi \geq 0} \int \|x-y\|^2 \, d\pi + \lambda_1 D(\pi\mathbf{1} \| \rho_t) + \lambda_2 D(\pi^\top\mathbf{1} \| \rho_{t+\Delta}) + \varepsilon \, \text{Ent}(\pi)$$

This simultaneously measures:
- How far mass moved (transport cost)
- How much mass appeared or disappeared (marginal penalties)
- How coherent the mapping is (entropic regularization)

---

## Why This Paradigm Is Powerful

The core abstraction—**pixel-level distance + dynamics for any segmentable structure**—solves fundamental limitations of current approaches:

| Current Limitation | Transport-Based Solution |
|--------------------|--------------------------|
| Embeddings are opaque | Pixel-level attribution shows *where* differences arise |
| Static snapshots only | Transport flow captures *dynamics* over time |
| Scalar metrics flatten heterogeneity | High-dimensional trajectory representations preserve variation |
| No principled way to compare distributions | OT provides mathematically grounded distance with physical interpretation |
| Can't distinguish migration failure from absence | Mass terms separate movement from appearance/disappearance |
| Different metrics for different structures | One universal metric applies to any segmentable structure |
| Can't ask "when" only "whether" | Temporal dynamics at three levels pinpoint divergence windows |

The result: a unified framework that handles obvious phenotypes (curved tails) and subtle ones (NC migration defects) with the same machinery, always providing both morphological distance and dynamics at pixel-level resolution.

---

## Related Work: OT on Masks and Morphometric Data

Optimal transport has been applied directly to segmentation mask–derived distributions in other domains. Gerber et al. ("Exploratory Population Analysis with Unbalanced Optimal Transport," 2018) and subsequent work (Gerber et al., "Optimal Transport Features for Morphometric Population Analysis," 2022/2023) use unbalanced OT on imaging-derived mass maps to separate tissue loss (mass change) from spatial displacement and to extract interpretable OT-based features for population-level analysis.

More directly on segmentation outputs, Liu et al. ("CellStitch: 3D Cellular Anisotropic Image Segmentation via Optimal Transport," 2023) computes optimal transport between 2D segmentation masks to stitch them into consistent 3D cell instances, demonstrating that OT can operate on mask objects as a practical matching primitive.

These precedents establish that OT on mask-derived distributions is a proven approach. What we propose extends this by applying OT to *time series* of masks to capture developmental dynamics—yielding per-transition embeddings that enable trajectory-based clustering and temporal divergence detection.

---

## Summary

This transport-based framework provides a **universal, pixel-based metric** that delivers both **morphological distance** (via transport cost) and **dynamics** (via transport flow). For any structure you can segment, you get:

1. **Per-structure readouts:** Quantify how any morphological unit transforms—obvious phenotypes like tail curvature, subtle ones like melanocyte migration, and mass changes like jaw growth.

2. **Temporal dynamics at multiple levels:** Detect when mutant development diverges by tracking mass dynamics, transport dynamics, or latent trajectory dynamics over time.

3. **Regional disentanglement:** Separate head from tail, melanocytes from jaw—ask targeted biological questions using the same metric everywhere.

This is the paradigm worth investing in. It directly addresses the biological questions we care about and provides interpretable, spatially-resolved, dynamics-aware readouts for complex, heterogeneous developmental phenotypes.
