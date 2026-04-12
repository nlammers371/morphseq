# Slice Sandbox Animation Guide

## Overview

Four GIFs showing the iteration-by-iteration evolution of the force law on synthetic 2-cluster data:

**Metrics-only animations** (showing sep_ratio, global_spread, energies):
- `ANIMATION_good_vs_collapse.gif` — Good params vs. catastrophic collapse
- `ANIMATION_good_vs_intermediate.gif` — Good params vs. moderate degradation

**Cluster position animations** (showing actual point cloud movement):
- `ANIMATION_clusters_good_vs_collapse.gif` — Good params vs. catastrophic collapse
- `ANIMATION_clusters_good_vs_intermediate.gif` — Good params vs. moderate degradation

All are 100-frame animations (every 3 iterations from 300 total).

**Recommendation**: Watch the cluster position animations first — they're the most intuitive.

---

## ANIMATION_clusters_good_vs_collapse.gif

**What you're seeing**: Actual point cloud positions (blue and red dots) over 300 iterations

**Left side (GOOD)**: k=20, eps_mult=0.005, lr=1e-4
- **Iter 0**: Two clusters overlap slightly (initial state, all embryos start close)
- **Iter ~30**: Clusters begin to separate clearly (attraction pulling within-cluster points)
- **Iter ~100**: Clusters fully separated with tight internal structure (both effects visible)
- **Iter ~300**: Clusters remain separated and compact (stable final configuration)

**Key visual**: Watch the individual points:
- Good run: Points move *outward from the opposite cluster* and *inward toward their own cluster* (bidirectional motion)
- The two clusters drift apart smoothly and stay apart

**Right side (BAD)**: k=5, eps_mult=0.10, lr=5e-4
- **Iter 0**: Two clusters present (same start)
- **Iter ~5**: Clusters immediately begin merging (strong repulsion pushes them apart wildly)
- **Iter ~20**: Points are spread out in a disorganized cloud (no coherent clusters)
- **Iter ~100+**: Everything mixed together in a homogeneous blob (complete loss of structure)

**Key visual**: Watch the overall shape:
- Bad run: Points spread radially outward (explosion) OR collapse inward (implosion), no clear cluster structure

---

## ANIMATION_clusters_good_vs_intermediate.gif

Same visualization but with intermediate parameters (not total collapse):

**Right side (INTERMEDIATE)**: k=5, eps_mult=0.02, lr=5e-4
- **Iter 0–50**: Clusters separate somewhat (beginning to work)
- **Iter 50–150**: Clusters oscillate and drift (metastable)
- **Iter 150–300**: Slow merging back together (eventual failure)

This is the "false hope" case — it looks like it's working for a while, then fails.

---

## ANIMATION_good_vs_collapse.gif (metrics)

**Left panel (GOOD)**: k=20, eps_mult=0.005, lr=1e-4
- sep_ratio: 4.72 → **7.13** (+51% improvement)
- collapse_score: **0.956** (cloud stays compact)
- Behavior: Clusters smoothly separate and stabilize

**Right panel (BAD)**: k=5, eps_mult=0.10, lr=5e-4
- sep_ratio: 4.72 → **1.02** (catastrophic collapse)
- collapse_score: **2.20** (cloud expands as it collapses!)
- Behavior: Initial oscillation, then immediate collapse to blob at iter 0

**What to observe**:
1. **Good run**: separation_ratio stays high throughout, increasing to 7+ by mid-run
2. **Bad run**: separation_ratio collapses instantly (best at iter 0), never recovers
3. **Good run**: global_spread stays ~3 (constant)
4. **Bad run**: global_spread jumps to 9.2 (2.8× expansion) due to mutual collapse

**Physics**:
- The bad run has epsilon_r ≈ 0.156 (repulsion strength)
- With k=5 (only 5 neighbors), attraction is local but still too strong
- Repulsion is *too large* relative to the narrow local attraction → system explodes outward
- Plus lr=5e-4 (too aggressive) amplifies the numerical instability

---

## ANIMATION_good_vs_intermediate.gif

**Left panel (GOOD)**: k=20, eps_mult=0.005, lr=1e-4
- sep_ratio: 4.72 → **7.13** (+51%)
- collapse_score: **0.956**
- Behavior: Smooth, stable improvement

**Right panel (INTERMEDIATE)**: k=5, eps_mult=0.02, lr=5e-4
- sep_ratio: 4.72 → **1.90** (−60% degradation from init, but not total collapse)
- collapse_score: **1.39** (slight expansion)
- Behavior: Initial oscillation, gradual drift toward blob

**What to observe**:
1. Both runs start similarly (same data, same forces at iter 0)
2. **Good run**: sep_ratio climbs smoothly to 7 and holds
3. **Intermediate run**: sep_ratio initially improves, then oscillates and declines
4. **Good run**: global_spread constant (0.956× initial)
5. **Intermediate run**: global_spread expands to 4× initial (cloud becomes diffuse during collapse)

**Physics**:
- eps_mult=0.02 with k=5 is a midpoint parameter set
- Strong enough repulsion to avoid instant collapse
- Weak enough attraction that local kNN neighborhoods don't fully stabilize
- Result: metastable for ~100 iters, then slow drift into collapse regime

**Key insight**: This shows the **phase transition** between good and bad parameters. Intermediate parameters are worse than both extremes because they sustain partial structure (misleading) before degrading (disaster).

---

## How to interpret the metrics columns

| Column | Meaning | Good | Bad |
|--------|---------|------|-----|
| **sep_ratio** | Cluster separation / within-cluster width | ↑ increase | ↓ decrease |
| **global_spread** | RMS distance from overall centroid | ≈ constant | ↑ expansion or → contraction |
| **collapse_score** | global_spread_final / global_spread_initial | 0.95–0.97 | <0.8 (collapse) or >1.2 (explosion) |
| **e_att** | Attraction energy (negative) | −2000 to −3000 | −400 to −1000 |
| **e_rep** | Repulsion energy (positive) | 150–200 | 150–400 |

**Good run signature**:
- sep_ratio ↑ over time (clusters separate)
- e_att negative and substantial (attraction working)
- e_rep modest but present (repulsion stabilizing)
- collapse_score ≈ 0.96 (cloud size stable)

**Bad run signature**:
- sep_ratio oscillates or ↓ (structure lost)
- e_att may be very negative (too much pull) or weak
- e_rep very large (repulsion fighting back but losing)
- collapse_score <0.8 or >2 (cloud shrinking or expanding)

---

## Why these visualizations matter

Traditional metrics tables (like `summary.csv`) show only the **endpoint** (iter 300):
- You see final sep_ratio and collapse_score
- You don't see *how* the system got there
- Intermediate oscillation is invisible

**Animations reveal**:
1. **Early warning signs** — Bad parameters fail immediately (sep_ratio collapses at iter 0)
2. **Stability** — Good parameters reach optimum and hold; bad ones drift
3. **Mechanism** — You can watch the clusters separate (good) or implode (bad)
4. **Overshoot** — You see whether the system oscillates or converges smoothly

---

## How to generate more animations

Use `animate_slice_sandbox.py`:

```bash
conda run -n segmentation_grounded_sam --no-capture-output python \\
  results/mcolon/20260329_pbx_crispant_analysis_cont/animate_slice_sandbox.py \\
  --good-run results/slice_sandbox_all_variants_v1/separated/run_0000 \\
  --bad-run results/slice_sandbox_all_variants_v1/separated/run_0037 \\
  --output-dir /tmp/my_animations
```

Find run IDs in `summary.csv`:
- **Best**: Highest `sep_ratio_final` (e.g., run_0000)
- **Worst**: Lowest `sep_ratio_final` or highest collapse_score (e.g., run_0037 or run_0049)
- **Intermediate**: Moderate values (e.g., run_0024)

---

## Summary

**ANIMATION_good_vs_collapse.gif** shows the difference between:
- A force law that **works** (preserves structure, improves separation)
- A force law that **fails** (instant collapse, no recovery)

**ANIMATION_good_vs_intermediate.gif** shows the **phase transition**:
- What happens when parameters are slightly off
- Why intermediate is often worse than just being wrong

Both highlight why **iteration-by-iteration tracking** (metrics_history.csv) is essential. The endpoint alone would miss these dynamics.
