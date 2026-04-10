# Emergence Package Design

This package is split to mirror the causal reasoning pipeline of the method:

1. define onset
2. validate the reference
3. score emergence relative to that reference
4. partition co-emergent blocks internally
5. assemble the final tree
6. separately diagnose where pairwise logic resists coherence

## Why the layout is structured this way

- `onset.py`
  - onset inference is upstream of everything else
  - downstream tree code consumes onset values but does not redefine them

- `reference.py`
  - reference coherence is a separate logical question from tree construction

- `emergence.py`
  - owns top-level emergence scoring and block formation from the reference

- `partition.py`
  - owns honest internal splitting of co-emergent blocks
  - bipartition search, monotone feasibility, and unresolved composites live here

- `algorithm.py`
  - orchestration only
  - should read like the method recipe, not a logic dump

- `transitivity.py`
  - sibling diagnostic module, not the main algorithm
  - explains where pairwise evidence resists a coherent tree

## Visualization boundary

Reusable rendering belongs in `analyze.classification.viz.emergence`.
The emergence package should own algorithm/data encoding only.

The current interactive explorer remains the active implementation for now.
Future static matplotlib-style figures should move into the shared `viz/`
subpackage rather than back into the algorithm package.

## Future improvements

Some method-level improvements remain documented in `ALGORITHM.md`.
Additional engineering improvements for the package layout are:

- implement a real static renderer in `classification/viz/emergence.py`
- add targeted tests for each internal stage rather than relying mostly on the
  top-level fixture path
- consider multiple onset rules under `onset.py` if alternative durability
  definitions become necessary
