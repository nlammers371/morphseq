You are a morphseq trajectory condensation expert. When the user asks about the
active force-based trajectory condensation package, use
`src/analyze/trajectory_condensation/`. Follow these rules exactly.

## Scope

- Use this for trajectory condensation, force diagnostics, saved-run
  visualization, iteration ranking, and principal-tree interpretation of
  condensed coordinates.
- Do not confuse this with `src/analyze/trajectory_analysis/`, which owns older
  DTW clustering and bootstrap projection workflows.
- Treat `results/.../trajectory_cosmology/` copies as experiment-local history.
  New reusable work should target `src/analyze/trajectory_condensation/`.

## Setup

```python
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

project_root = Path(__file__).resolve().parents[N]  # adjust N
sys.path.insert(0, str(project_root / "src"))

import analyze.trajectory_condensation as tc
```

## Read First

1. `src/analyze/trajectory_condensation/README.md`
2. `src/analyze/trajectory_condensation/ALGORITHM.md`
3. `src/analyze/trajectory_condensation/DESIGN.md`
4. `src/analyze/trajectory_condensation/viz/README_viz.md`

## Common Pattern

```python
data = tc.from_pairwise_margin_csv("results/.../raw_contrast_scores_long.csv")
x0 = tc.init_embedding.pca_init(data.features, data.mask)

config = tc.CondensationConfig(solver_max_iter=500)
result = tc.run_condensation(
    x0=x0,
    mask=data.mask,
    config=config,
    save_every=10,
)

run = tc.load_run("results/.../condensed_positions.npz", title="my run")
tc.render_run(run, "figures/my_run/")
```

## Visualization

Use `VIZ.md` in this skill and the package viz README for rendering details.
Do not add visualization policy to `condensation/engine/run.py` or
`condensation/forces/*`.
