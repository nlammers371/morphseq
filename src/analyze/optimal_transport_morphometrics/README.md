# Optimal transport morphometrics

This package is the embryo-specific layer on top of the reusable OT utilities in
[src/analyze/utils/optimal_transport](../utils/optimal_transport/README.md).

## Current ownership

- [src/analyze/utils/optimal_transport](../utils/optimal_transport/README.md)
  owns the reusable solver, working-grid seam, transports, and metrics.
- [src/analyze/utils/coord/grids/canonical.py](../utils/coord/grids/canonical.py)
  owns canonical embryo alignment.
- [uot_masks/](uot_masks/) owns embryo-specific I/O, preprocessing, wrappers,
  and visualization for the OT morphometrics workflow.
- [docs/](docs/) contains the design and contract notes for this area.

## What is canonical now

The most recent OT refactor moved the codebase toward a clean split:

1. geometry and canonical alignment live in the coord/canonical layer
2. working-grid preparation and solver execution live in reusable OT utilities
3. embryo-specific loading and orchestration live in `uot_masks/`
4. debug artifacts live under `results/.../canonical_aligner_debug/`

The key recent commits in this area are:

- `17e1d489` — docs/scripts for work-grid solver + canonical lift
- `6d989bff` — working-grid seam + work/canonical result types
- `53cd4158` — canonical template derived from OT outputs
- `6bedabeb` — output-grid / pair-frame API tightening

## Recommended reading order

1. [docs/DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md)
2. [../utils/optimal_transport/README.md](../utils/optimal_transport/README.md)
3. [uot_masks/__init__.py](uot_masks/__init__.py)
4. [uot_masks/run_transport.py](uot_masks/run_transport.py)
5. [uot_masks/run_timeseries.py](uot_masks/run_timeseries.py)

## What this package is not

- It is not the reusable OT solver layer.
- It is not the canonical alignment implementation.
- It is not the place to add new solver abstractions.

If a change belongs to the general OT engine, it should usually go into
`src/analyze/utils/optimal_transport/` instead.
