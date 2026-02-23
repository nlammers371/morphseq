"""Coordinate systems and geometry utilities.

This package defines stable coordinate-frame contracts and transformation
provenance for downstream analysis pipelines (e.g. UOT, ODE learning).

Public surface:
- `analyze.utils.coord.grids.canonical`: canonical-grid mapping ("to_canonical_grid_*")
- `analyze.utils.coord.register`: explicit registration ("register_to_fixed")
"""

from .types import (
    CanonicalFrameResult,
    CanonicalGrid,
    CanonicalImageResult,
    CanonicalMaskResult,
    Frame,
    RegisterResult,
)
from .transforms import GridTransform, TransformChain
