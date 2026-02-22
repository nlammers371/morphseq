"""
DEPRECATED: p0_viz — thin shim for backwards compatibility.

All visualization logic has moved to viz/phase0.py and viz/qc.py.
Import from viz directly:

    from viz import plot_cost_density_suite, plot_s_map, ...
    from viz import contract  # coordinate-contract wrappers

This shim will be removed in a future cleanup pass.
"""

import warnings
warnings.warn(
    "p0_viz is deprecated. Import from 'viz' instead: "
    "from viz import plot_cost_density_suite, plot_s_map, ...",
    DeprecationWarning,
    stacklevel=2,
)

from viz.phase0 import *  # noqa: F401,F403
from viz.phase0 import __all__  # noqa: F401
