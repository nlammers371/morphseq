"""
DEPRECATED: p0_qc — thin shim for backwards compatibility.

All QC logic has moved to viz/qc.py.
Import from viz directly:

    from viz import run_qc_suite, compute_iqr_outliers, ...

This shim will be removed in a future cleanup pass.
"""

import warnings
warnings.warn(
    "p0_qc is deprecated. Import from 'viz' instead: "
    "from viz import run_qc_suite, compute_iqr_outliers, ...",
    DeprecationWarning,
    stacklevel=2,
)

from viz.qc import *  # noqa: F401,F403
from viz.qc import __all__  # noqa: F401
