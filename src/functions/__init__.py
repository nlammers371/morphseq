"""Compatibility shims for legacy ``src.functions`` imports.

Historically, this package re-exported many submodules via eager ``import *``.
That makes importing *any* submodule under ``src.functions`` execute heavy/optional
imports (e.g. plotting dependencies like seaborn), which breaks lightweight
runtime contexts such as Build06's embedding-generation subprocess.

We keep best-effort re-exports, but make them non-fatal: if an optional
dependency is missing, importing ``src.functions`` still succeeds and only the
affected submodule's symbols are absent.
"""

from __future__ import annotations

import warnings


def _try_star_import(module: str) -> None:
    try:
        # NOTE: explicit imports so type-checkers/lint don't get confused.
        exec(f"from {module} import *")  # noqa: S102
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Optional legacy import failed: {module}: {e}")


_try_star_import("src.functions.utilities")
_try_star_import("src.functions.core_utils_segmentation")
_try_star_import("src.functions.custom_networks")
_try_star_import("src.functions.dataset_utils")
_try_star_import("src.functions.image_utils")
_try_star_import("src.functions.improved_build_splines")
_try_star_import("src.functions.plot_functions")
_try_star_import("src.functions.spline_fitting_v2")
_try_star_import("src.functions.spline_morph_spline_metrics")
_try_star_import("src.functions.embryo_df_performance_metrics")
