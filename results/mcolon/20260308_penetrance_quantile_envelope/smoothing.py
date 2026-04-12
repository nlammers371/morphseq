"""
smoothing.py — Generic LOESS and domain-specific quantile envelope frac selection.

Note on naming:
- ``loess_smooth`` is a generic math utility.
- ``select_quantile_curve_smoother`` is domain-specific (quantile envelope fitting);
  the name makes the abstraction honest and prevents misuse as a generic smoother selector.

Note on scope:
- ``compute_penetrance_by_time`` / ``mark_threshold_violations`` already exist in
  ``src/analyze/difference_detection/penetrance_threshold.py`` (embryo-level semantics).
  The functions here operate at the FRAME level and have different semantics; there is
  no collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Generic LOESS
# ---------------------------------------------------------------------------

def loess_smooth(
    x,
    y,
    frac: float,
    *,
    dropna: bool = True,
    require_sorted: bool = False,
) -> np.ndarray:
    """
    Locally weighted linear regression (LOESS) for 1-D data.

    Parameters
    ----------
    x, y : array-like
        Input coordinates. Must have the same length.
    frac : float
        Bandwidth fraction in (0, 1].
    dropna : bool
        If True (default), fit on non-NaN rows only; return NaN at NaN positions.
        If False, raise ValueError when any NaN is present.
    require_sorted : bool
        If True, raise ValueError if x is not monotone non-decreasing.

    Returns
    -------
    np.ndarray
        Same length as input; NaN where y was NaN (when dropna=True).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length; got {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError(f"Need at least 2 points; got {len(x)}")
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1]; got {frac}")
    if require_sorted and np.any(np.diff(x) < 0):
        raise ValueError("x is not monotone non-decreasing (require_sorted=True)")

    nan_mask = np.isnan(y) | np.isnan(x)
    if nan_mask.any():
        if not dropna:
            raise ValueError("NaN values present and dropna=False")
        xv, yv = x[~nan_mask], y[~nan_mask]
        smoothed_valid = _loess_core(xv, yv, frac)
        out = np.full(len(x), np.nan)
        out[~nan_mask] = smoothed_valid
        return out

    return _loess_core(x, y, frac)


def _loess_core(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
    """Tricube-weighted local linear regression evaluated at x."""
    return _loess_predict(x, y, x, frac)


def _loess_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    frac: float,
) -> np.ndarray:
    """Tricube-weighted local linear regression evaluated at x_eval."""
    n = len(x_train)
    h = max(int(np.ceil(frac * n)), 2)
    smoothed = np.empty(len(x_eval))
    for i, x0 in enumerate(x_eval):
        dists = np.abs(x_train - x0)
        idx = np.argsort(dists)[:h]
        d_max = dists[idx].max()
        if d_max == 0:
            smoothed[i] = y_train[idx[0]]
            continue
        u = dists[idx] / d_max
        w = (1 - u**3)**3
        xi, yi, wi = x_train[idx], y_train[idx], w
        wsum = wi.sum()
        wmean_x = (wi * xi).sum() / wsum
        wmean_y = (wi * yi).sum() / wsum
        beta = (
            (wi * (xi - wmean_x) * (yi - wmean_y)).sum()
            / max((wi * (xi - wmean_x)**2).sum(), 1e-12)
        )
        smoothed[i] = wmean_y + beta * (x0 - wmean_x)
    return smoothed


def _local_median(y, window: int, min_points: int) -> np.ndarray:
    """Centered rolling median using index neighborhoods."""
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    half = max(int(window) // 2, 0)
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        vals = y[lo:hi]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= min_points:
            out[i] = np.median(vals)
    return out


def detect_curve_inliers(
    x,
    y,
    *,
    window: int = 5,
    min_points: int = 3,
    sigma_threshold: float = 4.0,
    min_resid_fraction: float = 0.15,
):
    """
    Detect bins to keep for smoothing using a local-median residual screen.

    Bins with unusually large deviations from the local median are excluded
    from the LOESS fit but remain in the raw curve and output tables.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length; got {len(x)} vs {len(y)}")
    if len(y) == 0:
        return np.array([], dtype=bool), {"threshold": np.nan, "excluded_x": []}

    baseline = _local_median(y, window=window, min_points=min_points)
    ref_mask = ~np.isnan(baseline)
    residual = y - baseline
    if ref_mask.sum() < max(min_points, 3):
        return np.ones(len(y), dtype=bool), {
            "threshold": np.nan,
            "robust_sigma": np.nan,
            "value_range": float(y.max() - y.min()) if len(y) > 1 else 0.0,
            "excluded_x": [],
            "baseline": baseline,
            "residual": residual,
        }

    centered = residual[ref_mask] - np.median(residual[ref_mask])
    mad = float(np.median(np.abs(centered)))
    robust_sigma = 1.4826 * mad
    value_range = float(y.max() - y.min()) if len(y) > 1 else 0.0
    threshold = max(sigma_threshold * robust_sigma, min_resid_fraction * value_range, 1e-12)

    inliers = np.ones(len(y), dtype=bool)
    inliers[ref_mask] = np.abs(residual[ref_mask]) <= threshold
    return inliers, {
        "threshold": threshold,
        "robust_sigma": robust_sigma,
        "value_range": value_range,
        "excluded_x": x[~inliers].tolist(),
        "baseline": baseline,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# Roughness helper
# ---------------------------------------------------------------------------

def curve_roughness(y) -> float:
    """
    Discrete 2nd-difference roughness: std(diff(y, n=2)).

    Parameters
    ----------
    y : array-like, length >= 3

    Returns
    -------
    float
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        raise ValueError(f"Need at least 3 points for roughness; got {len(y)}")
    return float(np.diff(y, n=2).std())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SmoothedCurveSelection:
    """
    Result bundle from ``select_quantile_curve_smoother``.

    Attributes
    ----------
    selected_frac : float
    smoothed_y : np.ndarray
        Full-length output (NaN for unsupported / NaN-input positions).
    candidate_curves : dict[float, np.ndarray]
        frac → smoothed array evaluated at valid (non-NaN) positions only.
    diagnostics : dict[float, dict]
        frac → structured result dict with keys:
          passed, failed_checks, roughness, roughness_threshold,
          sign_change_rate, residual_range, value_range, any_negative.
    used_fallback : bool
        True if no candidate frac passed and fallback_frac was used.
    fit_mask : np.ndarray or None
        Boolean full-length mask indicating which rows were used to fit the
        selected smoother.
    fit_diagnostics : dict
        Diagnostics from the pre-LOESS inlier screen.
    """
    selected_frac: float
    smoothed_y: np.ndarray
    candidate_curves: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    used_fallback: bool = False
    fit_mask: np.ndarray | None = None
    fit_diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Domain-specific smoother selection
# ---------------------------------------------------------------------------

def select_quantile_curve_smoother(
    x,
    y,
    *,
    candidate_fracs,
    nonnegative: bool = False,
    fallback_frac: float = 0.10,
    fit_mask=None,
) -> SmoothedCurveSelection:
    """
    Domain-specific LOESS frac selection for quantile envelope fitting.

    Sweeps ``candidate_fracs`` (smallest first) and selects the smallest frac
    whose smoothed curve passes all shape-stability checks.  Not a generic
    utility — the name makes the abstraction honest.

    Parameters
    ----------
    x, y : array-like
        Time-bin centres and raw quantile values. y may contain NaN for
        unsupported bins.
    candidate_fracs : sequence of float
        Fracs to try, each in (0, 1].
    nonnegative : bool
        If True, reject fracs that produce any negative smoothed values.
    fallback_frac : float
        Frac to use if no candidate passes; must be in (0, 1].

    Returns
    -------
    SmoothedCurveSelection
    """
    candidate_fracs = list(candidate_fracs)
    if not candidate_fracs:
        raise ValueError("candidate_fracs must not be empty")
    if any(not (0 < f <= 1) for f in candidate_fracs):
        raise ValueError("All candidate_fracs must be in (0, 1]")
    if not (0 < fallback_frac <= 1):
        raise ValueError(f"fallback_frac must be in (0, 1]; got {fallback_frac}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError(f"Need at least 3 points; got {len(x)}")

    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if fit_mask is None:
        fit_mask = valid_mask.copy()
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if len(fit_mask) != len(x):
            raise ValueError(f"fit_mask must have length {len(x)}, got {len(fit_mask)}")
        fit_mask = valid_mask & fit_mask

    xv, yv = x[valid_mask], y[valid_mask]
    xf, yf = x[fit_mask], y[fit_mask]
    if len(xf) < 3:
        fit_mask = valid_mask.copy()
        xf, yf = xv, yv

    value_range = float(yf.max() - yf.min()) if len(yf) > 1 else 1.0
    roughness_threshold = 0.05 * value_range

    diagnostics: dict[float, dict] = {}
    candidate_curves: dict[float, np.ndarray] = {}
    selected_frac = None
    selected_sm = None

    for frac in sorted(candidate_fracs):
        sm_fit = _loess_predict(xf, yf, xf, frac)
        sm = _loess_predict(xf, yf, xv, frac)
        candidate_curves[frac] = sm

        failed_checks: list[str] = []

        # Check 1: non-negativity
        any_negative = bool(np.any(sm < 0))
        if nonnegative and any_negative:
            failed_checks.append("negativity")

        # Check 2: roughness
        roughness = curve_roughness(sm) if len(sm) >= 3 else 0.0
        if roughness > roughness_threshold:
            failed_checks.append("roughness")

        # Check 3: oscillation in residual
        residual = yf - sm_fit
        residual_range = float(residual.max() - residual.min()) if len(residual) > 1 else 0.0
        sign_changes = int(np.sum(np.diff(np.sign(residual - residual.mean())) != 0))
        sign_change_rate = sign_changes / max(len(sm_fit), 1)
        large_oscillation = (residual_range > 0.1 * value_range) and (sign_change_rate > 0.40)
        if large_oscillation:
            failed_checks.append("oscillation")

        passed = len(failed_checks) == 0
        diagnostics[frac] = {
            "passed": passed,
            "failed_checks": failed_checks,
            "roughness": roughness,
            "roughness_threshold": roughness_threshold,
            "sign_change_rate": sign_change_rate,
            "residual_range": residual_range,
            "value_range": value_range,
            "any_negative": any_negative,
        }

        if passed and selected_frac is None:
            selected_frac = frac
            selected_sm = sm

    used_fallback = selected_frac is None
    if used_fallback:
        selected_frac = fallback_frac
        if fallback_frac not in candidate_curves:
            selected_sm = _loess_predict(xf, yf, xv, fallback_frac)
            candidate_curves[fallback_frac] = selected_sm
        else:
            selected_sm = candidate_curves[fallback_frac]

    # Build full-length output (NaN for unsupported bins)
    smoothed_y = np.full(len(x), np.nan)
    smoothed_y[valid_mask] = selected_sm

    return SmoothedCurveSelection(
        selected_frac=selected_frac,
        smoothed_y=smoothed_y,
        candidate_curves=candidate_curves,
        diagnostics=diagnostics,
        used_fallback=used_fallback,
        fit_mask=fit_mask,
    )
