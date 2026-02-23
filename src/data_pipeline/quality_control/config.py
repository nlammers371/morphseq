"""
Quality control configuration and default parameters.

This module defines all default parameters used in QC operations to ensure
consistency across the codebase. All QC functions should import from this
module rather than hardcoding defaults.
"""

# Quality Control Defaults
QC_DEFAULTS = {
    # Death detection parameters
    "dead_lead_time_hours": 4.0,
    # Number of hours BEFORE detected death to retroactively flag embryo as compromised.
    # This buffer time accounts for the period when an embryo is already dying but
    # has not yet triggered the death detection threshold. Used in compute_dead_flag2_persistence()
    # to exclude data from compromised embryos.

    # Persistence threshold for dead flag confirmation
    "persistence_threshold": 0.80,
    # Fraction of time steps that must be flagged as dead to confirm death detection.
    # Default 0.80 means 80% of time steps after initial death detection must maintain
    # the dead status for biological persistence (death is permanent).

    # Minimum decline rate for death detection
    "min_decline_rate": 0.05,
    # Minimum rate of decline in fraction_alive required to trigger death inflection detection.
    # Prevents false positives from normal developmental variation.
}


def get_dead_lead_time() -> float:
    """
    Get the canonical dead_lead_time value.

    Returns
    -------
    float
        Hours of buffer time before death detection (default 4.0)
    """
    return QC_DEFAULTS["dead_lead_time_hours"]
