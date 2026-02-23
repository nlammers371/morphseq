"""Placeholder for future SplineFitter class.

This module is reserved for a future object-oriented API that will provide
a unified interface for spline fitting, projection, and analysis workflows.

Planned API (not yet implemented):
    >>> from src.analyze.spline_fitting import SplineFitter
    >>>
    >>> # Fit splines grouped by phenotype
    >>> fitter = SplineFitter(df, group_by='phenotype')
    >>> fitter.fit(bandwidth=0.5, n_bootstrap=100)
    >>>
    >>> # Access fitted splines
    >>> fitter.splines  # dict: phenotype -> spline DataFrame
    >>>
    >>> # Project new data onto fitted splines
    >>> fitter.project(new_df)
    >>> fitter.projections  # dict: phenotype -> projected DataFrame
    >>>
    >>> # Segment splines
    >>> fitter.segment(n_segments=5)
    >>> fitter.segments  # dict: phenotype -> segment info

Current Status:
    NOT IMPLEMENTED - Use functional API instead:
    - spline_fit_wrapper() for bootstrap fitting
    - project_points_onto_reference_spline() for projection
    - create_spline_segments_for_df() for segmentation

Why a future class?
    The current functional API works well but can be verbose for complex workflows.
    A SplineFitter class would:
    - Store fitted splines and metadata
    - Chain operations (fit -> project -> segment -> analyze)
    - Maintain state across operations
    - Provide cleaner API for grouped operations
    - Enable caching of expensive computations

Design Considerations:
    - Should it wrap or replace the functional API?
    - How to handle grouped vs single-spline workflows?
    - Should it integrate with trajectory_analysis.viz?
    - What caching/memoization is needed?

If you're implementing this:
    1. Start with simple fit() + project() workflow
    2. Add state management for splines/projections/segments
    3. Consider sklearn-style API: fit(X) then transform(X)
    4. Ensure backwards compatibility with functional API
    5. Add visualization methods or integrate with viz.py
"""

# Placeholder - no implementation yet
# Remove this comment when implementation begins

__all__ = []  # Nothing exported yet
