"""
Generic Plotting Utilities

Domain-agnostic plotting functions for time series and 3D visualization.

These functions use generic time-series algorithms from utils.timeseries
(DTW, DBA, interpolation) and have no trajectory_analysis dependencies.

Modules
=======
- feature_over_time : Faceted time series plotting (faceting-engine)
- plotting_3d : 3D scatter plots with trajectory lines

Functions
=========
Time Series Plotting:
- plot_feature_over_time : Plot a feature over time, colored by group

3D Plotting:
- plot_3d_scatter : 3D scatter plot with optional trajectory/mean lines

For domain-specific trajectory visualizations (genotype styling, phenotype colors),
see: src.analyze.trajectory_analysis.viz.plotting
"""

from .feature_over_time import plot_feature_over_time
from .proportions import plot_proportions
from .plotting_3d import plot_3d_scatter

__all__ = [
    # Generic API
    'plot_feature_over_time',
    'plot_3d_scatter',
    'plot_proportions',
]
