"""
Comprehensive Curvature Metrics for Embryo Shape Analysis

This module provides a collection of curvature-based metrics for quantifying
embryo morphology along the anterior-posterior axis.

Metrics are organized into categories:
1. Global metrics - Overall shape descriptors
2. Local metrics - Region-specific measurements
3. Distribution metrics - Statistical properties of curvature
4. Geometric metrics - Derived shape properties
"""

import numpy as np
from scipy import integrate, signal
from typing import Dict, Tuple, Optional


class CurvatureMetrics:
    """
    Compute comprehensive curvature metrics from embryo centerline.

    All metrics assume:
    - arc_length is in physical units (e.g., microns)
    - curvature is in inverse physical units (e.g., 1/microns)
    - arc_length[0] corresponds to one endpoint, arc_length[-1] to the other
    """

    def __init__(self, arc_length: np.ndarray, curvature: np.ndarray,
                 head_is_first: bool = True):
        """
        Initialize with curvature profile.

        Args:
            arc_length: Arc length parameter (N,) array in physical units
            curvature: Curvature values (N,) array in 1/physical units
            head_is_first: If True, arc_length[0] is head, else tail
        """
        self.arc_length = arc_length
        self.curvature = curvature
        self.head_is_first = head_is_first

        # Compute normalized arc length
        self.total_length = arc_length[-1] - arc_length[0]
        self.normalized_s = (arc_length - arc_length[0]) / self.total_length

        # Orient so head is always at s=0
        if not head_is_first:
            self.normalized_s = 1.0 - self.normalized_s
            self.curvature = self.curvature[::-1]

    # ========================================================================
    # GLOBAL METRICS - Overall shape descriptors
    # ========================================================================

    def mean_curvature(self) -> float:
        """
        Mean curvature along entire embryo.

        Higher values = more curved overall.
        Lower values = straighter overall.

        Returns:
            Mean curvature (1/length units)
        """
        return np.mean(self.curvature)

    def max_curvature(self) -> float:
        """
        Maximum curvature (point of strongest bending).

        Returns:
            Maximum curvature value (1/length units)
        """
        return np.max(self.curvature)

    def max_curvature_location(self) -> float:
        """
        Location of maximum curvature along normalized arc length.

        Returns:
            Normalized position (0=head, 1=tail)
        """
        max_idx = np.argmax(self.curvature)
        return self.normalized_s[max_idx]

    def total_curvature(self) -> float:
        """
        Total (integrated) curvature along embryo.

        This is the total angular change along the curve.
        For a closed circle: total curvature = 2π
        For a straight line: total curvature = 0

        Returns:
            Total curvature (dimensionless)
        """
        return np.trapz(self.curvature, self.arc_length)

    def total_absolute_curvature(self) -> float:
        """
        Total absolute curvature (always positive).

        Similar to total curvature but doesn't cancel out.
        Useful for comparing overall "bendiness" regardless of direction.

        Returns:
            Total absolute curvature (dimensionless)
        """
        return np.trapz(np.abs(self.curvature), self.arc_length)

    def end_to_end_distance(self, centerline_points: np.ndarray) -> float:
        """
        Straight-line distance from head to tail.

        Args:
            centerline_points: (N, 2) array of (x, y) coordinates

        Returns:
            Euclidean distance (length units)
        """
        if self.head_is_first:
            return np.linalg.norm(centerline_points[-1] - centerline_points[0])
        else:
            return np.linalg.norm(centerline_points[0] - centerline_points[-1])

    def tortuosity(self) -> float:
        """
        Tortuosity = arc_length / end_to_end_distance.

        Measures how "winding" the curve is.
        - Straight line: tortuosity = 1.0
        - Curved embryo: tortuosity > 1.0
        - Highly curved: tortuosity >> 1.0

        Note: Requires end_to_end_distance to be computed separately.

        Returns:
            Tortuosity (dimensionless, >= 1.0)
        """
        # This needs to be computed with centerline points
        # Placeholder - will be computed in comprehensive_metrics()
        pass

    # ========================================================================
    # DISTRIBUTION METRICS - Statistical properties
    # ========================================================================

    def curvature_std(self) -> float:
        """
        Standard deviation of curvature.

        High values = variable curvature along embryo
        Low values = uniform curvature

        Returns:
            Standard deviation (1/length units)
        """
        return np.std(self.curvature)

    def curvature_variance(self) -> float:
        """
        Variance of curvature.

        Returns:
            Variance (1/length units)²
        """
        return np.var(self.curvature)

    def curvature_range(self) -> float:
        """
        Range of curvature (max - min).

        Returns:
            Curvature range (1/length units)
        """
        return np.max(self.curvature) - np.min(self.curvature)

    def curvature_coefficient_of_variation(self) -> float:
        """
        Coefficient of variation = std / mean.

        Normalized measure of curvature variability.

        Returns:
            CV (dimensionless)
        """
        mean = self.mean_curvature()
        if mean == 0:
            return 0.0
        return self.curvature_std() / mean

    def curvature_skewness(self) -> float:
        """
        Skewness of curvature distribution.

        Positive = tail toward high curvature
        Negative = tail toward low curvature
        Zero = symmetric

        Returns:
            Skewness (dimensionless)
        """
        from scipy.stats import skew
        return skew(self.curvature)

    def curvature_kurtosis(self) -> float:
        """
        Kurtosis of curvature distribution.

        Measures "tailedness" of distribution.
        High kurtosis = sharp peaks, heavy tails

        Returns:
            Kurtosis (dimensionless)
        """
        from scipy.stats import kurtosis
        return kurtosis(self.curvature)

    # ========================================================================
    # LOCAL/REGIONAL METRICS - Region-specific measurements
    # ========================================================================

    def regional_mean_curvature(self, region_start: float = 0.0,
                                region_end: float = 1.0) -> float:
        """
        Mean curvature within a specific region.

        Args:
            region_start: Start position (0-1, normalized)
            region_end: End position (0-1, normalized)

        Returns:
            Mean curvature in region (1/length units)
        """
        mask = (self.normalized_s >= region_start) & (self.normalized_s <= region_end)
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(self.curvature[mask])

    def anterior_mean_curvature(self, anterior_fraction: float = 0.33) -> float:
        """
        Mean curvature in anterior region (head).

        Args:
            anterior_fraction: Fraction of embryo considered anterior

        Returns:
            Anterior mean curvature (1/length units)
        """
        return self.regional_mean_curvature(0.0, anterior_fraction)

    def trunk_mean_curvature(self, anterior_fraction: float = 0.33,
                            posterior_fraction: float = 0.33) -> float:
        """
        Mean curvature in trunk (middle) region.

        Args:
            anterior_fraction: Fraction of embryo that is anterior
            posterior_fraction: Fraction of embryo that is posterior

        Returns:
            Trunk mean curvature (1/length units)
        """
        return self.regional_mean_curvature(anterior_fraction,
                                           1.0 - posterior_fraction)

    def posterior_mean_curvature(self, posterior_fraction: float = 0.33) -> float:
        """
        Mean curvature in posterior region (tail).

        Args:
            posterior_fraction: Fraction of embryo considered posterior

        Returns:
            Posterior mean curvature (1/length units)
        """
        return self.regional_mean_curvature(1.0 - posterior_fraction, 1.0)

    def curvature_asymmetry(self, split_point: float = 0.5) -> float:
        """
        Asymmetry in curvature between anterior and posterior halves.

        Positive = posterior more curved
        Negative = anterior more curved
        Zero = symmetric

        Args:
            split_point: Where to split (default 0.5 = midpoint)

        Returns:
            Asymmetry (1/length units)
        """
        anterior = self.regional_mean_curvature(0.0, split_point)
        posterior = self.regional_mean_curvature(split_point, 1.0)
        return posterior - anterior

    # ========================================================================
    # PEAK DETECTION METRICS - Identifying bending points
    # ========================================================================

    def count_curvature_peaks(self, prominence: float = None) -> int:
        """
        Count number of local maxima (bending points) in curvature.

        Args:
            prominence: Minimum prominence for peak detection
                       (defaults to 0.2 * max_curvature)

        Returns:
            Number of peaks
        """
        if prominence is None:
            prominence = 0.2 * self.max_curvature()

        peaks, _ = signal.find_peaks(self.curvature, prominence=prominence)
        return len(peaks)

    def peak_locations(self, prominence: float = None) -> np.ndarray:
        """
        Get normalized locations of curvature peaks.

        Args:
            prominence: Minimum prominence for peak detection

        Returns:
            Array of peak locations (0-1, normalized)
        """
        if prominence is None:
            prominence = 0.2 * self.max_curvature()

        peaks, _ = signal.find_peaks(self.curvature, prominence=prominence)
        return self.normalized_s[peaks]

    def peak_spacing_variance(self, prominence: float = None) -> float:
        """
        Variance in spacing between curvature peaks.

        Low variance = regularly spaced bends
        High variance = irregularly spaced bends

        Args:
            prominence: Minimum prominence for peak detection

        Returns:
            Variance in peak spacing (dimensionless)
        """
        peak_locs = self.peak_locations(prominence)
        if len(peak_locs) < 2:
            return 0.0

        spacings = np.diff(peak_locs)
        return np.var(spacings)

    # ========================================================================
    # GEOMETRIC METRICS - Derived shape properties
    # ========================================================================

    def radius_of_curvature_stats(self) -> Dict[str, float]:
        """
        Statistics on radius of curvature (1/κ).

        Smaller radius = tighter bend
        Larger radius = gentler bend

        Returns:
            Dictionary with min, max, mean, median radius
        """
        # Avoid division by zero
        nonzero_curv = self.curvature[self.curvature > 1e-10]
        if len(nonzero_curv) == 0:
            return {'min': np.inf, 'max': np.inf, 'mean': np.inf, 'median': np.inf}

        radius = 1.0 / nonzero_curv

        return {
            'min': np.min(radius),
            'max': np.max(radius),
            'mean': np.mean(radius),
            'median': np.median(radius)
        }

    def effective_diameter(self) -> float:
        """
        Effective diameter if embryo were bent into a circle.

        Computed as: D = L / π, where L is total arc length

        Returns:
            Effective diameter (length units)
        """
        return self.total_length / np.pi

    def mean_bending_angle(self) -> float:
        """
        Mean angular change per unit length.

        Returns:
            Mean bending angle (radians/length unit)
        """
        return self.mean_curvature()

    def total_bending_angle(self) -> float:
        """
        Total angular change from head to tail.

        For a semicircle: π radians
        For a full circle: 2π radians
        For a straight line: 0 radians

        Returns:
            Total bending angle (radians)
        """
        return self.total_curvature()

    # ========================================================================
    # SMOOTHNESS METRICS - Curve smoothness/regularity
    # ========================================================================

    def curvature_gradient_magnitude(self) -> float:
        """
        Mean magnitude of curvature gradient (how rapidly curvature changes).

        High values = abrupt changes in curvature
        Low values = smooth, gradual curvature changes

        Returns:
            Mean curvature gradient (1/length units²)
        """
        curv_gradient = np.gradient(self.curvature, self.arc_length)
        return np.mean(np.abs(curv_gradient))

    def curvature_smoothness(self) -> float:
        """
        Inverse of curvature gradient - higher is smoother.

        Returns:
            Smoothness score (length units²)
        """
        grad_mag = self.curvature_gradient_magnitude()
        if grad_mag == 0:
            return np.inf
        return 1.0 / grad_mag

    def curvature_entropy(self) -> float:
        """
        Shannon entropy of curvature distribution.

        High entropy = uniform curvature distribution
        Low entropy = concentrated curvature values

        Returns:
            Entropy (bits)
        """
        from scipy.stats import entropy

        # Create histogram
        hist, _ = np.histogram(self.curvature, bins=50, density=True)
        # Normalize
        hist = hist / np.sum(hist)
        # Compute entropy (remove zeros to avoid log(0))
        hist = hist[hist > 0]

        return entropy(hist)

    # ========================================================================
    # COMPREHENSIVE METRICS - Get everything at once
    # ========================================================================

    def comprehensive_metrics(self, centerline_points: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all curvature metrics.

        Args:
            centerline_points: (N, 2) array of (x, y) coordinates (optional)
                              Required for tortuosity calculation

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Global metrics
        metrics['mean_curvature'] = self.mean_curvature()
        metrics['max_curvature'] = self.max_curvature()
        metrics['max_curvature_location'] = self.max_curvature_location()
        metrics['total_curvature'] = self.total_curvature()
        metrics['total_absolute_curvature'] = self.total_absolute_curvature()
        metrics['total_length'] = self.total_length

        # Distribution metrics
        metrics['curvature_std'] = self.curvature_std()
        metrics['curvature_variance'] = self.curvature_variance()
        metrics['curvature_range'] = self.curvature_range()
        metrics['curvature_cv'] = self.curvature_coefficient_of_variation()
        metrics['curvature_skewness'] = self.curvature_skewness()
        metrics['curvature_kurtosis'] = self.curvature_kurtosis()

        # Regional metrics
        metrics['anterior_mean_curvature'] = self.anterior_mean_curvature()
        metrics['trunk_mean_curvature'] = self.trunk_mean_curvature()
        metrics['posterior_mean_curvature'] = self.posterior_mean_curvature()
        metrics['curvature_asymmetry'] = self.curvature_asymmetry()

        # Peak detection
        metrics['n_curvature_peaks'] = self.count_curvature_peaks()
        peak_locs = self.peak_locations()
        metrics['peak_spacing_variance'] = self.peak_spacing_variance()
        if len(peak_locs) > 0:
            metrics['first_peak_location'] = peak_locs[0]
        else:
            metrics['first_peak_location'] = np.nan

        # Geometric metrics
        radius_stats = self.radius_of_curvature_stats()
        metrics['min_radius_of_curvature'] = radius_stats['min']
        metrics['max_radius_of_curvature'] = radius_stats['max']
        metrics['mean_radius_of_curvature'] = radius_stats['mean']
        metrics['median_radius_of_curvature'] = radius_stats['median']
        metrics['effective_diameter'] = self.effective_diameter()
        metrics['total_bending_angle'] = self.total_bending_angle()

        # Smoothness metrics
        metrics['curvature_gradient_magnitude'] = self.curvature_gradient_magnitude()
        metrics['curvature_smoothness'] = self.curvature_smoothness()
        metrics['curvature_entropy'] = self.curvature_entropy()

        # Tortuosity (if centerline points provided)
        if centerline_points is not None:
            end_to_end = self.end_to_end_distance(centerline_points)
            metrics['end_to_end_distance'] = end_to_end
            if end_to_end > 0:
                metrics['tortuosity'] = self.total_length / end_to_end
            else:
                metrics['tortuosity'] = np.inf

        return metrics

    def print_summary(self, centerline_points: Optional[np.ndarray] = None):
        """
        Print human-readable summary of curvature metrics.

        Args:
            centerline_points: (N, 2) array of (x, y) coordinates (optional)
        """
        metrics = self.comprehensive_metrics(centerline_points)

        print("="*70)
        print("CURVATURE METRICS SUMMARY")
        print("="*70)

        print("\n--- GLOBAL METRICS ---")
        print(f"Total arc length:        {metrics['total_length']:.2f}")
        print(f"Mean curvature:          {metrics['mean_curvature']:.6f}")
        print(f"Max curvature:           {metrics['max_curvature']:.6f}")
        print(f"  at position:           {metrics['max_curvature_location']:.3f} (0=head, 1=tail)")
        print(f"Total curvature:         {metrics['total_curvature']:.4f}")
        print(f"Total bending angle:     {metrics['total_bending_angle']:.4f} rad = {np.rad2deg(metrics['total_bending_angle']):.1f}°")

        if 'tortuosity' in metrics:
            print(f"End-to-end distance:     {metrics['end_to_end_distance']:.2f}")
            print(f"Tortuosity:              {metrics['tortuosity']:.3f}")

        print("\n--- DISTRIBUTION METRICS ---")
        print(f"Std deviation:           {metrics['curvature_std']:.6f}")
        print(f"Range:                   {metrics['curvature_range']:.6f}")
        print(f"Coefficient of variation:{metrics['curvature_cv']:.3f}")
        print(f"Skewness:                {metrics['curvature_skewness']:.3f}")
        print(f"Kurtosis:                {metrics['curvature_kurtosis']:.3f}")
        print(f"Entropy:                 {metrics['curvature_entropy']:.3f} bits")

        print("\n--- REGIONAL METRICS ---")
        print(f"Anterior (head) mean κ:  {metrics['anterior_mean_curvature']:.6f}")
        print(f"Trunk (middle) mean κ:   {metrics['trunk_mean_curvature']:.6f}")
        print(f"Posterior (tail) mean κ: {metrics['posterior_mean_curvature']:.6f}")
        print(f"Asymmetry (post-ant):    {metrics['curvature_asymmetry']:.6f}")

        print("\n--- PEAK DETECTION ---")
        print(f"Number of peaks:         {metrics['n_curvature_peaks']}")
        if not np.isnan(metrics['first_peak_location']):
            print(f"First peak at:           {metrics['first_peak_location']:.3f}")
        print(f"Peak spacing variance:   {metrics['peak_spacing_variance']:.6f}")

        print("\n--- GEOMETRIC METRICS ---")
        print(f"Min radius of curvature: {metrics['min_radius_of_curvature']:.2f}")
        print(f"Mean radius of curvature:{metrics['mean_radius_of_curvature']:.2f}")
        print(f"Max radius of curvature: {metrics['max_radius_of_curvature']:.2f}")
        print(f"Effective diameter:      {metrics['effective_diameter']:.2f}")

        print("\n--- SMOOTHNESS METRICS ---")
        print(f"Curvature gradient:      {metrics['curvature_gradient_magnitude']:.6f}")
        print(f"Smoothness score:        {metrics['curvature_smoothness']:.2f}")

        print("="*70)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_embryo_curvatures(metrics_list: list, embryo_labels: list):
    """
    Compare curvature metrics across multiple embryos.

    Args:
        metrics_list: List of metric dictionaries
        embryo_labels: List of embryo labels
    """
    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(metrics_list, index=embryo_labels)

    print("="*70)
    print("EMBRYO CURVATURE COMPARISON")
    print("="*70)

    # Key metrics for comparison
    key_metrics = [
        'mean_curvature',
        'max_curvature',
        'total_curvature',
        'tortuosity',
        'curvature_asymmetry',
        'anterior_mean_curvature',
        'trunk_mean_curvature',
        'posterior_mean_curvature'
    ]

    print(df[key_metrics].to_string())
    print("="*70)


def plot_curvature_profile_with_regions(arc_length: np.ndarray,
                                        curvature: np.ndarray,
                                        metrics: Dict[str, float],
                                        title: str = "Curvature Profile"):
    """
    Plot curvature profile with regional annotations.

    Args:
        arc_length: Arc length array
        curvature: Curvature array
        metrics: Dictionary of metrics
        title: Plot title
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize arc length
    s_norm = (arc_length - arc_length[0]) / (arc_length[-1] - arc_length[0])

    # Plot curvature
    ax.plot(s_norm, curvature, 'b-', linewidth=2, label='Curvature')
    ax.fill_between(s_norm, 0, curvature, alpha=0.3)

    # Add regional boundaries
    ax.axvline(0.33, color='gray', linestyle='--', alpha=0.5, label='Anterior/Trunk')
    ax.axvline(0.67, color='gray', linestyle='--', alpha=0.5, label='Trunk/Posterior')

    # Add mean line
    ax.axhline(metrics['mean_curvature'], color='r', linestyle='--',
               linewidth=2, label=f'Mean: {metrics["mean_curvature"]:.4f}')

    # Mark max curvature
    max_loc = metrics['max_curvature_location']
    max_val = metrics['max_curvature']
    ax.scatter([max_loc], [max_val], color='red', s=200, zorder=5,
              marker='*', edgecolors='black', linewidths=2,
              label=f'Max: {max_val:.4f} at s={max_loc:.2f}')

    # Labels
    ax.set_xlabel('Normalized Arc Length (0=head, 1=tail)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Curvature', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Demo: Generate synthetic curvature profile
    print("Curvature Metrics Module - Demo")
    print("="*70)

    # Create synthetic data (sine wave for curvature)
    s = np.linspace(0, 500, 200)  # arc length in microns
    curvature = 0.01 + 0.005 * np.sin(2 * np.pi * s / 200)  # sinusoidal curvature

    # Create metrics object
    cm = CurvatureMetrics(s, curvature, head_is_first=True)

    # Print summary
    cm.print_summary()

    # Get comprehensive metrics
    metrics = cm.comprehensive_metrics()

    print("\n✓ Demo complete. Use this module to analyze real embryo curvature data.")
