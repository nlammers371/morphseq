"""
Compare Clustering Methods v2

Compares posterior-based classification against original co-association method.
Loads pre-computed bootstrap results and applies both classification approaches.

Usage:
    python compare_methods_v2.py --k_range 2 5
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List
import sys

from bootstrap_posteriors import analyze_bootstrap_results
from adaptive_classification import (
    classify_embryos_2d,
    classify_embryos_adaptive,
    get_classification_summary
)


def load_bootstrap_data(k: int, base_dir: str = '../20251103_DTW_analysis/output/2_select_k/data') -> Dict:
    """Load pre-computed bootstrap results for given k."""
    filepath = Path(base_dir) / f'bootstrap_k{k}.pkl'
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def load_original_membership_results(k: int, base_dir: str = '../20251103_DTW_analysis/output/4_membership/data') -> Dict:
    """Load original co-association based membership results."""
    filepath = Path(base_dir) / 'membership_all_k.pkl'
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data.get(k, None)
    except FileNotFoundError:
        print(f"Warning: Could not find original membership results at {filepath}")
        return None


def compare_single_k(k: int,
                    threshold_max_p: float = 0.8,
                    threshold_log_odds: float = 0.7,
                    use_adaptive: bool = False) -> Dict:
    """
    Compare posterior-based vs. co-association classification for single k.

    Parameters
    ----------
    k : int
        Number of clusters
    threshold_max_p : float
        Threshold for max_p (posterior method)
    threshold_log_odds : float
        Threshold for log_odds_gap (posterior method)
    use_adaptive : bool
        Use adaptive per-cluster thresholds

    Returns
    -------
    results : dict containing:
        - 'k': Number of clusters
        - 'posterior_method': dict with posterior-based results
        - 'coassoc_method': dict with original co-association results
        - 'comparison': dict with comparative metrics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing k={k}")
    print(f"{'='*60}")

    # Load bootstrap data
    bootstrap_data = load_bootstrap_data(k)
    print(f"Loaded bootstrap data: {len(bootstrap_data['bootstrap_results'])} iterations")

    # Analyze posteriors
    print("Computing assignment posteriors...")
    posterior_results = analyze_bootstrap_results(bootstrap_data)

    # Classify using posterior method
    print("Classifying using posterior-based method...")
    if use_adaptive:
        classification_posterior = classify_embryos_adaptive(
            max_p=posterior_results['max_p'],
            log_odds_gap=posterior_results['log_odds_gap'],
            modal_cluster=posterior_results['modal_cluster'],
            base_threshold_max_p=threshold_max_p,
            base_threshold_log_odds=threshold_log_odds
        )
    else:
        classification_posterior = classify_embryos_2d(
            max_p=posterior_results['max_p'],
            log_odds_gap=posterior_results['log_odds_gap'],
            modal_cluster=posterior_results['modal_cluster'],
            threshold_max_p=threshold_max_p,
            threshold_log_odds=threshold_log_odds
        )

    summary_posterior = get_classification_summary(classification_posterior)

    print(f"Posterior method: {summary_posterior['n_core']} core "
          f"({summary_posterior['core_fraction']:.1%}), "
          f"{summary_posterior['n_uncertain']} uncertain, "
          f"{summary_posterior['n_outlier']} outliers")

    # Load original co-association results
    coassoc_results = load_original_membership_results(k)

    results = {
        'k': k,
        'posterior_method': {
            'classification': classification_posterior,
            'summary': summary_posterior,
            'posterior_data': posterior_results
        },
        'coassoc_method': coassoc_results,
        'comparison': {}
    }

    # Add comparison metrics if original results available
    if coassoc_results is not None:
        orig_summary = coassoc_results.get('summary', {})
        results['comparison'] = {
            'core_diff': summary_posterior['n_core'] - orig_summary.get('n_core', 0),
            'core_fraction_diff': summary_posterior['core_fraction'] - orig_summary.get('core_fraction', 0.0),
            'improvement_pct': (
                (summary_posterior['core_fraction'] - orig_summary.get('core_fraction', 0.0)) /
                orig_summary.get('core_fraction', 1e-6) * 100 if orig_summary.get('core_fraction', 0.0) > 0 else np.inf
            )
        }
        print(f"Original method: {orig_summary.get('n_core', 'N/A')} core "
              f"({orig_summary.get('core_fraction', 0.0):.1%})")
        print(f"Improvement: +{results['comparison']['core_diff']} core members "
              f"(+{results['comparison']['improvement_pct']:.1f}%)")

    return results


def compare_across_k(k_range: List[int],
                    threshold_max_p: float = 0.8,
                    threshold_log_odds: float = 0.7,
                    use_adaptive: bool = False,
                    output_dir: str = 'output/data') -> Dict:
    """
    Compare methods across multiple k values.

    Parameters
    ----------
    k_range : list of int
        Range of k values to analyze
    threshold_max_p : float
        Threshold for max_p
    threshold_log_odds : float
        Threshold for log_odds_gap
    use_adaptive : bool
        Use adaptive thresholds
    output_dir : str
        Directory to save results

    Returns
    -------
    all_results : dict
        Results for all k values
    """
    all_results = {}

    for k in k_range:
        try:
            results = compare_single_k(k, threshold_max_p, threshold_log_odds, use_adaptive)
            all_results[k] = results

            # Save individual k results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f'posteriors_k{k}.pkl', 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved results to {output_path / f'posteriors_k{k}.pkl'}")

        except Exception as e:
            print(f"Error processing k={k}: {e}")
            continue

    return all_results


def create_comparison_dataframe(all_results: Dict) -> pd.DataFrame:
    """
    Create DataFrame summarizing comparison across k values.

    Parameters
    ----------
    all_results : dict
        Results from compare_across_k()

    Returns
    -------
    df : pd.DataFrame
        Comparison summary table
    """
    rows = []

    for k, results in all_results.items():
        # Posterior method stats
        post_summary = results['posterior_method']['summary']
        post_data = results['posterior_method']['posterior_data']

        row = {
            'k': k,
            'method': 'Posterior',
            'n_core': post_summary['n_core'],
            'n_uncertain': post_summary['n_uncertain'],
            'n_outlier': post_summary['n_outlier'],
            'core_fraction': post_summary['core_fraction'],
            'mean_max_p': np.mean(post_data['max_p']),
            'mean_entropy': np.mean(post_data['entropy']),
            'mean_log_odds_gap': np.mean(post_data['log_odds_gap'])
        }
        rows.append(row)

        # Co-association method stats (if available)
        if results['coassoc_method'] is not None:
            coassoc_summary = results['coassoc_method'].get('summary', {})
            row_coassoc = {
                'k': k,
                'method': 'Co-association',
                'n_core': coassoc_summary.get('n_core', np.nan),
                'n_uncertain': coassoc_summary.get('n_uncertain', np.nan),
                'n_outlier': coassoc_summary.get('n_outlier', np.nan),
                'core_fraction': coassoc_summary.get('core_fraction', np.nan),
                'mean_max_p': np.nan,
                'mean_entropy': np.nan,
                'mean_log_odds_gap': np.nan
            }
            rows.append(row_coassoc)

    df = pd.DataFrame(rows)
    return df


def print_summary_table(df: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY ACROSS K VALUES")
    print("="*80)

    # Pivot to show both methods side by side
    pivot_core = df.pivot(index='k', columns='method', values='core_fraction')
    print("\nCore Membership Fraction:")
    print(pivot_core.to_string())

    pivot_n_core = df.pivot(index='k', columns='method', values='n_core')
    print("\nCore Member Count:")
    print(pivot_n_core.to_string())

    # Posterior-specific metrics
    post_df = df[df['method'] == 'Posterior']
    print("\nPosterior Method Metrics:")
    print(post_df[['k', 'mean_max_p', 'mean_entropy', 'mean_log_odds_gap']].to_string(index=False))

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare clustering quality assessment methods')
    parser.add_argument('--k_min', type=int, default=2, help='Minimum k value')
    parser.add_argument('--k_max', type=int, default=5, help='Maximum k value')
    parser.add_argument('--threshold_max_p', type=float, default=0.8,
                       help='Threshold for max_p (default: 0.8)')
    parser.add_argument('--threshold_log_odds', type=float, default=0.7,
                       help='Threshold for log_odds_gap (default: 0.7)')
    parser.add_argument('--adaptive', action='store_true',
                       help='Use adaptive per-cluster thresholds')
    parser.add_argument('--output_dir', type=str, default='output/data',
                       help='Output directory for results')

    args = parser.parse_args()

    k_range = list(range(args.k_min, args.k_max + 1))

    print("="*80)
    print("CLUSTERING QUALITY COMPARISON")
    print("="*80)
    print(f"K range: {k_range}")
    print(f"Threshold max_p: {args.threshold_max_p}")
    print(f"Threshold log_odds_gap: {args.threshold_log_odds}")
    print(f"Adaptive thresholds: {args.adaptive}")
    print("="*80)

    # Run comparison
    all_results = compare_across_k(
        k_range=k_range,
        threshold_max_p=args.threshold_max_p,
        threshold_log_odds=args.threshold_log_odds,
        use_adaptive=args.adaptive,
        output_dir=args.output_dir
    )

    # Create summary DataFrame
    df = create_comparison_dataframe(all_results)

    # Save DataFrame
    output_path = Path(args.output_dir)
    df.to_csv(output_path / 'comparison_summary.csv', index=False)
    print(f"\nSaved comparison summary to {output_path / 'comparison_summary.csv'}")

    # Save full results
    with open(output_path / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved full results to {output_path / 'all_results.pkl'}")

    # Print summary
    print_summary_table(df)

    print("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
