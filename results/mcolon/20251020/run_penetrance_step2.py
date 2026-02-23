#!/usr/bin/env python3
"""
Step 2: Regression Analysis for Incomplete Penetrance

Quantifies how much of the classifier's predicted mutant probability can be
explained by morphological distance from WT using regression models.

Fits both OLS and logit-transformed models to assess variance explained (R²)
and extract regression coefficients needed for Step 3 cutoff calculation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "20251016"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import penetrance analysis tools
from penetrance_analysis import (
    fit_ols_regression,
    fit_logit_regression,
    compute_regression_metrics,
    compute_residual_diagnostics,
    test_normality,
    test_heteroskedasticity,
    bootstrap_regression_ci,
    plot_regression_fit,
    plot_regression_diagnostics,
    plot_regression_comparison
)


def analyze_genotype_regression(
    genotype: str,
    data_dir: Path,
    plot_dir: Path,
    distance_col: str = 'mean_distance',
    prob_col: str = 'mean_prob'
) -> dict:
    """
    Run regression analysis for one genotype.

    Parameters
    ----------
    genotype : str
        Genotype name (e.g., 'cep290_homozygous')
    data_dir : Path
        Directory with per-embryo metrics CSV
    plot_dir : Path
        Directory for output plots
    distance_col : str
        Distance column name
    prob_col : str
        Probability column name

    Returns
    -------
    dict
        Regression results for both OLS and logit models
    """
    print("=" * 80)
    print(f"REGRESSION ANALYSIS: {genotype.upper()}")
    print("=" * 80)

    # Load per-embryo metrics from Step 1
    metrics_file = data_dir / f"{genotype}_per_embryo_metrics.csv"
    print(f"\nLoading: {metrics_file}")

    if not metrics_file.exists():
        raise FileNotFoundError(f"Per-embryo metrics not found: {metrics_file}")

    embryo_metrics = pd.read_csv(metrics_file)
    print(f"  Loaded {len(embryo_metrics)} embryos")

    # Extract data
    distances = embryo_metrics[distance_col].values
    probabilities = embryo_metrics[prob_col].values

    print(f"\n  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

    results = {}

    # ========================================
    # FIT OLS MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("FITTING OLS MODEL: prob = β0 + β1 * distance + ε")
    print("=" * 80)

    ols_results = fit_ols_regression(distances, probabilities)
    ols_metrics = compute_regression_metrics(ols_results, model_type='ols')

    print(f"\nOLS Regression Results:")
    print(f"  R² = {ols_metrics['r_squared']:.4f}")
    print(f"  Adjusted R² = {ols_metrics['r_squared_adj']:.4f}")
    print(f"  β₀ (Intercept) = {ols_metrics['beta0']:.4f} ± {ols_metrics['beta0_se']:.4f}")
    print(f"  β₁ (Slope) = {ols_metrics['beta1']:.4f} ± {ols_metrics['beta1_se']:.4f}")
    print(f"  95% CI for β₁: [{ols_metrics['beta1_ci_lower']:.4f}, {ols_metrics['beta1_ci_upper']:.4f}]")
    print(f"  F-statistic = {ols_metrics['f_statistic']:.2f}, p = {ols_metrics['f_pvalue']:.3e}")
    print(f"  AIC = {ols_metrics['aic']:.2f}")
    print(f"  BIC = {ols_metrics['bic']:.2f}")

    # Diagnostics
    ols_diagnostics = compute_residual_diagnostics(ols_results)
    normality_test = test_normality(ols_diagnostics['residuals'])
    hetero_test = test_heteroskedasticity(ols_results)

    print(f"\n  Normality test (Shapiro-Wilk):")
    print(f"    W = {normality_test['shapiro_statistic']:.4f}, p = {normality_test['shapiro_pvalue']:.3e}")

    print(f"  Heteroskedasticity test (Breusch-Pagan):")
    print(f"    LM = {hetero_test['bp_statistic']:.4f}, p = {hetero_test['bp_pvalue']:.3e}")

    # Store results
    results['ols'] = {
        'model': ols_results,
        'metrics': ols_metrics,
        'diagnostics': ols_diagnostics,
        'normality': normality_test,
        'heteroskedasticity': hetero_test
    }

    # ========================================
    # FIT LOGIT MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("FITTING LOGIT MODEL: logit(prob) = β0 + β1 * distance + ε")
    print("=" * 80)

    logit_results = fit_logit_regression(distances, probabilities)
    logit_metrics = compute_regression_metrics(logit_results, model_type='logit')

    print(f"\nLogit Regression Results:")
    print(f"  R² = {logit_metrics['r_squared']:.4f}")
    print(f"  Adjusted R² = {logit_metrics['r_squared_adj']:.4f}")
    print(f"  β₀ (Intercept) = {logit_metrics['beta0']:.4f} ± {logit_metrics['beta0_se']:.4f}")
    print(f"  β₁ (Slope) = {logit_metrics['beta1']:.4f} ± {logit_metrics['beta1_se']:.4f}")
    print(f"  95% CI for β₁: [{logit_metrics['beta1_ci_lower']:.4f}, {logit_metrics['beta1_ci_upper']:.4f}]")
    print(f"  F-statistic = {logit_metrics['f_statistic']:.2f}, p = {logit_metrics['f_pvalue']:.3e}")
    print(f"  AIC = {logit_metrics['aic']:.2f}")
    print(f"  BIC = {logit_metrics['bic']:.2f}")

    # Diagnostics
    logit_diagnostics = compute_residual_diagnostics(logit_results)
    logit_normality_test = test_normality(logit_diagnostics['residuals'])
    logit_hetero_test = test_heteroskedasticity(logit_results)

    print(f"\n  Normality test (Shapiro-Wilk):")
    print(f"    W = {logit_normality_test['shapiro_statistic']:.4f}, p = {logit_normality_test['shapiro_pvalue']:.3e}")

    print(f"  Heteroskedasticity test (Breusch-Pagan):")
    print(f"    LM = {logit_hetero_test['bp_statistic']:.4f}, p = {logit_hetero_test['bp_pvalue']:.3e}")

    # Store results
    results['logit'] = {
        'model': logit_results,
        'metrics': logit_metrics,
        'diagnostics': logit_diagnostics,
        'normality': logit_normality_test,
        'heteroskedasticity': logit_hetero_test
    }

    # ========================================
    # MODEL COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'OLS':<15} {'Logit':<15}")
    print("-" * 55)
    print(f"{'R²':<25} {ols_metrics['r_squared']:<15.4f} {logit_metrics['r_squared']:<15.4f}")
    print(f"{'Adjusted R²':<25} {ols_metrics['r_squared_adj']:<15.4f} {logit_metrics['r_squared_adj']:<15.4f}")
    print(f"{'AIC':<25} {ols_metrics['aic']:<15.2f} {logit_metrics['aic']:<15.2f}")
    print(f"{'BIC':<25} {ols_metrics['bic']:<15.2f} {logit_metrics['bic']:<15.2f}")
    print(f"{'Residual Std':<25} {ols_metrics['residual_std']:<15.4f} {logit_metrics['residual_std']:<15.4f}")

    # Determine best model (lower AIC is better)
    best_model = 'OLS' if ols_metrics['aic'] < logit_metrics['aic'] else 'Logit'
    print(f"\nBest model by AIC: {best_model}")

    # ========================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # ========================================
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)

    print("\nComputing bootstrap CIs for OLS model...")
    ols_boot_ci = bootstrap_regression_ci(
        distances,
        probabilities,
        model_type='ols',
        n_bootstrap=1000,
        random_state=42
    )

    print(f"  β₀ 95% CI: [{ols_boot_ci['beta0_ci'][0]:.4f}, {ols_boot_ci['beta0_ci'][1]:.4f}]")
    print(f"  β₁ 95% CI: [{ols_boot_ci['beta1_ci'][0]:.4f}, {ols_boot_ci['beta1_ci'][1]:.4f}]")
    print(f"  R² 95% CI: [{ols_boot_ci['r_squared_ci'][0]:.4f}, {ols_boot_ci['r_squared_ci'][1]:.4f}]")

    print("\nComputing bootstrap CIs for logit model...")
    logit_boot_ci = bootstrap_regression_ci(
        distances,
        probabilities,
        model_type='logit',
        n_bootstrap=1000,
        random_state=42
    )

    print(f"  β₀ 95% CI: [{logit_boot_ci['beta0_ci'][0]:.4f}, {logit_boot_ci['beta0_ci'][1]:.4f}]")
    print(f"  β₁ 95% CI: [{logit_boot_ci['beta1_ci'][0]:.4f}, {logit_boot_ci['beta1_ci'][1]:.4f}]")
    print(f"  R² 95% CI: [{logit_boot_ci['r_squared_ci'][0]:.4f}, {logit_boot_ci['r_squared_ci'][1]:.4f}]")

    results['ols']['bootstrap_ci'] = ols_boot_ci
    results['logit']['bootstrap_ci'] = logit_boot_ci

    # ========================================
    # GENERATE PLOTS
    # ========================================
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # OLS regression fit
    print("\n  Creating OLS regression fit plot...")
    fig = plot_regression_fit(
        embryo_metrics,
        ols_results,
        ols_metrics,
        genotype,
        model_type='ols',
        save_path=plot_dir / f"{genotype}_ols_fit.png"
    )
    plt.close(fig)

    # OLS diagnostics
    print("  Creating OLS diagnostic plots...")
    fig = plot_regression_diagnostics(
        embryo_metrics,
        ols_results,
        ols_diagnostics,
        genotype,
        save_path=plot_dir / f"{genotype}_ols_diagnostics.png"
    )
    plt.close(fig)

    # Logit regression fit
    print("  Creating logit regression fit plot...")
    fig = plot_regression_fit(
        embryo_metrics,
        logit_results,
        logit_metrics,
        genotype,
        model_type='logit',
        save_path=plot_dir / f"{genotype}_logit_fit.png"
    )
    plt.close(fig)

    # Logit diagnostics
    print("  Creating logit diagnostic plots...")
    fig = plot_regression_diagnostics(
        embryo_metrics,
        logit_results,
        logit_diagnostics,
        genotype,
        save_path=plot_dir / f"{genotype}_logit_diagnostics.png"
    )
    plt.close(fig)

    # ========================================
    # SAVE SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Create summary DataFrame for both models
    summary_rows = []

    for model_type in ['ols', 'logit']:
        metrics = results[model_type]['metrics']
        boot_ci = results[model_type]['bootstrap_ci']
        normality = results[model_type]['normality']
        hetero = results[model_type]['heteroskedasticity']

        summary_rows.append({
            'genotype': genotype,
            'model_type': model_type,
            **metrics,
            'beta0_boot_ci_lower': boot_ci['beta0_ci'][0],
            'beta0_boot_ci_upper': boot_ci['beta0_ci'][1],
            'beta1_boot_ci_lower': boot_ci['beta1_ci'][0],
            'beta1_boot_ci_upper': boot_ci['beta1_ci'][1],
            'r_squared_boot_ci_lower': boot_ci['r_squared_ci'][0],
            'r_squared_boot_ci_upper': boot_ci['r_squared_ci'][1],
            'shapiro_w': normality['shapiro_statistic'],
            'shapiro_p': normality['shapiro_pvalue'],
            'bp_statistic': hetero['bp_statistic'],
            'bp_pvalue': hetero['bp_pvalue']
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save to CSV
    output_file = data_dir / f"{genotype}_regression_summary.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return summary_df


def main():
    """Run Step 2 regression analysis for both CEP290 and TMEM67."""

    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020")
    data_dir = output_dir / "data" / "penetrance"
    plot_dir = output_dir / "plots" / "penetrance"

    print("\n" + "=" * 80)
    print("STEP 2: REGRESSION ANALYSIS FOR INCOMPLETE PENETRANCE")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Plot directory: {plot_dir}")

    all_results = []

    # Analyze CEP290
    cep290_summary = analyze_genotype_regression(
        genotype='cep290_homozygous',
        data_dir=data_dir,
        plot_dir=plot_dir
    )
    all_results.append(cep290_summary)

    # Analyze TMEM67
    tmem67_summary = analyze_genotype_regression(
        genotype='tmem67_homozygous',
        data_dir=data_dir,
        plot_dir=plot_dir
    )
    all_results.append(tmem67_summary)

    # ========================================
    # CREATE COMPARISON PLOT
    # ========================================
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOT")
    print("=" * 80)

    # Combine results
    df_all_results = pd.concat(all_results, ignore_index=True)

    # Create comparison plot
    fig = plot_regression_comparison(
        df_all_results,
        save_path=plot_dir / "regression_comparison.png"
    )
    plt.close(fig)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  DATA:")
    print("    - data/penetrance/cep290_homozygous_regression_summary.csv")
    print("    - data/penetrance/tmem67_homozygous_regression_summary.csv")
    print("  PLOTS:")
    print("    - plots/penetrance/cep290_homozygous_ols_fit.png")
    print("    - plots/penetrance/cep290_homozygous_ols_diagnostics.png")
    print("    - plots/penetrance/cep290_homozygous_logit_fit.png")
    print("    - plots/penetrance/cep290_homozygous_logit_diagnostics.png")
    print("    - plots/penetrance/tmem67_homozygous_ols_fit.png")
    print("    - plots/penetrance/tmem67_homozygous_ols_diagnostics.png")
    print("    - plots/penetrance/tmem67_homozygous_logit_fit.png")
    print("    - plots/penetrance/tmem67_homozygous_logit_diagnostics.png")
    print("    - plots/penetrance/regression_comparison.png")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    for _, row in df_all_results.iterrows():
        print(f"\n{row['genotype']} ({row['model_type'].upper()} model):")
        print(f"  R² = {row['r_squared']:.3f} [{row['r_squared_boot_ci_lower']:.3f}, {row['r_squared_boot_ci_upper']:.3f}]")
        print(f"  β₁ = {row['beta1']:.4f} [{row['beta1_boot_ci_lower']:.4f}, {row['beta1_boot_ci_upper']:.4f}]")
        print(f"  AIC = {row['aic']:.2f}")

        # Interpretation
        pct_variance = row['r_squared'] * 100
        print(f"  → Distance explains {pct_variance:.1f}% of variance in predicted probability")

    print("\n" + "=" * 80)
    print("NEXT STEP: Use regression coefficients to define penetrance cutoff (Step 3)")
    print("=" * 80)


if __name__ == "__main__":
    main()
