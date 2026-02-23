#!/usr/bin/env python3
"""
Build predictive models: embeddings ↔ curvature.

This script trains two reciprocal models:

1. Embeddings → Curvature
   - Can we predict curvature metrics from morphology embeddings?
   - Which embedding dimensions are most predictive of curvature?

2. Curvature → Embeddings
   - Can we reconstruct embedding space from simple curvature metrics?
   - Which curvature features are most informative about morphology?

Both models use leave-one-embryo-out (LOEO) cross-validation to avoid data
leakage and fairly evaluate generalization to unseen embryos.

Outputs:
- Model performance metrics (R², RMSE) by genotype
- Feature importance analysis
- Residual plots and error analysis
- Summary tables
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings

warnings.filterwarnings('ignore')

# Import data loading from this directory
from load_data import get_analysis_dataframe, get_genotype_short_name, get_genotype_color


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures' / '04_predictive_models'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables' / '04_predictive_models'
MODEL_DIR = RESULTS_DIR / 'outputs' / 'models' / '04_predictive_models'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to predict
CURVATURE_METRICS = ['arc_length_ratio', 'normalized_baseline_deviation']

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}

RIDGE_PARAMS = {
    'alpha': 1.0,
    'random_state': 42
}


# ============================================================================
# Model 1: Embeddings → Curvature
# ============================================================================

def train_embeddings_to_curvature(
    df,
    embedding_cols,
    curvature_metrics=None,
    model_type='random_forest'
):
    """
    Train model to predict curvature from embeddings.

    Uses leave-one-embryo-out cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
    embedding_cols : list of str
    curvature_metrics : list of str, optional
    model_type : {'random_forest', 'ridge'}

    Returns
    -------
    dict
        {
            'metrics': DataFrame of results,
            'models': dict of trained models,
            'predictions': dict of predictions vs actuals,
            'scalers': dict of scalers
        }
    """
    if curvature_metrics is None:
        curvature_metrics = CURVATURE_METRICS

    results = []
    models_dict = {}
    predictions_dict = {}
    scalers_dict = {}

    X = df[embedding_cols].values
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    for metric in curvature_metrics:
        y = df[metric].values
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Leave-one-embryo-out cross-validation
        embryo_ids = df['embryo_id'].unique()
        y_pred_list = []
        y_true_list = []
        genotypes_list = []

        for test_embryo in embryo_ids:
            # Split data
            train_mask = df['embryo_id'] != test_embryo
            test_mask = df['embryo_id'] == test_embryo

            X_train = X_scaled[train_mask]
            X_test = X_scaled[test_mask]
            y_train = y_scaled[train_mask]
            y_test = y_scaled[test_mask]

            # Train model
            if model_type == 'random_forest':
                model = RandomForestRegressor(**RF_PARAMS)
            else:  # ridge
                model = Ridge(**RIDGE_PARAMS)

            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Inverse transform to original scale
            y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

            y_pred_list.extend(y_pred_orig)
            y_true_list.extend(y_test_orig)
            genotypes_list.extend(df[test_mask]['genotype'].values)

        # Compute metrics
        y_pred_array = np.array(y_pred_list)
        y_true_array = np.array(y_true_list)

        r2 = r2_score(y_true_array, y_pred_array)
        rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
        mae = mean_absolute_error(y_true_array, y_pred_array)

        # Store results
        results.append({
            'metric': metric,
            'model_type': model_type,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_true_array),
        })

        predictions_dict[metric] = {
            'predicted': y_pred_array,
            'actual': y_true_array,
            'genotypes': np.array(genotypes_list)
        }

        # Train final model on all data for inference
        if model_type == 'random_forest':
            final_model = RandomForestRegressor(**RF_PARAMS)
        else:
            final_model = Ridge(**RIDGE_PARAMS)

        final_model.fit(X_scaled, y_scaled)
        models_dict[metric] = final_model
        scalers_dict[metric] = (scaler_X, scaler_y)

    return {
        'metrics': pd.DataFrame(results),
        'models': models_dict,
        'predictions': predictions_dict,
        'scalers': scalers_dict
    }


def get_feature_importance_embeddings_to_curvature(
    trained_models,
    embedding_cols,
    curvature_metrics=None
):
    """
    Extract feature importance for embeddings → curvature models.

    Parameters
    ----------
    trained_models : dict from train_embeddings_to_curvature
    embedding_cols : list of str
    curvature_metrics : list of str, optional

    Returns
    -------
    pd.DataFrame
        Feature importances for each embedding dimension per metric
    """
    if curvature_metrics is None:
        curvature_metrics = CURVATURE_METRICS

    importances = []

    for metric in curvature_metrics:
        model = trained_models['models'][metric]

        if hasattr(model, 'feature_importances_'):
            for emb_idx, emb_col in enumerate(embedding_cols):
                importances.append({
                    'metric': metric,
                    'embedding_dim': emb_col,
                    'importance': model.feature_importances_[emb_idx]
                })

    return pd.DataFrame(importances)


# ============================================================================
# Model 2: Curvature → Embeddings
# ============================================================================

def train_curvature_to_embeddings(
    df,
    embedding_cols,
    curvature_metrics=None,
    model_type='random_forest'
):
    """
    Train model to predict embeddings from curvature metrics.

    Uses leave-one-embryo-out cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
    embedding_cols : list of str
    curvature_metrics : list of str, optional
    model_type : {'random_forest', 'ridge'}

    Returns
    -------
    dict
        Results similar to train_embeddings_to_curvature
    """
    if curvature_metrics is None:
        curvature_metrics = CURVATURE_METRICS

    X = df[curvature_metrics].values
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    Y = df[embedding_cols].values
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    # Leave-one-embryo-out cross-validation
    embryo_ids = df['embryo_id'].unique()
    y_pred_list = []
    y_true_list = []
    genotypes_list = []

    for test_embryo in embryo_ids:
        train_mask = df['embryo_id'] != test_embryo
        test_mask = df['embryo_id'] == test_embryo

        X_train = X_scaled[train_mask]
        X_test = X_scaled[test_mask]
        Y_train = Y_scaled[train_mask]
        Y_test = Y_scaled[test_mask]

        # Train multi-output model
        if model_type == 'random_forest':
            base_model = RandomForestRegressor(**RF_PARAMS)
        else:
            base_model = Ridge(**RIDGE_PARAMS)

        model = MultiOutputRegressor(base_model)
        model.fit(X_train, Y_train)

        # Predict
        Y_pred = model.predict(X_test)

        # Inverse transform
        Y_test_orig = scaler_Y.inverse_transform(Y_test)
        Y_pred_orig = scaler_Y.inverse_transform(Y_pred)

        y_pred_list.extend(Y_pred_orig)
        y_true_list.extend(Y_test_orig)
        genotypes_list.extend(df[test_mask]['genotype'].values)

    # Compute aggregate metrics
    y_pred_array = np.array(y_pred_list)
    y_true_array = np.array(y_true_list)

    r2 = r2_score(y_true_array, y_pred_array)
    rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
    mae = mean_absolute_error(y_true_array, y_pred_array)

    result = {
        'metrics': pd.DataFrame([{
            'direction': 'Curvature → Embeddings',
            'model_type': model_type,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_true_array),
            'n_output_dims': len(embedding_cols)
        }]),
        'predictions': {
            'predicted': y_pred_array,
            'actual': y_true_array,
            'genotypes': np.array(genotypes_list)
        }
    }

    return result


# ============================================================================
# Visualization
# ============================================================================

def plot_model_performance(results_dict, save_dir=FIGURE_DIR):
    """
    Create comparison plots of model performance.

    Parameters
    ----------
    results_dict : dict
        Results from both models
    save_dir : Path
    """
    # Compile all metrics
    embed_to_curv = results_dict['embeddings_to_curvature']['metrics'].copy()
    embed_to_curv['direction'] = 'Embeddings → Curvature'

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = ['r2', 'rmse', 'mae']
    metric_labels = {'r2': 'R² Score', 'rmse': 'RMSE', 'mae': 'Mean Absolute Error'}

    for ax, metric in zip(axes, metrics_to_plot):
        data_to_plot = embed_to_curv[['metric', metric]].copy()
        data_to_plot.columns = ['Metric', 'Value']

        if metric == 'r2':
            ax.set_ylim(-0.1, 1.0)
        else:
            ax.set_ylim(bottom=0)

        sns.barplot(
            data=data_to_plot,
            x='Metric',
            y='Value',
            ax=ax,
            palette='Set2'
        )

        ax.set_title(f'{metric_labels[metric]}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Predictive Model Performance', fontweight='bold', fontsize=13)
    plt.tight_layout()

    save_path = save_dir / 'model_performance.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path


def plot_residuals(results_dict, save_dir=FIGURE_DIR):
    """
    Create residual plots for prediction errors.

    Parameters
    ----------
    results_dict : dict
    save_dir : Path
    """
    predictions = results_dict['embeddings_to_curvature']['predictions']

    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 5))

    if len(predictions) == 1:
        axes = [axes]

    for ax, (metric, pred_dict) in zip(axes, predictions.items()):
        y_true = pred_dict['actual']
        y_pred = pred_dict['predicted']
        genotypes = pred_dict['genotypes']

        residuals = y_true - y_pred

        # Scatter by genotype
        for genotype in np.unique(genotypes):
            mask = genotypes == genotype
            ax.scatter(
                y_pred[mask],
                residuals[mask],
                alpha=0.5,
                s=30,
                label=get_genotype_short_name(genotype),
                color=get_genotype_color(genotype)
            )

        # Zero line
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2)

        ax.set_xlabel('Predicted Value', fontsize=11)
        ax.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
        ax.set_title(metric, fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle('Prediction Residuals by Genotype', fontweight='bold', fontsize=13)
    plt.tight_layout()

    save_path = save_dir / 'residuals.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("PREDICTIVE MODELS: EMBEDDINGS ↔ CURVATURE")
    print("="*80)

    # Load data
    print("\nSTEP 1: LOADING DATA")
    df, metadata = get_analysis_dataframe()
    embedding_cols = metadata['embedding_cols']

    print(f"  Data shape: {df.shape}")
    print(f"  Embedding dimensions: {len(embedding_cols)}")
    print(f"  Curvature metrics: {CURVATURE_METRICS}")

    # Model 1: Embeddings → Curvature
    print("\nSTEP 2: TRAINING EMBEDDINGS → CURVATURE MODELS")

    results_embed_to_curv = train_embeddings_to_curvature(
        df,
        embedding_cols,
        curvature_metrics=CURVATURE_METRICS,
        model_type='random_forest'
    )

    print("\n  Performance Summary:")
    print(results_embed_to_curv['metrics'].to_string(index=False))

    # Feature importance
    print("\nSTEP 3: ANALYZING FEATURE IMPORTANCE")

    feature_imp = get_feature_importance_embeddings_to_curvature(
        results_embed_to_curv,
        embedding_cols,
        curvature_metrics=CURVATURE_METRICS
    )

    # Save top features
    for metric in CURVATURE_METRICS:
        metric_features = feature_imp[feature_imp['metric'] == metric].nlargest(10, 'importance')
        print(f"\n  Top 10 Embedding Dimensions for {metric}:")
        for _, row in metric_features.iterrows():
            print(f"    {row['embedding_dim']:20s}: {row['importance']:.4f}")

    feature_imp_file = TABLE_DIR / 'feature_importance_embeddings_to_curvature.csv'
    feature_imp.to_csv(feature_imp_file, index=False)
    print(f"\n  Saved to {feature_imp_file}")

    # Model 2: Curvature → Embeddings
    print("\nSTEP 4: TRAINING CURVATURE → EMBEDDINGS MODELS")

    results_curv_to_embed = train_curvature_to_embeddings(
        df,
        embedding_cols,
        curvature_metrics=CURVATURE_METRICS,
        model_type='random_forest'
    )

    print("\n  Performance Summary:")
    print(results_curv_to_embed['metrics'].to_string(index=False))

    # Save results
    print("\nSTEP 5: SAVING RESULTS")

    results_embed_to_curv['metrics'].to_csv(
        TABLE_DIR / 'model_performance_embeddings_to_curvature.csv',
        index=False
    )

    results_curv_to_embed['metrics'].to_csv(
        TABLE_DIR / 'model_performance_curvature_to_embeddings.csv',
        index=False
    )

    # Visualization
    print("\nSTEP 6: CREATING VISUALIZATIONS")

    results_dict = {
        'embeddings_to_curvature': results_embed_to_curv,
        'curvature_to_embeddings': results_curv_to_embed
    }

    perf_fig = plot_model_performance(results_dict)
    print(f"  Saved performance plot to {perf_fig}")

    residual_fig = plot_residuals(results_dict)
    print(f"  Saved residual plot to {residual_fig}")

    # Summary
    print("\n" + "="*80)
    print("PREDICTIVE MODEL ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"\n  Embeddings → Curvature:")
    for _, row in results_embed_to_curv['metrics'].iterrows():
        print(f"    {row['metric']:35s}: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}")

    print(f"\n  Curvature → Embeddings:")
    for _, row in results_curv_to_embed['metrics'].iterrows():
        print(f"    R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}")

    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables:  {TABLE_DIR}")
    print(f"  Models:  {MODEL_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
