# 5_assign_trajectory.py
"""Assign new trajectories to clusters using DTW or model-based methods."""

import numpy as np
from typing import Dict, Tuple, List
from scipy.optimize import minimize

# ============ CORE FUNCTIONS ============

def assign_by_dtw(trajectory: Tuple[np.ndarray, np.ndarray], 
                  medoids: List[Tuple], dtw_func: callable) -> int:
    """Assign trajectory to nearest medoid using DTW."""
    t_new, y_new = trajectory
    min_dist = np.inf
    best_cluster = -1
    
    for k, (t_med, y_med) in enumerate(medoids):
        dist = dtw_func(y_new, y_med)  # Assumes your DTW function
        if dist < min_dist:
            min_dist = dist
            best_cluster = k
    
    return best_cluster, min_dist


def compute_model_likelihood(t: np.ndarray, y: np.ndarray, 
                            cluster_model: Dict) -> float:
    """Compute likelihood of trajectory under mixed-effects model."""
    # Center time
    t_centered = t - cluster_model['t_center']
    
    # Estimate best-fitting random effects via MLE
    def neg_log_likelihood(params):
        b0, b1 = params
        y_pred = cluster_model['mean_spline'](t_centered) + b0 + b1 * t_centered
        residuals = y - y_pred
        sigma2 = cluster_model['variance_components']['sigma2']
        
        # Log likelihood of residuals
        ll_resid = -0.5 * len(t) * np.log(2*np.pi*sigma2) - np.sum(residuals**2)/(2*sigma2)
        
        # Log likelihood of random effects
        cov_matrix = cluster_model['variance_components']['cov_matrix']
        b_vec = np.array([b0, b1])
        ll_re = -0.5 * b_vec @ np.linalg.inv(cov_matrix) @ b_vec
        
        return -(ll_resid + ll_re)
    
    # Optimize
    result = minimize(neg_log_likelihood, x0=[0, 0], method='BFGS')
    
    # Return negative of minimum (i.e., maximum likelihood)
    return -result.fun, result.x


def assign_by_model(trajectory: Tuple[np.ndarray, np.ndarray],
                   cluster_models: Dict) -> Tuple[int, Dict]:
    """Assign trajectory to cluster with highest likelihood."""
    t, y = trajectory
    
    best_cluster = -1
    best_ll = -np.inf
    best_re = None
    all_results = {}
    
    for k, model in cluster_models.items():
        ll, re = compute_model_likelihood(t, y, model)
        all_results[k] = {'log_likelihood': ll, 'random_effects': re}
        
        if ll > best_ll:
            best_ll = ll
            best_cluster = k
            best_re = re
    
    # Convert to probabilities (softmax)
    ll_values = np.array([all_results[k]['log_likelihood'] for k in sorted(all_results.keys())])
    ll_values -= np.max(ll_values)  # Numerical stability
    probs = np.exp(ll_values) / np.sum(np.exp(ll_values))
    
    for i, k in enumerate(sorted(all_results.keys())):
        all_results[k]['probability'] = probs[i]
    
    return best_cluster, all_results


def compute_prediction_error(trajectory: Tuple[np.ndarray, np.ndarray],
                            cluster_model: Dict, random_effects: Dict) -> float:
    """Compute RMSE for trajectory under model."""
    t, y = trajectory
    t_centered = t - cluster_model['t_center']
    y_pred = cluster_model['mean_spline'](t_centered) + random_effects['b0'] + random_effects['b1'] * t_centered
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    return rmse


# ============ WRAPPER FUNCTIONS ============

def assign_trajectory(trajectory: Tuple[np.ndarray, np.ndarray],
                     cluster_models: Dict = None,
                     medoids: List[Tuple] = None,
                     dtw_func: callable = None,
                     method: str = 'both') -> Dict:
    """
    Assign a new trajectory to clusters.
    
    Args:
        trajectory: (t, y) tuple
        cluster_models: Dict of fitted models per cluster
        medoids: List of medoid trajectories per cluster
        dtw_func: Your DTW distance function
        method: 'dtw', 'model', or 'both'
    
    Returns:
        Assignment results with cluster, confidence, etc.
    """
    results = {}
    
    # DTW-based assignment
    if method in ['dtw', 'both'] and medoids is not None:
        cluster_dtw, dist = assign_by_dtw(trajectory, medoids, dtw_func)
        results['dtw'] = {
            'cluster': cluster_dtw,
            'distance': dist
        }
    
    # Model-based assignment
    if method in ['model', 'both'] and cluster_models is not None:
        cluster_model, model_results = assign_by_model(trajectory, cluster_models)
        results['model'] = {
            'cluster': cluster_model,
            'results_per_cluster': model_results
        }
        
        # Add prediction error
        best_re = model_results[cluster_model]['random_effects']
        rmse = compute_prediction_error(trajectory, cluster_models[cluster_model], 
                                       {'b0': best_re[0], 'b1': best_re[1]})
        results['model']['rmse'] = rmse
    
    # Agreement check
    if 'dtw' in results and 'model' in results:
        results['agreement'] = results['dtw']['cluster'] == results['model']['cluster']
    
    return results


def batch_assign(trajectories: List[Tuple], **kwargs) -> List[Dict]:
    """Assign multiple trajectories."""
    return [assign_trajectory(traj, **kwargs) for traj in trajectories]


def validate_assignments(test_trajectories: List[Tuple], 
                        test_labels: np.ndarray, **kwargs) -> Dict:
    """Validate assignment accuracy on test set."""
    assignments = batch_assign(test_trajectories, **kwargs)
    
    # Extract predictions
    dtw_preds = [a['dtw']['cluster'] for a in assignments if 'dtw' in a]
    model_preds = [a['model']['cluster'] for a in assignments if 'model' in a]
    
    # Compute accuracies
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    results = {}
    if dtw_preds:
        results['dtw_accuracy'] = accuracy_score(test_labels, dtw_preds)
        results['dtw_confusion'] = confusion_matrix(test_labels, dtw_preds)
    
    if model_preds:
        results['model_accuracy'] = accuracy_score(test_labels, model_preds)
        results['model_confusion'] = confusion_matrix(test_labels, model_preds)
    
    # Agreement between methods
    if dtw_preds and model_preds:
        results['method_agreement'] = accuracy_score(dtw_preds, model_preds)
    
    return results


# ============ PLOTTING FUNCTIONS (signatures only) ============

def plot_assignment(trajectory, cluster_model, random_effects, title="Trajectory Assignment"):
    """Plot new trajectory with assigned cluster fit."""
    pass

def plot_assignment_probabilities(model_results, title="Cluster Probabilities"):
    """Bar plot of assignment probabilities."""
    pass

def plot_confusion_matrix(confusion_mat, title="Assignment Confusion Matrix"):
    """Heatmap of confusion matrix."""
    pass

def plot_validation_comparison(validation_results, title="DTW vs Model Assignment"):
    """Compare assignment methods."""
    pass