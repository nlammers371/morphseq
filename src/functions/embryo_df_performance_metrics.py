# Embryo Performance Metrics Script

import os
import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    auc
)
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import CubicSpline
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import CubicSpline, interp1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline, interp1d


import plotly.graph_objects as go
import plotly.express as px


from datetime import datetime
import warnings






#this is the function that is used to save the paths, they are used the same way in all functions
#change this function so you can save them in the way you want. 
def results_path(sub_directory):
    """Creates a directory path with the current date and sub-directory."""
    today = datetime.now()
    month_year = today.strftime("%m_%Y")
    date = today.strftime("%m_%d_%Y")
    path = os.path.join('results', month_year, date, sub_directory)
    os.makedirs(path, exist_ok=True)  # Create directory if not exists
    return path


def ensure_embryo_id_column(df):
    """
    Ensures the 'embryo_id' column exists in the dataframe by extracting it from the 'snip_id' column.
    If 'embryo_id' already exists, no changes are made.
    
    Parameters:
    df (pd.DataFrame): The dataframe that may contain the 'snip_id' and 'embryo_id' columns.
    
    Returns:
    df (pd.DataFrame): The dataframe with the 'embryo_id' column ensured.
    """
    if 'embryo_id' not in df.columns:
        try:
            df['embryo_id'] = df['snip_id'].str.replace(r'_[^_]*$', '', regex=True)
        except KeyError as e:
            raise KeyError("'snip_id' column not found in the dataframe") from e
    return df

def split_train_test(df, test_size=0.2, random_state=42):
    """
    Splits a dataframe into training and test sets based on unique 'embryo_id'.

    Parameters:
    df (pd.DataFrame): The dataframe containing the 'embryo_id' column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The random seed for reproducibility.

    Returns:
    train_df (pd.DataFrame): The training dataframe.
    test_df (pd.DataFrame): The test dataframe.
    """
    df = ensure_embryo_id_column(df)
    unique_embryo_ids = df["embryo_id"].unique()
    train_ids, test_ids = train_test_split(unique_embryo_ids, test_size=test_size, random_state=random_state)
    train_df = df[df["embryo_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["embryo_id"].isin(test_ids)].reset_index(drop=True)
    
    return train_df, test_df, df

# # Split the data
# df_all_train, df_all_test, df_all = split_train_test(df_all)
# df_hld_train, df_hld_test, df_hld = split_train_test(df_hld)


def logistic_regression_multiclass(train_df, test_df, z_mu_biological_columns, pert_comparisons, tol=1e-3, balanced=False):
    """
    Perform logistic regression for a multiclass classification problem.

    Parameters:
    - train_df (pd.DataFrame): Training DataFrame.
    - test_df (pd.DataFrame): Test DataFrame.
    - z_mu_biological_columns (list): List of feature column names.
    - pert_comparisons (list): List of perturbation class names.
    - tol (float): Tolerance for stopping criteria. Defaults to 1e-3.
    - balanced (bool): Whether to balance the classes in training and test sets. Defaults to False.

    Returns:
    - y_test (np.ndarray): True labels for the test set.
    - y_pred_proba (np.ndarray): Predicted probabilities for the test set.
    - log_reg (LogisticRegression): Trained Logistic Regression model.
    - train_df (pd.DataFrame): Modified training DataFrame with 'class_num'.
    - test_df (pd.DataFrame): Modified test DataFrame with 'class_num'.
    """
    # Create a mapping for the perturbations to integer labels
    perturbation_to_label = {pert: int(i) for i, pert in enumerate(pert_comparisons)}

    # Add 'class_num' column to both train_df and test_df
    train_df['class_num'] = train_df['phenotype'].map(perturbation_to_label)
    test_df['class_num'] = test_df['phenotype'].map(perturbation_to_label)

    # Remove any rows where class_num is NaN
    train_df = train_df.dropna(subset=['class_num'])
    test_df = test_df.dropna(subset=['class_num'])

    # After mapping 'class_num' and dropping NaNs
    train_df['class_num'] = train_df['class_num'].astype(int)
    test_df['class_num'] = test_df['class_num'].astype(int)

    # Balance the classes in the training and test sets if balanced=True
    if balanced:
        # Balance the classes in the training set
        train_class_counts = train_df['class_num'].value_counts()
        min_train_class_size = train_class_counts.min()

        train_df_balanced = train_df.groupby('class_num').apply(
            lambda x: x.sample(n=min_train_class_size, random_state=42)
        ).reset_index(drop=True)

        # Balance the classes in the test set
        test_class_counts = test_df['class_num'].value_counts()
        min_test_class_size = test_class_counts.min()

        test_df_balanced = test_df.groupby('class_num').apply(
            lambda x: x.sample(n=min_test_class_size, random_state=42)
        ).reset_index(drop=True)

        # Update train_df and test_df with balanced data
        train_df = train_df_balanced
        test_df = test_df_balanced

    # Extract features and labels
    X_train = train_df[z_mu_biological_columns].values
    y_train = train_df['class_num'].values

    X_test = test_df[z_mu_biological_columns].values
    y_test = test_df['class_num'].values

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Logistic Regression for multiclass classification
    log_reg = LogisticRegression(
        C=10,
        l1_ratio=0.2,
        penalty='elasticnet',
        solver='saga',
        max_iter=250,  # Increased max_iter for convergence
        multi_class='multinomial',  # Important for multiclass problems
        random_state=42,
        tol=tol
    )

    # Fit the model on the training data
    log_reg.fit(X_train_scaled, y_train)

    # Predict probabilities on the test set
    y_pred_proba = log_reg.predict_proba(X_test_scaled)

    # Return results and modified dataframes
    return y_test, y_pred_proba, log_reg, train_df, test_df

# # --- Example usage with logistic_regression_multiclass (all suffix) ---
# y_test_all, y_pred_proba_all, log_reg_all, train_df_all, test_df_all = logistic_regression_multiclass(
#     df_all_train, df_all_test, z_mu_biological_columns, pert_comparisons)

# # --- Example usage with logistic_regression_multiclass (hld suffix) ---
# y_test_hld, y_pred_proba_hld, log_reg_hld, train_df_hld, test_df_hld = logistic_regression_multiclass(
#     df_hld_train, df_hld_test, z_mu_biological_columns, pert_comparisons)



# --- Function to Compute PR AUC Over Time Bins ---
def compute_pr_auc_over_time_bins(test_results_df, cls, num_bins=20, max_hpf=40):
    """
    Compute PR AUC over time bins for a specific class.

    Parameters:
    - test_results_df (pd.DataFrame): DataFrame containing 'predicted_stage_hpf', 'class_num', and 'y_pred_proba'.
    - cls (int): The class number to compute PR-AUC for.
    - num_bins (int): Number of time bins to divide the data into.
    - max_hpf (float or None): Maximum hpf value to include in the analysis. If None, no filtering is applied.

    Returns:
    - bin_centers (np.ndarray): The centers of the time bins.
    - pr_auc_list (list): The PR-AUC values for each time bin.
    """

    # Filter data based on max_hpf
    if max_hpf is not None:
        data = test_results_df[test_results_df['predicted_stage_hpf'] <= max_hpf]
    else:
        data = test_results_df

    # Define time bins
    time_min = data['predicted_stage_hpf'].min()
    time_max = data['predicted_stage_hpf'].max()
    bins = np.linspace(time_min, time_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    pr_auc_list = []

    # Compute PR-AUC for each time bin
    for i in range(num_bins):
        # Get the data in the current time bin
        bin_mask = (data['predicted_stage_hpf'] >= bins[i]) & (data['predicted_stage_hpf'] < bins[i + 1])
        bin_data = data[bin_mask]

        if not bin_data.empty:
            y_true = bin_data['class_num'] == cls
            y_scores = bin_data['y_pred_proba']

            # Compute number of positive and negative samples
            num_positive = np.sum(y_true)
            num_negative = np.sum(~y_true)

            # Append NaN if either num_positive or num_negative is zero
            if num_positive == 0 or num_negative == 0:
                pr_auc_list.append(np.nan)
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                pr_auc = auc(recall, precision)
                pr_auc_list.append(pr_auc)
        else:
            pr_auc_list.append(np.nan)

    return bin_centers, pr_auc_list

def pr_auc_over_time_multiclass(y_test, y_pred_proba, test_df, perturbations, num_bins=20, max_hpf=40):
    """
    Compute PR AUC over time bins for the multiclass problem without plotting.

    Parameters:
    - y_test (np.ndarray): True class labels.
    - y_pred_proba (np.ndarray): Predicted probabilities from the classifier.
    - test_df (pd.DataFrame): Test DataFrame containing 'predicted_stage_hpf' and 'class_num'.
    - perturbations (list): List of class names corresponding to the classes.
    - num_bins (int): Number of time bins to divide the data into.
    - max_hpf (float or None): Maximum hpf value to include in the analysis. If None, no filtering is applied.

    Returns:
    - results_dict (dict): Dictionary containing bin centers and PR-AUC lists for each perturbation.
    """

    # Prepare the test results dataframe with predicted probabilities and the true class labels
    test_results_df = test_df[['predicted_stage_hpf', 'class_num']].copy()

    # Add predicted probabilities to the DataFrame for each class
    for i, pert in enumerate(perturbations):
        test_results_df[f'y_pred_proba_{pert}'] = y_pred_proba[:, i]

    # Dictionary to store results
    results_dict = {}

    # Compute PR AUC for each perturbation
    for i, pert in enumerate(perturbations):
        # Prepare data for this perturbation
        data = test_results_df[['predicted_stage_hpf', 'class_num', f'y_pred_proba_{pert}']].copy()
        data.rename(columns={f'y_pred_proba_{pert}': 'y_pred_proba'}, inplace=True)

        # Compute PR AUC over time bins
        bin_centers, pr_auc_list = compute_pr_auc_over_time_bins(
            data, cls=i, num_bins=num_bins, max_hpf=max_hpf)

        # Store results
        results_dict[pert] = {
            'bin_centers': bin_centers,
            'pr_auc_list': pr_auc_list
        }

    return results_dict

def plot_pr_auc_over_time(results_dict, perturbations, dataset_label='', title="PR-AUC Over Time for Perturbations", plot=True, save_dir=None):
    """
    Plot PR AUC over time for the multiclass problem using the results from pr_auc_over_time_multiclass.

    Parameters:
    - results_dict (dict): Dictionary containing bin centers and PR-AUC lists for each perturbation.
    - perturbations (list): List of perturbation names.
    - dataset_label (str): Label to differentiate datasets (e.g., 'all', 'hld'). Default is ''.
    - title (str): Title of the plot.
    - plot (bool): Whether to display the plot. Defaults to True.
    - save (bool): Whether to save the plot. Defaults to False.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))

    # Plot PR AUC for each perturbation
    for pert in perturbations:
        bin_centers = results_dict[pert]['bin_centers']
        pr_auc_list = results_dict[pert]['pr_auc_list']
        plt.plot(bin_centers, pr_auc_list, label=f'{pert} PR AUC', marker='o')

    plt.xlabel("Time (hpf)")
    plt.ylabel("PR-AUC")
    plt.title(f"{title} ({dataset_label})" if dataset_label else title)
    plt.legend()
    plt.grid(True)

    # Save the PR-AUC plot if save=True
    if save_dir:
        filename = f"pr_auc_over_time_multiclass_{dataset_label}.png" if dataset_label else "pr_auc_over_time_multiclass.png"
        pr_auc_plot_path = os.path.join(save_dir, filename)
        plt.savefig(pr_auc_plot_path)
        print("PR-AUC plot saved to:", pr_auc_plot_path)

    # Display the plot if plot=True
    if plot:
        plt.show()

    # Close the plot to free memory
    plt.close()

def plot_pr_auc_comparison(results_dict1, results_dict2, perturbation, labels, title, filename_suffix=''):
    """
    Plot PR AUC over time for a specific perturbation from two different results_dicts.

    Parameters:
    - results_dict1 (dict): First results dictionary.
    - results_dict2 (dict): Second results dictionary.
    - perturbation (str): The perturbation to plot.
    - labels (tuple): A tuple of labels for the two datasets (e.g., ('All Data', 'Held-out Data')).
    - title (str): Title of the plot.
    - filename_suffix (str): Suffix to differentiate the filename.

    Returns:
    - None
    """

    plt.figure(figsize=(10, 6))

    # Dataset 1
    bin_centers1 = results_dict1[perturbation]['bin_centers']
    pr_auc_list1 = results_dict1[perturbation]['pr_auc_list']
    plt.plot(bin_centers1, pr_auc_list1, label=f'{labels[0]} - {perturbation}', marker='o')

    # Dataset 2
    bin_centers2 = results_dict2[perturbation]['bin_centers']
    pr_auc_list2 = results_dict2[perturbation]['pr_auc_list']
    plt.plot(bin_centers2, pr_auc_list2, label=f'{labels[1]} - {perturbation}', marker='x')

    plt.xlabel("Time (hpf)")
    plt.ylabel("PR-AUC")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the comparison plot
    filename = f"pr_auc_comparison_{perturbation}_{filename_suffix}.png" if filename_suffix else f"pr_auc_comparison_{perturbation}.png"
    pr_auc_plot_path = os.path.join(results_path("multiclass_classification_test"), filename)
    plt.savefig(pr_auc_plot_path)
    plt.close()

    print(f"Comparison plot saved to: {pr_auc_plot_path}")

def create_pr_auc_dataframe(results_dict_all, results_dict_hld, perturbations):
    """
    Create a dataframe containing per-bin PR-AUC values for each perturbation and each dataset.
    
    Parameters:
    - results_dict_all (dict): Results dictionary from 'all' dataset.
    - results_dict_hld (dict): Results dictionary from 'hld' dataset.
    - perturbations (list): List of perturbation names.
    
    Returns:
    - pr_auc_df (pd.DataFrame): Dataframe containing per-bin PR-AUC values.
    """
    data = []

    for pert in perturbations:
        # Get bin centers and PR-AUC values for both datasets
        bin_centers_all = results_dict_all[pert]['bin_centers']
        pr_auc_all = results_dict_all[pert]['pr_auc_list']
        
        bin_centers_hld = results_dict_hld[pert]['bin_centers']
        pr_auc_hld = results_dict_hld[pert]['pr_auc_list']

        # Round bin centers to the nearest integer
        bin_centers_all_rounded = np.round(bin_centers_all).astype(int)
        bin_centers_hld_rounded = np.round(bin_centers_hld).astype(int)

        # Ensure that the rounded bin centers match between datasets
        if not np.array_equal(bin_centers_all_rounded, bin_centers_hld_rounded):
            raise ValueError(f"Rounded bin centers for perturbation '{pert}' do not match between datasets.")
        
        # For each bin, record the PR-AUC values
        for i in range(len(bin_centers_all)):
            data.append({
                'perturbation': pert,
                'bin': i + 1,  # Bins numbered from 1 to num_bins
                'bin_center': bin_centers_all[i],
                'pr_auc_all': pr_auc_all[i],
                'pr_auc_hld': pr_auc_hld[i]
            })
    
    # Create DataFrame
    pr_auc_df = pd.DataFrame(data)
    
    return pr_auc_df

def compute_average_pr_auc(pr_auc_df):
    """
    Compute the average PR-AUC across bins for each perturbation and dataset.
    
    Parameters:
    - pr_auc_df (pd.DataFrame): Dataframe containing per-bin PR-AUC values.
    
    Returns:
    - avg_pr_auc_df (pd.DataFrame): Dataframe containing average PR-AUC and differences.
    """
    # Group by perturbation and compute mean PR-AUC, ignoring NaN values
    avg_pr_auc = pr_auc_df.groupby('perturbation').agg({
        'pr_auc_all': 'mean',
        'pr_auc_hld': 'mean'
    }).reset_index()
    
    # Calculate the difference
    avg_pr_auc['difference'] = avg_pr_auc['pr_auc_all'] - avg_pr_auc['pr_auc_hld']
    
    return avg_pr_auc

def plot_average_pr_auc_difference(avg_pr_auc_df, plot=True, save=False):
    """
    Plot the differences in average PR-AUC between the two datasets for each perturbation.
    
    Parameters:
    - avg_pr_auc_df (pd.DataFrame): DataFrame containing 'perturbation' and 'difference' columns.
    - plot (bool): Whether to display the plot interactively. Defaults to True.
    - save (bool): Whether to save the plot to a file. Defaults to False.
    
    Returns:
    - None
    """
    plt.figure(figsize=(12, 8))
    perturbations = avg_pr_auc_df['perturbation']
    differences = avg_pr_auc_df['difference']
    
    bars = plt.bar(perturbations, differences, color='skyblue', edgecolor='black')
    plt.xlabel('Perturbation', fontsize=14)
    plt.ylabel('Difference in Average PR-AUC (All - Held-out)', fontsize=14)
    plt.title('Difference in Average PR-AUC Between Datasets', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with difference values
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=12)
    
    # Save the plot if save=True
    if save:
        filename = "average_pr_auc_difference.png"
        plot_path = os.path.join(results_path("multiclass_classification_test"), filename)
        plt.savefig(plot_path, bbox_inches='tight')
        print("Average PR-AUC difference plot saved to:", plot_path)
    
    # Display the plot if plot=True
    if plot:
        plt.show()
    
    # Close the plot to free memory
    plt.close()

def save_dataframes(pr_auc_df, avg_pr_auc_df):
    """
    Save the dataframes as CSV files.
    
    Parameters:
    - pr_auc_df (pd.DataFrame): Dataframe containing per-bin PR-AUC values.
    - avg_pr_auc_df (pd.DataFrame): Dataframe containing average PR-AUC and differences.
    
    Returns:
    - None
    """
    # Define paths
    results_dir = results_path("multiclass_classification_test")
    per_bin_csv_path = os.path.join(results_dir, "per_bin_pr_auc_values.csv")
    avg_pr_auc_csv_path = os.path.join(results_dir, "average_pr_auc_differences.csv")
    
    # Save dataframes
    pr_auc_df.to_csv(per_bin_csv_path, index=False)
    avg_pr_auc_df.to_csv(avg_pr_auc_csv_path, index=False)
    
    print("Per-bin PR-AUC values saved to:", per_bin_csv_path)
    print("Average PR-AUC differences saved to:", avg_pr_auc_csv_path)


# # --- Example usage with logistic_regression_multiclass ---
# results_dict_all = pr_auc_over_time_multiclass(y_test_all, y_pred_proba_all, test_df_all, pert_comparisons)

# # Now is available globally in test_df, and you can use it
# results_dict_hld = pr_auc_over_time_multiclass(y_test_hld, y_pred_proba_hld, test_df_hld, pert_comparisons)


# plot_pr_auc_over_time(results_dict_all, pert_comparisons,dataset_label=f'all_test_{pert_comparisons}' , title="PR-AUC Over Time for All Data")

# plot_pr_auc_over_time(results_dict_hld, pert_comparisons, dataset_label= f'hld_test_{pert_comparisons}', title="PR-AUC Over Time for Held-out Data")


# # --- Main Processing ---
# # Step 1: Create per-bin PR-AUC dataframe
# pr_auc_df = create_pr_auc_dataframe(results_dict_all, results_dict_hld, pert_comparisons)

# # Step 2: Compute average PR-AUC and differences
# avg_pr_auc_df = compute_average_pr_auc(pr_auc_df)

# # Step 3: Plot the differences in average PR-AUC
# plot_average_pr_auc_difference(avg_pr_auc_df)

# # Step 4: Save dataframes as CSV files
# # save_dataframes(pr_auc_df, avg_pr_auc_df)


from sklearn.metrics import f1_score

def compute_f1_score_over_time_bins(data, cls, num_bins=20, max_hpf=40):
    """
    Compute F1 scores over time bins for a specific class.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'predicted_stage_hpf', 'y_true', and 'y_pred'.
    - num_bins (int): Number of time bins to divide the data into.
    - max_hpf (float or None): Maximum hpf value to include in the analysis. If None, no filtering is applied.

    Returns:
    - bin_centers (np.ndarray): The centers of the time bins.
    - f1_score_list (list): The F1 scores for each time bin.
    """
    # Filter data based on max_hpf
    if max_hpf is not None:
        data = data[data['predicted_stage_hpf'] <= max_hpf]

    # Check if data is empty after filtering
    if data.empty:
        print("No data available after filtering by max_hpf.")
        return np.array([]), []

    # Define time bins
    time_min = data['predicted_stage_hpf'].min()
    time_max = data['predicted_stage_hpf'].max()
    bins = np.linspace(time_min, time_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    f1_score_list = []

    # Compute F1 score for each time bin
    for i in range(num_bins):
        # Get the data in the current time bin
        bin_mask = (data['predicted_stage_hpf'] >= bins[i]) & (data['predicted_stage_hpf'] < bins[i + 1])
        bin_data = data[bin_mask]

        if not bin_data.empty:
            y_true  = bin_data['class_num'] == cls
            y_pred  = bin_data['y_pred']    == cls

            # Compute number of positive and negative samples
            num_positive = np.sum(y_true)
            num_negative = np.sum(~y_true)

            # Append NaN if either num_positive or num_negative is zero
            if num_positive == 0 or num_negative == 0:
                f1_score_list.append(np.nan)
            else:
                f1 = f1_score(y_true, y_pred)
                f1_score_list.append(f1)
        else:
            f1_score_list.append(np.nan)

    return bin_centers, f1_score_list

def f1_score_over_time_multiclass(y_test, y_pred_proba, test_df, perturbations, num_bins=20, max_hpf=40):
    """
    Compute F1 scores over time bins for the multiclass problem without plotting.

    Parameters:
    - y_test (np.ndarray): True class labels.
    - y_pred_proba (np.ndarray): Predicted probabilities from the classifier.
    - test_df (pd.DataFrame): Test DataFrame containing 'predicted_stage_hpf' and 'class_num'.
    - perturbations (list): List of class names corresponding to the classes.
    - num_bins (int): Number of time bins to divide the data into.
    - max_hpf (float or None): Maximum hpf value to include in the analysis. If None, no filtering is applied.

    Returns:
    - results_dict (dict): Dictionary containing bin centers and F1 score lists for each perturbation.
    """
    # Prepare the test results dataframe with predicted probabilities and the true class labels
    test_results_df = test_df[['predicted_stage_hpf', 'class_num']].copy()

    # Add predicted probabilities to the DataFrame for each class
    for i, pert in enumerate(perturbations):
        test_results_df[f'y_pred_proba_{pert}'] = y_pred_proba[:, i]

    # Add predicted class labels (argmax over predicted probabilities)
    y_pred = np.argmax(y_pred_proba, axis=1)
    test_results_df['y_pred'] = y_pred.astype(int)
    test_results_df['y_true'] = y_test.astype(int)

    # Dictionary to store results
    results_dict = {}

    # Compute F1 scores for each perturbation
    for i, pert in enumerate(perturbations):
        # Prepare data for this perturbation
        data = test_results_df[['predicted_stage_hpf', 'class_num', f'y_pred_proba_{pert}', 'y_true', 'y_pred']].copy()
        data.rename(columns={f'y_pred_proba_{pert}': 'y_pred_proba'}, inplace=True)

        # Compute F1 score over time bins
        bin_centers, f1_score_list = compute_f1_score_over_time_bins(
            data, cls=i, num_bins=num_bins, max_hpf=max_hpf)

        # Store results
        results_dict[pert] = {
            'bin_centers': bin_centers,
            'f1_score_list': f1_score_list
        }

    return results_dict

def plot_f1_score_over_time(results_dict, perturbations, dataset_label='', title="F1 Score Over Time for Perturbations", plot=True, save_dir=None):
    """
    Plot F1 scores over time for the multiclass problem using the results from f1_score_over_time_multiclass.

    Parameters:
    - results_dict (dict): Dictionary containing bin centers and F1 score lists for each perturbation.
    - perturbations (list): List of perturbation names.
    - dataset_label (str): Label to differentiate datasets (e.g., 'all', 'hld'). Default is ''.
    - title (str): Title of the plot.
    - plot (bool): Whether to display the plot. Defaults to True.
    - save (bool): Whether to save the plot. Defaults to False.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))

    # Plot F1 score for each perturbation
    for pert in perturbations:
        bin_centers = results_dict[pert]['bin_centers']
        f1_score_list = results_dict[pert]['f1_score_list']
        plt.plot(bin_centers, f1_score_list, label=f'{pert} F1 Score', marker='o')

    plt.xlabel("Time (hpf)")
    plt.ylabel("F1 Score")
    plt.title(f"{title} ({dataset_label})" if dataset_label else title)
    plt.legend()
    plt.grid(True)

    # Save the F1 score plot if save=True
    if save_dir:
        # Replace spaces with underscores in the filename
        filename = f"f1_score_over_time_multiclass_{dataset_label}.png" if dataset_label else "f1_score_over_time_multiclass.png"
        filename = filename.replace(" ", "_")
        
        f1_score_plot_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f1_score_plot_path)
        print("F1 score plot saved to:", f1_score_plot_path)

    # Display the plot if plot=True
    if plot:
        plt.show()

    # Close the plot to free memory
    plt.close()

def plot_f1_score_comparison(results_dict1, results_dict2, perturbation, labels, title, filename_suffix=''):
    """
    Plot F1 scores over time for a specific perturbation from two different results_dicts.

    Parameters:
    - results_dict1 (dict): First results dictionary.
    - results_dict2 (dict): Second results dictionary.
    - perturbation (str): The perturbation to plot.
    - labels (tuple): A tuple of labels for the two datasets (e.g., ('All Data', 'Held-out Data')).
    - title (str): Title of the plot.
    - filename_suffix (str): Suffix to differentiate the filename.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))

    # Dataset 1
    bin_centers1 = results_dict1[perturbation]['bin_centers']
    f1_score_list1 = results_dict1[perturbation]['f1_score_list']
    plt.plot(bin_centers1, f1_score_list1, labels=f'{labels[0]} - {perturbation}', marker='o')

    # Dataset 2
    bin_centers2 = results_dict2[perturbation]['bin_centers']
    f1_score_list2 = results_dict2[perturbation]['f1_score_list']
    plt.plot(bin_centers2, f1_score_list2, label=f'{labels[1]} - {perturbation}', marker='x')

    plt.xlabel("Time (hpf)")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the comparison plot
    filename = f"f1_score_comparison_{perturbation}_{filename_suffix}.png" if filename_suffix else f"f1_score_comparison_{perturbation}.png"
    f1_score_plot_path = os.path.join(results_path("multiclass_classification_test"), filename)
    plt.savefig(f1_score_plot_path)
    plt.close()

    print(f"Comparison plot saved to: {f1_score_plot_path}")

def create_f1_score_dataframe(results_dict_all, results_dict_hld, perturbations):
    """
    Create a dataframe containing per-bin F1 scores and their differences for each perturbation and each dataset.
    
    Parameters:
    - results_dict_all (dict): Results dictionary from 'all' dataset.
    - results_dict_hld (dict): Results dictionary from 'hld' dataset.
    - perturbations (list): List of perturbation names.
    
    Returns:
    - f1_score_df (pd.DataFrame): Dataframe containing per-bin F1 scores and differences.
    """
    data = []

    for pert in perturbations:
        # Get bin centers and F1 scores for both datasets
        bin_centers_all = results_dict_all[pert]['bin_centers']
        f1_score_all = results_dict_all[pert]['f1_score_list']
        
        bin_centers_hld = results_dict_hld[pert]['bin_centers']
        f1_score_hld = results_dict_hld[pert]['f1_score_list']

        # Round bin centers to the nearest integer
        bin_centers_all_rounded = np.round(bin_centers_all).astype(int)
        bin_centers_hld_rounded = np.round(bin_centers_hld).astype(int)

        # Ensure that the rounded bin centers match between datasets
        if not np.array_equal(bin_centers_all_rounded, bin_centers_hld_rounded):
            raise ValueError(f"Bin centers for perturbation '{pert}' do not match between datasets.")
        
        # For each bin, record the F1 scores and their difference
        for i in range(len(bin_centers_all)):
            difference = f1_score_all[i] - f1_score_hld[i]
            data.append({
                'Perturbation': pert,
                'bin': i + 1,  # Bins numbered from 1 to num_bins
                'bin_center': bin_centers_all_rounded[i],
                'F1_score_all': f1_score_all[i],
                'F1_score_hld': f1_score_hld[i],
                'F1_all_hld_diff': difference
            })

    # Create DataFrame
    f1_score_df = pd.DataFrame(data)
    
    return f1_score_df

def compute_average_f1_score(f1_score_df):
    """
    Compute the average F1 score across bins for each perturbation and dataset, and calculate the differences.
    
    Parameters:
    - f1_score_df (pd.DataFrame): Dataframe containing per-bin F1 scores.
    
    Returns:
    - avg_f1_score_df (pd.DataFrame): Dataframe containing average F1 scores and differences.
    """
    # Group by perturbation and compute mean F1 score, ignoring NaN values
    avg_f1_score = f1_score_df.groupby('Perturbation').agg({
        'F1_score_all': 'mean',
        'F1_score_hld': 'mean'
    }).reset_index()
    
    # Calculate the difference
    avg_f1_score['F1_all_hld_diff'] = avg_f1_score['F1_score_all'] - avg_f1_score['F1_score_hld']
    
    return avg_f1_score

def plot_average_f1_score_difference(avg_f1_score_df, plot=True, save=False):
    """
    Plot the differences in average F1 scores between the two datasets for each perturbation.
    
    Parameters:
    - avg_f1_score_df (pd.DataFrame): DataFrame containing 'perturbation' and 'difference' columns.
    - plot (bool): Whether to display the plot interactively. Defaults to True.
    - save (bool): Whether to save the plot to a file. Defaults to False.
    
    Returns:
    - None
    """
    plt.figure(figsize=(12, 8))
    perturbations = avg_f1_score_df['Perturbation']
    differences = avg_f1_score_df['F1_all_hld_diff']
    
    bars = plt.bar(perturbations, differences, color='skyblue', edgecolor='black')
    plt.xlabel('Perturbation', fontsize=14)
    plt.ylabel('Difference in Average F1 Score (All - Held-out)', fontsize=14)
    plt.title('Difference in Average F1 Score Between Datasets', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with difference values
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=12)
    
    # Save the plot if save=True
    if save:
        filename = "average_f1_score_difference.png"
        plot_path = os.path.join(results_path("multiclass_classification_test"), filename)
        plt.savefig(plot_path, bbox_inches='tight')
        print("Average F1 Score difference plot saved to:", plot_path)
    
    # Display the plot if plot=True
    if plot:
        plt.show()
    
    # Close the plot to free memory
    plt.close()


# Assuming you have y_test, y_pred, test_df, and perturbations defined

# Compute F1 scores over time for the multiclass problem

# ##all
# results_dict_all = f1_score_over_time_multiclass(y_test_all, y_pred_proba_all, test_df_all, pert_comparisons, num_bins=20, max_hpf=40)
# dataset_label = 'all_perts_F1'
# plot_f1_score_over_time(results_dict_all, pert_comparisons, dataset_label=dataset_label, title="F1 Score Over Time for Perturbations")

# #hld
# results_dict_hld = f1_score_over_time_multiclass(y_test_hld, y_pred_proba_hld, test_df_hld, pert_comparisons, num_bins=20, max_hpf=40)
# dataset_label = 'hld_gdf3_lmx1b_perts_F1'
# plot_f1_score_over_time(results_dict_hld, pert_comparisons, dataset_label=dataset_label, title="F1 Score Over Time for Perturbations")


# # --- Main Processing ---
# # Step 1: Create per-bin F1 score dataframe
# f1_score_df = create_f1_score_dataframe(results_dict_all, results_dict_hld, pert_comparisons)

# # # Step 2: Compute average F1 scores and differences
# avg_f1_score_df = compute_average_f1_score(f1_score_df)

# # # Step 3: Plot the differences in average F1 scores
# plot_average_f1_score_difference(avg_f1_score_df)


# # f1_score_df

# # avg_f1_score_df

# import numpy as np
# import pandas as pd


# # List of DataFrames and titles for processing
# dataframes = [
#     (df_all, 'Original Data'),
#     (df_hld, 'No lmx1b gdf3')
# ]

# random_state = 100

# # List of perturbations to process
# pert_comparisons

# # Subsample fraction
# subsample_fraction = 0.05  # Adjust as needed

# Functions to compute metrics
def compute_graph_metrics(z_mu_data):
    """
    Compute graph metrics for the given data.
    """
    k_neighbors = 10
    knn = NearestNeighbors(n_neighbors=min(k_neighbors, len(z_mu_data)))
    knn.fit(z_mu_data)
    knn_graph = knn.kneighbors_graph(z_mu_data, mode='connectivity')
    G = nx.Graph(knn_graph)

    metrics = {}
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        # Compute metrics on the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc).copy()
        metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
    metrics['clustering_coeff'] = nx.average_clustering(G)
    return metrics

def compute_histogram(distances, bins=30):
    """
    Compute histogram counts and bin edges for the given distances.
    """
    counts, bin_edges = np.histogram(distances, bins=bins, density=True)
    return counts, bin_edges

def compute_kl_divergence(ref_counts, counts):
    """
    Compute KL divergence between the reference histogram counts and current counts.
    """
    epsilon = 1e-10  # To prevent division by zero
    P = ref_counts + epsilon
    Q = counts + epsilon
    P /= np.sum(P)
    Q /= np.sum(Q)
    KL_div = np.sum(P * np.log(P / Q))
    return KL_div

def compute_metrics_for_dataframes(dataframes, comparisons, z_mu_biological_columns,subsample_fraction=0.05, random_state=100):
    """
    Compute metrics for given dataframes and comparisons.

    Parameters:
    - dataframes: List of tuples (DataFrame, title)
    - comparisons: List of perturbations to process
    - subsample_fraction: Fraction of data to subsample
    - random_state: Seed for reproducibility

    Returns:
    - metrics_intra_df: DataFrame containing intra-perturbation metrics
    - metrics_inter_df: DataFrame containing inter-perturbation metrics
    """
    # Prepare dictionaries to store results
    reference_histograms = {'intra': {}, 'inter': {}}
    metrics_intra_list = []
    metrics_inter_list = []

    # Loop over intra- and inter-perturbation distances
    for distance_type in ['intra', 'inter']:
        # Loop over each DataFrame and its title
        for df_idx, (df, df_title) in enumerate(dataframes):
            # Sample data for all perturbations
            subsampled_data = {}
            for perturbation in comparisons:
                df_pert = df[df['phenotype'] == perturbation].copy()
                if df_pert.empty:
                    # Skip perturbations not present in the DataFrame
                    continue
                embryo_ids = df_pert['embryo_id'].unique()

                # Initialize a list to store samples from each embryo_id
                samples_per_embryo = []

                for embryo_id in embryo_ids:
                    df_embryo = df_pert[df_pert['embryo_id'] == embryo_id]
                    # Ensure at least one data point is sampled from each embryo_id
                    n_samples = max(int(len(df_embryo) * subsample_fraction), 1)
                    df_embryo_sampled = df_embryo.sample(n=n_samples, random_state=random_state)
                    samples_per_embryo.append(df_embryo_sampled)

                # Combine samples from all embryo_ids for this perturbation
                df_pert_sampled = pd.concat(samples_per_embryo, ignore_index=True)
                subsampled_data[perturbation] = df_pert_sampled

            # Combine all subsampled data into a global dataset
            df_sampled = pd.concat(subsampled_data.values(), ignore_index=True)
            z_mu_data_sampled = df_sampled[z_mu_biological_columns].values
            # Compute global mean and std for z-scoring
            full_distances_sampled = pdist(z_mu_data_sampled, metric='euclidean')
            mean_dist = np.mean(full_distances_sampled)
            std_dist = np.std(full_distances_sampled)

            # Loop over the perturbations to compare
            for perturbation in comparisons:
                if perturbation not in subsampled_data:
                    continue  # Skip if perturbation data is missing

                df_pert_sampled = subsampled_data[perturbation]
                z_mu_data_pert = df_pert_sampled[z_mu_biological_columns].values

                # Ensure there are enough points
                if len(z_mu_data_pert) < 2:
                    continue

                if distance_type == 'intra':
                    # Intra-perturbation distances
                    distances = pdist(z_mu_data_pert, metric='euclidean')
                    z_mu_data_for_metrics = z_mu_data_pert  # For graph metrics
                else:
                    # Inter-perturbation distances
                    # Sample points from other perturbations
                    other_perturbations = [p for p in comparisons if p != perturbation and p in subsampled_data]
                    if not other_perturbations:
                        continue
                    df_other = pd.concat([subsampled_data[p] for p in other_perturbations], ignore_index=True)
                    z_mu_data_other = df_other[z_mu_biological_columns].values

                    # Ensure same number of points
                    n_samples = len(z_mu_data_pert)
                    if len(z_mu_data_other) > n_samples:
                        indices = np.random.choice(len(z_mu_data_other), n_samples, replace=False)
                        z_mu_data_other = z_mu_data_other[indices]
                    elif len(z_mu_data_other) < 2:
                        continue

                    # Compute distances between z_mu_data_pert and z_mu_data_other
                    distances = cdist(z_mu_data_pert, z_mu_data_other, metric='euclidean').flatten()
                    # For graph metrics, combine both datasets
                    z_mu_data_for_metrics = np.vstack((z_mu_data_pert, z_mu_data_other))

                # Z-score distances
                distances_z = (distances - mean_dist) / std_dist

                # Compute metrics
                metrics = compute_graph_metrics(z_mu_data_for_metrics)

                # Compute histogram and KL divergence
                # Use the same bin edges across all datasets for consistency
                if df_title == 'Original Data':
                    # Compute histogram counts and bin edges for the reference dataset
                    counts, bin_edges = compute_histogram(distances_z, bins=30)
                    # Store the reference histogram
                    reference_histograms[distance_type][perturbation] = {'counts': counts, 'bin_edges': bin_edges}
                    KL_div = 0.0  # KL divergence is zero for the reference
                else:
                    # Use the reference histogram's bin edges
                    ref_hist = reference_histograms[distance_type].get(perturbation)
                    if ref_hist is None:
                        continue

                    bin_edges = ref_hist['bin_edges']

                    # Compute histogram counts using the same bin edges
                    counts, _ = np.histogram(distances_z, bins=bin_edges, density=True)

                    # Compute KL divergence
                    KL_div = compute_kl_divergence(ref_hist['counts'], counts)

                # Store metrics in the appropriate list
                metrics_entry = {
                    'DataFrame': df_title,
                    'Perturbation': perturbation,
                    'Avg_Path_Length': metrics['avg_path_length'],
                    'Clustering_Coeff': metrics['clustering_coeff'],
                    'KL_Divergence': KL_div
                }

                if distance_type == 'intra':
                    metrics_intra_list.append(metrics_entry)
                else:
                    metrics_inter_list.append(metrics_entry)

    # Convert the lists of metrics to DataFrames
    metrics_intra_df = pd.DataFrame(metrics_intra_list)
    metrics_inter_df = pd.DataFrame(metrics_inter_list)

    return metrics_intra_df, metrics_inter_df

def compute_differences(metrics_intra_df, metrics_inter_df, reference_title, comparison_title):
    """
    Compute the differences between the metrics of the comparison DataFrame and the reference DataFrame.

    Parameters:
    - metrics_intra_df: DataFrame containing intra-perturbation metrics
    - metrics_inter_df: DataFrame containing inter-perturbation metrics
    - reference_title: Title of the reference DataFrame (e.g., 'Original Data')
    - comparison_title: Title of the comparison DataFrame (e.g., 'No lmx1b gdf3')

    Returns:
    - diff_intra: DataFrame containing differences in intra-perturbation metrics
    - diff_inter: DataFrame containing differences in inter-perturbation metrics
    """
    # Intra-Perturbation Differences
    metrics_intra_reference = metrics_intra_df[metrics_intra_df['DataFrame'] == reference_title]
    metrics_intra_comparison = metrics_intra_df[metrics_intra_df['DataFrame'] == comparison_title]

    # Merge on 'Perturbation'
    merged_intra = pd.merge(
        metrics_intra_reference,
        metrics_intra_comparison,
        on='Perturbation',
        suffixes=('_reference', '_comparison')
    )

    # Compute differences
    merged_intra['Avg_Path_Length_Diff_intra'] = merged_intra['Avg_Path_Length_reference'] - merged_intra['Avg_Path_Length_comparison']
    merged_intra['Clustering_Coeff_Diff_intra'] = merged_intra['Clustering_Coeff_reference'] - merged_intra['Clustering_Coeff_comparison']
    merged_intra['KL_Divergence_Diff_intra'] = merged_intra['KL_Divergence_reference'] - merged_intra['KL_Divergence_comparison']

    # Select relevant columns
    diff_intra = merged_intra[['Perturbation', 'Avg_Path_Length_Diff_intra', 'Clustering_Coeff_Diff_intra', 'KL_Divergence_Diff_intra']]

    # Inter-Perturbation Differences
    metrics_inter_reference = metrics_inter_df[metrics_inter_df['DataFrame'] == reference_title]
    metrics_inter_comparison = metrics_inter_df[metrics_inter_df['DataFrame'] == comparison_title]

    # Merge on 'Perturbation'
    merged_inter = pd.merge(
        metrics_inter_reference,
        metrics_inter_comparison,
        on='Perturbation',
        suffixes=('_reference', '_comparison')
    )

    # Compute differences
    merged_inter['Avg_Path_Length_Diff_inter'] = merged_inter['Avg_Path_Length_reference'] - merged_inter['Avg_Path_Length_comparison']
    merged_inter['Clustering_Coeff_Diff_inter'] = merged_inter['Clustering_Coeff_reference'] - merged_inter['Clustering_Coeff_comparison']
    merged_inter['KL_Divergence_Diff_inter'] = merged_inter['KL_Divergence_reference'] - merged_inter['KL_Divergence_comparison']

    # Select relevant columns
    diff_inter = merged_inter[['Perturbation', 'Avg_Path_Length_Diff_inter', 'Clustering_Coeff_Diff_inter', 'KL_Divergence_Diff_inter']]

    return diff_intra, diff_inter

def plot_differences_together(diff_df, distance_type, pert_comparisons, plot=True, save=False, save_path=""):
    """
    Plot differences in metrics together in a grid for a given distance type (Intra or Inter).
    
    Parameters:
    - diff_df (pd.DataFrame): DataFrame containing the differences in metrics (diff_intra or diff_inter).
    - distance_type (str): String indicating the type ('Intra' or 'Inter').
    - pert_comparisons (list): List of perturbations to ensure consistent ordering.
    - plot (bool): Whether to display the plots interactively. Defaults to True.
    - save (bool): Whether to save the plots to files. Defaults to False.
    - save_path (str): Directory path where plots will be saved if `save=True`.

    Returns:
    - None
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    
    sns.set(style="whitegrid")
    
    # Metrics to plot
    metric_names = ['Avg_Path_Length_Diff', 'Clustering_Coeff_Diff', 'KL_Divergence_Diff']
    metric_labels = {
        'Avg_Path_Length_Diff': 'Average Path Length',
        'Clustering_Coeff_Diff': 'Clustering Coefficient',
        'KL_Divergence_Diff': 'KL Divergence'
    }
    
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=(6*num_metrics, 6))
    
    # Ensure axes is iterable
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_names):
        # Extract relevant data and set categorical order for 'Perturbation'
        data = diff_df[['Perturbation', metric]].dropna().copy()
        data['Perturbation'] = pd.Categorical(data['Perturbation'], categories=pert_comparisons, ordered=True)
        data = data.sort_values('Perturbation')
        data = data.rename(columns={metric: 'Difference'})
        
        # Create the bar plot
        sns.barplot(
            data=data,
            x='Perturbation',
            y='Difference',
            palette='Blues_d',
            edgecolor='black',
            ax=ax
        )
        
        ax.set_xlabel('Perturbation', fontsize=14)
        ax.set_ylabel('Difference', fontsize=14)
        ax.set_title(metric_labels[metric], fontsize=16)
        ax.set_xticklabels(pert_comparisons, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate bars with difference values
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):  # Avoid NaN heights
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=12)
    
    # Add a global title
    fig.suptitle(f'Differences in Metrics Between Datasets ({distance_type} Perturbation)', fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    
    # Save the plot if save=True
    if save:
        if not save_path:
            save_path = "path_to_save_directory"  # Replace with your desired path
        filename = f"metrics_difference_{distance_type.lower()}.png"
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path, bbox_inches='tight')
        print(f"Metrics difference plot saved to: {full_save_path}")
    
    # Display the plot if plot=True
    if plot:
        plt.show()
    
    # Close the plot to free memory
    plt.close()

# # Compute the metrics
# metrics_intra_df, metrics_inter_df = compute_metrics_for_dataframes(
#     dataframes=dataframes,
#     comparisons=pert_comparisons,
#     subsample_fraction=subsample_fraction,
#     random_state=random_state
# )

# # Compute the differences
# diff_intra, diff_inter = compute_differences(
#     metrics_intra_df,
#     metrics_inter_df,
#     reference_title='Original Data',
#     comparison_title='No lmx1b gdf3'
# )


# # Plot differences together for intra-perturbation
# plot_differences_together(diff_intra, 'Intra',pert_comparisons)

# # Plot differences together for inter-perturbation
# plot_differences_together(diff_inter, 'Inter',pert_comparisons)

# # Add suffix "_intra" to columns in metrics_intra_df if they don't already have it
# metrics_intra_df.columns = [
#     f"{col}_intra" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_intra") else col
#     for col in metrics_intra_df.columns
# ]

# # Add suffix "_inter" to columns in metrics_inter_df if they don't already have it
# metrics_inter_df.columns = [
#     f"{col}_inter" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_inter") else col
#     for col in metrics_inter_df.columns
# ]


# diff_intra.columns = [
#     f"{col}_intra" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_intra") else col
#     for col in diff_intra.columns
# ]

# # Add suffix "_inter" to columns in metrics_inter_df if they don't already have it
# diff_inter.columns = [
#     f"{col}_inter" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_inter") else col
#     for col in diff_inter.columns
# ]


# diff_inter

# # Set number of neighbors (k) and subsample fraction
# k_neighbors = 20
# subsample_fraction = 0.1  # Adjust as needed

# # List of DataFrames and titles for comparison
# dataframes = [
#     (df_hld, 'No lmx1b gdf3')
# ]

# Function to randomly subsample
def random_subsample(df, fraction, random_state=42):
    return df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)

# Function to compute Jaccard similarities and return separate DataFrames
def compute_jaccard_similarities(orig_df, dataframes, comparisons, z_mu_biological_columns, k_neighbors=20, subsample_fraction=0.1):
    # Initialize lists to store results
    jaccard_results_global = []
    jaccard_results_inter = []
    jaccard_results_intra = []
    
    # Subsample original DataFrame once
    orig_subsampled = random_subsample(orig_df, subsample_fraction)
    
    # Fit NearestNeighbors model on original data (Global Structure)
    knn_orig = NearestNeighbors(n_neighbors=100)  # Use a large number to ensure enough neighbors after masking
    knn_orig.fit(orig_subsampled[z_mu_biological_columns].values)
    
    # Loop over each DataFrame in dataframes
    for df, df_label in dataframes:
        print(f"Processing DataFrame: {df_label}")
        
        # Subsample comparison DataFrame
        comp_subsampled = random_subsample(df, subsample_fraction)
        
        # Fit NearestNeighbors model on comparison data (Global Structure)
        knn_comp = NearestNeighbors(n_neighbors=100)  # Use a large number to ensure enough neighbors after masking
        knn_comp.fit(comp_subsampled[z_mu_biological_columns].values)
        
        # Loop over each perturbation
        for perturbation in comparisons:
            print(f"  Processing perturbation: {perturbation}")
            
            # Get subset of data for the current perturbation
            orig_subset = orig_subsampled[orig_subsampled['phenotype'] == perturbation]
            comp_subset = comp_subsampled[comp_subsampled['phenotype'] == perturbation]
            
            if orig_subset.empty or comp_subset.empty:
                print(f"    No data for perturbation {perturbation}")
                continue
            
            # Identify common snip_ids in both subsets
            common_snip_ids = set(orig_subset['snip_id']).intersection(set(comp_subset['snip_id']))
            if not common_snip_ids:
                print(f"    No common snip_ids for perturbation {perturbation}")
                continue
            
            # Initialize lists to store the similarity scores
            jaccard_similarities_all = []
            jaccard_similarities_inter = []
            jaccard_similarities_intra = []
            
            # Loop over the common snip_ids to compare nearest neighbors
            for snip_id in common_snip_ids:
                # Get the data points for this snip_id
                point_orig = orig_subset[orig_subset['snip_id'] == snip_id][z_mu_biological_columns].values
                point_comp = comp_subset[comp_subset['snip_id'] == snip_id][z_mu_biological_columns].values
                
                if point_orig.size == 0 or point_comp.size == 0:
                    continue  # Skip if no data for this snip_id
                
                # -- Global Structure --
                neighbors_orig_all = knn_orig.kneighbors(point_orig, return_distance=False)[0]
                neighbors_comp_all = knn_comp.kneighbors(point_comp, return_distance=False)[0]
                
                orig_neighbors_snip_ids_all = orig_subsampled.iloc[neighbors_orig_all]['snip_id'].values
                comp_neighbors_snip_ids_all = comp_subsampled.iloc[neighbors_comp_all]['snip_id'].values
                
                # Take the first k_neighbors
                orig_neighbors_snip_ids_all = orig_neighbors_snip_ids_all[:k_neighbors]
                comp_neighbors_snip_ids_all = comp_neighbors_snip_ids_all[:k_neighbors]
                
                # Calculate Jaccard similarity using the union of neighbor sets
                intersection_all = set(orig_neighbors_snip_ids_all).intersection(set(comp_neighbors_snip_ids_all))
                union_all = set(orig_neighbors_snip_ids_all).union(set(comp_neighbors_snip_ids_all))
                jaccard_similarity_all = len(intersection_all) / len(union_all)
                jaccard_similarities_all.append(jaccard_similarity_all)
                
                # -- Inter-Class Structure (exclude same perturbation) --
                # Filter neighbors to exclude same perturbation
                orig_neighbors_inter = orig_subsampled.iloc[neighbors_orig_all]
                comp_neighbors_inter = comp_subsampled.iloc[neighbors_comp_all]
                
                orig_neighbors_snip_ids_inter = orig_neighbors_inter[orig_neighbors_inter['phenotype'] != perturbation]['snip_id'].values
                comp_neighbors_snip_ids_inter = comp_neighbors_inter[comp_neighbors_inter['phenotype'] != perturbation]['snip_id'].values
                
                # Ensure we have enough neighbors
                orig_neighbors_snip_ids_inter = orig_neighbors_snip_ids_inter[:k_neighbors]
                comp_neighbors_snip_ids_inter = comp_neighbors_snip_ids_inter[:k_neighbors]
                
                if len(orig_neighbors_snip_ids_inter) > 0 and len(comp_neighbors_snip_ids_inter) > 0:
                    intersection_inter = set(orig_neighbors_snip_ids_inter).intersection(set(comp_neighbors_snip_ids_inter))
                    union_inter = set(orig_neighbors_snip_ids_inter).union(set(comp_neighbors_snip_ids_inter))
                    jaccard_similarity_inter = len(intersection_inter) / len(union_inter)
                    jaccard_similarities_inter.append(jaccard_similarity_inter)
                
                # -- Intra-Class Structure (only same perturbation) --
                # Filter neighbors to include only same perturbation
                orig_neighbors_intra = orig_subsampled.iloc[neighbors_orig_all]
                comp_neighbors_intra = comp_subsampled.iloc[neighbors_comp_all]
                
                orig_neighbors_snip_ids_intra = orig_neighbors_intra[orig_neighbors_intra['phenotype'] == perturbation]['snip_id'].values
                comp_neighbors_snip_ids_intra = comp_neighbors_intra[comp_neighbors_intra['phenotype'] == perturbation]['snip_id'].values
                
                # Ensure we have enough neighbors
                orig_neighbors_snip_ids_intra = orig_neighbors_snip_ids_intra[:k_neighbors]
                comp_neighbors_snip_ids_intra = comp_neighbors_snip_ids_intra[:k_neighbors]
                
                if len(orig_neighbors_snip_ids_intra) > 0 and len(comp_neighbors_snip_ids_intra) > 0:
                    intersection_intra = set(orig_neighbors_snip_ids_intra).intersection(set(comp_neighbors_snip_ids_intra))
                    union_intra = set(orig_neighbors_snip_ids_intra).union(set(comp_neighbors_snip_ids_intra))
                    jaccard_similarity_intra = len(intersection_intra) / len(union_intra)
                    jaccard_similarities_intra.append(jaccard_similarity_intra)
            
            # Store the average Jaccard similarities in the respective results lists
            if jaccard_similarities_all:
                avg_similarity_all = np.mean(jaccard_similarities_all)
                jaccard_results_global.append({
                    'DataFrame': df_label,
                    'Perturbation': perturbation,
                    'Jaccard Similarity': avg_similarity_all
                })
            if jaccard_similarities_inter:
                avg_similarity_inter = np.mean(jaccard_similarities_inter)
                jaccard_results_inter.append({
                    'DataFrame': df_label,
                    'Perturbation': perturbation,
                    'Jaccard Similarity': avg_similarity_inter
                })
            if jaccard_similarities_intra:
                avg_similarity_intra = np.mean(jaccard_similarities_intra)
                jaccard_results_intra.append({
                    'DataFrame': df_label,
                    'Perturbation': perturbation,
                    'Jaccard Similarity': avg_similarity_intra
                })
    
    # Convert respaired_models_and_metrics_dfults to DataFrames
    results_df_global = pd.DataFrame(jaccard_results_global)
    results_df_inter = pd.DataFrame(jaccard_results_inter)
    results_df_intra = pd.DataFrame(jaccard_results_intra)
    
    return results_df_global, results_df_inter, results_df_intra

# Function to plot the results

def plot_jaccard_results(results_df_global, results_df_inter, results_df_intra, pert_comparisons, plot=True, save=False, save_path=""):
    """
    Plot Jaccard similarity results together in a grid for different structure types.

    Parameters:
    - results_df_global (pd.DataFrame): DataFrame containing global structure Jaccard similarities.
    - results_df_inter (pd.DataFrame): DataFrame containing inter-class structure Jaccard similarities.
    - results_df_intra (pd.DataFrame): DataFrame containing intra-class structure Jaccard similarities.
    - pert_comparisons (list): List of perturbations to ensure consistent ordering.
    - plot (bool): Whether to display the plots interactively. Defaults to True.
    - save (bool): Whether to save the plots to files. Defaults to False.
    - save_path (str): Directory path where plots will be saved if `save=True`.

    Returns:
    - None
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    sns.set(style="whitegrid")

    # Define the order of the datasets for consistent coloring
    dataset_order = results_df_global['DataFrame'].unique()
    
    # Create a color palette
    palette = sns.color_palette('tab10', n_colors=len(dataset_order))
    color_dict = dict(zip(dataset_order, palette))

    # Metrics to plot
    plot_configs = [
        (results_df_global, 'Point-Similarity; Global Structure', 'Global'),
        (results_df_inter, 'Point-Similarity; Inter-Class Structure', 'Inter'),
        (results_df_intra, 'Point-Similarity; Intra-Class Structure', 'Intra')
    ]

    # Ensure 'Perturbation' is categorical with the defined order
    for df in [results_df_global, results_df_inter, results_df_intra]:
        df['Perturbation'] = pd.Categorical(df['Perturbation'], categories=pert_comparisons, ordered=True)

    num_plots = len(plot_configs)
    fig, axes = plt.subplots(1, num_plots, figsize=(8*num_plots, 6), sharey=True)

    # Ensure axes is iterable
    if num_plots == 1:
        axes = [axes]

    for ax, (df, title, suffix) in zip(axes, plot_configs):
        # Create the bar plot
        sns.barplot(
            data=df,
            x='Perturbation',
            y='Jaccard Similarity',
            hue='DataFrame',
            hue_order=dataset_order,
            ax=ax,
            ci=None,
            palette=color_dict
        )
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Perturbations', fontsize=14)
        ax.set_ylabel('Average Jaccard Similarity' if ax == axes[0] else '', fontsize=14)
        ax.set_xticklabels(pert_comparisons, rotation=45, ha='right')  # Use defined order explicitly
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate bars with similarity values
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):  # Avoid NaN heights
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=12)
        
        # Remove legend from all but the last subplot
        if ax != axes[-1]:
            ax.get_legend().remove()

    # Add a single legend for the entire figure
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title='Dataset', loc='upper right')

    # Add a global title
    fig.suptitle('Jaccard Similarity Across Different Structures', fontsize=18)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle

    # Save the plot if save=True
    if save:
        if not save_path:
            save_path = "path_to_save_directory"  # Replace with your desired path
        filename = f"jaccard_similarity_results.png"
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path, bbox_inches='tight')
        print(f"Jaccard similarity plot saved to: {full_save_path}")

    # Display the plot if plot=True
    if plot:
        plt.show()

    # Close the plot to free memory
    plt.close()

# # Usage example:

# # Compute the Jaccard similarities and get the separate results DataFrames
# results_df_global, results_df_inter, results_df_intra = compute_jaccard_similarities(
#     orig_df=df_all,
#     dataframes=dataframes,
#     comparisons=pert_comparisons,
#     z_mu_biological_columns=z_mu_biological_columns,
#     k_neighbors=k_neighbors,
#     subsample_fraction=subsample_fraction
# )

# # Plot the results
# plot_jaccard_results(results_df_global, results_df_inter, results_df_intra)

# # Add suffix "_inter" to columns in metrics_inter_df if they don't already have it
# results_df_global.columns = [
#     f"{col}_global" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_global") else col
#     for col in results_df_global.columns
# ]

# results_df_intra.columns = [
#     f"{col}_intra" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_intra") else col
#     for col in results_df_intra.columns
# ]

# # Add suffix "_inter" to columns in metrics_inter_df if they don't already have it
# results_df_inter.columns = [
#     f"{col}_inter" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_inter") else col
#     for col in results_df_inter.columns
# ]


# # List of dataframes to merge
# dfs_to_merge = [avg_f1_score_df, diff_intra, diff_inter, results_df_global, results_df_intra, results_df_inter]

# # Set 'Perturbation' as the index and drop 'DataFrame' columns
# dfs_to_merge = [df.set_index('Perturbation').drop(columns='DataFrame', errors='ignore') for df in dfs_to_merge]

# # Merge all dataframes on the 'Perturbation' index
# core_performance_metrics = pd.concat(dfs_to_merge, axis=1)

# # Reset index if needed
# core_performance_metrics.reset_index(inplace=True)

# core_performance_metrics

# f1_score_df

# # Drop the 'DataFrame' and 'Perturbation' columns from metrics_inter_df
# metrics_inter_df_dropped = metrics_inter_df.drop(columns=['DataFrame', 'Perturbation'])

# # Append 'DataFrame' and 'Perturbation' columns from metrics_intra_df to metrics_inter_df_dropped
# distance_metrics_intra_inter = pd.concat([metrics_intra_df, metrics_inter_df_dropped], axis=1)

# distance_metrics_intra_inter

# # 1 
# core_performance_metrics
# # 2
# f1_score_df
# # 3
# distance_metrics_intra_inter

#Note: there are various plotting functions, they have two options Saving, and Plotting, 
## If you dont want to plot them ignore them
## If you want to save the plots, then you need to manipulate the "results_path" function so that they are saved to the desired path



def produce_performance_metrics(
    df_all,
    df_hld,
    pert_comparisons,
    logreg_tol=1e-3,
    subsample_fraction=0.05,
    num_bins=20,
    max_hpf=40,
    random_state=100,
    plot=True,
    k_neighbors=5,
    subsample_fraction_jaccard=0.1
):
    """
    Analyze model data by performing logistic regression, computing F1 scores over time,
    calculating distance metrics, and computing Jaccard similarities.

    Parameters:
        df_all (pd.DataFrame): DataFrame for all data (no holdout)
        df_hld (pd.DataFrame): DataFrame with holdout data
        pert_comparisons (list): List of perturbations to compare
        logreg_tol (float): Tolerance for logistic regression
        subsample_fraction (float): Fraction for subsampling in metrics computation
        num_bins (int): Number of bins for time-based F1 score computation
        max_hpf (float): Maximum hours post-fertilization for binning
        random_state (int): Random state for reproducibility
        plot (bool): Whether to generate plots
        k_neighbors (int): Number of neighbors for Jaccard similarity measurements
        subsample_fraction_jaccard (float): Subsample fraction for Jaccard similarity computation

    Returns:
        core_performance_metrics (pd.DataFrame): Core performance metrics
        distance_metrics_intra_inter (pd.DataFrame): Intra and inter distance metrics
        metrics_inter_df (pd.DataFrame): Inter-metric results
    """
    # Split the data
    df_all_train, df_all_test, df_all = split_train_test(df_all)
    df_hld_train, df_hld_test, df_hld = split_train_test(df_hld)

    # Define the columns
    z_mu_columns = [col for col in df_all.columns if 'z_mu' in col]
    z_mu_biological_columns = [col for col in z_mu_columns if "b" in col]
    
    ### Logistic Regression ###

    # Logistic Regression for 'all' data
    y_test_all, y_pred_proba_all, log_reg_all, train_df_all, test_df_all = logistic_regression_multiclass(
        df_all_train, df_all_test, z_mu_biological_columns, pert_comparisons, tol=logreg_tol
    )

    # Logistic Regression for 'hld' data
    y_test_hld, y_pred_proba_hld, log_reg_hld, train_df_hld, test_df_hld = logistic_regression_multiclass(
        df_hld_train, df_hld_test, z_mu_biological_columns, pert_comparisons, tol=logreg_tol
    )
    
    ### F1 Scores ###
    # Compute F1 scores over time for 'all' data
    results_dict_all = f1_score_over_time_multiclass(
        y_test_all, y_pred_proba_all, test_df_all, pert_comparisons, num_bins=num_bins, max_hpf=max_hpf
    )
    dataset_label_all = 'all_perts_F1'
    if plot:
        plot_f1_score_over_time(
            results_dict_all, pert_comparisons, dataset_label=dataset_label_all, title="F1 Score Over Time for Perturbations"
        )

    # Compute F1 scores over time for 'hld' data
    results_dict_hld = f1_score_over_time_multiclass(
        y_test_hld, y_pred_proba_hld, test_df_hld, pert_comparisons, num_bins=num_bins, max_hpf=max_hpf
    )
    dataset_label_hld = 'hld_gdf3_lmx1b_perts_F1'

    
    #Compute average F1 scores and differences
    f1_score_df = create_f1_score_dataframe(results_dict_all, results_dict_hld, pert_comparisons)
    
    avg_f1_score_df = compute_average_f1_score(f1_score_df)
    
    if plot:
        plot_f1_score_over_time(
            results_dict_hld, pert_comparisons, dataset_label=dataset_label_hld, title="F1 Score Over Time for Perturbations"
        )
        plot_average_f1_score_difference(avg_f1_score_df)


    ### Distance Metrics ###
    # Prepare dataframes for distance metrics computation
    dataframes = [
        (df_all, 'Original Data'),
        (df_hld, 'No lmx1b gdf3')
    ]

    # Compute the metrics
    metrics_intra_df, metrics_inter_df = compute_metrics_for_dataframes(
        dataframes=dataframes,
        comparisons=pert_comparisons,
        z_mu_biological_columns=z_mu_biological_columns,
        subsample_fraction=subsample_fraction,
        random_state=random_state
    )

    # Compute the differences of the metrics
    diff_intra, diff_inter = compute_differences(
        metrics_intra_df,
        metrics_inter_df,
        reference_title='Original Data',
        comparison_title='No lmx1b gdf3'
    )


    ### Jaccard Similarity Measurements ###
    # List of DataFrames and titles for comparison
    dataframes_jaccard = [
        (df_hld, 'No lmx1b gdf3')
    ]

    # Compute the Jaccard similarities and get the separate results DataFrames
    results_df_global, results_df_inter, results_df_intra = compute_jaccard_similarities(
        orig_df=df_all,
        dataframes=dataframes_jaccard,
        comparisons=pert_comparisons,
        z_mu_biological_columns=z_mu_biological_columns,
        k_neighbors=k_neighbors,
        subsample_fraction=subsample_fraction_jaccard,
    )

    if plot:
        # Plot the results
        plot_jaccard_results(results_df_global, results_df_inter, results_df_intra)


    # Add suffixes to result dataframes
    results_df_global.columns = [
        f"{col}_global" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_global") else col
        for col in results_df_global.columns
    ]

    results_df_intra.columns = [
        f"{col}_intra" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_intra") else col
        for col in results_df_intra.columns
    ]

    results_df_inter.columns = [
        f"{col}_inter" if col not in ['DataFrame', 'Perturbation'] and not col.endswith("_inter") else col
        for col in results_df_inter.columns
    ]


    # Clean up the intra and inter distance metrics by dropping unnecessary columns
    organized_df = pd.merge(
            metrics_inter_df[metrics_inter_df["DataFrame"]=="Original Data"],
            metrics_inter_df[metrics_inter_df["DataFrame"]=="No lmx1b gdf3"],   
            on='Perturbation',
            suffixes=('_nohld', '_hld'),
            how='inner'  # Ensures only matching rows are kept, change to 'outer' or 'left' if needed
        )
    organized_df = organized_df.drop(columns=["KL_Divergence_nohld","DataFrame_hld","DataFrame_nohld"])
    metrics_inter_df_clean = organized_df.rename(columns={"KL_Divergence_hld": "KL_Divergence"})


    # Combine the modified groups back into a single dataframe
    organized_df = pd.merge(
            metrics_intra_df[metrics_intra_df["DataFrame"]=="Original Data"],
            metrics_intra_df[metrics_intra_df["DataFrame"]=="No lmx1b gdf3"],   
            on='Perturbation',
            suffixes=('_nohld', '_hld'),
            how='inner'  # Ensures only matching rows are kept, change to 'outer' or 'left' if needed
        )
    organized_df = organized_df.drop(columns=["KL_Divergence_nohld","DataFrame_hld","DataFrame_nohld"])
    metrics_intra_df_clean = organized_df.rename(columns={"KL_Divergence_hld": "KL_Divergence"})

    # Merge the cleaned DataFrames on the 'Perturbation' column
    distance_metrics_intra_inter = pd.merge(
        metrics_intra_df_clean,
        metrics_inter_df_clean,
        on='Perturbation',
        suffixes=('_intra', '_inter'),
        how='inner'  # Ensures only matching rows are kept, change to 'outer' or 'left' if needed
    )

    ### Merge DataFrames ###
    # List of dataframes to merge
    dfs_to_merge = [avg_f1_score_df, diff_intra, diff_inter, results_df_global, results_df_intra, results_df_inter]

    # Set 'Perturbation' as the index and drop 'DataFrame' columns
    dfs_to_merge = [df.set_index('Perturbation').drop(columns='DataFrame', errors='ignore') for df in dfs_to_merge]

    # Merge all dataframes on the 'Perturbation' index
    core_performance_metrics = pd.concat(dfs_to_merge, axis=1).reset_index()

    # Merging dataframes on 'Perturbation' column
    core_performance_metrics = core_performance_metrics.merge(
        distance_metrics_intra_inter,
        on="Perturbation",
        how="inner"  # Change to 'left', 'right', or 'outer' as needed
    )

    # Calculate mean for numeric columns in core_performance_metrics
    summary_stats = core_performance_metrics.mean(numeric_only=True).to_frame().T
    summary_stats['Perturbation'] = "avg_pert"

    
    # Stack the two DataFrames
    core_performance_metrics = pd.concat([core_performance_metrics, summary_stats], ignore_index=True)


    # Return the results
    return core_performance_metrics



# #example ussage of function to produce all the core perfomance metrics. 
# # Call the function
# core_performance_metrics, distance_metrics_intra_inter, metrics_inter_df = produce_perfomance_metrics(
#     df_all,
#     df_hld,
#     pert_comparisons,
#     logreg_tol=1e-3,
#     subsample_fraction=0.05,
#     subsample_fraction_jaccard=0.1,
#     num_bins=20,
#     max_hpf=40,
#     random_state=100,
#     plot=True,
#     k_neighbors=5
# )


def plot_trajectories_3d(splines_final,save_dir=None):
    """
    Plots PCA trajectories for different perturbations and datasets in a 3D Plotly plot.

    Parameters:
    splines_final (pd.DataFrame): DataFrame containing the trajectory data with columns
                                  ['dataset', 'Perturbation', 'point_index', 'PCA_1', 'PCA_2', 'PCA_3']

    Returns:
    None
    """
    # Define perturbations and their corresponding colors
    pert_comparisons = ["wnt-i", "tgfb-i", "wt", "lmx1b", "gdf3"]
    
    color_map = {
        "wnt-i": "red",
        "tgfb-i": "green",
        "wt": "blue",
        "lmx1b": "orange",
        "gdf3": "purple"
    }
    
    # Define dataset styles with dash styles
    dataset_styles = {
        "all": {"dash": "solid", "name": "all"},
        "hld": {"dash": "dash", "name": "hld"},
        "hld_aligned": {"dash": "dot", "name": "hld aligned"}
    }
    
    # Initialize the figure
    fig = go.Figure()

    # Iterate over each perturbation
    for pert in pert_comparisons:
        pert_data = splines_final[splines_final['Perturbation'] == pert]
        color = color_map.get(pert, "black")  # Default to black if perturbation not found
        
        # Iterate over each dataset
        for dataset, style in dataset_styles.items():
            dataset_data = pert_data[pert_data['dataset'] == dataset]
            
            if dataset_data.empty:
                continue  # Skip if there's no data for this dataset
            
            # Sort by point_index to ensure proper trajectory
            dataset_data = dataset_data.sort_values(by='point_index')
            
            # Add trace
            fig.add_trace(
                go.Scatter3d(
                    x=dataset_data['PCA_1'],
                    y=dataset_data['PCA_2'],
                    z=dataset_data['PCA_3'],
                    mode='lines',
                    name=f"{pert} - {style['name']}",
                    line=dict(color=color, dash=style['dash'], width=4),
                    )
                )
    if save_dir:
        fig.write_html(os.path.join(save_dir,"model_splines.html"))    
 
    # Show the plot
    return fig

def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    return np.dot(a, b) / (a_norm * b_norm)

class LocalPrincipalCurve:
    def __init__(self, bandwidth=0.5, max_iter=100, tol=1e-4, angle_penalty_exp=2, h=None):
        """
        Initialize the Local Principal Curve solver.
        """
        self.bandwidth = bandwidth
        self.h = h if h is not None else self.bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.angle_penalty_exp = angle_penalty_exp

        self.initializations = []
        self.paths = []
        self.cubic_splines_eq = []
        self.cubic_splines = []

    def _kernel_weights(self, dataset, x):
        dists = np.linalg.norm(dataset - x, axis=1)
        weights = np.exp(- (dists**2) / (2 * self.bandwidth**2))
        w = weights / np.sum(weights)
        return w

    def _local_center_of_mass(self, dataset, x):
        w = self._kernel_weights(dataset, x)
        mu = np.sum(dataset.T * w, axis=1)
        return mu

    def _local_covariance(self, dataset, x, mu):
        w = self._kernel_weights(dataset, x)
        centered = dataset - mu
        cov = np.zeros((dataset.shape[1], dataset.shape[1]))
        for i in range(len(dataset)):
            cov += w[i] * np.outer(centered[i], centered[i])
        return cov

    def _principal_component(self, cov, prev_vec=None):
        vals, vecs = np.linalg.eig(cov)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        gamma = vecs[:, 0]  # first principal component

        # Sign/direction handling
        if prev_vec is not None and np.linalg.norm(prev_vec) != 0:
            cos_alpha = np.dot(gamma, prev_vec) / (np.linalg.norm(gamma)*np.linalg.norm(prev_vec))
            if cos_alpha < 0:
                gamma = -gamma

            # Angle penalization
            cos_alpha = np.dot(gamma, prev_vec) / (np.linalg.norm(gamma)*np.linalg.norm(prev_vec))
            a_x = (abs(cos_alpha))**self.angle_penalty_exp
            gamma = a_x * gamma + (1 - a_x) * prev_vec
            gamma /= np.linalg.norm(gamma)

        return gamma

    def _forward_run(self, dataset, x_start):
        x = x_start
        path_x = [x]
        prev_gamma = None

        for _ in range(self.max_iter):
            mu = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            x_new = mu + self.h * gamma

            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new)
                break

            path_x.append(x_new)
            x = x_new
            prev_gamma = gamma

        return np.array(path_x)

    def _backward_run(self, dataset, x0, gamma0):
        x = x0
        path_x = [x]
        prev_gamma = -gamma0

        for _ in range(self.max_iter):
            mu = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            x_new = mu + self.h * gamma
            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new)
                break

            path_x.append(x_new)
            x = x_new
            prev_gamma = gamma

        return np.array(path_x)

    def _find_starting_point(self, dataset, start_point):
        if start_point is None:
            idx = np.random.choice(len(dataset))
            return dataset[idx], idx
        else:
            diffs = dataset - start_point
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            closest_pt = dataset[min_idx]
            if not np.allclose(closest_pt, start_point):
                print(f"Starting point not in dataset. Using closest point: {closest_pt}")
            return closest_pt, min_idx

    def fit(self, dataset, start_points=None, end_point=None, remove_similar_end_start_points=True):
        """
        Fit LPC on the dataset. Optionally provide:
         - start_points: array of shape (d,) or a single point of shape (d,)
         - end_point: single point of shape (d,), only allowed if a start_point is provided.
        """
        dataset = np.array(dataset)
        self.paths = []
        self.initializations = []

        if end_point is not None and start_points is None:
            raise ValueError("end_point provided but no start_points given. end_point only allowed if start_point is provided.")

        # Ensure start_points is a list
        if start_points is not None and not isinstance(start_points, (list, tuple)):
            start_points = [start_points]

        if end_point is not None and (start_points is None or len(start_points) != 1):
            raise ValueError("If end_point is provided, exactly one start_point must be provided.")

        for sp in (start_points if start_points is not None else [None]):
            x0, _ = self._find_starting_point(dataset, sp)

            forward_path = self._forward_run(dataset, x0)
            if len(forward_path) > 1:
                initial_gamma_direction = (forward_path[1] - forward_path[0]) / self.h
            else:
                initial_gamma_direction = np.zeros(dataset.shape[1])

            if np.linalg.norm(initial_gamma_direction) > 0:
                backward_path = self._backward_run(dataset, x0, initial_gamma_direction)
                full_path = np.vstack([backward_path[::-1], forward_path[1:]])
            else:
                full_path = forward_path

            # Check orientation
            dist_start_to_first = np.linalg.norm(x0 - full_path[0])
            dist_start_to_last = np.linalg.norm(x0 - full_path[-1])
            if dist_start_to_last < dist_start_to_first:
                full_path = full_path[::-1]

            if remove_similar_end_start_points:
                start_pt = full_path[0]
                end_pt = full_path[-1]

                dist_to_start = np.linalg.norm(full_path - start_pt, axis=1)
                dist_to_end = np.linalg.norm(full_path - end_pt, axis=1)

                mask = np.ones(len(full_path), dtype=bool)
                mask[(dist_to_start < self.tol) | (dist_to_end < self.tol)] = False
                mask[0] = True
                mask[-1] = True
                full_path = full_path[mask]

            self.paths.append(full_path)
            self.initializations.append(x0)

        # Fit splines and compute equal arc-length
        self._fit_cubic_splines_eq()
        self._compute_equal_arc_length_spline_points()

        # If end_point provided, correct for the looping back issue
        if end_point is not None:
            try:
                # Assuming a single path scenario
                spline_points = self.cubic_splines[0]
                
                # 1) Find closest point on cubic_spline to end_point
                dists = np.linalg.norm(spline_points - end_point, axis=1)
                closest_idx = np.argmin(dists)

                # 2) Determine end_direction_vector using points around closest_idx
                # We'll take up to 3 points: [closest_idx-1, closest_idx, closest_idx+1]
                # If closest_idx is at the boundary, adjust accordingly
                if closest_idx == 0:
                    # At start, use next two points if available
                    if len(spline_points) > 2:
                        p0 = spline_points[closest_idx]
                        p1 = spline_points[closest_idx + 1]
                        p2 = spline_points[closest_idx + 2]
                        end_direction_vector = ((p1 - p0) + (p2 - p1)) / 2.0
                    else:
                        # If very short, just fallback
                        end_direction_vector = np.array([1, 0, 0])
                elif closest_idx == len(spline_points) - 1:
                    # At the end, we might not have a point after it
                    # use the two points before it if possible
                    if len(spline_points) > 2:
                        p_end = spline_points[closest_idx]
                        p_endm1 = spline_points[closest_idx - 1]
                        p_endm2 = spline_points[closest_idx - 2]
                        end_direction_vector = ((p_end - p_endm1) + (p_endm1 - p_endm2)) / 2.0
                    else:
                        end_direction_vector = np.array([1, 0, 0])
                else:
                    # Middle somewhere, use prev and next
                    p_before = spline_points[closest_idx - 1]
                    p_mid = spline_points[closest_idx]
                    p_after = spline_points[closest_idx + 1]
                    end_direction_vector = ((p_mid - p_before) + (p_after - p_mid)) / 2.0

                # Normalize end_direction_vector
                norm_edv = np.linalg.norm(end_direction_vector)
                if norm_edv > 0:
                    end_direction_vector = end_direction_vector / norm_edv
                else:
                    warnings.warn("end_direction_vector has zero magnitude. Using default direction.")
                    end_direction_vector = np.array([1, 0, 0])

                # 3) Check directionality after closest_idx
                # We'll look at pairs of points (p_j, p_{j+1}) for j > closest_idx
                cutoff_index = None
                for j in range(closest_idx + 1, len(spline_points) - 1):
                    seg_vec = spline_points[j + 1] - spline_points[j]
                    csim = cosine_similarity(seg_vec, end_direction_vector)
                    if csim < 0.5:
                        cutoff_index = j + 1
                        break

                # If we found a cutoff_index, truncate the spline
                if cutoff_index is not None:
                    spline_points = spline_points[:cutoff_index]

                    # Refit with truncated spline_points
                    self.paths = [spline_points]
                    self._fit_cubic_splines_eq()
                    self._compute_equal_arc_length_spline_points()

            except (ValueError, IndexError, TypeError) as e:
                # Log a warning and exit the if block gracefully
                warnings.warn(
                    f"Error processing spline with end_point: {e}. Skipping spline adjustment."
                )
                # Optionally, you can log more details for debugging
                # For example:
                # warnings.warn(f"Error processing spline: {e}. spline_points shape: {spline_points.shape}, end_point shape: {np.shape(end_point)}")
                return  # Exit the if block

        return self.paths

    def _fit_cubic_splines_eq(self):
        self.cubic_splines_eq = []
        for path in self.paths:
            if len(path) < 4:
                self.cubic_splines_eq.append(None)
                continue
            t = np.arange(len(path))
            splines_dict = {}
            for dim in range(path.shape[1]):
                splines_dict[dim] = CubicSpline(t, path[:, dim])
            self.cubic_splines_eq.append(splines_dict)

    def _compute_cubic_spline_points(self, num_points=500):
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            path = self.paths[i]
            t_values = np.linspace(0, len(path) - 1, num_points)
            spline_points = self.evaluate_cubic_spline(i, t_values)
            self.cubic_splines.append(spline_points)

    def evaluate_cubic_spline(self, path_idx, t_values):
        if path_idx >= len(self.cubic_splines_eq) or self.cubic_splines_eq[path_idx] is None:
            raise ValueError(f"No cubic spline found for path index {path_idx}.")
        spline = self.cubic_splines_eq[path_idx]
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T  # Fixed line
        return points

    def compute_arc_length(self, spline, t_min, t_max, num_samples=10000):
        t_values = np.linspace(t_min, t_max, num_samples)
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T  # Fixed line

        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_length = np.insert(np.cumsum(distances), 0, 0.0)
        return t_values, cumulative_length

    def get_uniformly_spaced_points(self, spline, num_points):
        path_length = len(spline[0].x)
        t_min = 0
        t_max = path_length - 1

        t_vals_dense, cum_length = self.compute_arc_length(spline, t_min, t_max, num_samples=5000)
        total_length = cum_length[-1]
        desired_distances = np.linspace(0, total_length, num_points)
        t_for_dist = interp1d(cum_length, t_vals_dense, kind='linear')(desired_distances)

        uniform_points = np.array([spline[dim](t_for_dist) for dim in sorted(spline.keys())]).T  # Fixed line
        return uniform_points

    def _compute_equal_arc_length_spline_points(self, num_points=500):
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            spline_points = self.get_uniformly_spaced_points(eq, num_points)
            self.cubic_splines.append(spline_points)

    def plot_path_3d(self, path_idx=0, dataset=None):
        dataset = np.array(dataset)
        path = self.paths[path_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if dataset is not None:
            ax.scatter(dataset[:,0], dataset[:,1], dataset[:,2], alpha=0.5, label='Data')
        ax.plot(path[:,0], path[:,1], path[:,2], 'r-', label='Local Principal Curve')
        ax.legend()
        plt.show()

    def plot_cubic_spline_3d(self, path_idx, show_path=True):
        if path_idx >= len(self.paths):
            raise IndexError(f"Path index {path_idx} is out of range. Total paths: {len(self.paths)}.")
        path = self.paths[path_idx]
        spline_points = self.cubic_splines[path_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if show_path:
            ax.scatter(path[:, 0], path[:, 1], path[:, 2], label="LPC Path", alpha=0.5)
        ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], color="red", label="Cubic Spline")
        ax.legend()
        plt.show()


def extract_spline(splines_df, dataset_label, perturbation):
    sdf = splines_df[(splines_df["dataset"] == dataset_label) & (splines_df["Perturbation"] == perturbation)]
    sdf = sdf.sort_values("point_index")
    points = sdf[["PCA_1", "PCA_2", "PCA_3"]].values
    return points

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def mean_l1_error(a, b):
# a and b are Nx3 arrays of points.
# Compute L1 distance for each point pair: sum of absolute differences across coordinates
# Then take the mean over all points.
    return np.mean(np.sum(np.abs(a - b), axis=1))
    
def centroid(X):
    return np.mean(X, axis=0)

def rmsd(X, Y):
    return np.sqrt(np.mean(np.sum((X - Y)**2, axis=1)))

def quaternion_alignment(P, Q):
    """
    Compute the optimal rotation using quaternions that aligns Q onto P.
    Returns rotation matrix R and translation vector t.
    """
    # Ensure P and Q have the same shape
    assert P.shape == Q.shape, "P and Q must have the same shape"
    
    # 1. Compute centroids and center the points
    P_cent = centroid(P)
    Q_cent = centroid(Q)
    P_prime = P - P_cent
    Q_prime = Q - Q_cent
    
    # 2. Construct correlation matrix M
    M = Q_prime.T @ P_prime
    
    # 3. Construct the Kearsley (Davenport) 4x4 matrix K
    # Refer to the equations above
    A = np.array([
        [ M[0,0]+M[1,1]+M[2,2],   M[1,2]-M[2,1],         M[2,0]-M[0,2],         M[0,1]-M[1,0]       ],
        [ M[1,2]-M[2,1],         M[0,0]-M[1,1]-M[2,2],  M[0,1]+M[1,0],         M[0,2]+M[2,0]       ],
        [ M[2,0]-M[0,2],         M[0,1]+M[1,0],         M[1,1]-M[0,0]-M[2,2],  M[1,2]+M[2,1]       ],
        [ M[0,1]-M[1,0],         M[0,2]+M[2,0],         M[1,2]+M[2,1],         M[2,2]-M[0,0]-M[1,1]]
    ], dtype=np.float64)
    A = A / 3.0
    
    # 4. Find the eigenvector of A with the highest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    max_idx = np.argmax(eigenvalues)
    q = eigenvectors[:, max_idx]
    q = q / np.linalg.norm(q)
    
    # 5. Convert quaternion q into rotation matrix R
    # Quaternion format: q = [q0, q1, q2, q3]
    q0, q1, q2, q3 = q
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
        [2*(q2*q1 + q0*q3),             q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q3*q1 - q0*q2),             2*(q3*q2 + q0*q1),             q0**2 - q1**2 - q2**2 + q3**2]
    ])
    
    # 6. Compute translation
    t = P_cent - R @ Q_cent
    
    return R, t

def _segment_direction_metrics(data_a, data_b, k=10):
    """
    Compute SegmentColinearity and SegmentCovariance for two given sets of points `data_a` and `data_b`.
    Both data_a and data_b are np.ndarray of shape (n, 3).

    If there aren't enough points for k segments, returns (np.nan, np.nan).
    """
    min_len = min(len(data_a), len(data_b))
    data_a = data_a[:min_len]
    data_b = data_b[:min_len]

    if min_len < k + 1 or min_len == 0:
        return (np.nan, np.nan)

    # Define segments using data_b
    segment_indices = np.linspace(0, min_len - 1, k + 1, dtype=int)

    aligned_segment_vecs = []
    all_segment_vecs = []

    for i in range(k):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i + 1]

        start_b = data_b[start_idx]
        end_b = data_b[end_idx]

        # Find closest points in data_a to start_b and end_b
        start_dists = np.linalg.norm(data_a - start_b, axis=1)
        closest_start_idx = np.argmin(start_dists)
        closest_start_a = data_a[closest_start_idx]

        end_dists = np.linalg.norm(data_a - end_b, axis=1)
        closest_end_idx = np.argmin(end_dists)
        closest_end_a = data_a[closest_end_idx]

        # Construct vectors
        vec_a = closest_end_a - closest_start_a
        vec_b = end_b - start_b

        # Normalize
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a > 0:
            vec_a = vec_a / norm_a
        else:
            vec_a = np.zeros(3)
        if norm_b > 0:
            vec_b = vec_b / norm_b
        else:
            vec_b = np.zeros(3)

        aligned_segment_vecs.append(vec_a)
        all_segment_vecs.append(vec_b)

    aligned_segment_vecs = np.array(aligned_segment_vecs)
    all_segment_vecs = np.array(all_segment_vecs)

    # Cosine similarities
    cos_sims = []
    for i in range(len(aligned_segment_vecs)):
        va = aligned_segment_vecs[i].reshape(1, -1)
        vb = all_segment_vecs[i].reshape(1, -1)
        sim = cosine_similarity(va, vb)[0][0]
        cos_sims.append(sim)

    avg_cosine_sim = np.mean(cos_sims) if len(cos_sims) > 0 else np.nan

    # Covariances
    covariances = []
    for dim_idx in range(3):
        dim_a = aligned_segment_vecs[:, dim_idx]
        dim_b = all_segment_vecs[:, dim_idx]
        if len(dim_a) > 1:
            cov = np.cov(dim_a, dim_b, bias=True)[0, 1]
        else:
            cov = np.nan
        covariances.append(cov)
    avg_cov = np.nanmean(covariances) if len(covariances) > 0 else np.nan

    return (avg_cosine_sim, avg_cov)


    # Split the dataset into 'all' and 'hld_aligned'
    splines_all = splines_final_df[splines_final_df["dataset"] == "all"]
    splines_hld_aligned = splines_final_df[splines_final_df["dataset"] == "hld_aligned"]

def segment_direction_consistency(splines_final_df, k=10):
    """
    Step 1 (Across): For each perturbation present in both datasets, compute SegmentColinearity and SegmentCovariance
    by comparing splines_hld_aligned and splines_all.

    Step 2 (Within): Compute these metrics for all unique pairs of perturbations within each dataset
    (both splines_hld_aligned and splines_all separately).
    Then compute the mean and std of these pairwise metrics for each dataset.

    Returns:
    - across_df: DataFrame with ['Perturbation', 'SegmentColinearity', 'SegmentCovariance']
    - within_hld_aligned_df: DataFrame with ['Metric', 'Mean', 'Std'] for pairwise metrics within splines_hld_aligned
    - within_all_df: DataFrame with ['Metric', 'Mean', 'Std'] for pairwise metrics within splines_all
    """

    splines_all = splines_final_df[splines_final_df["dataset"] == "all"]
    splines_hld = splines_final_df[splines_final_df["dataset"] == "hld"]
    splines_hld_aligned = splines_final_df[splines_final_df["dataset"] == "hld_aligned"]
    
    pca_columns = ["PCA_1", "PCA_2", "PCA_3"]
    for col in pca_columns:
        if col not in splines_hld_aligned.columns or col not in splines_all.columns:
            raise ValueError(f"Missing required PCA column: {col}")

            

    # Across computations
    perts_aligned = set(splines_hld_aligned["Perturbation"].unique())
    perts_all = set(splines_all["Perturbation"].unique())
    common_perts = perts_aligned.intersection(perts_all)

    across_results = []
    for pert in common_perts:
        data_a_df = splines_hld_aligned[splines_hld_aligned["Perturbation"] == pert].sort_values("point_index")
        data_b_df = splines_all[splines_all["Perturbation"] == pert].sort_values("point_index")
        data_a = data_a_df[pca_columns].values
        data_b = data_b_df[pca_columns].values

        sim, cov = _segment_direction_metrics(data_a, data_b, k=k)
        across_results.append({"Perturbation": pert, "SegmentColinearity": sim, "SegmentCovariance": cov})

    across_df = pd.DataFrame(across_results)

    # Calculate column means (excluding the Perturbation column)
    mean_row = across_df.iloc[:, 1:].mean()
    mean_row["Perturbation"] = "avg_pert"

    # Append the mean row to the DataFrame
    across_df = pd.concat([across_df, pd.DataFrame([mean_row])], ignore_index=True)


    # Within computations for splines_hld
    perts_in_aligned = list(perts_aligned)
    within_values_colinearity_hld = []
    within_values_covariance_hld  = []

    for i in range(len(perts_in_aligned)):
        for j in range(i+1, len(perts_in_aligned)):
            pert1 = perts_in_aligned[i]
            pert2 = perts_in_aligned[j]

            data_pert1 = splines_hld[splines_hld["Perturbation"] == pert1].sort_values("point_index")[pca_columns].values
            data_pert2 = splines_hld[splines_hld["Perturbation"] == pert2].sort_values("point_index")[pca_columns].values

            sim, cov = _segment_direction_metrics(data_pert1, data_pert2, k=k)
            if not np.isnan(sim):
                within_values_colinearity_hld.append(sim)
            if not np.isnan(cov):
                within_values_covariance_hld.append(cov)

    metrics_hld = []
    for metric_name, vals in [("SegmentColinearity", within_values_colinearity_hld), 
                              ("SegmentCovariance",  within_values_covariance_hld)]:
        mean_val = np.nanmean(vals) if len(vals) > 0 else np.nan
        std_val = np.nanstd(vals) if len(vals) > 0 else np.nan
        metrics_hld.append({"Metric": metric_name, "Mean": mean_val, "Std": std_val})

    within_hld_df = pd.DataFrame(metrics_hld)

    # Within computations for splines_all
    perts_in_all = list(perts_all)
    within_values_colinearity_all = []
    within_values_covariance_all = []

    for i in range(len(perts_in_all)):
        for j in range(i+1, len(perts_in_all)):
            pert1 = perts_in_all[i]
            pert2 = perts_in_all[j]

            data_pert1 = splines_all[splines_all["Perturbation"] == pert1].sort_values("point_index")[pca_columns].values
            data_pert2 = splines_all[splines_all["Perturbation"] == pert2].sort_values("point_index")[pca_columns].values

            sim, cov = _segment_direction_metrics(data_pert1, data_pert2, k=k)
            if not np.isnan(sim):
                within_values_colinearity_all.append(sim)
            if not np.isnan(cov):
                within_values_covariance_all.append(cov)

    metrics_all_list = []
    for metric_name, vals in [("SegmentColinearity", within_values_colinearity_all), 
                              ("SegmentCovariance", within_values_covariance_all)]:
        mean_val = np.nanmean(vals) if len(vals) > 0 else np.nan
        std_val = np.nanstd(vals) if len(vals) > 0 else np.nan
        metrics_all_list.append({"Metric": metric_name, "Mean": mean_val, "Std": std_val})

    within_all_df = pd.DataFrame(metrics_all_list)

    return across_df, within_hld_df, within_all_df

def calculate_dispersion_metrics(splines_final_df, n=5):
    """
    Calculates dispersion metrics for each dataset, including:
    - Dispersion Coefficient (slope of dispersion vs. point_index, normalized to [0, 1])
    - Initial Dispersion (average dispersion of the first n points)
    - Last Dispersion (average dispersion of the last n points)

    Parameters:
    - splines_final_df (pd.DataFrame): DataFrame containing all PCA trajectories with 'dataset' column.
    - n (int): Number of initial and last points to consider for initial and last dispersion.

    Returns:
    - pd.DataFrame: DataFrame with columns ['Dataset', 'disp_coefficient', 'dispersion_first_n', 'dispersion_last_n'].
    """
    # Extract subsets
    splines_all = splines_final_df[splines_final_df["dataset"] == "all"]
    splines_hld = splines_final_df[splines_final_df["dataset"] == "hld"]
    splines_hld_aligned = splines_final_df[splines_final_df["dataset"] == "hld_aligned"]

    # Ensure PCA columns are present
    pca_columns = ["PCA_1", "PCA_2", "PCA_3"]
    for col in pca_columns:
        if col not in splines_final_df.columns:
            raise ValueError(f"Missing required PCA column: {col}")

    # Get unique datasets
    datasets = splines_final_df["dataset"].unique()

    # Initialize list to store results
    results = []

    for dataset in datasets:
        if dataset == "hld_aligned":
            continue
        # Filter data for the current dataset
        dataset_df = splines_final_df[splines_final_df["dataset"] == dataset]

        # Get unique point_indices
        point_indices = sorted(dataset_df["point_index"].unique())

        # Initialize lists to store dispersion and point_index
        dispersion_list = []
        point_index_list = []

        # Initialize lists to store initial and last dispersions
        initial_dispersions = []
        last_dispersions = []

        for pid in point_indices:
            # Filter data for the current point_index
            point_df = dataset_df[dataset_df["point_index"] == pid]

            # Calculate dispersion: average Euclidean distance from centroid
            dispersion = compute_dispersion(point_df, pca_columns)

            # Append to lists
            dispersion_list.append(dispersion)
            point_index_list.append(pid)

            # If within first n points, store for initial dispersion
            if pid < n:
                initial_dispersions.append(dispersion)

            # If within last n points, store for last dispersion
            if pid >= max(point_indices) - n + 1:
                last_dispersions.append(dispersion)

        # Check if there are enough points for regression
        if len(point_index_list) < 2:
            print(f"Warning: Dataset '{dataset}' has less than 2 unique point_indices. Setting disp_coefficient to NaN.")
            disp_coefficient = np.nan
        else:
            # Prepare data for linear regression
            X = np.array(point_index_list).reshape(-1, 1)  # Shape: (num_points, 1)
            y = np.array(dispersion_list)  # Shape: (num_points,)

            # Fit linear regression
            reg = LinearRegression().fit(X, y)
            disp_coefficient = reg.coef_[0]
            disp_coefficient *= len(point_indices)  # Normalize to [0, 1]

        # Calculate average initial dispersion
        dispersion_first_n = np.mean(initial_dispersions) if initial_dispersions else np.nan
        if np.isnan(dispersion_first_n):
            print(f"Warning: Dataset '{dataset}' has no points within the first {n} point_indices.")

        # Calculate average last dispersion
        dispersion_last_n = np.mean(last_dispersions) if last_dispersions else np.nan
        if np.isnan(dispersion_last_n):
            print(f"Warning: Dataset '{dataset}' has no points within the last {n} point_indices.")

        # Append results
        results.append({
            "Dataset": dataset,
            "disp_coefficient": disp_coefficient,
            "dispersion_first_n": dispersion_first_n,
            "dispersion_last_n": dispersion_last_n
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)



    return results_df

def compute_dispersion(df, pca_columns):
    """
    Computes the average Euclidean distance of points from their centroid.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing PCA coordinates.
    - pca_columns (list): List of PCA column names.
    
    Returns:
    - float: Average Euclidean distance (dispersion).
    """
    if df.empty:
        return np.nan
    
    # Calculate centroid
    centroid = df[pca_columns].mean().values
    
    # Calculate Euclidean distances from centroid
    distances = np.linalg.norm(df[pca_columns].values - centroid, axis=1)
    
    # Return average distance
    return distances.mean()

import pandas as pd

# -------------------------------
# Helper Functions
# -------------------------------

def rename_within_metrics(df, suffix, key):
    """Renames columns in within metrics DataFrame with a given suffix."""
    renamed_df = df[["Metric", "Mean"]].copy()
    renamed_df["Metric"] += suffix  # Add suffix
    renamed_df = renamed_df.set_index("Metric").T  # Transpose for easy appending
    renamed_df.insert(0, "model_index", key)  # Add model_index
    return renamed_df

def process_dispersion_metrics(df, key):
    """Processes and renames dispersion metrics DataFrame."""
    disp_all = df[df["Dataset"] == "all"].drop("Dataset", axis=1)
    disp_all.columns = [col + "_all" for col in disp_all.columns]
    disp_hld = df[df["Dataset"] == "hld"].drop("Dataset", axis=1)
    disp_hld.columns = [col + "_hld" for col in disp_hld.columns]
    
    combined_df = pd.concat([disp_all.reset_index(drop=True), disp_hld.reset_index(drop=True)], axis=1)
    combined_df.insert(0, "model_index", key)  # Add model_index
    return combined_df

def process_segment_direction(splines_final_df, key):
    """Calculates and processes segment direction consistency metrics."""
    across_seg_df, within_hld_seg_df, within_all_seg_df = segment_direction_consistency(splines_final_df, k=100)
    across_seg_df.insert(0, "model_index", key)  # Add model_index
    
    within_hld_renamed = rename_within_metrics(within_hld_seg_df, "_mean_within_hld", key)
    within_all_renamed = rename_within_metrics(within_all_seg_df, "_mean_within_all", key)
    
    within_seg_measures = pd.concat([within_hld_renamed, within_all_renamed], axis=1)
    return across_seg_df, within_seg_measures

def combine_results_dict(results_dict):
    """
    Combines the results dictionary into a single DataFrame.
    Handles duplicate 'model_index' columns by ensuring uniqueness during merge.
    """
    final_list_of_dfs = []
    
    for model_index, metrics in results_dict.items():
        # Start with across_seg_df as the base since it has multiple perturbations
        if "across_seg_df" not in metrics:
            continue  # If for some reason this key doesn't have across_seg_df, skip
        
        base_df = metrics["across_seg_df"].copy()

        # Drop duplicate 'model_index' columns from other metrics before merging
        if "within_seg_measures" in metrics:
            temp_within = metrics["within_seg_measures"].copy()
            temp_within = temp_within.loc[:, ~temp_within.columns.duplicated()]  # Remove duplicate columns
            base_df = base_df.merge(temp_within, on="model_index", how="left")
        
        if "dispersion_metrics" in metrics:
            temp_disp = metrics["dispersion_metrics"].copy()
            temp_disp = temp_disp.loc[:, ~temp_disp.columns.duplicated()]  # Remove duplicate columns
            base_df = base_df.merge(temp_disp, on="model_index", how="left")

        # Append to list
        final_list_of_dfs.append(base_df)

    # Concatenate all model results
    if final_list_of_dfs:
        final_results_df = pd.concat(final_list_of_dfs, ignore_index=True)
    else:
        final_results_df = pd.DataFrame()

    return final_results_df

# Example usage:
# results_df = combine_results_dict(results_dict)
# This will produce a DataFrame with each row corresponding to a (model_index, Perturbation) pair,
# and columns from across_seg_df, within_seg_measures, and dispersion_metrics.




def plot_avg_predictions_multiclass(
    test_df, 
    y_pred_proba, 
    pert_comparisons, 
    pert_plotting=None, 
    window_size=3, 
    max_hpf=40,
    plot=True,
    save_dir=None,
    filename=None,
    highlight_embryos=None
):
    """
    Visualizes average model predictions over time for each embryo in a multiclass scenario using Plotly.
    
    Parameters:
    - test_df (pd.DataFrame): Contains 'embryo_id', 'snip_id', 'class_num', and 'predicted_stage_hpf'.
    - y_pred_proba (np.ndarray): Predicted probabilities from the model.
    - pert_comparisons (list): All perturbations/classes used in the model.
    - pert_plotting (list, optional): Subset of perturbations to plot. Defaults to all.
    - window_size (int): Sliding window size for averaging. Defaults to 3.
    - max_hpf (int): Maximum hours post-fertilization (hpf) to display. Defaults to 40.
    - plot (bool): Whether to display the plot interactively. Defaults to True.
    - save_dir (str, optional): Directory to save the plot to. If provided, the plot will be saved.
    - filename (str, optional): Filename to save the plot as. Defaults to "average_model_predictions_multiclass.html".
    - highlight_embryos (list of [str, str] or [str], optional): List of embryo IDs with optional colors. Defaults to None.
    
    Returns:
    - str or None: Path to the saved file if save_dir is provided, otherwise None
    """
    def generate_pastel_colors(n):
        """Generates n unique pastel colors."""
        cmap = plt.cm.gist_rainbow
        return [mcolors.rgb2hex(cmap(i / n)) for i in range(n)]

    def rgba_to_rgba_str(rgba_tuple):
        r, g, b, a = rgba_tuple
        return f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})'

    # Ensure 'snip_id' is present in test_df for hover information
    if 'snip_id' not in test_df.columns:
        if 'embryo_id' in test_df.columns:
            test_df['snip_id'] = test_df['embryo_id']
        else:
            raise ValueError("DataFrame must contain 'snip_id' or 'embryo_id' columns for hover information.")
    
    # Use pert_comparisons as provided without reordering
    pert_comparisons = list(pert_comparisons)
    
    # Set perturbations to plot (also without reordering)
    if pert_plotting is None:
        pert_plotting = pert_comparisons
    else:
        pert_plotting = list(pert_plotting)
    
    # Map perturbations to labels and vice versa
    perturbation_to_label = {pert: idx for idx, pert in enumerate(pert_comparisons)}
    label_to_perturbation = {idx: pert for idx, pert in enumerate(pert_comparisons)}
    
    # Assign colors using tab10 colormap
    cmap = plt.get_cmap('tab10')
    colors = [rgba_to_rgba_str(cmap(i % 10)) for i in range(len(pert_plotting))]
    perturbation_to_color = {pert: colors[i] for i, pert in enumerate(pert_plotting)}
    
    # Add predicted probabilities to test_df
    proba_columns = [f'proba_{i}' for i in range(y_pred_proba.shape[1])]
    proba_df = pd.DataFrame(y_pred_proba, columns=proba_columns).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    test_df = pd.concat([test_df, proba_df], axis=1)
    
    # Filter test_df based on pert_plotting and max_hpf
    class_nums_to_plot = [perturbation_to_label[pert] for pert in pert_plotting]
    test_df_filtered = test_df[
        (test_df['class_num'].isin(class_nums_to_plot)) & 
        (test_df['predicted_stage_hpf'] <= max_hpf)
    ]
    
    # Process highlighted embryos
    highlight_dict = {}
    pastel_colors = generate_pastel_colors(20)  # Generate 20 unique pastel colors
    
    if highlight_embryos:
        for i, entry in enumerate(highlight_embryos):
            # Process each highlight entry
            if isinstance(entry, list):
                if len(entry) == 1:  # Only embryo_id provided
                    embryo_id, color = entry[0], pastel_colors[i % len(pastel_colors)]
                elif len(entry) == 2:  # Both embryo_id and color provided
                    embryo_id, color = entry
                else:
                    print(f"Invalid highlight entry: {entry}. Skipping.")
                    continue
            else:  # Single string entry
                embryo_id, color = entry, pastel_colors[i % len(pastel_colors)]
                
            # Add to highlight_dict if embryo exists in the filtered data
            if embryo_id in test_df_filtered['embryo_id'].values:
                highlight_dict[embryo_id] = color
            else:
                print(f"Embryo ID '{embryo_id}' not found in the filtered DataFrame. Skipping.")
    
    # Separate regular and highlighted embryos
    highlighted_ids = set(highlight_dict.keys())
    regular_embryo_ids = [eid for eid in test_df_filtered['embryo_id'].unique() if eid not in highlighted_ids]
    
    # Initialize Plotly figure
    fig = go.Figure()
    
    # Function to plot embryos (regular or highlighted)
    def plot_embryos(embryo_ids, color_mapping, is_highlight=False):
        for embryo_id in embryo_ids:
            embryo_data = test_df_filtered[test_df_filtered['embryo_id'] == embryo_id].copy()
            
            # Get the embryo's perturbation label
            embryo_class_num = int(embryo_data['class_num'].iloc[0])
            embryo_perturbation = label_to_perturbation.get(embryo_class_num, None)
            
            if embryo_perturbation not in pert_plotting:
                continue
            
            # Sort by 'predicted_stage_hpf'
            embryo_data.sort_values('predicted_stage_hpf', inplace=True)
            
            # Get prediction probabilities based on classification type
            proba_col = f'proba_{embryo_class_num}'
            
            if proba_col not in embryo_data.columns:
                print(f"Warning: {proba_col} not found. Skipping embryo_id {embryo_id}.")
                continue
                
            # Use direct probabilities (simplified)
            prediction = embryo_data[proba_col]
            
            # Apply sliding window average
            embryo_data['avg_prediction'] = prediction.rolling(
                window=window_size, min_periods=1
            ).mean()
            embryo_data['avg_time'] = embryo_data['predicted_stage_hpf'].rolling(
                window=window_size, min_periods=1
            ).mean()
            
            # Determine styling based on highlight status
            if is_highlight:
                color = color_mapping.get(embryo_id, perturbation_to_color[embryo_perturbation])
                marker_size = 10
                line_width = 5
            else:
                color = perturbation_to_color[embryo_perturbation]
                marker_size = 4
                line_width = 2
            
            # Add line trace
            fig.add_trace(
                go.Scatter(
                    x=embryo_data['avg_time'],
                    y=embryo_data['avg_prediction'],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Add scatter trace with hover info
            fig.add_trace(
                go.Scatter(
                    x=embryo_data['avg_time'],
                    y=embryo_data['avg_prediction'],
                    mode='markers',
                    marker=dict(color=color, size=marker_size),
                    showlegend=False,
                    hovertemplate=(
                        'Snip ID: %{customdata}<br>'
                        'Mean Predicted Stage (hpf): %{x:.2f}<br>'
                        'Avg Prediction Probability: %{y:.2f}<extra></extra>'
                    ),
                    customdata=embryo_data['snip_id']
                )
            )
    
    # Plot regular embryos first
    plot_embryos(regular_embryo_ids, perturbation_to_color, is_highlight=False)
    
    # Plot highlighted embryos last with specified colors
    if highlighted_ids:
        plot_embryos(list(highlighted_ids), highlight_dict, is_highlight=True)
    
    # Add legend entries for perturbations
    for pert in pert_plotting:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=perturbation_to_color[pert], size=10),
                legendgroup=pert,
                showlegend=True,
                name=pert
            )
        )
    
    # Add legend entries for highlighted embryos
    for embryo_id, color in highlight_dict.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, size=10),
                legendgroup=f'Highlight: {embryo_id}',
                showlegend=True,
                name=f'Highlight: {embryo_id}',
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Average Model Predictions Over Time per Embryo',
        xaxis_title='Mean Predicted Stage (hpf)',
        yaxis_title='Avg Prediction Probability',
        legend_title='Embryo Type',
        font=dict(size=14),
        title_font_size=18,
        legend=dict(
            itemsizing='constant',
            font=dict(size=12)
        ),
        template='plotly_white',
        hovermode='closest',
        width=1000,
        height=700
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',
        tickfont=dict(size=12)
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',
        tickfont=dict(size=12)
    )
    
    # Save the plot if save_dir is provided
    saved_path = None
    if save_dir:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Use provided filename or default
        plot_filename = filename or "average_model_predictions_multiclass.html"
        
        # Create full path
        plot_path = os.path.join(save_dir, plot_filename)
        
        # Save the plot
        fig.write_html(plot_path)
        print(f"Average Model Predictions plot saved to: {plot_path}")
        saved_path = plot_path
    
    # Display the plot if plot=True
    if plot:
        fig.show()
        
    return saved_path
def split_train_test_stratified(df, pert_pair, pert_column='phenotype', embryo_id_column='embryo_id', test_size=0.2, random_state=42):
    """
    Splits a dataframe into training and test sets based on unique 'embryo_id',
    ensuring each perturbation in the pair has a proper train/test split.

    Parameters:
    df (pd.DataFrame): The dataframe containing the embryo data
    pert_pair (list): List of two perturbations to compare
    pert_column (str): Column name for perturbation/phenotype
    embryo_id_column (str): Column name for embryo ID
    test_size (float): The proportion of the dataset to include in the test split
    random_state (int): The random seed for reproducibility

    Returns:
    train_df (pd.DataFrame): The training dataframe containing only the filtered perturbations
    test_df (pd.DataFrame): The test dataframe containing only the filtered perturbations
    """
    # Filter dataframe to only include the perturbations in the pair
    filtered_df = df[df[pert_column].isin(pert_pair)].copy()
    
    if filtered_df.empty:
        raise ValueError(f"No data found for perturbations: {pert_pair}")
    
    # Get unique embryo IDs for each perturbation
    train_ids = []
    test_ids = []
    
    for pert in pert_pair:
        pert_df = filtered_df[filtered_df[pert_column] == pert]
        
        if pert_df.empty:
            print(f"Warning: No data found for perturbation '{pert}'")
            continue
            
        pert_embryo_ids = pert_df[embryo_id_column].unique()
        
        if len(pert_embryo_ids) == 0:
            print(f"Warning: No embryo IDs found for perturbation '{pert}'")
            continue
            
        # Split embryo IDs for this perturbation
        pert_train_ids, pert_test_ids = train_test_split(
            pert_embryo_ids, 
            test_size=test_size,
            random_state=random_state
        )
        
        train_ids.extend(pert_train_ids)
        test_ids.extend(pert_test_ids)
    
    # Create train and test dataframes
    train_df = filtered_df[filtered_df[embryo_id_column].isin(train_ids)].reset_index(drop=True)
    test_df = filtered_df[filtered_df[embryo_id_column].isin(test_ids)].reset_index(drop=True)
    
    # Check if we have both perturbations in train and test
    train_perts = train_df[pert_column].unique()
    test_perts = test_df[pert_column].unique()
    
    if len(train_perts) < len(pert_pair) or len(test_perts) < len(pert_pair):
        missing_in_train = set(pert_pair) - set(train_perts)
        missing_in_test = set(pert_pair) - set(test_perts)
        
        print(f"Warning: Not all perturbations present in both splits:")
        if missing_in_train:
            print(f"  Missing in train: {missing_in_train}")
        if missing_in_test:
            print(f"  Missing in test: {missing_in_test}")
    
    # Print statistics about the split
    print(f"Split statistics for {pert_pair}:")
    for pert in pert_pair:
        n_train = sum(train_df[pert_column] == pert)
        n_test = sum(test_df[pert_column] == pert)
        total = n_train + n_test
        if total > 0:
            train_pct = n_train / total * 100
            test_pct = n_test / total * 100
            print(f"  {pert}: {n_train} train ({train_pct:.1f}%), {n_test} test ({test_pct:.1f}%)")
        else:
            print(f"  {pert}: No data found")
    
    return train_df, test_df
