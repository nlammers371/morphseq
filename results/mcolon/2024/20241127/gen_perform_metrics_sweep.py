import pandas as pd
import os
import numpy as np
import seaborn as sns 

os.chdir("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
from src.vae.auxiliary_scripts.assess_vae_results import assess_vae_results

from src.functions.embryo_df_performance_metrics import (split_train_test, 
                                                        logistic_regression_multiclass, 
                                                        f1_score_over_time_multiclass,
                                                        plot_f1_score_over_time,
                                                        create_f1_score_dataframe,
                                                        compute_average_f1_score,
                                                        plot_average_f1_score_difference,
                                                        compute_metrics_for_dataframes,
                                                        compute_differences,
                                                        compute_graph_metrics,
                                                        compute_histogram,
                                                        compute_kl_divergence,
                                                        plot_differences_together,
                                                        random_subsample,
                                                        compute_jaccard_similarities,
                                                        plot_jaccard_results,
                                                        produce_perfomance_metrics,)


#these arefunctions that will search for the embryo_stats_csv file and then attempt to create it
def find_embryo_stats_csv(model_path):
    """
    Recursively search for the 'embryo_stats_df.csv' file starting from model_path.
    Parameters:
        model_path (str): The base path to start searching.
    Returns:
        str: The full path to the 'embryo_stats_df.csv' file if found, otherwise None.
    """
    target_file = 'embryo_stats_df.csv'
    
    for root, dirs, files in os.walk(model_path):
        if target_file in files and 'figures' in root:
            return os.path.join(root, target_file)
    
    print(f"File '{target_file}' not found under {model_path}")
    return None


def get_embryo_df_path(model_path):
    embryo_df_path = find_embryo_stats_csv(model_path)
    
    if embryo_df_path is None:
        print(f"File 'embryo_stats_df.csv' not found under {model_path}")
        print("Attempting to generate 'embryo_stats_df.csv'...")
        
        path_parts = model_path.strip(os.sep).split(os.sep)
        
        try:
            model_name = os.path.basename(model_path)
            # a couple of parameters for the model assessment script
            overwrite_flag = False # will skip if it detects the exprected output data already
            n_image_figures = 100  # make qualitative side-by-side reconstruction figures
            assess_vae_results(root, train_folder, model_name, n_image_figures=n_image_figures, overwrite_flag=overwrite_flag, batch_size=64, skip_figures_flag=True)

            # Try to find the 'embryo_stats_df.csv' again
            embryo_df_path = find_embryo_stats_csv(model_path)
            
            if embryo_df_path is None:
                print(f"Failed to generate 'embryo_stats_df.csv' under {model_path}")
        except ValueError:
            print(f"Could not parse 'model_path' to extract necessary components: {model_path}")
            embryo_df_path = None
        except ImportError:
            print("Could not import 'assess_vae_results'. Please ensure it is correctly imported.")
            embryo_df_path = None
        except Exception as e:
            print(f"An error occurred while generating 'embryo_stats_df.csv': {e}")
            embryo_df_path = None
    return embryo_df_path







# set key path parameters
root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/" # path to top of the data directory
train_folder = "20241107_ds" # name of 'master' training folder that contains all runs
sweep_df_path = os.path.join(root, "metadata", "parameter_sweeps", "sweep01", "")

# load dataframes for each component of the sweep--I've broken it up into 6 chunks, 5 of which are currently running on the cluster (other is on workstation)
df_list = []
for block_num in range(0, 5):
    # load df
    temp_df = pd.read_csv(sweep_df_path + f"sweep01_{block_num:02}.csv")
    # keep only runs that have finished
    temp_df = temp_df.loc[temp_df["completed"]==1, :]
    # add to list
    df_list.append(temp_df)

# combine
sweep_df = pd.concat(df_list, axis=0, ignore_index=True)

# Subset the dataframe to only include completed entries
completed_df = sweep_df[sweep_df['completed'] == True].copy()

# Apply the function to generate the 'embryo_df_path' column
completed_df['embryo_df_path'] = completed_df['model_path'].apply(find_embryo_stats_csv) # change to get_embryo_df_path when you want to calc the embryo_df

parameter_cols = [
    'metric_loss_type',
    'margin',
    'metric_weight',
    'self_target_prob',
    'time_only_flag',
    'temperature',
    'learning_rate',
    'latent_dim',
    'beta',
    'batch_size',
    'zn_frac'
]

# Group the dataframe by parameter columns
grouped = completed_df.groupby(parameter_cols)

# Initialize a list to collect new rows
new_rows = []

# Iterate over each group
for group_keys, group_df in grouped:
    holdout_flags = group_df['holdout_flag'].unique()
    # Check if both holdout_flag values 0 and 1 are present
    if set(holdout_flags) == {0, 1}:
        # Extract rows for holdout_flag 0 and 1
        hld_row = group_df[group_df['holdout_flag'] == 1].iloc[0]
        nohld_row = group_df[group_df['holdout_flag'] == 0].iloc[0]
        
        # Create a new row with shared parameters and separate paths
        new_row = dict(zip(parameter_cols, group_keys))
        new_row['model_path_hld'] = hld_row['model_path']
        new_row['model_path_nohld'] = nohld_row['model_path']
        new_row['embryo_df_path_hld'] = hld_row['embryo_df_path']
        new_row['embryo_df_path_nohld'] = nohld_row['embryo_df_path']
        
        new_rows.append(new_row)

# Create a new dataframe from the list of new rows
paired_models_df = pd.DataFrame(new_rows)

# Remove rows where 'embryo_df_path_hld' or 'embryo_df_path_nohld' is None
paired_models_df = paired_models_df[
    paired_models_df['embryo_df_path_hld'].notnull() &
    paired_models_df['embryo_df_path_nohld'].notnull()
]

# Ensure 'embryo_df_path_hld' and 'embryo_df_path_nohld' are not the same
duplicates = paired_models_df[paired_models_df['embryo_df_path_hld'] == paired_models_df['embryo_df_path_nohld']]
if not duplicates.empty:
    print("Duplicate embryo_df_path found in the following rows:\n", duplicates)

# Remove rows with duplicate paths
paired_models_df = paired_models_df[paired_models_df['embryo_df_path_hld'] != paired_models_df['embryo_df_path_nohld']]

# Define the list of parameter columns

# Dictionaries to store outputs
distance_metrics_dict = {}
metrics_inter_df_dict = {}
core_performance_metrics = {}
core_performance_metrics_summary_list = []

# Loop over each row in paired_models_df
for i in range(len(paired_models_df)):
    row = paired_models_df.iloc[i]
    
    # Read the dataframes
    df_all = pd.read_csv(row["embryo_df_path_nohld"])
    df_hld = pd.read_csv(row["embryo_df_path_hld"])
    
    # Call the function
    core_performance_metrics, distance_metrics_intra_inter, metrics_inter_df = produce_performance_metrics(
        df_all,
        df_hld,
        pert_comparisons,
        logreg_tol=1e-3,
        subsample_fraction=0.05,
        subsample_fraction_jaccard=0.1,
        num_bins=20,
        max_hpf=40,
        random_state=100,
        plot=True,
        k_neighbors=5
    )
    
    # Compute summary statistics for core_performance_metrics
    # For example, compute the mean of numeric columns
    summary_stats = core_performance_metrics.mean(numeric_only=True)

    #now because we did the mean, add the suffic _mean to the column names
    summary_stats = summary_stats.to_dict()

    # rename columns because these are the mean values 
    summary_stats = {f"{col}_mean": value for col, value in summary_stats.items()}
    
    # Add an identifier for the model
    summary_stats['model_index'] = i
    
    # Append to the list
    core_performance_metrics_summary_list.append(summary_stats)
    
    # Store the other outputs in dictionaries
    distance_metrics_dict[i]    = distance_metrics_intra_inter
    metrics_inter_df_dict[i]    = metrics_inter_df
    core_performance_metrics[i] = core_performance_metrics

# After the loop, create a DataFrame from core_performance_metrics_list
core_performance_metrics_summary_list_df = pd.DataFrame(core_performance_metrics_summary_list)

# Merge core_performance_metrics_df back to paired_models_df on 'model_index'
paired_models_df = paired_models_df.reset_index().rename(columns={'index': 'model_index'})
paired_models_df = paired_models_df.merge(core_performance_metrics_df, on='model_index', how='left')