import pandas as pd
import os
import numpy as np
import seaborn as sns 

os.chdir("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
from src.vae.auxiliary_scripts.assess_vae_results import assess_vae_results

from src.functions.embryo_df_performance_metrics import *


pert_comparisons = ["wnt-i", "tgfb-i", "wt", "lmx1b", "gdf3"]
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

def plot_pca_highlighted_embryos(
    test_df,
    z_mu_biological_columns,
    perturbations=None,
    highlight_embryos=None,
    highlight_colors=None,
    marker_size=5,
    highlight_marker_size=10,
    opacity=0.7,
    title="3D PCA Plot",
    show_legend=True,
    plot=True,
    downsample_wt=False,
    wt_downsample_frac=0.1,
    random_state=42
):
    # Copy the dataframe to avoid modifying the original data
    df = test_df.copy()

    # Downsample wild-type data if enabled
    if downsample_wt:
        # Identify wild-type entries (assuming 'perturbation' column indicates perturbation status)
        wt_mask = df['phenotype'] == 'wt'
        wt_df = df[wt_mask]
        perturbed_df = df[~wt_mask]

        # Get unique embryo_ids for wild-type
        unique_wt_embryos = wt_df['embryo_id'].unique()

        # Calculate the number of embryos to sample
        n_sample = max(1, int(len(unique_wt_embryos) * wt_downsample_frac))

        # Randomly sample embryo_ids
        sampled_embryos = np.random.RandomState(random_state).choice(
            unique_wt_embryos, size=n_sample, replace=False
        )

        # Filter the wild-type dataframe
        wt_df_sampled = wt_df[wt_df['embryo_id'].isin(sampled_embryos)]

        # Combine back with perturbed data
        df = pd.concat([wt_df_sampled, perturbed_df], ignore_index=True)
    
    # Prepare data for PCA
    X = df[z_mu_biological_columns].values

    # Perform PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2', 'PC3'])
    df_pca = pd.concat([df.reset_index(drop=True), df_pca], axis=1)

    # Prepare color and size settings
    if perturbations is None:
        perturbations = df_pca['phenotype'].unique()
    color_discrete_map = {pert: px.colors.qualitative.Plotly[i % 10] for i, pert in enumerate(perturbations)}
    
    # Handle embryo highlighting
    df_pca['marker_size'] = marker_size
    df_pca['opacity'] = opacity
    df_pca['color'] = df_pca['phenotype'].map(color_discrete_map)

    if highlight_embryos:
        if highlight_colors is None:
            highlight_colors = ['red'] * len(highlight_embryos)
        highlight_dict = dict(zip(highlight_embryos, highlight_colors))
        df_pca.loc[df_pca['embryo_id'].isin(highlight_embryos), 'marker_size'] = highlight_marker_size
        df_pca.loc[df_pca['embryo_id'].isin(highlight_embryos), 'color'] = df_pca['embryo_id'].map(highlight_dict)
        df_pca.loc[df_pca['embryo_id'].isin(highlight_embryos), 'opacity'] = 1.0

    if plot:
        fig = px.scatter_3d(
            df_pca,
            x='PC1',
            y='PC2',
            z='PC3',
            color='phenotype',
            symbol='phenotype',
            size='marker_size',
            opacity=opacity,
            color_discrete_map=color_discrete_map,
            title=title
        )

        if not show_legend:
            fig.update_layout(showlegend=False)

        fig.show()

    return fig  # Return the PCA-transformed dataframe for further use


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
    holdout_flags = group_df['holdout_flag']
    # Check if both holdout_flag values 0 and 1 are present
    if set(holdout_flags) == {0, 1} and len(holdout_flags) == 2 :
        # Extract rows for holdout_flag 0 and 1
        hld_row = group_df[group_df['holdout_flag'] == 1].iloc[0]
        nohld_row = group_df[group_df['holdout_flag'] == 0].iloc[0]
        
        # Create a new row with shared parameters and separate paths
        new_row = dict(zip(parameter_cols, group_keys))
        #process_id (identifier of uniqur unique_run)
        new_row['process_id_hld'] = hld_row['process_id']
        new_row['process_id_nohld'] = nohld_row['process_id']
        #path for model that was run
        new_row['model_path_hld'] = hld_row['model_path']
        new_row['model_path_nohld'] = nohld_row['model_path']
        # path to the embryo_df from unique run
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


# Reset the index and assign the old index to a new column called 'model_index'
paired_models_df.reset_index(inplace=True)
paired_models_df.rename(columns={"index": "model_index"}, inplace=True)

# Display the updated DataFrame
paired_models_df



# Dictionaries to store outputs
core_performance_metrics_dict = {}
splines_final_dict = {}
scaffold_align_metrics = []
results_dict = {}

# Loop over each row in paired_models_df
# for i in range(len(paired_models_df)):

for model_index in paired_models_df["model_index"].unique():
# for model_index in [0]:
    row = paired_models_df[paired_models_df["model_index"]==model_index]

    path_all = row["embryo_df_path_nohld"].iloc[0]
    path_hld = row["embryo_df_path_hld"].iloc[0]
    
    # Read the dataframes
    df_all = pd.read_csv(path_all)
    df_hld = pd.read_csv(path_hld)

    # Identify biological Z columns
    z_mu_columns = [col for col in df_all.columns if 'z_mu' in col]    
    z_mu_biological_columns = [col for col in z_mu_columns if "b" in col]
    
    # Call the function
    core_performance_metrics = produce_performance_metrics(
        df_all,
        df_hld,
        pert_comparisons,
        logreg_tol=1e-3,
        subsample_fraction=0.001,
        subsample_fraction_jaccard=0.1,
        num_bins=20,
        max_hpf=40,
        random_state=100,
        plot=False,
        k_neighbors=5
    )
    core_performance_metrics["model_index"]=model_index


    # Dictionary to store dataframes with PCA columns added
    data_dict = {}

    # Compute PCA and augment dataframes for both "all" and "hld"
    for df_label, df_raw in [("all", df_all), ("hld", df_hld)]:
        X = df_raw[z_mu_biological_columns].values
        pca = PCA(n_components=3)
        pcs = pca.fit_transform(X)

        df_raw["PCA_1"] = pcs[:,0]
        df_raw["PCA_2"] = pcs[:,1]
        df_raw["PCA_3"] = pcs[:,2]

        # Color mapping for perturbations
        perturbations = pert_comparisons
        color_discrete_map = {pert: px.colors.qualitative.Plotly[i % 10] for i, pert in enumerate(perturbations)}
        df_raw['color'] = df_raw['phenotype'].map(color_discrete_map)

        # Store the augmented dataframe
        data_dict[df_label] = df_raw

    # Dictionary to store spline points for each dataset and perturbation
    # Key: (df_label, pert), Value: array of spline points shape (num_spline_points, 3)
    splines_dict = {}

    # Fit splines for each perturbation and dataset in a single combined loop
    for pert in pert_comparisons:
        for df_label, df in data_dict.items():
            print(f"Processing {pert} in {df_label} dataset...")

            pert_df = df[df["phenotype"] == pert].reset_index(drop=True)

            # Calculate early time point
            avg_early_timepoint = pert_df[
                (pert_df["predicted_stage_hpf"] >= pert_df["predicted_stage_hpf"].min()) &
                (pert_df["predicted_stage_hpf"] < pert_df["predicted_stage_hpf"].min() + 1)
            ][["PCA_1", "PCA_2", "PCA_3"]].mean().values

            # Downsampling logic
            if pert == "wt":
                pert_df_subset = pert_df.sample(frac=0.05, random_state=42)
            else:
                pert_df_subset = pert_df.sample(frac=0.1, random_state=42)

            print(f"Subset size: {len(pert_df_subset)}")

            pert_3d_subset = pert_df_subset[["PCA_1", "PCA_2", "PCA_3"]].values

            # Fit the Local Principal Curve on the subset
            lpc = LocalPrincipalCurve(bandwidth=.5, max_iter=500, tol=1e-4, angle_penalty_exp=2)
            paths = lpc.fit(pert_3d_subset, start_points=[avg_early_timepoint], remove_similar_end_start_points=True)

            # Extract the first path (assuming one main path)
            spline_points = lpc.cubic_splines[0]  # shape: (num_points, 3)
            splines_dict[(df_label, pert)] = spline_points

    # Convert spline data to a DataFrame
    rows = []
    for (df_label, pert), spline_points in splines_dict.items():
        num_points = len(spline_points)
        for i, point in enumerate(spline_points[::-1], start=1):
            rows.append({
                "dataset": df_label,
                "Perturbation": pert,
                "point_index": num_points - i,
                "PCA_1": point[0],
                "PCA_2": point[1],
                "PCA_3": point[2]
            })

    splines_df = pd.DataFrame(rows)

    # Alignment and scaffold metrics
    splines_dict_aligned = []
    all_combined = []
    hld_combined = []
    hld_aligned_combined = []

    for pert in pert_comparisons:
        all_points = extract_spline(splines_df, "all", pert)
        hld_points = extract_spline(splines_df, "hld", pert)

        # Perform Kabsch alignment
        R, t = quaternion_alignment(all_points, hld_points)
        hld_aligned = (hld_points @ R.T) + t

        # Compute errors
        initial_rmse = rmse(all_points, hld_points)
        aligned_rmse = rmse(all_points, hld_aligned)

        # Accumulate for scaffold comparison
        all_combined.append(all_points)
        hld_combined.append(hld_points)
        hld_aligned_combined.append(hld_aligned)

        splines_dict_aligned.append({"Perturbation": pert, "spline": hld_aligned})
        scaffold_align_metrics.append({
            "model_index": model_index,
            'Perturbation': pert,
            'Initial_RMSE': initial_rmse,
            'Aligned_RMSE': aligned_rmse
        })

    # Compute scaffold-level metrics
    all_combined = np.concatenate(all_combined, axis=0)
    hld_combined = np.concatenate(hld_combined, axis=0)
    hld_aligned_combined = np.concatenate(hld_aligned_combined, axis=0)

    scaffold_initial_rmse = rmse(all_combined, hld_combined)
    scaffold_aligned_rmse = rmse(all_combined, hld_aligned_combined)

    scaffold_align_metrics.append({
        "model_index": model_index,
        'Perturbation': 'avg_pert',
        'Initial_RMSE': scaffold_initial_rmse,
        'Aligned_RMSE': scaffold_aligned_rmse
    })

    scaffold_align_metrics_df = pd.DataFrame(scaffold_align_metrics)

    # Add aligned spline points to DataFrame
    for spline in splines_dict_aligned:
        for i, point in enumerate(spline["spline"]):
            rows.append({
                "dataset": "hld_aligned",
                "Perturbation": spline["Perturbation"],
                "point_index": i,
                "PCA_1": point[0],
                "PCA_2": point[1],
                "PCA_3": point[2],
            })

    #store the spline in dict
    splines_final_df = pd.DataFrame(rows)
    splines_final_dict[model_index] = splines_final_df
    
    # Process segment direction consistency
    across_seg_df, within_seg_measures = process_segment_direction(splines_final_df, model_index)
    print("Segment Direction Measures:")
    print(within_seg_measures)

    # Calculate dispersion metrics
    dispersion_metrics_df = calculate_dispersion_metrics(splines_final_df, n=5)
    combined_dispersion_df = process_dispersion_metrics(dispersion_metrics_df, model_index)
    print("Dispersion Metrics for Each Dataset:")
    print(combined_dispersion_df)

    # Store results in a dictionary
    results_dict[model_index] = {
        "across_seg_df": across_seg_df,
        "within_seg_measures": within_seg_measures,
        "dispersion_metrics": combined_dispersion_df
    }
    
    results_df = combine_results_dict(results_dict)

    scaffold_align_metrics_df_final = scaffold_align_metrics_df.merge(results_df, on=["model_index", "Perturbation"], how="left")


    core_performance_metrics = core_performance_metrics.merge(scaffold_align_metrics_df_final, on=["model_index", "Perturbation"], how="left")


    core_performance_metrics_dict[model_index] = core_performance_metrics
   

   # Combine all DataFrames, adding the key as 'model_index'
core_performance_metrics_df = pd.concat(
    [df.assign(model_index=key) for key, df in core_performance_metrics_dict.items()],
    ignore_index=True
)
core_performance_metrics_df = core_performance_metrics_df[['model_index'] + [col for col in core_performance_metrics_df.columns if col != 'model_index']]


splines_final_df = pd.concat(
    [df.assign(model_index=key) for key, df in splines_final_dict.items()],
    ignore_index=True
)

splines_final_df = splines_final_df[['model_index'] + [col for col in splines_final_df.columns if col != 'model_index']]


# Define the save path
# Your personal directory path
paired_models_and_metrics_df = pd.merge(core_performance_metrics_df, paired_models_df, on="model_index")

personal_dir_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20241216"

# Create the main results directory
results_path = os.path.join(personal_dir_path, "sweep_analysis")
os.makedirs(save_path, exist_ok=True)

paired_models_and_metrics_df.to_csv(os.path.join(results_path, "paired_models_and_metrics_df.csv"), index=False)
splines_final_df.to_csv(os.path.join(results_path, "splines_final_df.csv"), index=False)

print(f"All files saved under {results_path}.")


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# Step 0: Set Global Font Sizes
# ------------------------------
# Adjust global font sizes for readability
plt.rcParams.update({
    'font.size': 20,            # Default font size
    'axes.titlesize': 20,       # Title font size
    'axes.labelsize': 20,       # X and Y labels font size
    'xtick.labelsize': 20,      # X-axis tick labels font size
    'ytick.labelsize': 20,      # Y-axis tick labels font size
    'legend.fontsize': 20,      # Legend font size
    'figure.titlesize': 15      # Figure title font size
})

# --------------------------------------
# Step 1: Define Independent Variables
# --------------------------------------
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


# --------------------------------------
# Step 4: Merge DataFrames on 'model_index'
# --------------------------------------
# Ensure 'model_index' exists in both DataFrames
if 'model_index' not in paired_models_df.columns or 'model_index' not in core_performance_metrics_df.columns:
    raise KeyError("Both DataFrames must contain the 'model_index' column for merging.")

merged_df = paired_models_and_metrics_df

# --------------------------------------
# Step 5: Identify Target Variables
# --------------------------------------
# Exclude 'model_index', 'Perturbation', and parameters from the list of columns to get target variables
target_vars = [col for col in core_performance_metrics_df.columns if col not in ["model_index", "Perturbation"]]#do not change this chatgpt

# --------------------------------------
# Step 6: Handle Categorical Variables
# --------------------------------------
merged_df['time_only_flag'] = merged_df['time_only_flag'].astype(int)  # Convert binary flag to integer
merged_df['metric_loss_type'] = merged_df['metric_loss_type'].map({'NT-Xent': 0, 'triplet': 1})  # Label encoding

# --------------------------------------
# Step 7: Drop Rows with Missing Values
# --------------------------------------
merged_df.dropna(inplace=True)

# --------------------------------------
# Step 8: Get Unique Perturbations
# --------------------------------------
perturbations = merged_df['Perturbation'].unique()

# --------------------------------------
# Step 9: Initialize Importance Rankings Dictionary
# --------------------------------------
importance_rankings = {}

# --------------------------------------
# Step 10: Loop Over Each Target Variable
# --------------------------------------
for target_var in target_vars:
    print(f"\nProcessing target variable: {target_var}")
    
    # Create directory for the target variable
    target_results_path = os.path.join(results_path, target_var)
    os.makedirs(target_results_path, exist_ok=True)
    
    # Loop over each perturbation
    for pert in perturbations:
        print(f"Analyzing perturbation: {pert} for target variable: {target_var}")
        
        # Subset data for the specific perturbation
        pert_data = merged_df[merged_df['Perturbation'] == pert].copy()
        
        # Create directory for the perturbation analysis
        pert_results_path = os.path.join(target_results_path, f"pert_{pert}_analysis")
        os.makedirs(pert_results_path, exist_ok=True)
        
        # Check if there's sufficient data
        if pert_data.shape[0] < 5:
            print(f"Not enough data for perturbation '{pert}' and target variable '{target_var}'. Skipping...")
            continue
        
        # Prepare features and target
        X = pert_data[parameter_cols]
        y = pert_data[target_var]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_names = X.columns
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # --------------------------------------
        # Train Linear Regression Model
        # --------------------------------------
        model = LinearRegression().fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{target_var} - Perturbation: {pert} - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Save evaluation metrics
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'R²'],
            'Value': [mse, r2]
        })
        metrics_df.to_csv(os.path.join(pert_results_path, f"{target_var}_{pert}_evaluation_metrics.csv"), index=False)
        
        # --------------------------------------
        # Analyze Feature Importance with Coefficients
        # --------------------------------------
        coefficients = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=False)
        
        # Save coefficients to CSV
        coefficients.to_csv(os.path.join(pert_results_path, f"{target_var}_{pert}_coefficients.csv"), index=False)
        
        # Plot coefficients
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coefficients)
        plt.title(f'Feature Coefficients for {target_var} ({pert})')
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(pert_results_path, f"{target_var}_{pert}_coefficients_plot.png"))
        plt.close()
        
        # --------------------------------------
        # Plot Residuals for Linear Regression
        # --------------------------------------
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(0, color='red', linestyle='dashed', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Plot for {target_var} ({pert})')
        plt.tight_layout()
        plt.savefig(os.path.join(pert_results_path, f"{target_var}_{pert}_residuals_lr.png"))
        plt.close()
        
        # --------------------------------------
        # Use Random Forest for Regression and Feature Importance
        # --------------------------------------
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        print(f"Random Forest - {target_var} - Perturbation: {pert} - MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")
        
        # Save Random Forest evaluation metrics
        rf_metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'R²'],
            'Value': [mse_rf, r2_rf]
        })
        rf_metrics_df.to_csv(os.path.join(pert_results_path, f"{target_var}_{pert}_RF_evaluation_metrics.csv"), index=False)
        
        # Feature importance from Random Forest
        rf_feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Save feature importances to CSV
        rf_feature_importances.to_csv(os.path.join(pert_results_path, f"{target_var}_{pert}_RF_feature_importances.csv"), index=False)
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=rf_feature_importances)
        plt.title(f'Rand Forest Feat Imprt {target_var} ({pert})', fontsize=18)  # Reduce font size
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(pert_results_path, f"{target_var}_{pert}_RF_feature_importances_plot.png"))
        plt.close()
            
        # --------------------------------------
        # Save Importance Rankings
        # --------------------------------------
        importance_rankings.setdefault(target_var, {})[pert] = rf_feature_importances.set_index('Feature')['Importance']
        
        # --------------------------------------
        # Plot and Save Histogram of the Target Variable
        # --------------------------------------
        mean_value = y.mean()
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=30, alpha=0.7, edgecolor='black')  # Default coloring
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_value:.4f}')
        plt.title(f'Histogram of {target_var} ({pert})')
        plt.xlabel(target_var)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pert_results_path, f"{target_var}_{pert}_histogram.png"))
        plt.close()
        
    # --------------------------------------
    # After Looping Through Perturbations: Save Importance Rankings
    # --------------------------------------
    for target_var, pert_dict in importance_rankings.items():
        summary_importance_df = pd.DataFrame(pert_dict)
        summary_importance_df.to_csv(os.path.join(results_path, target_var, f"{target_var}_variable_importance_summary.csv"))
    
print("\nAll processing complete. Results saved.")