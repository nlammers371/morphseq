import pandas as pd
import os
from src._Archive.vae import assess_vae_results
from tqdm import tqdm 

# set key path parameters
root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/" # path to top of the data directory
train_folder = "20241107_ds" # name of 'master' training folder that contains all runs
sweep_df_path = os.path.join(root, "metadata", "parameter_sweeps", "sweep01", "")
out_df_path = os.path.join(root, "metadata", "parameter_sweeps", "")

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

out_df_name = out_df_path + "sweep01_compeleted_runs.csv"
if os.path.isdir(out_df_name):
    out_df = pd.read_csv(out_df_name)

    # remove cases where assessment has already been run
    sweep_df = sweep_df.merge(out_df.loc[:, ["process_id"]], how="left", on="process_id", indicator=True)
    sweep_df = sweep_df.loc[sweep_df["_merge"]==False, :]
    sweep_df.drop(labels=["_merge"], axis=1, inplace=True)
    sweep_df.reset_index(inplace=True, drop=True)

    df_list = [out_df]
else:
    df_list = []

# now, get training folder path
for i in tqdm(range(sweep_df.shape[0])):

    row = sweep_df.loc[[i], :]
    model_path = row["model_path"].values[0]
    model_name = os.path.basename(model_path)

    # a couple of parameters for the model assessment script
    overwrite_flag = False # will skip if it detects the exprected output data already
    n_image_figures = 100  # make qualitative side-by-side reconstruction figures

    results_path = assess_vae_results(root, train_folder, model_name, n_image_figures=n_image_figures,
                                                overwrite_flag=overwrite_flag, batch_size=64)
    
    # update 
    row["results_path"] = results_path
    df_list.append(row)

    # save
    out_df = pd.concat(df_list, axis=0, ignore_index=True)
    out_df.to_csv(out_df_name, index=False)

    

