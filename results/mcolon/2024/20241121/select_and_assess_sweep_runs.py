import pandas as pd
import os
import numpy as np

os.chdir("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
from src.vae.auxiliary_scripts.assess_vae_results import assess_vae_results


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

# ok, now we can filter for sweeps of interest using the fields in the dataframe
# metric_loss_type: nt-xent or triplet
# margin: sets length scale for metric loss in latent space
# metric_weight: relative weight of metric loss in the overall objective
# self_target_prob: fraction of self-comparisons vs other-comparisons
# time_only_flag: if 1, only uses time for metric loss
# holdout_flag: 0 if all data, 1 if lmx/gdf3 held out
# beta: weight of gaussian normalization term in the loss

# an example
metric_loss_type = "triplet"
margin = 0.1
metric_weight = 25
self_target_prob = 0.5
time_only_flag = 0
holdout_flag = 0 
beta = 1

# boolean filter. NOTE THAT SOME COMBOS WILL NOT BE PRESENT (yet)
df_filter = (sweep_df["metric_loss_type"] == metric_loss_type) & (sweep_df["margin"] == margin) & (sweep_df["metric_weight"] == metric_weight) & \
            (sweep_df["self_target_prob"] == self_target_prob) & (sweep_df["time_only_flag"] == time_only_flag) & (sweep_df["holdout_flag"] == holdout_flag) & \
            (sweep_df["beta"] == beta)

# now, get training folder path
if np.any(df_filter):
    model_path = sweep_df.loc[df_filter, "model_path"].values[0]
    model_name = os.path.basename(model_path)
    # a couple of parameters for the model assessment script
    overwrite_flag = False # will skip if it detects the exprected output data already
    n_image_figures = 100  # make qualitative side-by-side reconstruction figures
    #had to change skip figures to true because i didnt have write access. 
    assess_vae_results(root, train_folder, model_name, n_image_figures=n_image_figures, overwrite_flag=overwrite_flag, batch_size=64, skip_figures_flag=True)
