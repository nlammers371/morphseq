import os
from src._Archive.vae.auxiliary_scripts.train_vae import train_vae
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm

#######################
# Script to test whether recent tweaks to model architecture are impacting performance

def main():

    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    train_folder = "20241107_ds"

    # Params that I will hold fixed for now
    # latent_dim = 100
    n_conv_layers = 5
    n_epochs = 150
    # input_dim = (1, 576, 256)
    input_dim = (1, 288, 128)
    # learning_rate = 1e-4
    model_type = "SeqVAE"
    distance_metric = "euclidean"
    n_workers = 8
    # train_suffix = "sweep01"

    # set paths to metric key(s)
    metric_key_path = os.path.join(root, "metadata/combined_metadata_files/curation/perturbation_metric_key_sweep01.csv")

    # path to file designating inclusion/exclusion of different perturbations
    pert_key_path_list = [os.path.join(root, "metadata/combined_metadata_files/curation/perturbation_train_key_sweep01.csv"),
                          os.path.join(root, "metadata/combined_metadata_files/curation/perturbation_train_key_sweep01_no_gdf3_lmx1b.csv")]
    
    # Read in table with params to test
    in_path = os.path.join(root, "metadata", "parameter_sweeps", "hyperparam_sweep01_df.csv")
    hyperparam_df = pd.read_csv(in_path)

    ####################################################################
    # Set index range to run (this is the only thing that should change)\
    script_path = __file__
    # Extract just the script name
    increment = 100
    script_name = os.path.basename(script_path).replace(".py", "")
    sweep_num = int(script_name.replace("sweep01_", ""))
    index_list = np.arange(sweep_num*increment, (sweep_num+1)*increment)
    out_path = os.path.join(root, "metadata", "parameter_sweeps", "sweep01", f"sweep01_{sweep_num:02}.csv")
    ###################################################################

    if not os.path.exists(out_path):
        temp_df = hyperparam_df.loc[np.isin(hyperparam_df["process_id"].to_numpy(), index_list)].reset_index(drop=True)
        temp_df.to_csv(out_path, index=False)
        run_indices = range(temp_df.shape[0])
    else:
        temp_df = pd.read_csv(out_path)
        run_indices = np.where(temp_df["completed"]==0)[0]

    for i in tqdm(run_indices):

        temp_df = pd.read_csv(out_path)

        params_iter = temp_df.loc[i, :]
        
        metric_loss_type = params_iter["metric_loss_type"]
        margin = params_iter["margin"]
        metric_weight = params_iter["metric_weight"] # 50, 2, 1000] #[0.0001, 0.001, 0.01, 0.1, 1]
        self_target_prob = params_iter["self_target_prob"]
        beta = params_iter["beta"]
        time_only_flag = params_iter["time_only_flag"]
        holdout_flag = params_iter["holdout_flag"]

        latent_dim = params_iter["latent_dim"]
        temperature = params_iter["temperature"]
        learning_rate = params_iter["learning_rate"]
        zn_frac = params_iter["zn_frac"]
        
        # set inclusion/exclusion key holdout flag
        pert_key_path = pert_key_path_list[holdout_flag]

        # set batch size based off of training type
        # metric_loss_type = "NT-Xent"
        if metric_loss_type == "triplet":
            # batch_size = 2048
            batch_size = 512
        else:
            batch_size = 1536
            # batch_size = 2048

        # generate suffix for train folder
        train_suffix = f"sweep_01_block{sweep_num:02}_iter{i:03}"

        # train
        save_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type, 
                    latent_dim=latent_dim, batch_size=batch_size, input_dim=input_dim,
                    n_preload_workers=n_workers, n_load_workers=n_workers, metric_loss_type=metric_loss_type,
                    distance_metric=distance_metric, n_epochs=n_epochs, margin=margin, metric_weight=metric_weight, 
                    learning_rate=learning_rate, n_conv_layers=n_conv_layers, self_target_prob=self_target_prob,
                    beta=beta, time_only_flag=time_only_flag, temperature=temperature, zn_frac=zn_frac,
                    metric_key_path=metric_key_path, pert_time_key_path=pert_key_path)
        
        temp_df.loc[i, "model_path"] = save_dir
        temp_df.loc[i, "completed"] = 1

        temp_df.to_csv(out_path, index=False)



if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()