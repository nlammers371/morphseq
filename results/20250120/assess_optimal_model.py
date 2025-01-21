from src.vae.auxiliary_scripts.assess_vae_results import assess_vae_results


# set key path parameters
root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/" # path to top of the data directory
train_folder = "20241107_ds" # name of 'master' training folder that contains all runs

model_name = "SeqVAE_z100_ne150_sweep_01_block01_iter030" # best model from parameter sweeps

overwrite_flag = True # will skip if it detects the exprected output data already
n_image_figures = 100  # make qualitative side-by-side reconstruction figures

results_path = assess_vae_results(root, train_folder, model_name, n_image_figures=n_image_figures,
                                                overwrite_flag=overwrite_flag, batch_size=64)
    
