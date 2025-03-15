import os
import sys

code_root = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq"
sys.path.insert(0, code_root)

from src.vae.auxiliary_scripts.assess_vae_results import assess_vae_results

def main():
    # set key path parameters
    # root = "Y:\\projects\\data\\morphseq\\" # path to top of the data directory
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    train_folder = "20250315_ds" # name of 'master' training folder that contains all runs

    model_name = "SeqVAE_z100_ne150_sweep_01_block01_iter030" # best model from parameter sweeps

    overwrite_flag = True # will skip if it detects the exprected output data already
    # n_image_figures = 100  # make qualitative side-by-side reconstruction figures

    assess_vae_results(root, train_folder, model_name, skip_figures_flag=True,
                                                overwrite_flag=overwrite_flag, batch_size=64)

if __name__ == '__main__':
    main()
    
