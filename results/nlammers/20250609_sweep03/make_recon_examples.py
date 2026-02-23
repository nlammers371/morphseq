from src.analyze.get_recon_examples import recon_wrapper
import os
from glob2 import glob

if __name__ == "__main__":

    root = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/20241107_ds/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20241107_ds/"
    tr_root = os.path.join(root, "training_outputs")
    dir_list = sorted(glob(os.path.join(tr_root, "sweep03_*")))
    out_path = os.path.join(root, "sweep03_figs")
    for dir_name in dir_list:
        hydra_path = os.path.join(tr_root, dir_name, "")
        recon_wrapper(hydra_run_path=hydra_path,
                      out_path=out_path,
                      run_type="multirun")