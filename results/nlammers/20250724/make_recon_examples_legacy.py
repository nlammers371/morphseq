import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.analyze.get_recon_examples import recon_wrapper
import os
# from glob2 import glob

if __name__ == "__main__":

    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models/")
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20241107_ds/"
    # sweep_list = ["sweep10"]#["sweep05", "sweep04", "sweep03", "sweep02"] # ,
    # tr_root = root / "training_outputs" 
    # for sweep in sweep_list:
    #     dir_list = sorted(tr_root.glob(f"{sweep}_*"))
    #     out_path = os.path.join(root, f"{sweep}_figs")
    #     for dir_name in dir_list:
    model_class = "legacy"
    model_name = "20241107_ds_sweep01_optimum"
    hydra_path = root / model_class / model_name  
    out_path = root / f"{model_name}_figs"
    # hydra_path = os.path.join(tr_root, dir_name, "")
    recon_wrapper(hydra_run_path=hydra_path,
                    out_path=out_path,
                    run_type=None,
                    model_class=model_class)