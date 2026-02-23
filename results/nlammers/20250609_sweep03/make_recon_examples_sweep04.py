import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.analyze.get_recon_examples import recon_wrapper
import os
from glob2 import glob

if __name__ == "__main__":

    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/")
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20241107_ds/"
    tr_root = root / "training_outputs" 
    dir_list = sorted(tr_root.glob("sweep04_*"))
    out_path = os.path.join(root, "sweep04_figs")
    for dir_name in dir_list:
        hydra_path = os.path.join(tr_root, dir_name, "")
        recon_wrapper(hydra_run_path=hydra_path,
                      out_path=out_path,
                      run_type="multirun")