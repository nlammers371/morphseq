import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from dev.flux.flux_run_utils import train_vector_nn



if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    train_vector_nn(
        root=root,
        run_name= "test_potentia",
        wandb_entity="trap-zf-ml",
        wandb_project="dev-flux",
        use_pca=True,
        ) 