from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

# from src.build.pipeline_objects import Experiment, ExperimentManager
from src.analyze.analysis_utils import calculate_morph_embeddings

if __name__ == "__main__":
    
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    model_name = "20241107_ds_sweep01_optimum"
    # later_than = 20250501
    experiments = ["20250716_chem4_28C_T00_1158", "20250716_chem4_28C_T01_1400",
                   "20250716_chem4_34C_T00_1014", "20250716_chem4_35C_T00_1045"]

    calculate_morph_embeddings(data_root=data_root, model_class="legacy", model_name=model_name, experiments=experiments)