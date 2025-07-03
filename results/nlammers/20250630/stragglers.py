import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.pipeline_objects import Experiment, ExperimentManager

if __name__ == "__main__":
    
    # export everything dated later than XX
    experiments = ['20250612_24hpf_wfs1_ctcf']#, '20250622_chem_28C_T00_1425', '20250622_chem_34C_T00_1256', '20250622_chem_35C_T00_1223_check']
    force_update = True

    # call pipeline functions
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    # load master experiment log
    manager = ExperimentManager(root=root)

    # Export 
    manager.export_experiments(experiments=experiments, force_update=force_update)
    manager.stitch_experiments(experiments=experiments, force_update=force_update)