import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.pipeline_objects import Experiment, ExperimentManager

if __name__ == "__main__":
    
    # export everything dated later than XX
    # later_than = 20250501
    experiments = ["20250716_chem4_35C_T00_1045"] #["20250716_chem4_28C_T00_1158", "20250716_chem4_28C_T01_1400",
                #    "20250716_chem4_34C_T00_1024", "20250716_chem4_35C_T00_1045"]
    # experiments = ["20250215"]

    # experiments = ['20250624_chem02_35C_T00_1216', '20250624_chem02_35C_T01_1711', '20250625_chem02_35C_T02_1228']
    # call pipeline functions
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    # load master experiment log
    manager = ExperimentManager(root=root)
    
    # Export 
    manager.export_experiments(experiments=experiments, force_update=True)
    # manager.export_experiment_metadata(experiments=experiments)
    manager.stitch_experiments(experiments=experiments, force_update=True)
    # manager.stitch_z_experiments(later_than=later_than, force_update=False)
    manager.get_embryo_stats(experiments=experiments, force_update=True)
