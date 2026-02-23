import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.pipeline_objects import Experiment, ExperimentManager

if __name__ == "__main__":
    
    # export everything dated later than XX
    later_than = 20230616
    earlier_than = 20230901
    force_update = True

    # call pipeline functions
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    # load master experiment log
    manager = ExperimentManager(root=root)

    # Export 
    manager.export_experiments(later_than=later_than, earlier_than=earlier_than, force_update=force_update)