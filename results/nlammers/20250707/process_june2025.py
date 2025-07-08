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
    experiments = ["20240812"]#, "20250703_chem3_28C_T00_1325", "20250703_chem3_34C_T00_1131", "20250703_chem3_34C_T01_1457", 
                #    "20250703_chem3_35C_T00_1101", "20250703_chem3_35C_T01_1437",
                #    '20250612_24hpf_ctrl_atf6', '20250612_24hpf_wfs1_ctcf', '20250612_30hpf_ctrl_atf6', '20250612_30hpf_wfs1_ctcf', 
                #     '20250612_36hpf_ctrl_atf6', '20250612_36hpf_wfs1_ctcf', '20250622_chem_28C_T00_1425', '20250622_chem_28C_T01_1658', 
                #     '20250622_chem_34C_T00_1256', '20250622_chem_34C_T01_1632', '20250622_chem_35C_T00_1223_check', '20250622_chem_35C_T01_1605', 
                #     '20250623_chem_28C_T02_1259', '20250623_chem_34C_T02_1231', '20250623_chem_35C_T02_1204', '20250624_chem02_28C_T00_1356', 
                #     '20250624_chem02_28C_T01_1808', '20250624_chem02_34C_T00_1243', '20250624_chem02_34C_T01_1739', '20250624_chem02_35C_T00_1216', 
                #     '20250624_chem02_35C_T01_1711', '20250625_chem02_28C_T02_1332', '20250625_chem02_34C_T02_1301', '20250625_chem02_35C_T02_1228']

    # call pipeline functions
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    # load master experiment log
    manager = ExperimentManager(root=root)

    # Export 
    # manager.export_experiments(experiments=experiments, force_update=False)
    # manager.stitch_experiments(experiments=experiments, force_update=False)
    # manager.stitch_z_experiments(later_than=later_than, force_update=False)
    manager.get_embryo_stats(experiments=experiments, force_update=True)
