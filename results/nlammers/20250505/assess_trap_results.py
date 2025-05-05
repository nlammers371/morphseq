from src.assess.assess_hydra_results import assess_hydra_results
import os

if __name__ == "__main__":
    dir_list = ["ntxent_squeeze_percep005_beta01_margin2_20250505_001400",
                "ntxent_squeeze_percep005_beta01_margin2_T1_20250505_051059"]

    root = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/20241107_ds/hydra_outputs/"
    for dir_name in dir_list:
        hydra_path = os.path.join(root, dir_name, "")
        assess_hydra_results(hydra_run_path=hydra_path,
                             run_type="run",
                             overwrite_flag=True)