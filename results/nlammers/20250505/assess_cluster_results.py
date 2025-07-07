from src.analyze.assess_hydra_results import assess_hydra_results
import os

if __name__ == "__main__":
    dir_list = ["ntxent_00_n20_T10_bio_20250504_213211"]
                #["ntxent_00_n20_T10_m2_bio_20250505_005919", "ntxent_percep02_20250504_234742",
                # "ntxent_percep005_20250505_054253", "ldm_long_20250504_235948"]

    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20241107_ds/hydra_outputs/"
    for dir_name in dir_list:
        hydra_path = os.path.join(root, dir_name, "")
        assess_hydra_results(hydra_run_path=hydra_path,
                             run_type="run",
                             overwrite_flag=True)