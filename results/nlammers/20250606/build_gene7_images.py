import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01A_compile_keyence_images import stitch_ff_from_keyence # build_ff_from_keyence
from src.build.build01A_compile_keyence_images import build_ff_from_keyence
from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
import multiprocessing

# Aim here is to rebuild all Keyence datafiles using the most recent version of the image pipeline

def main():
    overwrite = True
    # data_root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    n_workers = 1
    # dir_list = ["20250529_24hpf_ctrl_atf6"]#, "20250529_24hpf_wfs1_ctcf", "20250529_30hpf_ctrl_atf6",
                # "20250529_30hpf_wfs1_ctcf", "20250529_36hpf_ctrl_atf6", "20250529_36hpf_wfs1_ctcf", 
                # "20250529_36hpf_extras_wfs1_ctcf"]

    dir_list = ["20250529_24hpf_wfs1_ctcf"]
    orientation_list = ["horizontal"]*len(dir_list)

    # build FF images
    # build_ff_from_keyence(data_root, n_workers=8, overwrite=overwrite, dir_list=dir_list)
    # build_ff_from_keyence(data_root, n_workers=8, par_flag=False, overwrite_flag=overwrite, dir_list=dir_list)

    # stitch FF images
    stitch_ff_from_keyence(data_root, n_workers=n_workers, overwrite=overwrite, orientation_list=orientation_list, dir_list=dir_list)

    # stitch_z_from_keyence(data_root, dir_list=dir_list, orientation_list=orientation_list, n_workers=n_workers, overwrite=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
