import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1
import multiprocessing

def main():
    # overwrite_flag = False
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    # data_root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    dir_list = ["20250215"]
    # reducing kept z slices because of OOF issue with 1206 experiment
    # build FF images
    build_ff_from_yx1(data_root=data_root, overwrite=True,  dir_list=dir_list)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
