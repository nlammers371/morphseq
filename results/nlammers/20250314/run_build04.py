import os
import sys

code_root = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq"
sys.path.insert(0, code_root)

import multiprocessing
from src.build.build04_perform_embryo_qc import perform_embryo_qc
from src.build.build05_make_training_snips import make_image_snips

def main():
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    # root = "Y:\\projects\\data\\morphseq\\"
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    perform_embryo_qc(root)

    train_name = "20250315_ds"
    make_image_snips(root, train_name, rs_factor=0.5, label_var=None, overwrite_flag=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()