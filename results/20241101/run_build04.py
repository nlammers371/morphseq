import multiprocessing
from src.build.build04_perform_embryo_qc import perform_embryo_qc

def main():
    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    perform_embryo_qc(root)

    # train_name = "20240710_ds"
    # make_image_snips(root, train_name, rs_factor=0.5, label_var=None, overwrite_flag=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()