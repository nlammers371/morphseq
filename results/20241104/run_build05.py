import multiprocessing
from src.build.build05_make_training_snips import make_image_snips

def main():
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    train_name = "20241107_ds"
    make_image_snips(root, train_name, rs_factor=0.5, label_var=None, overwrite_flag=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()