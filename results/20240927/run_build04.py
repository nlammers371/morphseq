from src.build.build03A_process_embryos_main_par import build_well_metadata_master, segment_wells, compile_embryo_stats, extract_embryo_snips
import multiprocessing
from src.build.build02A_adjust_ff_contrast import adjust_contrast_wrapper
from src.build.build02B_segment_bf_main import apply_unet
from src.build.build04_perform_embryo_qc import perform_embryo_qc
from src.build.build05A_make_training_images_ff import make_image_snips

def main():
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"

    perform_embryo_qc(root)

    # train_name = "20240710_ds"
    # make_image_snips(root, train_name, rs_factor=0.5, label_var=None, overwrite_flag=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()