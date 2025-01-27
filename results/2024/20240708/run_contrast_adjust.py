from src.build._Archive.build02A_adjust_ff_contrast import adjust_contrast_wrapper
from src.build.build03A_process_embryos_main_par import segment_wells, compile_embryo_stats, extract_embryo_snips

def main():
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    adjust_contrast_wrapper(root, par_flag=True, overwrite_flag=False)

if __name__ == '__main__':
    main()