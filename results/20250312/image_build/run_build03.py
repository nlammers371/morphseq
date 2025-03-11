import os
import sys

code_root = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq"
sys.path.insert(0, code_root)

from src.build.build03A_process_embryos_main_par import build_well_metadata_master, segment_wells, compile_embryo_stats, extract_embryo_snips
import multiprocessing

def main():
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    # root = "Y:\\projects\\data\\morphseq\\"
    # print("Rebuilding metadata...")
    # build_well_metadata_master(root)
    # #
    # # # print("Segmenting wells...")
    # segment_wells(root, par_flag=True, overwrite_well_stats=False)
    #
    # # print("Compiling stats...")
    compile_embryo_stats(root, overwrite_flag=False, par_flag=True)
    
    print("Extracting snips...")
    extract_embryo_snips(root, par_flag=True, outscale=6.5, dl_rad_um=100, overwrite_flag=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()