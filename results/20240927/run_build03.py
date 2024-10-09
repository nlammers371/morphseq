from src.build.build03A_process_embryos_main_par import build_well_metadata_master, segment_wells, compile_embryo_stats, extract_embryo_snips
import multiprocessing

def main():
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    print("Extracting snips...")
    extract_embryo_snips(root, par_flag=True, outscale=6.5, dl_rad_um=100, overwrite_flag=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()