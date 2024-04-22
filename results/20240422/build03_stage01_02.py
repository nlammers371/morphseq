from src.build.build03_process_embryos_main_par import build_well_metadata_master, segment_wells

def main():
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    
    # print('Compiling well metadata...')
    build_well_metadata_master(root)
    # #
    # # print('Compiling embryo metadata...')
    segment_wells(root, par_flag=True, overwrite_well_stats=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()