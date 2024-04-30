from src.build.build04_perform_embryo_qc import perform_embryo_qc
from src.build.build03_process_embryos_main_par import segment_wells, compile_embryo_stats, extract_embryo_snips

def main():
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"

    # segment_wells(root, par_flag=False, overwrite_well_stats=False)
    # compile_embryo_stats(root, overwrite_flag=True)
    extract_embryo_snips(root, par_flag=True, outscale=6.5, dl_rad_um=75, overwrite_flag=True)
    #
    perform_embryo_qc(root)

if __name__ == '__main__':
    main()