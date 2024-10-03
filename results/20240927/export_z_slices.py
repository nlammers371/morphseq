from src.build.build03B_export_z_snips import extract_embryo_z_snips
import multiprocessing
import os

def main():
   
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    # print(os.cpu_count())
    # print('Compiling well metadata...')
    extract_embryo_z_snips(root, par_flag=True, overwrite_flag=True, outscale=6.5, dl_rad_um=10)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()