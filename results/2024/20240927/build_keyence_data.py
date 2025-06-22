from src.build.build01A_compile_keyence_torch import build_ff_from_keyence, stitch_ff_from_keyence
import multiprocessing

# Aim here is to rebuild all Keyence datafiles using the most recent version of the image pipeline

def main():
    overwrite_flag = True
    # data_root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    n_workers = 8
    dir_list = ["20240813_extras"] #["20240509_18ss", "20240509_24hpf", "20240510", "20240619", "20240620", "20240717", "20240718", "20240724", "20240725", "20240726", "20240813_24hpf", "20240813_30hpf", "20240813_36hpf"]
    orientation_list = ["horizontal"] #* len(dir_list)
    # build FF images
    build_ff_from_keyence(data_root, n_workers=n_workers, par_flag=True, overwrite_flag=overwrite_flag, dir_list=dir_list)

    # stitch FF images
    stitch_ff_from_keyence(data_root, n_workers=n_workers, par_flag=True, overwrite_flag=overwrite_flag, orientation_list=orientation_list, dir_list=dir_list)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
