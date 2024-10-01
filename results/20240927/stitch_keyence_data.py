from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
import multiprocessing

# Aim here is to rebuild all Keyence datafiles using the most recent version of the image pipeline

def main():
    overwrite_flag = False
    # data_root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    # n_workers = 8
    dir_list = ["20240619", "20240620"] #["20240726", "20240813_24hpf", "20240813_30hpf", "20240813_36hpf"] # "20240813_extras", "20240509_18ss", "20240509_24hpf", "20240510", "20240717", "20240718", "20240724", "20240725", 
    orientation_list = ["horizontal"] * len(dir_list)

    # stitch z slices
    stitch_z_from_keyence(data_root, orientation_list=orientation_list, par_flag=False, n_workers=4, overwrite_flag=True, dir_list=dir_list, write_dir=None)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
