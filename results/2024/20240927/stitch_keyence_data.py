from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
import multiprocessing

# Aim here is to rebuild all Keyence datafiles using the most recent version of the image pipeline

def main():

    # data_root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    # n_workers = 8
    dir_list = ["20231207", "20231208", "20240509_18ss", "20240509_24hpf", "20240510", "20240717", "20240718", "20240724", "20240725", "20240726", "20240813_24hpf", "20240813_30hpf", "20240813_36hpf", "20240813_extras"]
    orientation_list = ["vertical"]*2 + ["horizontal"] * (len(dir_list) - 2)

    # stitch z slices
    stitch_z_from_keyence(data_root, orientation_list=orientation_list, par_flag=True, n_workers=4, overwrite_flag=True, dir_list=dir_list, write_dir=None)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
