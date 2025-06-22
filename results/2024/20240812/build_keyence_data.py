from src.build.build01A_compile_keyence_torch import build_ff_from_keyence, stitch_ff_from_keyence
import multiprocessing


def main():
    overwrite_flag = True
    data_root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    dir_list = ["20231207", "20231208"]
    n_workers = 8
    # build FF images
    # build_ff_from_keyence(data_root, n_workers=n_workers, par_flag=False, overwrite_flag=overwrite_flag, dir_list=dir_list)

    # stitch FF images
    stitch_ff_from_keyence(data_root, n_workers=n_workers, par_flag=True, overwrite_flag=overwrite_flag, dir_list=dir_list)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
