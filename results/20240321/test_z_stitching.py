from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
import multiprocessing


def main():
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = None
    n_workers = 8
    overwrite_flag = False
    # build FF images
    # build_ff_from_keyence(data_root, n_workers=n_workers, par_flag=True, overwrite_flag=overwrite_flag, dir_list=dir_list)

    # stitch FF images
    stitch_z_from_keyence(data_root, n_workers=n_workers, par_flag=True, overwrite_flag=overwrite_flag, dir_list=dir_list)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
