from src.build.build01A_compile_keyence_images import build_ff_from_keyence, stitch_ff_from_keyence
import multiprocessing


def main():
    overwrite_flag = True
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["20231208"]
    n_workers = 4
    # build FF images
    build_ff_from_keyence(data_root, n_workers=n_workers, par_flag=False, overwrite_flag=overwrite_flag, dir_list=dir_list)

    # stitch FF images
    stitch_ff_from_keyence(data_root, n_workers=n_workers, par_flag=False, overwrite_flag=overwrite_flag, dir_list=dir_list)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
