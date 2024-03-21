from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1
import multiprocessing

def main():
    overwrite_flag = False
    # data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    data_root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    dir_list = ["20240314"]
    # reducing kept z slices because of OOF issue with 1206 experiment
    # build FF images
    build_ff_from_yx1(data_root=data_root, dir_list=dir_list, overwrite_flag=overwrite_flag, metadata_only_flag=False,
                      par_flag=True, n_z_keep_in=None, n_workers=8)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
