from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1

overwrite_flag = True
data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
dir_list = ["20231206"]
# reducing kept z slices because of OOF issue with 1206 experiment
# build FF images
build_ff_from_yx1(data_root=data_root, dir_list=dir_list, overwrite_flag=overwrite_flag, metadata_only_flag=False, n_z_keep_in=8)
