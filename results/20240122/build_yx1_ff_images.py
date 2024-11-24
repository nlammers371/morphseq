from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1

overwrite_flag = True
data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
dir_list = ["20231206"] #["20231110", "20231206", "20231215", "20231218"]
# build FF images
# build_ff_from_keyence(data_root, write_dir=write_dir, overwrite_flag=True, dir_list=dir_list, ch_to_use=4)
# stitch FF images
build_ff_from_yx1(data_root=data_root, dir_list=dir_list, overwrite_flag=overwrite_flag)