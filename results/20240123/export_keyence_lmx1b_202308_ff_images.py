from src.build.build01A_compile_keyence_images_par import build_ff_from_keyence, stitch_ff_from_keyence

overwrite_flag = True

data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
dir_list = ["lmx1b_20230830", "lmx1b_20230831"]
# build FF images
build_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list, ch_to_use=1, no_timelapse_flag=True)
# stitch FF images
stitch_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)