from src.build.build01A_compile_keyence_images import build_ff_from_keyence, stitch_ff_from_keyence


overwrite_flag = True

data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
dir_list = None
n_workers = 8
# build FF images
# build_ff_from_keyence(data_root, n_workers=n_workers, par_flag=True, overwrite_flag=overwrite_flag, dir_list=dir_list)

# stitch FF images
stitch_ff_from_keyence(data_root, n_workers=n_workers, par_flag=n_workers>1, overwrite_flag=overwrite_flag, dir_list=dir_list)