from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
# dir_list = ["20230525", "20231207"]
# build FF images

stitch_z_from_keyence(root, overwrite_flag=True)