from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
dir_list = ["20230615", "20230620", "20230622", "20230627", "20230629", "20230830", "20230831", "20231207", "20231208"]
# build FF images

stitch_z_from_keyence(root, overwrite_flag=True, dir_list=dir_list)