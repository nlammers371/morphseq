import os
import sys

code_root = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq"
sys.path.insert(0, code_root)

from src.build.build02B_segment_bf_main import apply_unet


root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
# root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
n_workers = 1
overwrite = False

model_name_vec = ["mask_v0_0100", "via_v1_0100", "yolk_v1_0050", "focus_v0_0100", "bubble_v0_0100"] #, "unet_yolk_v1_0050", "unet_focus_v2_0050"]
checkpoint_path = None #"/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/unet_training/UNET_training_mask/training/mask_raw_v2_checkpoints/epoch=82-step=3817.ckpt"
segment_list = ["20240812"]#"20250703_chem3_28C_T00_1325", "20250703_chem3_34C_T00_1131", "20250703_chem3_34C_T01_1457", "20250703_chem3_35C_T00_1101", "20250703_chem3_35C_T01_1437"]
# segment_list = ['20250612_24hpf_ctrl_atf6', '20250612_24hpf_wfs1_ctcf', '20250612_30hpf_ctrl_atf6', '20250612_30hpf_wfs1_ctcf', 
#                 '20250612_36hpf_ctrl_atf6', '20250612_36hpf_wfs1_ctcf', '20250622_chem_28C_T00_1425', '20250622_chem_28C_T01_1658', 
#                 '20250622_chem_34C_T00_1256', '20250622_chem_34C_T01_1632', '20250622_chem_35C_T00_1223_check', '20250622_chem_35C_T01_1605', 
#                 '20250623_chem_28C_T02_1259', '20250623_chem_34C_T02_1231', '20250623_chem_35C_T02_1204', '20250624_chem02_28C_T00_1356', 
#                 '20250624_chem02_28C_T01_1808', '20250624_chem02_34C_T00_1243', '20250624_chem02_34C_T01_1739', '20250624_chem02_35C_T00_1216', 
#                 '20250624_chem02_35C_T01_1711', '20250625_chem02_28C_T02_1332', '20250625_chem02_34C_T02_1301', '20250625_chem02_35C_T02_1228']

for m, model_name in enumerate(model_name_vec):
    apply_unet(root=root, model_name=model_name,  n_classes=1, checkpoint_path=checkpoint_path, 
               n_workers=n_workers, overwrite_flag=overwrite, make_sample_figures=True, n_sample_figures=100)