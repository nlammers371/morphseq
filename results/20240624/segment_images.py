from src.build.build02B_segment_bf_main import apply_unet

# root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data"
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
n_workers = 2
overwrite = False

model_name_vec = ["via_v1_0100", "focus_v0_0100"] #, "unet_yolk_v1_0050", "unet_focus_v2_0050"]
checkpoint_path = None #"/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/unet_training/UNET_training_mask/training/mask_raw_v2_checkpoints/epoch=82-step=3817.ckpt"
segment_list = None
for m, model_name in enumerate(model_name_vec):
    apply_unet(root=root, model_name=model_name,  n_classes=1, checkpoint_path=checkpoint_path,
               n_workers=n_workers, overwrite_flag=overwrite, make_sample_figures=True, n_sample_figures=500)