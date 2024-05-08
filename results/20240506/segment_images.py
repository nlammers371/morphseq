from src.build.build02B_segment_bf_main import apply_unet

# root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data"
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/"
n_workers = 0
overwrite = True

n_class_vec = [1]
model_name_vec = ["unet_yolk_v2_0025"] #, "unet_yolk_v1_0050", "unet_focus_v2_0050"]
segment_list = None
for m, model_name in enumerate(model_name_vec):
    apply_unet(root=root, model_name=model_name,  n_classes=n_class_vec[m],
               n_workers=n_workers, overwrite_flag=overwrite, make_sample_figures=True)