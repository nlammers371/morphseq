from src.build.build02_segment_bf_main import apply_unet

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data"
microscope = "YX1"
n_workers = 4
overwrite = True

n_class_vec = [2, 1, 1, 1]
model_name_vec = ["unet_emb_v4_0050", "unet_bubble_v0_0050", "unet_yolk_v0_0050", "unet_focus_v2_0050"]
segment_list = ["test_case"]

for m, model_name in enumerate(model_name_vec):
    apply_unet(root=root, microscope=microscope, model_name=model_name, segment_list=segment_list, n_classes=n_class_vec[m], n_workers=n_workers, overwrite_flag=overwrite,
    make_sample_figures=True, n_sample_figures=1000)

# first, apply classifier to identify living and dead embryos, as well bubbles
# n_classes = 2
# model_name = "unet_emb_v4_0050"
# apply_unet(root, microscope, model_name, n_classes)

# # now apply bubble classifier
# n_classes = 1
# model_name = "unet_bubble_v0_0050"
# apply_unet(root, model_name, n_classes)

# # now apply yolk classifier
# n_classes = 1
# model_name = "unet_yolk_v0_0050"
# apply_unet(root, model_name, n_classes)

# # now apply classifier to flag out-of-focus embryos
# n_classes = 1
# model_name = "unet_focus_v2_0050"
# apply_unet(root, model_name, n_classes)