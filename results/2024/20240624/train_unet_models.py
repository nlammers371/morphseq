import os

from src.ml_preprocessing.train_embryo_morph_segmenter import train_unet_classifier

def main():


    model_type_vec = ["via", "yolk", ]
    model_name_vec = ["via_v1_", "yolk_v1_", ] #, "bubble_v0", "focus_v0"]

    # option to load previously trained model
    # pretrained_model = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/segmentation_models/20240507/unet_yolk_v0_0050"
    n_epoch_vec = [100, 50]
    # Set path to data
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/"

    for m in range(len(model_type_vec)):
        model_name = model_name_vec[m]
        model_type = model_type_vec[m]
        train_unet_classifier(image_root=root, model_type=model_type, model_name=model_name,
                              n_epoch=n_epoch_vec[m], pretrained_model=None)

if __name__ == '__main__':
    main()