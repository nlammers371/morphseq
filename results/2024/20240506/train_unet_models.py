from src.segmentation.ml_preprocessing.train_embryo_morph_segmenter import train_unet_classifier

def main():

    model_name = 'unet_yolk_v2_'

    # option to load previously trained model
    pretrained_model = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/segmentation_models/20240507/unet_yolk_v0_0050"
    n_epochs = 25
    # Set path to data
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/"

    seed_str = str(932) #str(126) + "_v2"

    train_unet_classifier(image_root=root, model_type="yolk", model_name=model_name, seed_str=seed_str,
                          n_epoch=n_epochs, pretrained_model=pretrained_model)

if __name__ == '__main__':
    main()