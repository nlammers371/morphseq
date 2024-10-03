from src.build.build03A_process_embryos_main_par import build_well_metadata_master, segment_wells, compile_embryo_stats, extract_embryo_snips
import multiprocessing
import multiprocessing
from src.build.build02A_adjust_ff_contrast import adjust_contrast_wrapper
from src.build.build02B_segment_bf_main import apply_unet
from src.build.build04_perform_embryo_qc import perform_embryo_qc
from src.build.build05A_make_training_images_ff import make_image_snips

def main():
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"

    ##############
    # Segmentation
    model_name_vec = ["mask_v0_0100", "via_v1_0100", "yolk_v1_0050", "focus_v0_0100", "bubble_v0_0100"] # , "bubble_raw_v0_0050", "focus_raw_v0_0050"]#'["mask_v0_0050", "bubble_v0_0050", "focus_v0_0050"] #, "via_v0_0050"] #, "unet_yolk_v1_0050", "unet_focus_v2_0050"]
    checkpoint_path = None  # "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/unet_training/UNET_training_mask/training/mask_raw_v2_checkpoints/epoch=82-step=3817.ckpt"
    for m, model_name in enumerate(model_name_vec):
        apply_unet(root=root, model_name=model_name, n_classes=1, checkpoint_path=checkpoint_path,
                   n_workers=2, overwrite_flag=False, make_sample_figures=True, n_sample_figures=5000)

    ##############
    # Image metadata extraction and segmentation
    print('Compiling well metadata...')
    build_well_metadata_master(root)

    print('Compiling embryo metadata...')
    segment_wells(root, par_flag=True, overwrite_well_stats=True)
    #
    compile_embryo_stats(root, overwrite_flag=True)
    extract_embryo_snips(root, par_flag=True, outscale=6.5, dl_rad_um=10, overwrite_flag=True)

    ##############
    # # Snip QC and export
    # perform_embryo_qc(root)
    #
    # train_name = "20240627"
    # make_image_snips(root, train_name, rs_factor=0.5, label_var=None, overwrite_flag=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()