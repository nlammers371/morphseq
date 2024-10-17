from src.build.build03A_process_embryos_main_par import build_well_metadata_master, segment_wells, compile_embryo_stats, extract_embryo_snips
import multiprocessing
from src.build._Archive.build02A_adjust_ff_contrast import adjust_contrast_wrapper
from src.build.build02B_segment_bf_main import apply_unet
def main():
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"

    #build 2
    # adjust_contrast_wrapper(root, par_flag=True, overwrite_flag=False)

    # n_class_vec = [2, 1, 1, 1]
    # model_name_vec = ["unet_emb_v4_0050", "unet_bubble_v0_0050", "unet_yolk_v0_0050", "unet_focus_v2_0050"]
    #
    # for m, model_name in enumerate(model_name_vec):
    #     apply_unet(root=root, model_name=model_name, n_classes=n_class_vec[m], n_workers=2,
    #                overwrite_flag=False, make_sample_figures=True)

    # build 3
    # print('Compiling well metadata...')
    # build_well_metadata_master(root)

    # # # # # print('Compiling embryo metadata...')
    # segment_wells(root, par_flag=False, overwrite_well_stats=False)
    # compile_embryo_stats(root, overwrite_flag=True)
    # extract_embryo_snips(root, par_flag=True, outscale=6.5, dl_rad_um=10, overwrite_flag=True)



if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()