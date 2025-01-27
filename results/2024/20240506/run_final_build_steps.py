from src.build.build04_perform_embryo_qc import perform_embryo_qc
from src.build.build05A_make_training_images_ff import make_image_snips

def main():
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"

    # segment_wells(root, par_flag=False, overwrite_well_stats=False)
    # compile_embryo_stats(root, overwrite_flag=True)
    # extract_embryo_snips(root, par_flag=True, outscale=6.5, dl_rad_um=75, overwrite_flag=True)
    #
    # perform_embryo_qc(root)

    train_name = "20240508"
    make_image_snips(root, train_name, label_var=None, rs_factor=1.0, overwrite_flag=False)

if __name__ == '__main__':
    main()