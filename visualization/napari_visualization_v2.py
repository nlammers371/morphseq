from aicsimageio import AICSImage
import numpy as np
import napari
# from skimage.transform import resize
import os
import glob2 as glob


# root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/raw_image_data/YX1/"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/hcr/peripheral_nerves_bmpi/"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/hcr/raw_data/20240329/"
root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/David/20250227/"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/hcr/raw_data/"


# keyword = "mNeon"
image_list = sorted(glob.glob(os.path.join(root, "*.nd2")))

viewer = napari.Viewer(ndisplay=3)

for image_ind in range(len(image_list)):
    full_filename = image_list[image_ind]
    name = os.path.basename(full_filename).replace(".nd2", "")
    imObject = AICSImage(full_filename)

    imData = np.squeeze(imObject.data)

    # Extract pixel sizes and bit_depth
    res_raw = imObject.physical_pixel_sizes
    scale_vec = np.asarray(res_raw)

    if "mNeon" in name:
        viewer.add_image(imData, colormap="Green", name=name, scale=tuple(scale_vec))
    else:
        viewer.add_image(imData, colormap="Red", name=name, scale=tuple(scale_vec))

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()