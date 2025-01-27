from aicsimageio import AICSImage
import numpy as np
import napari
# from skimage.transform import resize
import os
import glob2 as glob


# root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/raw_image_data/YX1/"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/hcr/peripheral_nerves_bmpi/"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/hcr/raw_data/20240329/"
root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/raw_data/20240726/"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/hcr/raw_data/"
image_list = sorted(glob.glob(os.path.join(root, "*.nd2")))
#
image_ind = 1

full_filename = image_list[image_ind]
# full_filename = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecFin/HCR_Data/raw/2022_12_21 HCR Prdm1a Robo3 Fgf10a/2022_12_21 HCR Prdm1a Robo3 Fgf10a_3_decon.czi"
print(full_filename)
imObject = AICSImage(full_filename)

imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)

# channel_names = ["DAPI", "Thsb1", "MyoD", "Emilin"]
# colors = ["gray", "green", "magenta", "cyan"]
#
viewer = napari.Viewer()
viewer.add_image(imData, channel_axis=0, scale=tuple(scale_vec))

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()