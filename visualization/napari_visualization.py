from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
import os
import nd2
# read the image data
from ome_zarr.io import parse_url

root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/raw_image_data/YX1/"
date = "20240126"
image_name = "B09_lmx1b_72hpf_emilin_10x.nd2"

full_filename = os.path.join(root, date, image_name)

imObject = AICSImage(full_filename)

imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)

shape_curr = imData.shape

#
viewer = napari.view_image(imData, colormap="magenta", scale=tuple(scale_vec))

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()