from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
import os
import glob2 as glob
import nd2
# read the image data
from ome_zarr.io import parse_url

# root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/raw_image_data/YX1/"
root = "/Volumes/Sequoia/20240401_phil_hcr/Thbs4b Myod Emilin3a 2024_01_10/"
image_list = sorted(glob.glob(os.path.join(root, "*.czi")))

image_ind = 6

full_filename = image_list[image_ind]
print(full_filename)
imObject = AICSImage(full_filename)

imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)

channel_names = ["DAPI", "Thsb1", "MyoD", "Emilin"]
colors = ["gray", "green", "magenta", "cyan"]
#
viewer = napari.Viewer()
viewer.add_image(imData, channel_axis=0, name=channel_names[::-1], colormap=colors[::-1], scale=tuple(scale_vec))

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()