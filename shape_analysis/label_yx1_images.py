from aicsimageio import AICSImage
from ome_zarr.reader import Reader
import numpy as np
import napari
# read the image data
from ome_zarr.io import parse_url


readPathLabels = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/training_images_and_labels/30hpf_labels/lbData2.tif"
# reader_lb = Reader(parse_url(readPathLabels))

from ome_zarr.reader import Reader
from PIL import Image
full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/timeseries.nd2"
# lb_path = "/Users/nick/predicted_labels3.tif"
#
# imLB = AICSImage(lb_path)
# lbData = np.squeeze(imLB.data)
# full_filename = "/Volumes/LaCie/YX1/20230112/RNA100_GFP_4x_wholeEmbryo_highResZ.nd2"
imObject = AICSImage(full_filename)
imObject.set_scene("XYPos:17")
imData = np.squeeze(imObject.get_image_data("CZYX", T=13))
# imData = np.squeeze(imObject.data)
# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
res_array = np.asarray(res_raw)

lbObject = AICSImage(readPathLabels)
label_data = lbObject.data
#
# # with open("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", 'wb') as f:
# # np.save("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", imData)
#
viewer = napari.view_image(imData, colormap="green", scale=res_array)
labels_layer = viewer.add_labels(label_data, name='segmentation', scale=res_array)
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()