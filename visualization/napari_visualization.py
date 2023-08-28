from aicsimageio import AICSImage
import numpy as np
import napari
# read the image data
from ome_zarr.io import parse_url


# full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/yx1_temp/10x_emilin3a-mScarlet_notochord_zstep05_bright_bf_0005.nd2"
# full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/yx1_temp/20230817/20x_E03_48hpf_tdTom_fin.nd2"
full_filename = "/Volumes/LaCie/40x_fin_tdTom_ZF_pec_fin.nd2"
# full_filename = "/Volumes/LaCie/tdTom_ZF_pec_fin.nd2"
# full_filename = "/Volumes/LaCie/40x_tile_fin_tdTom_ZF_pec_fin001.nd2"
imObject = AICSImage(full_filename)
# imObject.set_scene("XYPos:4")
# imData = np.squeeze(imObject.get_image_data("TZYX", C=0))
imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
res_array = np.asarray(res_raw)

#
# # with open("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", 'wb') as f:
# # np.save("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", imData)
#
viewer = napari.view_image(imData, colormap="green", scale=res_array)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()