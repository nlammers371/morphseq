from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
# read the image data
from ome_zarr.io import parse_url


# full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/yx1_temp/10x_emilin3a-mScarlet_notochord_zstep05_bright_bf_0005.nd2"
# full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/yx1_temp/20230817/20x_E03_48hpf_tdTom_fin.nd2"
# full_filename = "D:\\Nick\\pecfin\\20230817\\40x_fin_tdTom_ZF_pec_fin.nd2" #"/Volumes/LaCie/40x_fin_tdTom_ZF_pec_fin.nd2"
# full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/YX1/20230830/tdTom_54hpf_pecfin_40x.nd2"
full_filename = "/Volumes/LaCie/40x_tile_fin_tdTom_ZF_pec_fin001.nd2"
imObject = AICSImage(full_filename)
# imObject.set_scene("XYPos:6")
# imData = np.squeeze(imObject.get_image_data("TZYX", C=0))
imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)

shape_curr = imData.shape
rs_factor = scale_vec[0] / scale_vec[1]
shape_new = np.asarray(shape_curr)
shape_new[0] = np.round(shape_new[0]*rs_factor).astype(int)
# shape_new = np.round(shape_new / 2).astype(int) # downsize by a factor of 2
imData_rs = resize(imData, shape_new, order=1, anti_aliasing=False)
#
# # with open("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", 'wb') as f:
# # np.save("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", imData)
#
viewer = napari.view_image(imData_rs, colormap="magenta")#, scale=res_array)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()