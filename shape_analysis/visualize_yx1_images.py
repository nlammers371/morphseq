from aicsimageio import AICSImage
import numpy as np
import napari

full_filename = "/Volumes/LaCie/YX1/20230112/RNA300_GFP_10x_wholeEmbryo.nd2"
# full_filename = "/Volumes/LaCie/YX1/20230112/RNA100_GFP_4x_wholeEmbryo_highResZ.nd2"
imObject = AICSImage(full_filename)
imData = imObject.data
# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
res_array = np.asarray(res_raw)

full_viewer = napari.view_image(imData, colormap="green", scale=res_array)

if __name__ == '__main__':
    napari.run()