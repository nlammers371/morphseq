from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import napari
import numpy as np
# from napari_animation import Animation
from skimage.measure import label, regionprops, regionprops_table
from aicsimageio import AICSImage

# set parameters
readPath = "/Volumes/BLUEDRIVE/20230525/zf_bact2-tdTom_fin_48hpf_zoom_out.czi"

#############
# Main image
#############

# load in raw czi file
imObject = AICSImage(readPath)
imData = np.squeeze(imObject.data)



# pull second-smallest image and experiment
#im_3 = np.asarray(image_data[level])
# calculate upper resolution limit for display
#res_upper = np.percentile(im_3[3, :, :, :], 99.999)

#colormaps = [channel_metadata[i]["color"] for i in range(len(channel_metadata))]
colormaps = ["gray"]

viewer = napari.view_image(imData)
# labels_layer = viewer.add_labels(label_data[0], name='segmentation', scale=scale_vec)

# viewer.theme = "dark"

if __name__ == '__main__':
    napari.run()