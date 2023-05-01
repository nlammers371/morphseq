from aicsimageio import AICSImage
import numpy as np
import napari
import apoc
import os
from tifffile import imsave

print("running now")
# set path to image
image_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/timeseries.nd2"
imObject = AICSImage(image_path)

# make path to save output
seg_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/seg_30hpf_classifier/"
if os.path.isdir(seg_path)==False:
    os.makedirs(seg_path)

# set path to classifier
# classifier_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/training_24hpf/SG_24hpf.cl"
classifier_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/30hpf_labels/SG_30hpf.cl"
# apoc.erase_classifier(classifier_path)
output_probability_of_class = 2
# clf = apoc.ProbabilityMapper(opencl_filename=classifier_path, output_probability_of_class=output_probability_of_class)
clf = apoc.ObjectSegmenter(opencl_filename=classifier_path, positive_class_identifier=2)

# for now, just segment one ilustrative time series
scene_index = 17
imObject.set_scene("XYPos:17")
n_time_points = imObject.dims["T"][0]
for t in range(n_time_points):

    # extract image
    imData = np.squeeze(imObject.get_image_data("CZYX", T=t))

    # generate prediction
    result = clf.predict(image=imData)

    outName = seg_path + f'Embryo{scene_index:03}_' + f'T{t:03}.tif'
    imsave(outName, result)

if __name__ == '__main__':
    napari.run()