from aicsimageio import AICSImage
import skimage.io as io
import time
import numpy as np
import pims

# path = "/net/trapnell/vol1/trapnell-inst/tdTom_40X_pecfin_timeseries_20231214_well000_t004_probs.tif"
path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/YX1/20231218/tbxta_4x_timeseries.nd2"
# reader = pims.open(path, series=0)
imObject = AICSImage(path)
print("starting load...")
start = time.time()
data = imObject.data
# np.squeeze(imObject.get_image_data("CZYX", T=0))
print(time.time() - start)
print("blah")