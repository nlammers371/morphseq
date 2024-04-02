import os
import glob as glob
from src.functions.utilities import path_leaf
from tqdm import tqdm

date_list = ["20231110", "20231206", "20231218"]
root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images"

for date in tqdm(date_list):
    image_list = sorted(glob.glob(os.path.join(root, date, "*.png")))
    for image_path in image_list:
        image_name = path_leaf(image_path)
        image_name_new = image_name.replace("ff_", "")
        os.rename(image_path, os.path.join(root, date, image_name_new))
