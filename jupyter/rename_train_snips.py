import os
import glob2 as glob
import numpy as np
from src.functions.utilities import path_leaf
import re

thresh = 1718000000
snip_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/bf_embryo_masks/"
file_list = sorted(glob.glob(snip_path  + "*"))
m_time_list = np.asarray([os.path.getmtime(path) for path in file_list])
rm_indices = np.where(m_time_list<thresh)[0]
file_name_list = [path_leaf(f) for f in file_list]

file_name_list = [path_leaf(f) for f in file_list]
dir_list = [os.path.dirname(file) for file in file_list]
new_file_name_list = [re.sub("([\(\[]).*?([\)\]])", "", file).replace(" ", "") for file in file_name_list]
new_dir_list = [os.path.join(dir_list[i], new_file_name_list[i]) for i in range(len(dir_list))]
[os.rename(file_list[i], new_dir_list[i]) for i in range(len(file_list))]