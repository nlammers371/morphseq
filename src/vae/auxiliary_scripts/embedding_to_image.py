
import sys
sys.path.append('/net/trapnell/vol1/home/mdcolon/proj/morphseq') 
from src.functions.dataset_utils import *
import os
from src.vae.models.auto_model import AutoModel
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from skimage import io


def set_inputs_to_device(device, inputs: Dict[str, Any]):
    inputs_on_device = inputs
    if device == "cuda":
        cuda_inputs = dict.fromkeys(inputs)
        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                cuda_inputs[key] = inputs[key].cuda()
            else:
                cuda_inputs[key] = inputs[key]
        inputs_on_device = cuda_inputs
    return inputs_on_device

#outpath dir where to put images and subfolders

# Script to generate image reconstructions from latent space vectors
def assess_image_set(image_path, embryo_data_path, trained_model_path, out_path, image_prefix_list="", rs_factor=1.0,
                     batch_size=64):
    
    rs_flag = rs_factor != 1
    # check for GPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # load your data here
    print("Loading embryo data...")
    embryo_df = pd.read_csv(os.path.join(embryo_data_path), index_col=0) #ASk Nick it says metadata but us this the actual file 

    # load the model
    print("Loading model...")
    trained_model = AutoModel.load_from_folder(os.path.join(trained_model_path, 'final_model'))
    trained_model = trained_model.to(device)
    
    # make subdir for images
    image_path = os.path.join(out_path, "images_reconstructions")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # make subdir for side-by-side comparison figures
    recon_fig_path = os.path.join(out_path, "recon_figs")
    if not os.path.isdir(recon_fig_path):
        os.makedirs(recon_fig_path)


    # iterate through df...you may wnat to batch this somehow so you can pass multiple mu vecs at once
    for n in tqdm(range(embryo_df.shape[0])): #ASK nick how it knows which columns to get 

        mu, log_var = ... # EXTRACT MU HERE

        recon_x_out = trained_model.decoder(mu)["reconstruction"]

        # x = x.detach().cpu()
        recon_x_out = recon_x_out.detach().cpu()

    

        for b in range(x.shape[0]):
            # save just the recon on its own
            int_recon_out = (np.squeeze(np.asarray(recon_x_out[b, 0, :, :]))*255).astype(np.uint8)
            io.imsave(fname=os.path.join(image_path, image_snip_names[b] + '_loss.jpg'), arr=int_recon_out)       

assess_image_set(trained_model_path="/net/trapnell/vol1/home/mdcolon/proj/morphseq/models/SeqVAE_training_2024-05-11_21-10-40",
                  out_path="/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20240812/embedding_imgs",
                    embryo_data_path="/net/trapnell/vol1/home/mdcolon/proj/fishcaster/data/test_data_wpred_08_08_2024.csv")

# embryo_data_path="/net/trapnell/vol1/home/mdcolon/proj/fishcaster/data/test_data_wpred_08_08_2024.csv"



image_path = os.path.join(out_path, "images_reconstructions")
if not os.path.isdir(image_path):
    os.makedirs(image_path)



trained_model_path="/net/trapnell/vol1/home/mdcolon/proj/morphseq/models/SeqVAE_training_2024-05-11_21-10-40"
out_path="/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20240813/gen_paths_u_lambda_40"
# load your data here
print("Loading embryo data...")
base_embryo_df = pd.read_csv(os.path.join("/net/trapnell/vol1/home/mdcolon/proj/fishcaster/results/08_2024/08_13_2024/gen_t0_uniform_results/sel_ref_df_lamda_20.cvsv"), index_col=0)

gen_embryo_df  = pd.read_csv(os.path.join("/net/trapnell/vol1/home/mdcolon/proj/fishcaster/results/08_2024/08_13_2024/gen_t0_uniform_results/gen_df_lamda_40.csv"), index_col=0) 

z_mu_columns      = [col for col in base_embryo_df.columns if 'z_mu' in col][0:100]
# z_mu_pred_columns =  [col+"_pred" for col in z_mu_columns]

z_mu_df = base_embryo_df[z_mu_columns]
z_mu_pred_df = gen_embryo_df[z_mu_columns]


# check for GPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# load the model
print("Loading model...")
trained_model = AutoModel.load_from_folder(os.path.join(trained_model_path, 'final_model'))
trained_model = trained_model.to(device)


def combine_images(img1, img2):
    # Assumes both images have the same height
    img_height = img1.shape[0]
    img_width1 = img1.shape[1]
    img_width2 = img2.shape[1]
    # adding image data to new numpy obj
    combined_img = np.zeros((img_height, img_width1 + img_width2), dtype=np.uint8)
    combined_img[:, :img_width1] = img1
    combined_img[:, img_width1:] = img2
    return combined_img

# from PIL import Image, ImageDraw, ImageFont
image_path = out_path
for i in range(len(z_mu_df)):
    mu_orig = z_mu_df.iloc[i]
    mu_orig = torch.tensor(mu_orig, dtype=torch.float32).unsqueeze(0)
    recon_x_out_orig = trained_model.decoder(mu_orig)["reconstruction"].detach().numpy()
    int_recon_out_orig = (np.squeeze(np.asarray(recon_x_out_orig[0, 0, :, :])) * 255).astype(np.uint8)
    mu_pred = z_mu_pred_df.iloc[i]
    mu_pred = torch.tensor(mu_pred, dtype=torch.float32).unsqueeze(0)
    recon_x_out_pred = trained_model.decoder(mu_pred)["reconstruction"].detach().numpy()
    int_recon_out_pred = (np.squeeze(np.asarray(recon_x_out_pred[0, 0, :, :])) * 255).astype(np.uint8)
    # Combine the original and predicted images side-by-side
    combined_image = combine_images(int_recon_out_orig, int_recon_out_pred)
    # Save the combined image
    io.imsave(fname=os.path.join(image_path, f'orig_vs_emb_image_{i}.jpg'), arr=combined_image)


# embryo_df[embryo_df["embryo_id"]=="20231110_H04_e00"]


from PIL import Image
import os
import re

def natural_sort_key(filename):
    # Extract numerical parts from the filename
    return [int(part) if part.isdigit() else part for part in re.split('(\d+)', filename)]


# Define the path to the folder containing the JPG images
image_folder = '"/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20240813/gen_paths_u_lambda_5"'
output_gif = 'lambda_5_uniform_proj.gif'

# Get all the image files in the folder and sort them by name
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')], key=natural_sort_key)

# Load the images into a list
images = [Image.open(os.path.join(image_folder, file)) for file in image_files]

# Save as a GIF
images[0].save(output_gif, save_all=True, append_images=images[1:], duration=500, loop=0)


# Define the path to the folder containing the JPG images
image_folder = '"/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20240813/gen_paths_u_lambda_10"'
output_gif = 'lambda_10_uniform_proj.gif'

# Get all the image files in the folder and sort them by name
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')], key=natural_sort_key)

# Load the images into a list
images = [Image.open(os.path.join(image_folder, file)) for file in image_files]

# Save as a GIF
images[0].save(output_gif, save_all=True, append_images=images[1:], duration=500, loop=0)




# Define the path to the folder containing the JPG images
image_folder = '/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20240813/gen_paths_u_lambda_40'
output_gif = 'lambda_40_uniform_proj.gif'

# Get all the image files in the folder and sort them by name
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')], key=natural_sort_key)

# Load the images into a list
images = [Image.open(os.path.join(image_folder, file)) for file in image_files]

# Save as a GIF
images[0].save(output_gif, save_all=True, append_images=images[1:], duration=500, loop=0)


