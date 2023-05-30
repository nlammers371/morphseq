# save the model

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
import pathlib
from model.conv_vae import VAE

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set path to model and data
model_name = 'vae_depth_z050'
db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/"
datadir = db_path + "Nick/morphSeq/data/vae_20230522/depth_images_res014/"
modeldir = db_path + "Nick/morphSeq/data/vae_20230522/" + model_name + "/"
zd = int(model_name[-3:])
# datadir = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/vae_test/max_images_res014/"
# modeldir = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/vae_test/" + model_name + "/"
outdir = modeldir + "depth_images_res014_pd/" + model_name + "/"

if not os.path.isdir(outdir):
    os.makedirs(outdir)

batch_size = 1

# build data sampler
im_data = datasets.ImageFolder(datadir, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                            transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(im_data, batch_size=batch_size)
n_plot = 250
# Model class must be defined somewhere
net = VAE().to(device)
net.load_state_dict(torch.load(modeldir + model_name, map_location=device))
net.eval()

mu_array = np.empty((len(data_loader), zd))

with torch.no_grad():
    for idx, data in enumerate(random.sample(list(data_loader), n_plot)):
        im_path = im_data.samples[idx][0]
        im_path = pathlib.PureWindowsPath(im_path)
        im_path = im_path.as_posix()

        im_name = im_path.replace(datadir + 'class0/', '')
        im_name = im_name.replace('.tif', '')
        im_name = im_name.replace('.tiff', '')

        imgs, _ = data
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])


        plt.subplot(121)
        plt.imshow(np.squeeze(img))
        plt.xticks([])
        plt.yticks([])
        plt.title('Original')

        out, mu, logVAR = net(imgs)
        mu_array[idx, :] = mu.cpu()
        outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])

        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))

        plt.xticks([])
        plt.yticks([])
        plt.title('Predicted')
        plt.savefig(outdir + im_name + 'vae_pd.tif')

