# save the model
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler

from conv_vae import VAE

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set path to model and data
modeldir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/vae_test/"
datadir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/vae_test/max_images_res014/"
outdir = modeldir + "max_images_res014_pd/"

if not os.path.isdir(outdir):
    os.makedirs(outdir)

batch_size = 1

# build data sampler
im_data = datasets.ImageFolder(datadir, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                            transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(im_data, batch_size=batch_size)

# Model class must be defined somewhere
net = VAE().to(device)
net.load_state_dict(torch.load(modeldir + '20230516_vae01'))
net.eval()

iter_i = 0
mu_array = np.empty((len(data_loader), 50))

with torch.no_grad():
    for idx, data in enumerate(list(data_loader)):
        im_path = im_data.samples[idx][0]
        im_name = im_path.replace(datadir + 'class0/', '')

        imgs, _ = data
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
        plt.title('Original')

        plt.subplot(121)
        plt.imshow(np.squeeze(img))
        plt.xticks([])
        plt.yticks([])

        out, mu_array[idx, :], logVAR = net(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])

        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))

        plt.xticks([])
        plt.yticks([])
        plt.title('Predicted')
        plt.savefig(outdir + im_name + 'vae_pd.tif')

        iter_i += 1
