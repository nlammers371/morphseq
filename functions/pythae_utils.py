from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
import torch.nn as nn
import torch
from math import floor
import numpy as np


# define transforms
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor() # the data must be tensors
])

def make_dynamic_rs_transform(im_dims):
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((im_dims[0], im_dims[1])),
        transforms.ToTensor(),
    ])
    return data_transform

#########3
# Define a custom dataset class
class MyCustomDataset(datasets.ImageFolder):

    def __init__(self, root, return_name=False, transform=None, target_transform=None):
        self.return_name = return_name
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X, Y = super().__getitem__(index)

        if not self.return_name:
            return DatasetOutput(
                data=X
            )
        else:
            return DatasetOutput(data=X), self.samples[index]
##########
# Define custom convolutional encoder that allows for variable input size
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, output_padding=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h_out = floor((h_w[0]-1)*stride - 2*pad + dilation*(kernel_size[0] - 1) + output_padding + 1)
    w_out = floor((h_w[1]-1)*stride - 2*pad + dilation*(kernel_size[1] - 1) + output_padding + 1)
    return h_out, w_out

class Encoder_Conv_VAE_FLEX(BaseEncoder):
    def __init__(self, init_config, n_conv_layers=4, n_out_channels=16):
        BaseEncoder.__init__(self)

        stride = 2  # I'm keeping this fixed at 2 for now
        kernel_size = 4 # Keep fixed at

        self.n_out_channels = n_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_conv_layers = n_conv_layers

        self.input_dim = init_config.input_dim
        self.latent_dim = init_config.latent_dim
        self.n_channels = self.input_dim[0]

        # get predicted output size of base image
        [ht, wt] = self.input_dim[1:]
        n_iter_layers = np.min([n_conv_layers, 6])
        for n in range(n_iter_layers):
            [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)

        if n_conv_layers > 7:
            raise Exception("Networks deeper than 7 convolutional layers are not currently supported.")
        # use this to calculate feature size
        featureDim = ht*wt*n_out_channels*2**(n_conv_layers-1)

        self.conv_layers = nn.Sequential()

        for n in range(n_conv_layers):
            if n == 0:
                n_in = self.n_channels
            else:
                n_in = n_out_channels*2**(n-1)
            n_out = n_out_channels*2**n

            if (n == 0) and (n_conv_layers == 7):
                self.conv_layers.append(nn.Conv2d(n_in, out_channels=n_out, kernel_size=5, stride=1, padding=2))  # preserves size
            else:
                self.conv_layers.append(nn.Conv2d(n_in, out_channels=n_out, kernel_size=kernel_size, stride=stride, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(n_out))
            self.conv_layers.append(nn.ReLU())

        # add latent layers
        self.embedding = nn.Linear(featureDim, self.latent_dim)
        self.log_var = nn.Linear(featureDim, self.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class Decoder_Conv_AE_FLEX(BaseDecoder):
    def __init__(self, encoder_config):
        BaseDecoder.__init__(self)

        n_out_channels = encoder_config.n_out_channels
        kernel_size = encoder_config.kernel_size
        stride = encoder_config.stride
        n_conv_layers = encoder_config.n_conv_layers

        self.input_dim = encoder_config.input_dim  # (1, 28, 28)
        self.latent_dim = encoder_config.latent_dim
        self.n_channels = self.input_dim[0]

        # get predicted output size of base image
        [ht, wt] = self.input_dim[1:]
        for n in range(n_conv_layers):
            [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)
        self.h_base = ht*2  # NL: factor of 2 is because we have one fewer conv layer in decoder
        self.w_base = wt*2

        # use this to calculate feature size
        featureDim = 4 * ht * wt * n_out_channels * 2 ** (n_conv_layers - 1)
        self.featureDim = featureDim

        # self.fc = nn.Linear(self.latent_dim, featureDim * 4 * 4)  # not sure where this factor of 16 comes from
        self.fc = nn.Linear(self.latent_dim, featureDim)

        self.deconv_layers = nn.Sequential()
        for n in range(1, n_conv_layers):
            p_ind = n_conv_layers - n
            if n == n_conv_layers-1:
                n_out = self.n_channels
            else:
                n_out = n_out_channels*2**(p_ind-1)
            n_in = n_out_channels*2**p_ind

            self.deconv_layers.append(nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding=1))

            if n == n_conv_layers-1:
                self.deconv_layers.append(nn.Sigmoid())
            else:
                self.deconv_layers.append(nn.BatchNorm2d(n_out))
                self.deconv_layers.append(nn.ReLU())

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], int(self.featureDim / self.w_base / self.h_base), self.h_base, self.w_base)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output


class Decoder_Conv_AE_FLEX_Matched(BaseDecoder):
    def __init__(self, encoder_config):
        BaseDecoder.__init__(self)

        n_out_channels = encoder_config.n_out_channels
        kernel_size = encoder_config.kernel_size
        stride = encoder_config.stride
        n_conv_layers = encoder_config.n_conv_layers

        self.input_dim = encoder_config.input_dim  # (1, 28, 28)
        self.latent_dim = encoder_config.latent_dim
        self.n_channels = self.input_dim[0]

        # get predicted output size of base image
        [ht, wt] = self.input_dim[1:]
        n_iter_layers = np.min([n_conv_layers, 6])
        for n in range(n_iter_layers):
            [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)
        self.h_base = ht
        self.w_base = wt

        # use this to calculate feature size
        featureDim = ht * wt * n_out_channels * 2 ** (n_conv_layers - 1)
        self.featureDim = featureDim

        # self.fc = nn.Linear(self.latent_dim, featureDim * 4 * 4)  # not sure where this factor of 16 comes from
        self.fc = nn.Linear(self.latent_dim, featureDim)

        self.deconv_layers = nn.Sequential()
        for n in range(n_conv_layers):
            p_ind = n_conv_layers - n - 1
            if n == n_conv_layers - 1:
                n_out = self.n_channels
            else:
                n_out = n_out_channels * 2 ** (p_ind - 1)
            n_in = n_out_channels * 2 ** p_ind
            if (n == n_conv_layers-1) and (n_conv_layers == 7):
                self.deconv_layers.append(nn.ConvTranspose2d(n_in, n_out, 5, 1, padding=2))  # size-preserving
            else:
                self.deconv_layers.append(nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding=1))

            if n == n_conv_layers - 1:
                self.deconv_layers.append(nn.Sigmoid())
            else:
                self.deconv_layers.append(nn.BatchNorm2d(n_out))
                self.deconv_layers.append(nn.ReLU())

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], int(self.featureDim / self.w_base / self.h_base), self.h_base, self.w_base)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output