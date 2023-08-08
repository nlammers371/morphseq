from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
import torch.nn as nn
import torch
from math import floor


# define transforms
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor() # the data must be tensors
])

data_transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize((128, 128),antialias=False)# the data must be tensors
])

#########3
# Define a custom dataset class

class MyCustomDataset(datasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X, _ = super().__getitem__(index)

        return DatasetOutput(
            data=X
        )


##########
# Define custom convolutional encoder that allows for variable input size
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, output_padding=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h_out = floor((h_w[0]-1)*stride - 2*pad + dilation*(kernel_size[0] - 1) + output_padding + 1)
    w_out = floor((h_w[1]-1)*stride - 2*pad + dilation*(kernel_size[1] - 1) + output_padding + 1)
    return h_out, w_out

class Encoder_Conv_VAE_FLEX(BaseEncoder):
    def __init__(self, input_dim=(1, 28, 28), latent_dim=10, n_out_channels=16, stride=2, kernel_size=4):
        BaseEncoder.__init__(self)

        self.input_dim = input_dim # (1, 28, 28)
        self.latent_dim = latent_dim
        self.n_channels = input_dim[1]

        # get predicted output size of base image
        n_conv_layers = 4
        [ht, wt] = input_dim[1:]
        for n in range(n_conv_layers):
            [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)

        # use this to calculate feature size
        featureDim = ht*wt*n_out_channels*2**(n_conv_layers-1)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, out_channels=n_out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(n_out_channels),
            nn.ReLU(),
            nn.Conv2d(n_out_channels, 2*n_out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(2*n_out_channels),
            nn.ReLU(),
            nn.Conv2d(2*n_out_channels, 4*n_out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(4*n_out_channels),
            nn.ReLU(),
            nn.Conv2d(4*n_out_channels, 8*n_out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(8*n_out_channels),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(featureDim, latent_dim)
        self.log_var = nn.Linear(featureDim, latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class Decoder_Conv_AE_FLEX(BaseDecoder):
    def __init__(self, input_dim=(1, 28, 28), latent_dim=10, n_out_channels=32, stride=2, kernel_size=4):
        BaseDecoder.__init__(self)

        self.input_dim = input_dim  # (1, 28, 28)
        self.latent_dim = latent_dim
        self.n_channels = input_dim[1]

        # get predicted output size of base image
        n_conv_layers = 4
        [ht, wt] = input_dim[1:]
        for n in range(n_conv_layers):
            [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)

        # use this to calculate feature size
        featureDim = ht * wt * n_out_channels * 2 ** (n_conv_layers - 1)
        self.featureDim = featureDim

        self.fc = nn.Linear(latent_dim, featureDim * 4 * 4)  # not sure where this factor of 16 comes from
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(featureDim, featureDim/2, 3, 2, padding=1),
            nn.BatchNorm2d(featureDim/2),
            nn.ReLU(),
            nn.ConvTranspose2d(featureDim/2, featureDim/4, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(featureDim/4),
            nn.ReLU(),
            nn.ConvTranspose2d(featureDim/2, self.n_channels, 3, 2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], self.featureDim, 4, 4)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output