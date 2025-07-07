import sys
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
from src._Archive.vae import BaseEncoder, BaseDecoder
from src._Archive.vae import ModelOutput
import torch.nn as nn
import torch
import numpy as np
from src.functions.utilities import conv_output_shape
from scipy.stats import ortho_group

# Define an encoder class with tuneable variables for the number of convolutional layers ad the depth of the conv kernels
class Encoder_Conv_VAE(BaseEncoder):
    def __init__(self, init_config):
        BaseEncoder.__init__(self)

        stride = 2  # I'm keeping this fixed at 2 for now
        kernel_size = 4  # Keep fixed at 4

        self.n_out_channels = init_config.n_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_conv_layers = init_config.n_conv_layers
        self.orth_subspace_flag = init_config.orth_flag
        self.input_dim = init_config.input_dim
        self.latent_dim = init_config.latent_dim
        self.n_channels = self.input_dim[0]

        n_out_channels = self.n_out_channels
        n_conv_layers = self.n_conv_layers
        # get predicted output size of base image
        [ht, wt] = self.input_dim[1:]
        n_iter_layers = np.min([self.n_conv_layers, 6])
        for n in range(n_iter_layers):
            [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)

        if n_conv_layers > 7:
            raise Exception("Networks deeper than 7 convolutional layers are not currently supported.")

        # Calculate how many features we will generate
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
        if not self.orth_subspace_flag:
            self.embedding = nn.Linear(featureDim, self.latent_dim)
        else:   # in this case we add an additional linear layer
            self.embedding0 = nn.Linear(featureDim, self.latent_dim)
            self.embedding = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

            # fix A to be orthogonal (and not trainable)
            m = ortho_group.rvs(dim=self.latent_dim).astype('float32')
            with torch.no_grad():
                self.embedding.weight = nn.Parameter(
                    torch.from_numpy(m), requires_grad=False)
                # self.B.weight = nn.Parameter(
                #     torch.from_numpy(m[n_labels:, :]), requires_grad=False)

        self.log_var = nn.Linear(featureDim, self.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        if not self.orth_subspace_flag:
            output = ModelOutput(
                embedding=self.embedding(h1),
                log_covariance=self.log_var(h1)
            )
        else:
            h2 = self.embedding0(h1)
            output = ModelOutput(
                embedding=self.embedding(h2),
                log_covariance=self.log_var(h1),
                weight_matrix=self.embedding.weight   # return weights so that we can apply orthogonality constraint
            )
        return output

# Defines a "matched" decoder class that inherits key features from its paired encoder
class Decoder_Conv_VAE(BaseDecoder):
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



#
#
# class Encoder_Conv_VAE_FLEX(BaseEncoder):
#     def __init__(self, init_config, n_conv_layers=4, n_out_channels=16):
#         BaseEncoder.__init__(self)
#
#         stride = 2  # I'm keeping this fixed at 2 for now
#         kernel_size = 4 # Keep fixed at
#
#         self.n_out_channels = n_out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.n_conv_layers = n_conv_layers
#
#         self.input_dim = init_config.input_dim
#         self.latent_dim = init_config.latent_dim
#         self.n_channels = self.input_dim[0]
#
#         # get predicted output size of base image
#         [ht, wt] = self.input_dim[1:]
#         n_iter_layers = np.min([n_conv_layers, 6])
#         for n in range(n_iter_layers):
#             [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)
#
#         if n_conv_layers > 7:
#             raise Exception("Networks deeper than 7 convolutional layers are not currently supported.")
#         # use this to calculate feature size
#         featureDim = ht*wt*n_out_channels*2**(n_conv_layers-1)
#
#         self.conv_layers = nn.Sequential()
#
#         for n in range(n_conv_layers):
#             if n == 0:
#                 n_in = self.n_channels
#             else:
#                 n_in = n_out_channels*2**(n-1)
#             n_out = n_out_channels*2**n
#
#             if (n == 0) and (n_conv_layers == 7):
#                 self.conv_layers.append(nn.Conv2d(n_in, out_channels=n_out, kernel_size=5, stride=1, padding=2))  # preserves size
#             else:
#                 self.conv_layers.append(nn.Conv2d(n_in, out_channels=n_out, kernel_size=kernel_size, stride=stride, padding=1))
#             self.conv_layers.append(nn.BatchNorm2d(n_out))
#             self.conv_layers.append(nn.ReLU())
#
#         # add latent layers
#         self.embedding = nn.Linear(featureDim, self.latent_dim)
#         self.log_var = nn.Linear(featureDim, self.latent_dim)
#
#     def forward(self, x: torch.Tensor):
#         h1 = self.conv_layers(x).reshape(x.shape[0], -1)
#         output = ModelOutput(
#             embedding=self.embedding(h1),
#             log_covariance=self.log_var(h1)
#         )
#         return output
#
# # Write decoder class that allows for variable number of convolutional layers
# class Decoder_Conv_AE_FLEX(BaseDecoder):
#     def __init__(self, encoder_config):
#         BaseDecoder.__init__(self)
#
#         n_out_channels = encoder_config.n_out_channels
#         kernel_size = encoder_config.kernel_size
#         stride = encoder_config.stride
#         n_conv_layers = encoder_config.n_conv_layers
#
#         self.input_dim = encoder_config.input_dim  # (1, 28, 28)
#         self.latent_dim = encoder_config.latent_dim
#         self.n_channels = self.input_dim[0]
#
#         # get predicted output size of base image
#         [ht, wt] = self.input_dim[1:]
#         for n in range(n_conv_layers):
#             [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)
#         self.h_base = ht*2  # NL: factor of 2 is because we have one fewer conv layer in decoder
#         self.w_base = wt*2
#
#         # use this to calculate feature size
#         featureDim = 4 * ht * wt * n_out_channels * 2 ** (n_conv_layers - 1)
#         self.featureDim = featureDim
#
#         # self.fc = nn.Linear(self.latent_dim, featureDim * 4 * 4)  # not sure where this factor of 16 comes from
#         self.fc = nn.Linear(self.latent_dim, featureDim)
#
#         self.deconv_layers = nn.Sequential()
#         for n in range(1, n_conv_layers):
#             p_ind = n_conv_layers - n
#             if n == n_conv_layers-1:
#                 n_out = self.n_channels
#             else:
#                 n_out = n_out_channels*2**(p_ind-1)
#             n_in = n_out_channels*2**p_ind
#
#             self.deconv_layers.append(nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding=1))
#
#             if n == n_conv_layers-1:
#                 self.deconv_layers.append(nn.Sigmoid())
#             else:
#                 self.deconv_layers.append(nn.BatchNorm2d(n_out))
#                 self.deconv_layers.append(nn.ReLU())
#
#     def forward(self, z: torch.Tensor):
#         h1 = self.fc(z).reshape(z.shape[0], int(self.featureDim / self.w_base / self.h_base), self.h_base, self.w_base)
#         output = ModelOutput(reconstruction=self.deconv_layers(h1))
#
#         return output
#
#
# class Decoder_Conv_AE_FLEX_Matched(BaseDecoder):
#     def __init__(self, encoder_config):
#         BaseDecoder.__init__(self)
#
#         n_out_channels = encoder_config.n_out_channels
#         kernel_size = encoder_config.kernel_size
#         stride = encoder_config.stride
#         n_conv_layers = encoder_config.n_conv_layers
#
#         self.input_dim = encoder_config.input_dim  # (1, 28, 28)
#         self.latent_dim = encoder_config.latent_dim
#         self.n_channels = self.input_dim[0]
#
#         # get predicted output size of base image
#         [ht, wt] = self.input_dim[1:]
#         n_iter_layers = np.min([n_conv_layers, 6])
#         for n in range(n_iter_layers):
#             [ht, wt] = conv_output_shape([ht, wt], kernel_size=kernel_size, stride=stride, pad=1)
#         self.h_base = ht
#         self.w_base = wt
#
#         # use this to calculate feature size
#         featureDim = ht * wt * n_out_channels * 2 ** (n_conv_layers - 1)
#         self.featureDim = featureDim
#
#         # self.fc = nn.Linear(self.latent_dim, featureDim * 4 * 4)  # not sure where this factor of 16 comes from
#         self.fc = nn.Linear(self.latent_dim, featureDim)
#
#         self.deconv_layers = nn.Sequential()
#         for n in range(n_conv_layers):
#             p_ind = n_conv_layers - n - 1
#             if n == n_conv_layers - 1:
#                 n_out = self.n_channels
#             else:
#                 n_out = n_out_channels * 2 ** (p_ind - 1)
#             n_in = n_out_channels * 2 ** p_ind
#             if (n == n_conv_layers-1) and (n_conv_layers == 7):
#                 self.deconv_layers.append(nn.ConvTranspose2d(n_in, n_out, 5, 1, padding=2))  # size-preserving
#             else:
#                 self.deconv_layers.append(nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding=1))
#
#             if n == n_conv_layers - 1:
#                 self.deconv_layers.append(nn.Sigmoid())
#             else:
#                 self.deconv_layers.append(nn.BatchNorm2d(n_out))
#                 self.deconv_layers.append(nn.ReLU())
#
#     def forward(self, z: torch.Tensor):
#         h1 = self.fc(z).reshape(z.shape[0], int(self.featureDim / self.w_base / self.h_base), self.h_base, self.w_base)
#         output = ModelOutput(reconstruction=self.deconv_layers(h1))
#
#         return output