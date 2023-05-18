"""
The following is an import of PyTorch libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import random


"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def load_split_train_test(datadir, valid_size = .2, batch_size=64):
    # train_transforms = transforms.Compose([transforms.Resize(224),
    #                                    transforms.ToTensor(),
    #                                    ])
    # test_transforms = transforms.Compose([transforms.Resize(224),
    #                                   transforms.ToTensor(),
    #                                   ])
    train_data = datasets.ImageFolder(datadir, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                             transforms.ToTensor()]))
    test_data = datasets.ImageFolder(datadir, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                            transforms.ToTensor()]))
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return trainloader, testloader

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=1, h=128, w=256, zDim=50, kernel_size=5, stride=2, out_channels=16):
        super(VAE, self).__init__()

        # calculate feature size
        h1, w1 = conv_output_shape([h, w], kernel_size=kernel_size, stride=stride)
        h2, w2 = conv_output_shape([h1, w1], kernel_size=kernel_size, stride=stride)

        featureDim = h2*w2*32
        self.featureDim = featureDim
        self.im_size0 = [h, w]
        self.im_size2 = [h2, w2]

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, out_channels, 5, stride=stride)
        self.encConv2 = nn.Conv2d(out_channels, 2*out_channels, 5, stride=stride)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        op = 0
        if stride == 2:
            op = 1
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(2*out_channels, out_channels, 5, stride=stride, output_padding=op) # NL: output padding is needed to recover original shape
        self.decConv2 = nn.ConvTranspose2d(out_channels, imgChannels, 5, stride=stride, output_padding=op)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, self.featureDim)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        outChannels = int(self.featureDim / self.im_size2[0] / self.im_size2[1])
        x = x.view(-1, outChannels, self.im_size2[0], self.im_size2[1])
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)

        return out, mu, logVar


if __name__ == "__main__":

    """
    Determine if any GPUs are available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datadir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/vae_test/max_images_res014/"
    modeldir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/vae_test/"

    """
    Initialize Hyperparameters
    """
    batch_size = 50
    learning_rate = 1e-3
    num_epochs = 30

    train_loader, test_loader = load_split_train_test(datadir, valid_size=.3, batch_size=batch_size)

    """
    Initialize the network and the Adam optimizer
    """
    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader, 0):
            imgs, _ = data
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss {}'.format(epoch, loss))

    """
    The following part takes a random image from test loader to feed into the VAE.
    Both the original image and generated image from the distribution are shown.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import random

    net.eval()
    with torch.no_grad():
        for data in random.sample(list(test_loader), 1):

            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])

            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, logVAR = net(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])

            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.show()



    # save the model
    torch.save(net.state_dict(), modeldir + '20230516_vae_z050')