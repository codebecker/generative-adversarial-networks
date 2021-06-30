# -*- coding: utf-8 -*-

import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, nz=100, batchnorm=True):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 28x28 images
        with pixel intensity ranging from -1 to +1 for Mnist and 32x32 images for cifar.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.nz = nz                  #latentdimension
        self.batchnorm = batchnorm
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        # Project the input
        self.linear1 = nn.Linear(self.nz, 256*7*7, bias=False)
        self.bn1d1 = nn.BatchNorm1d(256*7*7) if self.batchnorm else None
        self.leaky_relu = nn.LeakyReLU()

        # Convolutions output dim:[(W−K+2P)/S]+1 W = width of Input, K= Kernel Size(Square) P=Padding, S= Stride
        self.conv1 = nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,      #ouput dim Mnist:[(7−5+2*2)/1]+1 =7
                stride=1,
                padding=2,
                bias=False)
        self.bn2d1 = nn.BatchNorm2d(128) if self.batchnorm else None
# Transpose Convolutions output dim:[(W−(K-1)+2P-1)/S]+1 ,W = width of Input, K= Kernel Size(Square) P=Padding, S= Stride
# or Transpose Convolutions output dim:(W−1)*S-2P+(K-1)+1 ,W = width of Input, K= Kernel Size(Square) P=Padding, S= Stride
        self.conv2 = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,         #ouput dim Mnist:(7-1)*2-2*1+(4-1)+1=14
                stride=2,              #ouput dim Cifar:(7-1)*2-2*0+(4-1)+1=16
                padding=0,
                bias=False)
        self.bn2d2 = nn.BatchNorm2d(64) if self.batchnorm else None

        self.conv3 = nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,     #ouput channel for cifar-> RGB
                kernel_size=4,     #ouput dim Mnist:(14-1)*2-2*1+(4-1)+1=28
                stride=2,          #ouput dim Cifar:(16-1)*2-2*1+(4-1)+1=32
                padding=1,
                bias=False)
        self.tanh = nn.Tanh()
        #self.relu = F.relu()

    def forward(self, input_tensor, embedding=None):
        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(input_tensor)
        intermediate = self.bn1d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = intermediate.view((-1, 256, 7, 7))

        intermediate = self.conv1(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d2(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv3(intermediate)
        output_tensor = self.tanh(intermediate)
        #output_tensor = F.relu(intermediate)
        return output_tensor
    

class Discriminator(nn.Module):
    def __init__(self):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels for Mnist and 3 channel 32x32 pixels for Cifar.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        # Convolutions output dim:[(W−K+2P)/S]+1 W = width of Input, K= Kernel Size(Square) P=Padding, S= Stride
        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,            #ouput dim Mnist:[(28−5+2*2)/2]+1 =14,5 =14
                stride=2,                 #ouput dim Cifar:[(32−5+2*0)/2]+1 =14,5 =14
                padding=0,
                bias=True)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout_2d = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=128,        
                kernel_size=5,          #ouput dim Mnist:[(14−5+2*2)/2]+1 = 7,5 =7
                stride=2,               #ouput dim Cifar:[(14−5+2*2)/2]+1 =7,5 =7
                padding=2,
                bias=True)

        self.linear1 = nn.Linear(128*7*7, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, embedding=None):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = intermediate.view((-1, 128*7*7))
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)

        return output_tensor
    
    