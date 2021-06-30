# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import optim


class Generator(nn.Module):
    def __init__(self, embedding_dim=300, nz=100, batchnorm=True):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 28x28 images
        with pixel intensity ranging from -1 to +1.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.factor = 7
        self.nz = nz          #latentdimension   
        self.nz_dim = 256   
        self.embedding_dim = embedding_dim
        self.projected_embedded_dim = 128
        self.batchnorm = batchnorm
        self.latent_dim = self.nz_dim + self.projected_embedded_dim
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""

        #projection of embedding to lower dimensional space
        self.linear0 = nn.Linear(self.embedding_dim, self.projected_embedded_dim*self.factor*self.factor)
        self.bn0d0 = nn.BatchNorm1d(num_features=self.projected_embedded_dim*self.factor*self.factor) if self.batchnorm else None
        self.leaky_relu0 = nn.LeakyReLU()


        # Project the input
        self.linear1 = nn.Linear(self.nz, self.nz_dim*self.factor*self.factor, bias=False)
        self.bn1d1 = nn.BatchNorm1d(self.nz_dim*self.factor*self.factor) if self.batchnorm else None
        self.leaky_relu = nn.LeakyReLU()

        # Convolutions
        self.conv1 = nn.Conv2d(
                in_channels=self.latent_dim,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False)
        self.bn2d1 = nn.BatchNorm2d(128) if self.batchnorm else None

        self.conv2 = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False)
        self.bn2d2 = nn.BatchNorm2d(64) if self.batchnorm else None

        self.conv3 = nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False)
        self.tanh = nn.Tanh()

    def forward(self, noise, embed_vector):
        """ Project embedding to lower dimension"""
        projection = self.linear0(embed_vector)
        projection = self.bn0d0(projection)
        projection = self.leaky_relu0(projection)
        projection = projection.view((-1, self.projected_embedded_dim, self.factor, self.factor))

        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(noise)
        intermediate = self.bn1d1(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = intermediate.view((-1, self.nz_dim, self.factor, self.factor))

        #concat both inputs
        concat = torch.cat([intermediate, projection], 1)

        intermediate = self.conv1(concat)
        if self.batchnorm:
            intermediate = self.bn2d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d2(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv3(intermediate)
        output_tensor = self.tanh(intermediate)
        return output_tensor
    

class Discriminator(nn.Module):
    def __init__(self, embedding_dim=300):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self.factor = 7
        self.nz_dim = 256
        self.embedding_dim = embedding_dim #Fasttext uses 300  
        self.projected_embedding_dim = 128 #propsed in paper.
        self.latent_dim = self.projected_embedding_dim * self.factor * self.factor *2
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.projection_linear = nn.Linear(in_features=self.embedding_dim, out_features=self.projected_embedding_dim)
        self.projection_bn = nn.BatchNorm1d(num_features=self.projected_embedding_dim)
        self.projection_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout_2d = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True)

        self.linear1 = nn.Linear(in_features=self.latent_dim, out_features=1024, bias=True)
        self.leaky_relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.leaky_relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(in_features=512, out_features=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, image, embed_vector):
        #step 1: reduce dim of embedding
        projection = self.projection_linear(embed_vector)
        projection = self.projection_bn(projection)
        projection = self.projection_leaky_relu(projection)

        projection = projection.repeat(1,1, self.factor,self.factor)
        projection = projection.view((-1, 128*self.factor*self.factor))

        #step 2: pass to CNN
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(image)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        intermediate = intermediate.view((-1, 128*self.factor*self.factor))

        #step 3 concatenate depthwise with intermediate 
        concat = torch.cat([intermediate, projection], 1)

        #step 4 final layer
        final = self.linear1(concat)
        final = self.leaky_relu1(final)
        final = self.linear2(final)
        final = self.leaky_relu2(final)
        final = self.linear3(final)
        output_tensor = self.sigmoid(final)

        return output_tensor