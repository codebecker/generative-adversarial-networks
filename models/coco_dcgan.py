

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
        self.factor = 7
        self.nz = nz          #latentdimension   
        self.nz_dim = 256       
        #self.embedding_dim = embedding_dim #Fasttext uses 300  
        #self.projected_embedded_dim = 128 #propsed in paper.
        #self.latent_dim = self.nz_dim + self.projected_embedded_dim

        self.batchnorm = batchnorm
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        #projection of embedding to lower dimensional space

        # self.linear0 = nn.Linear(self.embedding, self.projected_embed_dim)
        # self.bn0d0 = nn.BatchNorm1d(num_features=self.projected_embed_dim) if self.batchnorm else None
        # self.linear0 = nn.Linear(self.embedding_dim, self.projected_embedded_dim*self.factor*self.factor)
        # self.bn0d0 = nn.BatchNorm1d(num_features=self.projected_embedded_dim*self.factor*self.factor) if self.batchnorm else None
        # self.leaky_relu0 = nn.LeakyReLU()

        # Project the input
        self.linear1 = nn.Linear(self.nz, self.nz_dim*self.factor*self.factor, bias=False)
        self.bn1d1 = nn.BatchNorm1d(self.nz_dim*self.factor*self.factor) if self.batchnorm else None
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Convolutions output dim:[(W−K+2P)/S]+1 W = width of Input, K= Kernel Size(Square) P=Padding, S= Stride
        self.conv1 = nn.Conv2d(
                #in_channels = self.latent_dim, # input concatenated noice and embeding
                in_channels=self.nz_dim,
                out_channels=128,
                kernel_size=5,      #ouput dim COCO:[(7−5+2*2)/1]+1 =7
                stride=1,
                padding=2,
                bias=False)
        self.bn2d1 = nn.BatchNorm2d(128) if self.batchnorm else None

# or Transpose Convolutions output dim:(W−1)*S-2P+(K-1)+1 ,W = width of Input, K= Kernel Size(Square) P=Padding, S= Stride
        self.conv2 = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=108,
                kernel_size=4,         
                stride=2,              #ouput dim COCO:(7-1)*2-2*0+(4-1)+1=16
                padding=0,
                bias=False)
        self.bn2d2 = nn.BatchNorm2d(108) if self.batchnorm else None

        self.conv3 = nn.ConvTranspose2d(
                in_channels=108,
                out_channels=87,     
                kernel_size=4,     
                stride=2,          #ouput dim COCO:(16-1)*2-2*1+(4-1)+1=32
                padding=1,
                bias=False)
        self.bn2d3 = nn.BatchNorm2d(87) if self.batchnorm else None
        
        self.conv4 = nn.ConvTranspose2d(
                in_channels=87,
                out_channels=66,     
                kernel_size=4,     #ouput dim COCO:(32-1)*2-2*1+(4-1)+1=64
                stride=2,          
                padding=1,
                bias=False)
        self.bn2d4 = nn.BatchNorm2d(66) if self.batchnorm else None
        
        self.conv5 = nn.ConvTranspose2d(
                in_channels=66,
                out_channels=45,    
                kernel_size=4,     
                stride=2,          #ouput dim COCO:(64-1)*2-2*1+(4-1)+1=128
                padding=1,
                bias=False)
        self.bn2d5 = nn.BatchNorm2d(45) if self.batchnorm else None
        
        self.conv6 = nn.ConvTranspose2d(
                in_channels=45,
                out_channels=24,     
                kernel_size=4,     #
                stride=2,          #ouput dim COCO:(128-1)*2-2*1+(4-1)+1=256
                padding=1,
                bias=False)
        self.bn2d6 = nn.BatchNorm2d(24) if self.batchnorm else None
        
        self.conv7 = nn.ConvTranspose2d(
                in_channels=24,
                out_channels=3,     #Output channel RGB
                kernel_size=4,     
                stride=2,          #ouput dim COCO:(256-1)*2-2*1+(4-1)+1=512
                padding=1,
                bias=False)
        
        self.tanh = nn.Tanh()
        #self.relu = F.relu()

    #def forward(self, z, embed_vector):
    def forward(self, z):
        """ Project embedding to lower dimension"""
        # projection = self.linear0(embed_vector)
        # projection = self.bn0d0(projection)
        # projection = self.leaky_relu0(projection)

        # projection = projection.unsqueeze(2).unsqueeze(3)
        #projection = projection.view((-1, self.projected_embedded_dim, self.factor, self.factor))

        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(z)
        intermediate = self.bn1d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = intermediate.view((-1, self.nz_dim, self.factor, self.factor))

        # print(intermediate.shape)
        # print(projection.shape)
        #concat both inputs
        #concat = torch.cat([intermediate, projection], 1)
        #print(concat.shape)
        

        #pass concated input vector into CNN
        intermediate = self.conv1(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        
        intermediate = self.conv3(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        
        intermediate = self.conv4(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d4(intermediate)
        intermediate = self.leaky_relu(intermediate)
        
        intermediate = self.conv5(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d5(intermediate)
        intermediate = self.leaky_relu(intermediate)
        
        intermediate = self.conv6(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d6(intermediate)
        intermediate = self.leaky_relu(intermediate)
        

        intermediate = self.conv7(intermediate)
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
        self.factor = 7
        self.nz_dim = 256
        #self.embedding_dim = embedding_dim #Fasttext uses 300  
        self.projected_embedding_dim = 128 #propsed in paper.
        self.latent_dim = self.projected_embedding_dim * self.factor * self.factor # *2

        self._init_modules()

    def _init_modules(self):
        # self.projection_linear = nn.Linear(in_features=self.embedding_dim, out_features=self.projected_embedding_dim)
        # self.projection_bn = nn.BatchNorm1d(num_features=self.projected_embedding_dim)
        # self.projection_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        """Initialize the modules."""
        # Convolutions output dim:[(W−K+2P)/S]+1 W = width of Input, K= Kernel Size(Square) P=Padding, S= Stride
        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=5,            #
                stride=2,                 #ouput dim COCO:[(512−5+2*2)/2]+1 =256 =256
                padding=2,
                bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout_2d = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(
                in_channels=24,
                out_channels=45,        
                kernel_size=5,         
                stride=2,               #ouput dim COCO:[(256−5+2*2)/2]+1 =128,5 =128
                padding=2,
                bias=True)
        
        self.conv3 = nn.Conv2d(
                in_channels=45,
                out_channels=66,        
                kernel_size=5,         
                stride=2,               #ouput dim COCO:[(128−5+2*2)/2]+1 = 64
                padding=2,
                bias=True)
        
        self.conv4 = nn.Conv2d(
                in_channels=66,
                out_channels=87,        
                kernel_size=5,         
                stride=2,               #ouput dim COCO:[(64−5+2*0)/2]+1  =32
                padding=2,
                bias=True)
        
        self.conv5 = nn.Conv2d(
                in_channels=87,
                out_channels=108,        
                kernel_size=5,         
                stride=2,               #ouput dim COCO:[(32−5+2*2)/2]+1  =16
                padding=2,
                bias=True)
        
        self.conv6 = nn.Conv2d(
                in_channels=108,
                out_channels=128,        
                kernel_size=5,         
                stride=2,               #ouput dim COCO:[(16−5+2*2)/2]+1 =7,5 =7
                padding=1,
                bias=True)
        

        # self.linear1 = nn.Linear(in_features=self.latent_dim, out_features=1024, bias=True)
        # self.leaky_relu1 = nn.LeakyReLU()
        # self.linear2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        # self.leaky_relu2 = nn.LeakyReLU()
        # self.linear3 = nn.Linear(in_features=512, out_features=1, bias=True)

        # self.sigmoid = nn.Sigmoid()
        
        self.linear1 = nn.Linear(128*7*7, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

   # def forward(self, image, embed_vector):
    def forward(self, image):

        #step 1: reduce dim of embedding
        # projection = self.projection_linear(embed_vector)
        # projection = self.projection_bn(projection)
        # projection = self.projection_leaky_relu(projection)

        # projection = projection.repeat(1,1, self.factor,self.factor)
        # projection = projection.view((-1, 128*self.factor*self.factor))


        #step 2: "no changes" conv 1 and conv 2 
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(image)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        
        intermediate = self.conv3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        
        intermediate = self.conv4(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        
        intermediate = self.conv5(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        
        intermediate = self.conv6(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)
        
        
        # intermediate = intermediate.view((-1, 128*self.factor*self.factor))
        
        # intermediate = self.linear1(intermediate)
        # intermediate = self.leaky_relu1(intermediate)
        # intermediate = self.linear2(intermediate)
        # intermediate = self.leaky_relu2(intermediate)
        # intermediate = self.linear3(intermediate)
        # output_tensor = self.sigmoid(intermediate)

        #step 3 concatenate depthwise with intermediate
        #concat = torch.cat([intermediate, projection], 1)

        #step 4 reduce dimension of concatenation
        # final = self.linear1(concat)
        # final = self.leaky_relu1(final)
        # final = self.linear2(final)
        # final = self.leaky_relu2(final)
        # final = self.linear3(final)
        # output_tensor = self.sigmoid(final)
        
        intermediate = intermediate.view((-1, 128*7*7))
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)

        return output_tensor
    