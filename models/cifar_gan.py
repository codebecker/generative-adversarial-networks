import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, width=32, depth=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.width = width
        self.depth = depth
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),

            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),

            nn.Linear(4096, (self.width * self.width * self.depth)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, self.width, self.width)

class Discriminator(nn.Module):
    def __init__(self, n_input=3072):
        super(Discriminator, self).__init__()
        self.n_input = n_input
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, self.n_input)
        return self.main(x)