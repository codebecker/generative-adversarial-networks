from models import MNIST_GAN

class model_factory(object):

    @staticmethod
    def generator_factory(type, nz):
        if type == 'gan':
            return MNIST_GAN.Generator(nz)

    @staticmethod
    def discriminator_factory(type):
        if type == 'gan':
            return MNIST_GAN.Discriminator()
        