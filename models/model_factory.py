from models import cifar_gan, mnist_gan

class model_factory(object):

    @staticmethod
    def generator_factory(type, nz):
        if 'mnist' in type:
            return mnist_gan.Generator(nz)
        if type == 'cifar':
            return cifar_gan.Generator(nz)

    @staticmethod
    def discriminator_factory(type):
        if 'mnist' in type:
            return mnist_gan.Discriminator()
        if type == 'cifar':
            return cifar_gan.Discriminator()
        