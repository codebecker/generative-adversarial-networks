from models import cifar_dcgan, mnist_dcgan, cifar_gan, mnist_gan

class model_factory(object):

    @staticmethod
    def generator_factory(type, nz):
        if 'mnist' in type:
            return mnist_dcgan.Generator(nz)
        if type == 'cifar10':
            return cifar_dcgan.Generator(nz)
        if type == 'cifar100':
            return cifar_dcgan.Generator(nz)

    @staticmethod
    def discriminator_factory(type):
        if 'mnist' in type:
            return mnist_dcgan.Discriminator()
        if type == 'cifar10':
            return cifar_dcgan.Discriminator()
        if type == 'cifar100':
            return cifar_dcgan.Discriminator()
        