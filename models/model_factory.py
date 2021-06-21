from models import cifar_dcgan, mnist_dcgan, t2i_mnist_dcgan, t2i_cifar_dcgan ,t2i_coco_dcgan, coco_dcgan , coco_dcgan_big, cifar_dcgan_big

class model_factory(object):

    @staticmethod
    def generator_factory(type, nz, textToImage=False, embedding_dim = None):
        if 'mnist' in type:
            if textToImage:
                return t2i_mnist_dcgan.Generator(embedding_dim, nz)
            else:
                return mnist_dcgan.Generator(nz)
        if type == 'cifar10': #or type == 'coco':
            if textToImage:
                return t2i_cifar_dcgan.Generator(embedding_dim, nz)
            else:
                #return cifar_dcgan.Generator(nz)
                return cifar_dcgan_big.Generator(nz)

        if type == 'cifar100':
            return cifar_dcgan.Generator(nz)
        
        if type == 'coco':
            if textToImage:
                return t2i_coco_dcgan.Generator(embedding_dim, nz)
            else:
                #return coco_dcgan.Generator(nz)
                #return coco_dcgan_big.Generator(nz)
                return cifar_dcgan.Generator(nz)

    @staticmethod
    def discriminator_factory(type, textToImage=False, embedding_dim = None):
        if 'mnist' in type:
            if textToImage:
                return t2i_mnist_dcgan.Discriminator(embedding_dim)
            else:
                return mnist_dcgan.Discriminator()
        if type == 'cifar10': #or type == 'coco':
            if textToImage:
                return t2i_cifar_dcgan.Discriminator(embedding_dim)
            else:
                #return cifar_dcgan.Discriminator()
                return cifar_dcgan_big.Discriminator()
        if type == 'cifar100':
            return cifar_dcgan.Discriminator()
        
        if type == 'coco':
            if textToImage:
                return t2i_coco_dcgan.Discriminator(embedding_dim)
            else:
                #return coco_dcgan.Discriminator()
                #return coco_dcgan_big.Discriminator()
                return cifar_dcgan.Discriminator()