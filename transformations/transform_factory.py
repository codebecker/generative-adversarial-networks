from transformations import mnist_transformation, cifar_transformation, fmnist_transformation

class transform_factory(object):

    @staticmethod
    def transform_factory(type):
        if type == 'fmnist':
            return fmnist_transformation.Transformation()
        if type == 'mnist':
            return mnist_transformation.Transformation()
        if type == 'cifar10':
            return cifar_transformation.Transformation()
        if type == 'cifar100':
            return cifar_transformation.Transformation()