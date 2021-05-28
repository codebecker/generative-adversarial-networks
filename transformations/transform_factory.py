from transformations import mnist_transformation, cifar_transformation

class transform_factory(object):

    @staticmethod
    def transform_factory(type):
        if type == 'fmnist':
            return mnist_transformation.Transformation()
        if type == 'cifar10':
            return cifar_transformation.Transforamtion()