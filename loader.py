import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import os

class loader():
    def __init__(self, type, transformation) -> None:
        self.dir = "./data/"
        self.train = True
        self.download = True
        self.__dataset = self.__getDataset(type, transformation)
      
    def __getDataset(self, type, transformation):
        if type == "fmnist":
            return datasets.FashionMNIST(
                    root=self.dir,
                    train=self.train,
                    download=self.download,
                    transform=transformation
                    )

        if type == "mnist":
            return datasets.MNIST(
                    root=self.dir,
                    train=self.train,
                    download=self.download,
                    transform=transformation
                    )
        
        if type == "cifar10":
            return datasets.CIFAR10(
                    root=self.dir,
                    train=self.train,
                    download=self.download,
                    transform=transformation
                    )
        if type == "cifar100":
            return datasets.CIFAR100(
                    root=self.dir,
                    train=self.train,
                    download=self.download,
                    transform=transformation
                    )
        else:
            print("unknown dataset")
            exit()

    def getDataset(self):
        return self.__dataset
    