import torchvision.datasets as datasets
from data.coco.cocoDataSet import cocoDataSet

class loader():
    def __init__(self, type, transformation, *args) -> None:
        self.dir = "./data/"
        self.train = True
        self.download = True
        if args:
            self.opts = args
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
        if type == "coco":
            t2i = self.opts[0]
            categories = self.opts[1]
            size = self.opts[2]
            return cocoDataSet(
                    transform = transformation,
                    t2i=t2i,
                    categories=categories,
                    size=size
                    )
        else:
            print("unknown dataset")
            exit()

    def getDataset(self):
        return self.__dataset
    