
import torchvision.transforms as transforms
class Transformation():
    def __init__(self):
        self.__compose =  transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
                            ])

    def get_compose(self):
        return self.__compose