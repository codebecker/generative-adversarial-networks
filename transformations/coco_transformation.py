
import torchvision.transforms as transforms
class Transformation():
    def __init__(self):
        self.__compose = transforms.Compose([
                                            transforms.CenterCrop(300),
                                            transforms.ToTensor()
                                            ])
                                            # transforms.Normalize((0.485, 0.456, 0.406),
        #                                                           (0.229, 0.224, 0.225))
        #
    def get_compose(self):
        return self.__compose