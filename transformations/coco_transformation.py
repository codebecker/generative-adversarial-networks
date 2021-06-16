
import torchvision.transforms as transforms
class Transformation():
    def __init__(self):
        self.__compose = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.CenterCrop(100),
                        transforms.Resize((32,32)),
                        transforms.ToTensor()
                        ])
    def get_compose(self):
        return self.__compose