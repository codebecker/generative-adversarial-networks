from numpy.core.fromnumeric import size
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset
import numpy as np
import skimage.io as io

class cocoDataSet(Dataset):
    def __init__(self, transform, t2i=False, categories=['cat'], size=32):
        self.__data_dir ='./data/coco/'
        self.__dataset_name = 'val2017'
        self.transform = transform
        self.__categories = categories
        self.__ids = []
        self.__ds_size = size
        self.__t2i = t2i

        self.__getIDsFromCategory()
        if self.__ids:
            self.data, self.labels, self.imgs = self.__loadImages()

    def __getIDsFromCategory(self):
        annFile_categories='{}annotations/instances_{}.json'.format(self.__data_dir, self.__dataset_name)
        annFile_captions = '{}/annotations/captions_{}.json'.format(self.__data_dir, self.__dataset_name)

        self.coco_categories=COCO(annFile_categories)
        self.coco_caps=COCO(annFile_captions)

        catIds = self.coco_categories.getCatIds(catNms=self.__categories)
        imgIds = self.coco_categories.getImgIds(catIds=catIds)
        print('# IDs in COCO categorie {}: {}'.format(self.__categories, len(imgIds)))
        if self.__ds_size <= len(imgIds):
            self.__ids = imgIds[:self.__ds_size]
        else:
            print(f'Warning: Defined dataset size is {self.__ds_size}, but there are only {len(imgIds)} Images available')
            self.__ids = imgIds

    def __loadImages(self):
        data = []
        all_captions = []
        imgs = []
        for idx in self.__ids:
            img = self.coco_categories.loadImgs(idx)[0]
            imgs.append(img)
            # use url to load image
            I = io.imread(img['coco_url']) # shape of np array (480, 640, 3)
            data.append(I)

            # load and display caption annotations
            annIds = self.coco_caps.getAnnIds(imgIds=img['id'])
            anns = self.coco_caps.loadAnns(annIds)
            captions = []
            for entries in anns:
                captions.append((entries['caption']))
            all_captions.append(captions)
        data = np.array(data, dtype=object)
        return data, all_captions, imgs

    def __len__(self):
        return self.__ds_size

    def __getitem__(self, idx):
        #load image
        img = self.imgs[idx]
        image = io.imread(img['coco_url'])
        image = self.transform(image)

        #load labels
        annIds = self.coco_caps.getAnnIds(imgIds=img['id'])
        anns = self.coco_caps.loadAnns(annIds)
        #get random
        label = anns[np.random.choice(5, 1)[0]]['caption']

        if self.__t2i:
            #TODO embedding
            pass
        return image, label
