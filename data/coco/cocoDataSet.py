from numpy.core.fromnumeric import size
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset
import numpy as np
import skimage.io as io
from embeddings.embeddingsLoaderFactory import embeddingsLoaderFactory
import pickle
import random

class cocoDataSet(Dataset):
    def __init__(self, transform, t2i=False, categories=['cat'], size=32, embedding_type = 'distilbert-base-uncased'):
        self.__data_dir ='./data/coco/'
        self.__dataset_name = 'val2017'
        self.transform = transform
        self.__categories = categories
        self.__ids = []
        self.__ds_size = size
        self.__t2i = t2i
        self.__embed_dim = None
        self.__embedding_type = embedding_type
        if self.__t2i:
            self.__embedding_model = embeddingsLoaderFactory.embeddingLoader(type=self.__embedding_type)
            self.__embed_dim = self.__embedding_model.getSize()

        self.__getIDsFromCategory()
        if self.__ids:
            self.__imgs, self.__emb_dict = self.__loadImages()
            if self.__t2i:
                self.writeToFile(self.__emb_dict)

    def __getIDsFromCategory(self):
        annFile_categories='{}annotations/instances_{}.json'.format(self.__data_dir, self.__dataset_name)
        annFile_captions = '{}/annotations/captions_{}.json'.format(self.__data_dir, self.__dataset_name)

        self.coco_categories=COCO(annFile_categories)
        self.coco_caps=COCO(annFile_captions)
        catIds = []
        imgIds = []
        for cat in self.__categories:
            catIds.append(self.coco_categories.getCatIds(catNms=cat))

        for cat_idx, catId in enumerate(catIds):
            tmp_Ids = self.coco_categories.getImgIds(catIds=catId)
            imgIds += tmp_Ids
            print("Coco Dataset: Append categorie {} with {} entries".format(self.__categories[cat_idx], len(tmp_Ids)))
        random.shuffle(imgIds)

        if self.__ds_size <= len(imgIds):
            self.__ids = imgIds[:self.__ds_size]
        else:
            print(f'Warning Coco Dataset: Defined dataset size is {self.__ds_size}, but there are only {len(imgIds)} Images available')
            self.__ids = imgIds

    def __loadImages(self):
        imgs = []
        emb_dict = dict()
        for idx, image_ids in enumerate(self.__ids):
            img = self.coco_categories.loadImgs(image_ids)[0]
            imgs.append(img)
            # use url to load image
            I = io.imread(img['coco_url']) # shape of np array (480, 640, 3)

            # load and display caption annotations
            annIds = self.coco_caps.getAnnIds(imgIds=img['id'])
            anns = self.coco_caps.loadAnns(annIds)
            captions = []
            for entries in anns:
                captions.append((entries['caption']))
            if self.__t2i:
                single_emb_dict = self.__getEmbeddings(captions)
                emb_dict[idx] = single_emb_dict

        return imgs, emb_dict

    def __getEmbeddings(self, captions):
        emb_dict = dict()
        for idx, caption in enumerate(captions):
            embedding = self.__embedding_model.getEmbedding(caption).detach().numpy()
            emb_dict[idx] = embedding
        return emb_dict


    def __len__(self):
        return self.__ds_size

    def __getitem__(self, idx):
        #load image
        img = self.__imgs[idx]
        image = io.imread(img['coco_url'])
        image = self.transform(image)
        if self.__t2i:
            #return just the number of the label
            label = np.random.choice(5, 1)[0]
            return image, idx, label
        else:
            return image, idx


    def getEmbeddingDim(self) -> int:
        return self.__embed_dim

    def writeToFile(self, dictionary):
        path_dir = './embeddings/'
        if self.__embedding_type == 'fasttext':
            path_dir += 'fasttext/'
        else:
            path_dir += 'transformers/' + 'coco_'+ self.__embedding_type

        with open(path_dir + '_embeddings.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
