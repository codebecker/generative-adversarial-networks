from transformers import AutoTokenizer, AutoModel
import torch
from embeddings.embeddingLoaderInterface import embeddingLoaderInterface

class transformerLoader(embeddingLoaderInterface):
    def __init__(self, model_name='distilbert-base-uncased') -> None:  
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModel.from_pretrained(model_name)
        self.__config = self.__model.config

    def getEmbedding(self, sentence):
        input_ids = torch.tensor(self.__tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
        return self.__model(input_ids)[0][0][0]

    def getSize(self) -> int:
        return self.__config.dim

