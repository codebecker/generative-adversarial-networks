import fasttext.util
import fasttext
from embeddings.embeddingLoaderInterface import embeddingLoaderInterface

class fasttextLoader(embeddingLoaderInterface):
    def __init__(self) -> None:
        fasttext.util.download_model('en', if_exists='ignore')  # English
        self.__ft = fasttext.load_model('cc.en.300.bin')

    def getSize(self) -> int :
        self.__ft.get_dimension()

    def getEmbedding(self, sentence):
        return  self.__ft.get_word_vector(sentence)

