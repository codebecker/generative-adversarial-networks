from embeddings.fasttext.fasttextLoader import fasttextLoader
from embeddings.transformers.transformerLoader import transformerLoader
class embeddingsLoaderFactory(object):

    @staticmethod
    def embeddingLoader(type):
        if type == 'fasttext':
            return fasttextLoader()
        else:
            return transformerLoader(type)