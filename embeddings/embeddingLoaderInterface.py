from abc import ABC, abstractmethod

class embeddingLoaderInterface():
   @abstractmethod
   def getEmbedding(self, sentence):
       pass
   @abstractmethod
   def getSize(self):
       pass