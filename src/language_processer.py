import os
import pickle

from gensim.models import Word2Vec
# from glove import Glove
# from scipy.spatial.distance import cosine

from trainer import Trainer


class LanguageProcesser:
    def __init__(self, recipes, elmo=False, word2vec=False):    # , glove=True
        self.trainer = Trainer()

        self.elmo_model = None
        self.word2vec_model = None
        self.glove_model = None

        if os.path.exists("../models/elmo.pickle"):
            pickle_in = open("/models/elmo.pickle", "rb")
            self.elmo_model = pickle.load(pickle_in)
        elif elmo:
            self.elmo_model = self.trainer.elmo_model(recipes)

        if os.path.exists("../models/word2vec.model"):
            self.word2vec_model = Word2Vec.load("../models/word2vec.model")
        elif word2vec:
            self.word2vec_model = self.trainer.word2vec_model(recipes)

        '''
        if os.path.exists("../models/glove.model"):
            self.glove_model = Glove.load("../models/glove.model")
        elif glove:
            self.glove_model = self.trainer.glove_model(recipes)
        '''

    def closest_words_word2vec(self, word, n=10):
        return self.word2vec_model.wv.most_similar(word, topn=n)

    '''
    def closest_words_glove(self, word, n=10):
        distances = [(cosine(self.glove_model.word_vectors[self.glove_model.dictionary[word]],
                      self.glove_model.word_vectors[self.glove_model.dictionary[X]]), X)
                      for X in self.glove_model.dictionary.keys()]
        sorted_distances = sorted(distances)

        return sorted_distances[1:n+1]    # Se elimina el primero porque es uno mismo
    '''

    def get_embedding_word2vec(self, word):
        if self.word2vec_model is not None:
            return self.word2vec_model.wv[word]
