import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models import Word2Vec
from glove import Corpus, Glove
from scipy.spatial.distance import cosine


class LanguageProcesser:
    def __init__(self):
        self.elmo_model = None
        self.word2vec_model = None
        self.glove_model = None

    def closest_words_glove(self, word, n=5):
        distances = [(cosine(self.glove_model.word_vectors[self.glove_model.dictionary[word]], self.glove_model.word_vectors[self.glove_model.dictionary[X]]), X) for X in self.glove_model.dictionary.keys()]
        sorted_distances = sorted(distances)

        return sorted_distances[1:n+1]    # Se elimina el primero porque es uno mismo
