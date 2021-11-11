import os
import torch

import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from sortedcollections import ValueSortedDict
from scipy.spatial import distance

from constants import PATH_ELMO_MODEL, PATH_BERT_MODEL, PATH_WORD2VEC_MODEL,\
    PATH_GLOVE_MODEL, PATH_WORD2VEC_PRETRAINED_MODEL
# from glove import Glove
from trainer import Trainer


class LanguageProcesser:
    def __init__(self, recipes, ing_vocab, elmo=False, bert=False, word2vec=False, glove=False, sg=0, pretrained=False):
        self.trainer = Trainer(ing_vocab)

        self.elmo_model = None
        self.bert_model = None
        self.word2vec_model = None
        self.glove_model = None

        if elmo:
            if os.path.exists(PATH_ELMO_MODEL):
                self.elmo_model = torch.load(PATH_ELMO_MODEL)
            else:
                self.elmo_model = self.trainer.elmo_model()
        
        if bert:
            if os.path.exists(PATH_BERT_MODEL):
                self.bert_model = torch.load(PATH_BERT_MODEL)
            else:
                self.bert_model = self.trainer.bert_model()

        if word2vec:
            if pretrained:
                if os.path.exists(PATH_WORD2VEC_PRETRAINED_MODEL):
                    self.word2vec_model = KeyedVectors.load(PATH_WORD2VEC_PRETRAINED_MODEL)
                else:
                    self.word2vec_model = api.load("glove-twitter-100")
                    self.word2vec_model.save(PATH_WORD2VEC_PRETRAINED_MODEL)
            else:
                if os.path.exists(PATH_WORD2VEC_MODEL):
                    self.word2vec_model = Word2Vec.load(PATH_WORD2VEC_MODEL).wv
                else:
                    self.word2vec_model = self.trainer.word2vec_model(recipes, sg).wv

        # if glove:
        #     if os.path.exists(PATH_GLOVE_MODEL):
        #         self.glove_model = Glove.load(PATH_GLOVE_MODEL)
        #     else:
        #         self.glove_model = self.trainer.glove_model(recipes)

    # Funciones para testing
    def closest_words_word2vec(self, word, n=10):
        return self.word2vec_model.most_similar(word, topn=n)

    def closest_words_elmo(self, word, n=10):
        if word not in self.elmo_model:
            print("Error: word", word, "out of vocabulary.")
            return []

        embedding = self.elmo_model[word]

        sorted_dict = ValueSortedDict()

        for (key, value) in self.elmo_model.items():
            cos_distance = distance.cosine(embedding, value)
            sorted_dict.__setitem__(key, cos_distance)

        return sorted_dict.items()[1:n + 1]  # Se quita el primero porque es el target (cos_distance = 0.0)

    def closest_words_bert(self, word, n=10):
        if word not in self.bert_model:
            print("Error: word", word, "out of vocabulary.")
            return []

        embedding = self.bert_model[word]

        sorted_dict = ValueSortedDict()

        for (key, value) in self.bert_model.items():
            cos_distance = distance.cosine(embedding, value)
            sorted_dict.__setitem__(key, cos_distance)

        return sorted_dict.items()[1:n + 1]  # Se quita el primero porque es el target (cos_distance = 0.0)
