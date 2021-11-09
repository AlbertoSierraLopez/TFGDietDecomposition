import os
import torch
import gensim.downloader as api

from constants import PATH_ELMO_MODEL, PATH_BERT_MODEL, PATH_WORD2VEC_MODEL, PATH_GLOVE_MODEL, PATH_WORD2VEC_PRETRAINED_MODEL
from gensim.models import Word2Vec, KeyedVectors
# from glove import Glove
from trainer import Trainer


class LanguageProcesser:
    def __init__(self, recipes, ing_vocab, elmo=False, bert=False, word2vec=False, glove=False, sg=0, pretrained=False):
        self.trainer = Trainer(ing_vocab)

        self.elmo_model = None
        self.bert_model = None
        self.word2vec_model = None
        self.glove_model = None
        '''
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
        '''
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

    def closest_words_word2vec(self, word, n=10):
        return self.word2vec_model.most_similar(word, topn=n)

    def get_embedding_word2vec(self, word):
        if self.word2vec_model is not None:
            return self.word2vec_model[word]
