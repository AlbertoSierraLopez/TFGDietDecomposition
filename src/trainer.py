import torch

from constants import PATH_ELMO_MODEL, PATH_BERT_MODEL, PATH_WORD2VEC_MODEL, PATH_GLOVE_MODEL
import numpy as np
# from allennlp.modules.elmo import batch_to_ids
# from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
# import BertEmbeddings as Bert
from gensim.models import Word2Vec
# from glove import Corpus, Glove


class Trainer:
    def __init__(self, ing_vocab):
        self.ing_vocab = ing_vocab

    def elmo_model(self):
        print("\tNo se puede entrenar ELMo en PyCharm...")
        return None
    #     print("\tEntrenando modelo ELMo...")
    #
    #     options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/" \
    #                    "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    #     weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/" \
    #                   "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    #     elmo = ElmoTokenEmbedder(options_file, weight_file)
    #
    #     embeddings = elmo(batch_to_ids(self.model_vocab))
    #
    #     elmo_model = dict()
    #
    #     for i in range(len(self.model_vocab)):
    #         elmo_model[self.model_vocab[i]] = torch.mean(embeddings[i].detach(), 0)
    #
    #     torch.save(elmo_model, PATH_ELMO_MODEL)
    #     return elmo_model

    def bert_model(self):
        print("\tNo se puede entrenar Bert en PyCharm...")
        return None
    #     print("\tEntrenando modelo Bert...")
    #
    #     bert_embeddings = Bert.BertEmbeddings()
    #
    #     embeddings = bert_embeddings(self.model_vocab)
    #
    #     bert_model = dict()
    #
    #     for i in range(len(self.model_vocab)):
    #         bert_model[self.model_vocab[i]] = list(embeddings[i]['embeddings_map'].values())[0]
    #
    #     torch.save(bert_model, PATH_BERT_MODEL)
    #     return bert_model

    @staticmethod
    def word2vec_model(recipes, sg):
        print("\tEntrenando modelo Word2Vec...")

        model = Word2Vec(min_count=20,
                         window=8,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=4,
                         sg=sg)
        model.build_vocab(recipes, progress_per=10000)
        model.train(recipes, total_examples=model.corpus_count, epochs=10, report_delay=1)

        model.save(PATH_WORD2VEC_MODEL)

        return model

    # @staticmethod
    # def glove_model(recipes):
    #     corpus = Corpus()
    #     corpus.fit(recipes, window=8)
    #     glove = Glove(no_components=1024, learning_rate=0.01)
    #
    #     glove.fit(corpus.matrix, epochs=64, no_threads=4, verbose=True)
    #     glove.add_dictionary(corpus.dictionary)
    #
    #     glove.save(PATH_GLOVE_MODEL)
    #
    #     return glove
