import pickle

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models import Word2Vec
from glove import Corpus, Glove


class Trainer:

    @staticmethod
    def elmo_model(recipes):
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        session = tf.compat.v1.Session()
        session.run(tf.compat.v1.global_variables_initializer())
        session.run(tf.compat.v1.tables_initializer())

        vectors = []
        for recipe in recipes:
            embeddings = elmo(inputs={"tokens": [recipe], "sequence_len": [len(recipe)]},
                              signature="tokens",
                              as_dict=True)["elmo"]

            vectors.append(session.run(embeddings))

        for vector in vectors:
            vector.shape = (np.shape(vector)[1], 1024)

        pickle_out = open("/models/elmo.pickle", "wb")
        pickle.dump(vectors, pickle_out)
        pickle_out.close()

        return np.concatenate(vectors, axis=0)

    @staticmethod
    def word2vec_model(recipes, sg=0):
        model = Word2Vec(min_count=20,
                         window=2,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=4,
                         sg=sg)
        model.build_vocab(recipes, progress_per=10000)
        model.train(recipes, total_examples=model.corpus_count, epochs=10, report_delay=1)

        model.save("/models/word2vec.model")

        return model

    @staticmethod
    def glove_model(recipes):
        corpus = Corpus()
        corpus.fit(recipes, window=8)
        glove = Glove(no_components=1024, learning_rate=0.01)

        glove.fit(corpus.matrix, epochs=64, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)

        glove.save("/models/glove.model")

        return glove
