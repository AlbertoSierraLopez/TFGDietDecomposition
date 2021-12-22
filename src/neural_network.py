import os
import numpy as np
import gensim.downloader as api

from constants import PATH_MLPC, PATH_WORD2VEC_PRETRAINED_MLPC_MODEL
from joblib import dump, load
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


class NeuralNetwork:
    def __init__(self, ing_vocab, vocab):
        # Modelo NLP
        if os.path.exists(PATH_WORD2VEC_PRETRAINED_MLPC_MODEL):
            self.nlp = KeyedVectors.load(PATH_WORD2VEC_PRETRAINED_MLPC_MODEL)
        else:
            self.nlp = api.load("glove-wiki-gigaword-100")
            self.nlp.save(PATH_WORD2VEC_PRETRAINED_MLPC_MODEL)

        # Ingredientes del vocabulario de una única palabra
        ingredient_words = [ingredient for ingredient in ing_vocab if len(ingredient.split()) < 2]
        word_freq = Counter(vocab)
        # Palabras que no son ingredientes
        regular_words = [key for (key, value) in word_freq.most_common(2000) if key not in ing_vocab]

        # No vamos a entrenar la red neuronal con palabras, sino con embeddings
        ingredient_embeddings = np.asarray([self.nlp[word] for word in ingredient_words if word in self.nlp])
        regular_embeddings = np.asarray([self.nlp[word] for word in regular_words if word in self.nlp])

        # Crear X e Y
        X = np.vstack((ingredient_embeddings, regular_embeddings))

        y_1 = np.ones(ingredient_embeddings.shape[0])
        y_0 = np.zeros(regular_embeddings.shape[0])
        y = np.concatenate((y_1, y_0))

        # Separar en train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        if os.path.exists(PATH_MLPC):
            print(">Red Neuronal cargada.")
            self.mlp = load(PATH_MLPC)
        else:
            print(">Creando Red Neuronal ...")
            # Multi Layer Perceptron Classifier:
            self.mlp = MLPClassifier(activation='relu', solver='adam', max_iter=500)
            self.mlp.fit(X_train, y_train)

            dump(self.mlp, PATH_MLPC)

            '''
            predict_test = self.mlp.predict(X_test)
            # Evaluar
            print("\tRed Neuronal:")
            print(classification_report(y_test, predict_test))
            '''

    def predict(self, token):
        if token in self.nlp:
            prediction = self.mlp.predict(self.nlp[token].reshape(1, 100))  # (, 100) -> (1, 100)
            return bool(prediction)     # Como la predicción siempre va a ser 0 o 1, podemos devolver directamente bool
        else:   # En caso de no existir en el modelo, directamente enviamos que no es ingrediente
            return False
