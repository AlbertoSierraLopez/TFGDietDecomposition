import os
import pickle

from time import time

from data_loader import DataLoader
from knowledge_manager import KnowledgeManager
from language_processer import LanguageProcesser
from tokenizer import Tokenizer

#               1  2  3  4  5  6  7  8  9 10 11
requirements = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

dataset = "RAW_recipes.csv"
start_time = time()

print(">Leyendo dataset: ", dataset, "...", sep='')
data_loader = DataLoader(path_csv="../datasets/"+dataset)
# data_loader.display_dataframe()

print(">Extrayendo columnas...")
tags = data_loader.get_column('tags')
recipes = data_loader.get_column('steps')
ingredients = data_loader.get_column('ingredients')

print(">Tokenizando recetas...")
tokenizer = Tokenizer()
if os.path.exists("../models/tokenized_recipes.pickle"):
    print("\tCargando tokens guardados...")
    pickle_in = open("../models/tokenized_recipes.pickle", "rb")
    tokenized_recipes = pickle.load(pickle_in)
    pickle_in.close()
else:
    print("\tCreando tokens nuevos...")
    tokenized_recipes = [tokenizer.nltk_tokenize(recipe) for recipe in recipes]
    pickle_out = open("../models/tokenized_recipes.pickle", "wb")
    pickle.dump(tokenized_recipes, pickle_out)
    pickle_out.close()

print(">Extrayendo vocabulario...")
vocab, word2ind, ind2word = tokenizer.get_vocab(tokenized_recipes)

print(">Elaborando grafo de conocimiento...")
knowledge_manager = KnowledgeManager(ingredients, tags)
KG_ing = knowledge_manager.KG_ing
KG_tag = knowledge_manager.KG_tag
print("\tIngredientes con los que se más veces se relaciona la 'manzana':", knowledge_manager.top_edges('apple'))
print("\tAlgunos ingredientes relacionados con 'vegan'", list(KG_tag['low-protein'])[:10])

print(">Entrenando modelo...")
nlp = LanguageProcesser(tokenized_recipes, word2vec=True)
print("\tPalabras más próximas a 'huevos':", nlp.closest_words_word2vec(word='eggs'))
print("\tPalabras más próximas a 'sal':", nlp.closest_words_word2vec(word='salt'))
print("\tPalabras más próximas a 'azúcar':", nlp.closest_words_word2vec(word='sugar'))
print("\tPalabras más próximas a 'jamón':", nlp.closest_words_word2vec(word='ham'))

print("\nTiempo transcurrido:", round(time() - start_time, 4), "segundos")
