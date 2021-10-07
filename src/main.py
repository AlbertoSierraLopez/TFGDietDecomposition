from time import time

from data_loader import DataLoader
from tokenizer import Tokenizer
from collections import Counter


requirement = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dataset = "RAW_recipes.csv"
start_time = time()

print(">Leyendo dataset: ", dataset, "...", sep='')
data_loader = DataLoader(path_csv="../datasets/"+dataset, cap=50)
# data_loader.display_dataframe()

print(">Extrayendo columnas...")
tags = data_loader.get_column('tags')
steps = data_loader.get_column('steps')
ingredients = data_loader.get_column('ingredients')

print(">Tokenizando recetas...")
tokenizer = Tokenizer()
tokenized_recipes = [tokenizer.nltk_tokenize(recipe) for recipe in steps]

print(">Extrayendo vocabulario...")
vocab, word2ind, ind2word = tokenizer.get_vocab(tokenized_recipes)



print("Tiempo transcurrido:", round(time() - start_time, 4), "segundos")
