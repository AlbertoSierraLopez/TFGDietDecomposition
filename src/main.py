from time import time

from data_loader import DataLoader
from tokenizer import Tokenizer
from knowledge_manager import KnowledgeManager


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

print(">Elaborando grafo de conocimiento...")
knowledge_manager = KnowledgeManager(ingredients, tags)
KG_ing = knowledge_manager.KG_ing
KG_tag = knowledge_manager.KG_tag
print("\tIngredientes con los que se relaciona la 'manzana':", KG_ing['apple'])
print("\tIngredientes relacionados con 'low-protein'", KG_tag['low-protein'])

print("\nTiempo transcurrido:", round(time() - start_time, 4), "segundos")
