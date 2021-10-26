import os
import pickle
import numpy as np

from time import time

from data_loader import DataLoader
from knowledge_manager import KnowledgeManager
from language_processer import LanguageProcesser
from ingredient_manager import IngredientManager
from statistics import Statistics
from tokenizer import Tokenizer

#                        1  2  3  4  5  6  7  8  9 10 11
requirements = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)

dataset = "RAW_recipes.csv"
start_time = time()
test_size = 5


# MODULO 1
print(">Leyendo dataset: ", dataset, "...", sep='')
data_loader = DataLoader(path_csv="../datasets/"+dataset, test_size=test_size)

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

# No se utiliza ahora mismo
print(">Extrayendo vocabulario...")
vocab, word2ind, ind2word = tokenizer.get_vocab(tokenized_recipes)

print(">Elaborando grafo de conocimiento...")
knowledge_manager = KnowledgeManager(ingredients, tags)
KG_ing = knowledge_manager.KG_ing
KG_tag = knowledge_manager.KG_tag
# print("\tIngredientes con los que se más veces se relaciona la 'manzana':", knowledge_manager.top_edges('apple'))
# print("\tAlgunos ingredientes relacionados con 'low-protein'", list(KG_tag['low-protein'])[:10])

print(">Entrenando modelo...")
nlp = LanguageProcesser(tokenized_recipes, word2vec=True)
# print("\tPalabras más próximas a 'huevos':", nlp.closest_words_word2vec(word='eggs'))
# print("\tPalabras más próximas a 'sal':", nlp.closest_words_word2vec(word='salt'))
# print("\tPalabras más próximas a 'azúcar':", nlp.closest_words_word2vec(word='sugar'))
# print("\tPalabras más próximas a 'jamón':", nlp.closest_words_word2vec(word='ham'))


test_recipes_generator = data_loader.test_recipes_generator()
ingredient_manager = IngredientManager(requirements, data_loader.get_list('ingredients'), wvmodel=nlp.word2vec_model,
                                       kg_ing=KG_ing, kg_tag=KG_tag)

key_mode = input("\n¿Desea procesar recetas una a una? (Y/N)\n").upper()

if not (key_mode == 'Y') and not (key_mode == 'S'):
    # Sacar estadísticas:
    print(">Calculando estadísticas...")
    statistics = Statistics(data_loader.test, test_recipes_generator, ingredient_manager, debug=True)
    print("\tTamaño del conjunto de test:", test_size, sep="\t")
    print("\tPrecisión:", statistics.compute_statistics(), sep="\t")

else:
    # Procesar recetas test una a una:
    counter = 1
    key_continue = 'Y'

    while (key_continue == 'Y') or (key_continue == 'S'):
        # MODULO 2
        print(">Leyendo entrada...")
        input_recipe, clean_recipe = next(test_recipes_generator)
        print("\tReceta:", clean_recipe, sep="\n\t")

        print(">Detectando ingredientes en la receta...")
        ingredient_manager.load_recipe(clean_recipe, tokenizer.nltk_tokenize(input_recipe), debug=True)
        print("\tIngredientes detectados:", ingredient_manager.ingredients)
        print("\tIngredientes incompatibles:", ingredient_manager.unwanted)
        print("\tInformación nutricional de la receta:", ingredient_manager.get_total_nutrients(), sep='\n')

        # MODULO 3
        print(">Buscando sustituciones...")
        new_recipe = ingredient_manager.replace_unwanted()
        print("\tReemplazos:", ingredient_manager.replacements, sep="\n\t")
        print("\tReceta válida:", new_recipe, end="\n", sep="\n\t")

        print(">Guardando receta...")
        with open('../output/in_recipes.txt', 'w+') as f:
            f.write(str(counter) + '.')
            f.write(clean_recipe)
        with open('../output/out_recipes.txt', 'w+') as f:
            f.write(str(counter) + '.')
            f.write(new_recipe)
        print("\tListo.")

        print(">Tiempo transcurrido:", round(time() - start_time, 4), "segundos")

        key_continue = input("\n¿Desea procesar otra receta? (Y/N)\n").upper()
        start_time = time()
        counter += 1

    print(">El archivo de recetas se encuentra en:")
    print("\t", os.path.abspath('../output/'))

print("\n>Proceso terminado")
exit()
