import numpy as np
from time import time
import os

from constants import DEBUG, GOLDEN, PATH_DATASET, PATH_OUTPUT, TEST_SIZE
from data_loader import DataLoader
from knowledge_manager import KnowledgeManager
from language_processer import LanguageProcesser
from ingredient_manager import IngredientManager
from statistics import Statistics
from tokenizer import Tokenizer

# Preparar carpetas del sistema
if not os.path.exists('../datasets'):
    print(">Directorio 'datasets' creado")
    os.mkdir('../datasets')
if not os.path.exists('../models'):
    print(">Directorio 'models' creado")
    os.mkdir('../models')
if not os.path.exists('../output'):
    print(">Directorio 'output' creado")
    os.mkdir('../output')

#                        1  2  3  4  5  6  7  8  9 10 11
requirements = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
start_time = time()

# MODULO 1
print(">Leyendo dataset: '", PATH_DATASET, "'...", sep='')
data_loader = DataLoader(PATH_DATASET)

print(">Extrayendo columnas...")
tags = data_loader.get_column('tags')
recipes = data_loader.get_column('steps')
ingredients = data_loader.get_column('ingredients')

print(">Tokenizando recetas...")
tokenizer = Tokenizer()
tokenized_recipes = tokenizer.get_tokenized_recipes(recipes)

# No se utiliza ahora mismo:
print(">Extrayendo vocabulario...")
vocab, word2ind, ind2word = tokenizer.get_vocab(tokenized_recipes, extended_vocab=False)
ing_vocab = tokenizer.get_ing_vocab(ingredients)

print(">Elaborando grafos de conocimiento...")
knowledge_manager = KnowledgeManager(ingredients, tags.tolist())
KG_ing = knowledge_manager.KG_ing
KG_tag = knowledge_manager.KG_tag

print(">Entrenando modelo...")
nlp = LanguageProcesser(tokenized_recipes, ing_vocab, elmo=True, bert=True, word2vec=True, sg=1, pretrained=True)
'''
if DEBUG:
    print("\tWord2Vec")
    print("\t\tPalabras más próximas a 'milk':", nlp.closest_words_word2vec(word='milk', n=5))
    print("\t\tPalabras más próximas a 'egg':", nlp.closest_words_word2vec(word='egg', n=5))
    print("\t\tPalabras más próximas a 'salt':", nlp.closest_words_word2vec(word='salt', n=5))
    print("\t\tPalabras más próximas a 'sugar':", nlp.closest_words_word2vec(word='sugar', n=5))
    print("\t\tPalabras más próximas a 'ham':", nlp.closest_words_word2vec(word='ham', n=5))
    print("\tELMo")
    print("\t\tPalabras más próximas a 'milk':", nlp.closest_words_elmo(word='milk', n=5))
    print("\t\tPalabras más próximas a 'egg':", nlp.closest_words_elmo(word='egg', n=5))
    print("\t\tPalabras más próximas a 'salt':", nlp.closest_words_elmo(word='salt', n=5))
    print("\t\tPalabras más próximas a 'sugar':", nlp.closest_words_elmo(word='sugar', n=5))
    print("\t\tPalabras más próximas a 'ham':", nlp.closest_words_elmo(word='ham', n=5))
    print("\tBert")
    print("\t\tPalabras más próximas a 'milk':", nlp.closest_words_bert(word='milk', n=5))
    print("\t\tPalabras más próximas a 'egg':", nlp.closest_words_bert(word='egg', n=5))
    print("\t\tPalabras más próximas a 'salt':", nlp.closest_words_bert(word='salt', n=5))
    print("\t\tPalabras más próximas a 'sugar':", nlp.closest_words_bert(word='sugar', n=5))
    print("\t\tPalabras más próximas a 'ham':", nlp.closest_words_bert(word='ham', n=5))
'''


print(">Tiempo transcurrido:", round(time() - start_time, 4), "segundos\n")
start_time = time()


print(">Inicializando Detector de Ingredientes...")
chunks = False
test_recipes_generator = data_loader.test_recipes_generator()
test_recipes_tuples = data_loader.get_test_recipes()
ingredient_manager = IngredientManager(requirements, ing_vocab, vocab, nlp_model=nlp.bert_model, model_type='bert',
                                       kg_ing=KG_ing, kg_tag=KG_tag, chunks=chunks)

# Sacar estadísticas:
statistics = Statistics(data_loader.test, test_recipes_tuples, ingredient_manager, chunks=chunks)
if DEBUG:
    print(">Calculando estadísticas...")
    print("\tTamaño del conjunto de test:", TEST_SIZE, sep="\t")
    precision, recall, f1 = statistics.compute_statistics()
    print("\tPrecisión:", precision)
    print("\tExhaustividad:", recall)
    print("\tF1:", f1)
    print(">Tiempo transcurrido:", round(time() - start_time, 4), "segundos\n")
    start_time = time()
if GOLDEN:
    print(">Procesando estándar...")
    statistics.process_golden_standard()
    print(">Tiempo transcurrido:", round(time() - start_time, 4), "segundos\n")
    start_time = time()


# Procesar recetas test una a una:
counter = 1
key_continue = input("\n¿Desea procesar recetas una a una? (Y/N)\n").upper()
file_in = open(PATH_OUTPUT + "in_recipes.txt", 'w+')
file_out = open(PATH_OUTPUT + "out_recipes.txt", 'w+')

while (key_continue == 'Y') or (key_continue == 'S'):
    # MODULO 2
    print(">Leyendo entrada...")
    input_recipe, clean_recipe = next(test_recipes_generator)
    print("\tReceta:", clean_recipe, sep="\n\t")

    print(">Detectando ingredientes en la receta...")
    ingredient_manager.load_recipe(clean_recipe, tokenizer.spacy_tokenize_pro_str(input_recipe))
    print("\tIngredientes detectados:", ingredient_manager.ingredients)
    print("\tIngredientes incompatibles:", ingredient_manager.unwanted)
    if DEBUG:
        print("\tInformación nutricional de la receta:", ingredient_manager.get_total_nutrients(), sep='\n')

    # MODULO 3
    print(">Buscando sustituciones...")
    new_recipe = ingredient_manager.replace_unwanted()
    print("\tReemplazos:", ingredient_manager.replacements, sep="\n\t")
    print("\tReceta válida:", new_recipe, end="\n", sep="\n\t")

    print(">Guardando receta...")
    file_in.write(str(counter) + '. ' + clean_recipe + '\n')
    file_out.write(str(counter) + '. ' + new_recipe + '\n')
    print("\tListo.")

    print(">Tiempo transcurrido:", round(time() - start_time, 4), "segundos")

    key_continue = input("\n¿Desea procesar otra receta? (Y/N)\n").upper()
    start_time = time()
    counter += 1

file_in.close()
file_out.close()
print(">El archivo de recetas se encuentra en:")
print("\t", os.path.abspath(PATH_OUTPUT))

print("\n>Proceso terminado")
exit()
