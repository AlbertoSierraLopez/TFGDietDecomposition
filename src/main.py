import numpy as np
from time import time
import os

from constants import DEBUG, GOLDEN, PATH_DATASET, PATH_OUTPUT, TEST_SIZE
import constants
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

## CONFIGURACION INICIAL
# Requisitos
input_requirements = []
while len(input_requirements) != 11:
    input_requirements = list(input("Introduce un array con las restricciones alimenticias:"
                                    "1. Dieta Vegetariana"
                                    "2. Dieta Vegana"
                                    "3. Dieta Hipocalórica"
                                    "4. Dieta Proteica"
                                    "5. Dieta Baja en Carbohidratos"
                                    "6. Dieta Baja en Sodio"
                                    "7. Intolerancia a la Lactosa"
                                    "8. Alergia a los Frutos Secos"
                                    "9. Alergia al Marisco"
                                    "10. Intolerancia al Gluten"
                                    "11. Celiaquía"
                                    "[___________]"))
requirements = np.array(input_requirements, dtype=bool)

# NLP
input_nlp = 0
input_sg = 0
input_pretrained = 0
while input_nlp not in [1, 2, 3]:
    input_nlp = int(input("Selecciona el modelo de NLP:\n"
                          "[1]: Word2Vec\n"
                          "[2]: Elmo\n"
                          "[3]: Bert\n"))
if input_nlp == 1:
    while input_sg not in [1, 2]:
        input_sg = int(input("Selecciona el modelo de embedding de Word2Vec:\n"
                             "[1]: CBOW\n"
                             "[2]: Skip-gram\n"))
    while input_pretrained not in [1, 2]:
        input_pretrained = int(input("Selecciona el modo de entrenamiento del modelo Word2Vec:\n"
                                     "[1]: Modelo Pre-Entrenado\n"
                                     "[2]: Entrenar Modelo Ahora\n"))

# Debug
constants.DEBUG = input("¿Desea procesar el conjunto de test al completo? [S/N]").upper() == 'S'

# Golden Standard
constants.GOLDEN = input("¿Desea procesar el golden standard? [S/N]").upper() == 'S'

# Detección de Ingredientes
input_detection = 0
while input_detection not in [1, 2, 3, 4, 5]:
    input_detection = int(input("Selecciona el método de detección de ingredientes:\n"
                                "[1]: Diccionario de Ingredientes\n"
                                "[2]: Distancia de Levenshtein (no-recomendado)\n"
                                "[3]: Red Neuronal\n"
                                "[4]: Ontología\n"
                                "[5]: Noun Chunks\n"))

## COMIENZO
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
if input_nlp == 1:
    nlp = LanguageProcesser(tokenized_recipes, ing_vocab, elmo=False, bert=False, word2vec=True,
                            sg=int(input_sg/2), pretrained=bool(int(input_pretrained/2)))
elif input_nlp == 2:
    nlp = LanguageProcesser(tokenized_recipes, ing_vocab, elmo=True, bert=False, word2vec=False)
elif input_nlp == 3:
    nlp = LanguageProcesser(tokenized_recipes, ing_vocab, elmo=False, bert=True, word2vec=False)
else:
    nlp = LanguageProcesser(tokenized_recipes, ing_vocab)

print(">Tiempo transcurrido:", round(time() - start_time, 4), "segundos\n")
start_time = time()


print(">Inicializando Detector de Ingredientes...")
test_recipes_generator = data_loader.test_recipes_generator()
test_recipes_tuples = data_loader.get_test_recipes()

nlp_model, nlp_name = nlp.get_model(input_nlp)
ingredient_manager = IngredientManager(requirements, ing_vocab, vocab, nlp_model=nlp_model, model_type=nlp_name,
                                       kg_ing=KG_ing, kg_tag=KG_tag, detection=input_detection)

# Sacar estadísticas:
statistics = Statistics(data_loader.test, test_recipes_tuples, ingredient_manager, chunks=input_detection==5)
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
