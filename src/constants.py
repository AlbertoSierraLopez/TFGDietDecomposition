import numpy as np

PATH_DATASET = "../datasets/RAW_recipes.csv"
PATH_ING_VOCAB = "../datasets/ingredients.json"
PATH_TOKENS = "../models/tokenized_recipes.pickle"
PATH_OUTPUT = "../output/"
PATH_TRAIN = "../datasets/train_recipes.csv"
PATH_TEST = "../datasets/test_recipes.csv"
PATH_CACHE = "../models/usda_cache.json"
PATH_IMP_ING = "../models/improved_ingredients.pickle"
PATH_KG_ING = "../models/kg_ingredients.pickle"
PATH_KG_TAG = "../models/kg_tags.pickle"
PATH_ELMO_MODEL = "../models/elmo.pickle"
PATH_WORD2VEC_MODEL = "../models/word2vec.model"
PATH_GLOVE_MODEL = "../models/glove.model"

TORCH_TOKENIZER = "basic_english"
SPACY_TOKENIZER = "C:/Users/Aussar/AppData/Local/Programs/Python/Python38/Lib/site-packages/en_core_web_sm/en_core_web_sm-3.1.0"
ELMO_MODULE = "https://tfhub.dev/google/elmo/2"
API_KEY = "dGv22hi1mexUfPPHzeKpENdiVUag9gnFMaEbbKio"

TEST_SIZE = 55
DEBUG = True
REQUIREMENT_LIST = np.array(['Dieta vegetariana', 'Dieta vegana', 'Dieta hipocalórica', 'Dieta proteica',
                             'Dieta baja en carbohidratos', 'Dieta baja en sodio', 'Intolerancia a la lactosa',
                             'Alergia los frutos secos', 'Alergia al marisco', 'Intolerancia al gluten', 'Celiaquía'])
