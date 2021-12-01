import numpy as np

PATH_DATASET = "../datasets/RAW_recipes.csv"
PATH_TOKENS = "../models/tokenized_recipes.pickle"
PATH_OUTPUT = "../output/"
PATH_TRAIN = "../datasets/train_recipes.csv"
PATH_TEST = "../datasets/test_recipes.csv"
PATH_CACHE = "../models/usda_cache.json"
PATH_KG_ING = "../models/kg_ingredients.pickle"
PATH_KG_TAG = "../models/kg_tags.pickle"
PATH_ELMO_MODEL = "../models/elmo_model.pt"
PATH_BERT_MODEL = "../models/bert_model.pt"
PATH_WORD2VEC_MODEL = "../models/word2vec.model"
PATH_WORD2VEC_PRETRAINED_MODEL = "../models/word2vec_pretrained.kv"
PATH_GLOVE_MODEL = "../models/glove.model"
PATH_MLPC = "../models/mlpclassifier.joblib"
PATH_WORD2VEC_PRETRAINED_MLPC_MODEL = "../models/word2vec_pretrained_mlpc.kv"

TORCH_TOKENIZER = "basic_english"
SPACY_TOKENIZER = "C:/Users/Aussar/AppData/Local/Programs/Python/Python38/Lib/site-packages/en_core_web_sm/en_core_web_sm-3.1.0"
API_KEY = "dGv22hi1mexUfPPHzeKpENdiVUag9gnFMaEbbKio"

TEST_SIZE = 55
DEBUG = True
REQUIREMENT_LIST = np.array(['Dieta vegetariana', 'Dieta vegana', 'Dieta hipocalórica', 'Dieta proteica',
                             'Dieta baja en carbohidratos', 'Dieta baja en sodio', 'Intolerancia a la lactosa',
                             'Alergia los frutos secos', 'Alergia al marisco', 'Intolerancia al gluten', 'Celiaquía'])
