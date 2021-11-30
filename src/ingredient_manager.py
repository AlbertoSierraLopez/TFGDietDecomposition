import json
import os
import requests
import nltk
import atexit
import pandas as pd

from constants import PATH_CACHE, REQUIREMENT_LIST, API_KEY
from fuzzywuzzy import fuzz
from sortedcollections import ValueSortedDict
from scipy.spatial import distance


class IngredientManager:
    def __init__(self, requirements, ing_vocab, nlp_model, model_type, kg_ing, kg_tag):
        self.recipe = None
        self.tokenized_recipe = None
        self.ingredients = None
        self.unwanted = None
        self.replacements = None

        self.requirements = REQUIREMENT_LIST[requirements]
        self.nlp_model = nlp_model
        self.model_type = model_type
        self.kg_ing = kg_ing
        self.kg_tag = kg_tag
        self.ing_vocab = ing_vocab

        # proxy cache
        if os.path.exists(PATH_CACHE):
            with open(PATH_CACHE) as json_file:
                self.usda_cache = json.load(json_file)
        else:
            self.usda_cache = dict()

        atexit.register(self.exit_handler)

    def load_recipe(self, recipe, tokenized_recipe):
        self.recipe = recipe
        self.tokenized_recipe = tokenized_recipe

        self.ingredients = self.detect_ingredients()
        self.unwanted = self.unwanted_ingredients()

        self.replacements = self.get_replacements()

# MODULO 2
    def detect_ingredients(self):
        return sorted(set([token for token in self.tokenized_recipe if self.is_ingredient_v1(token)]))

#   Detectores de ingredientes:
    # 1. Pertenece al vocabulario
    def is_ingredient_v1(self, token):
        return token in self.ing_vocab

    # 2. Usando la distancia de Levenshtein con el vocabulario
    def is_ingredient_v2(self, token):
        for ingredient in self.ing_vocab:
            lev_patial_ratio = fuzz.partial_ratio(token, ingredient)
            lev_ratio = fuzz.ratio(token, ingredient)
            # Comprobar que el termino del vocabulario contiene el token, y si es compuesto, no es muy diferente
            if lev_ratio > 80 and lev_patial_ratio == 100:
                return True
        return False

    def unwanted_ingredients(self):
        unwanted_ingredients = []

        for ingredient in self.ingredients:
            if not self.passes_requirements(ingredient):
                unwanted_ingredients.append(ingredient)

        return unwanted_ingredients

    # Si por cualquier razón, una propiedad del ingrediente no se puede evaluar,
    # se da por buena y el ingrediente pasa como válido
    def passes_requirements(self, food):
        nutrients = self.get_nutrients(food)
        category = self.get_food_category(food)
        ingredients = self.get_ingredients(food)

        if 'Dieta vegetariana' in self.requirements:
            if food not in self.kg_tag['vegan']:
                return False

        if 'Dieta vegana' in self.requirements:
            if food not in self.kg_tag['vegetarian']:
                return False

        if 'Dieta hipocalórica' in self.requirements and nutrients is not None:
            # La energía puede aparecer en julios, sólo la queremos en KCals:
            kcals = nutrients.loc[(nutrients['nutrientName'] == 'Energy') & (nutrients['unitName'] == 'KCAL')]
            if len(kcals) > 0 and kcals['value'].item() > 300.0:
                return False

        if 'Dieta proteica' in self.requirements and nutrients is not None:
            protein = nutrients.loc[nutrients['nutrientName'] == 'Protein']
            kcals = nutrients.loc[(nutrients['nutrientName'] == 'Energy') & (nutrients['unitName'] == 'KCAL')]
            if len(protein) > 0 and len(kcals) > 0:
                v_protein = protein['value'].item()
                v_kcals = kcals['value'].item()
                # Un alimento es alto en proteínas si éstas representan más del 16% de su aporte calórico
                if (4 * v_protein) < (0.16 * v_kcals):
                    return False

        if 'Dieta baja en carbohidratos' in self.requirements and nutrients is not None:
            carbs = nutrients.loc[nutrients['nutrientName'] == 'Carbohydrate, by difference']
            kcals = nutrients.loc[(nutrients['nutrientName'] == 'Energy') & (nutrients['unitName'] == 'KCAL')]
            if len(carbs) > 0 and len(kcals) > 0:
                v_carbs = carbs['value'].item()
                v_kcals = kcals['value'].item()
                # Un alimento es alto en carbs si éstos representan más del 65% de su aporte calórico
                if (4 * v_carbs) < (0.65 * v_kcals):
                    return False

        if 'Dieta baja en sodio' in self.requirements and nutrients is not None:
            sodium = nutrients.loc[nutrients['nutrientName'] == 'Sodium, Na']
            if len(sodium) > 0 and sodium['value'].item() > 140.0:
                return False

        if 'Intolerancia a la lactosa' in self.requirements and ingredients is not None:
            if 'milk' in ingredients or 'cream' in ingredients or 'butter' in ingredients:
                return False

        if 'Alergia los frutos secos' in self.requirements and category is not None:
            if 'nuts' in category:
                return False

        if 'Alergia al marisco' in self.requirements and category is not None:
            if 'seafood' in category:
                return False

        if 'Intolerancia al gluten' in self.requirements and ingredients is not None:
            if 'wheat' in ingredients:
                return False

        if 'Celiaquía' in self.requirements and ingredients is not None:
            if 'wheat' in ingredients or 'barley' in ingredients or 'rye' in ingredients:
                return False

        return True

    def query(self, food):
        if food in self.usda_cache:
            response = self.usda_cache[food]
        else:
            payload = {'query': food, 'dataType': 'Survey (FNDDS)', 'requireAllWords': True}
            response = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search?api_key=+' + API_KEY, params=payload)
            self.usda_cache[food] = response.json()
        return response.json()

    # Devuelve la primera fila de la query como dataframe
    def get_info(self, food):
        response = self.query(food)

        food_df = pd.json_normalize(response['foods'])
        if len(food_df) > 0:
            # Cogemos la primera fila de la query (asumimos que la más parecida a la búsqueda):
            row = food_df.iloc[0]
            # Eliminar columnas sin valor (ya se comprueba si existe columna más adelante):
            clean_row = row.replace(["", " "], float("NaN"))
            return clean_row.dropna()
        else:
            return None

    # Devuelve la columna 'Food Nutrients' de la primera fila de la query, en forma de dataframe
    def get_nutrients(self, food):
        info = self.get_info(food)

        if info is None or 'foodNutrients' not in info:
            return None

        nutrients = info['foodNutrients']
        return pd.json_normalize(nutrients)

    # Devuelve la columna 'Food Category' de la primera fila de la query, tokenizada
    def get_food_category(self, food):
        info = self.get_info(food)

        if info is None or 'foodCategory' not in info:
            return None

        food_cat = info['foodCategory']
        return [token.lower() for token in nltk.word_tokenize(food_cat)]        # Devolver en minuscula

    # Devuelve la columna 'Ingredients' de la primera fila de la query, tokenizada
    def get_ingredients(self, food):
        info = self.get_info(food)

        if info is None or 'ingredients' not in info:
            return None

        ingredients = info['ingredients']
        return [token.lower() for token in nltk.word_tokenize(ingredients)]     # Devolver en minuscula

    def get_total_nutrients(self):
        total_nutrients = None

        for ingredient in self.ingredients:
            nutrients = self.get_nutrients(ingredient)

            if total_nutrients is None:
                total_nutrients = nutrients
            else:
                join = pd.concat([total_nutrients, nutrients], join='inner')
                total_nutrients = join.groupby(["nutrientId", "nutrientName", "nutrientNumber", "unitName"],
                                               as_index=False).sum()
        return total_nutrients

# MODULO 3
    def replace_unwanted(self):
        if self.recipe is None:
            raise Exception("No recipe is loaded.")

        new_recipe = []
        for token in nltk.word_tokenize(self.recipe):
            if token in self.unwanted:
                token = self.replacements[token]
            new_recipe.append(token)

        return nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(new_recipe)

    def get_replacements(self):
        replacements = dict()

        for ingredient in self.unwanted:
            replacements[ingredient] = self.find_replacement(ingredient)

        return replacements

    def find_replacement(self, ingredient):
        # Comprueba que exista la palabra en el modelo NLP:
        if ingredient in self.nlp_model:
            # Comprueba los 25 ingredientes más cercanos:
            for (alternative, similarity) in self.most_similar(ingredient, topn=25):
                # Comprobar que es un ingrediente y que es válido
                if (alternative in self.ing_vocab) and (self.passes_requirements(alternative)):
                    return alternative
        return '<None>'     # Si no encuentra nada, lo mejor es eliminar el ingrediente de la receta y no sustituirlo

    def most_similar(self, target, topn=25):
        if self.model_type in ['elmo', 'bert']:
            if target not in self.nlp_model:
                print("Error: word", target, "out of vocabulary.")
                return []

            embedding = self.nlp_model[target]

            sorted_dict = ValueSortedDict()

            for (key, value) in self.nlp_model.items():
                cos_distance = distance.cosine(embedding, value)
                sorted_dict.__setitem__(key, cos_distance)

            return sorted_dict.items()[1:topn+1]    # Se quita el primero porque es el target (cos_distance = 0.0)

        elif self.model_type in ['word2vec']:
            return self.nlp_model.most_similar(target, topn=25)

        elif self.model_type in ['glove']:
            distances = [(distance.cosine(self.nlp_model.word_vectors[self.nlp_model.dictionary[target]],
                                          self.nlp_model.word_vectors[self.nlp_model.dictionary[X]]), X)
                         for X in self.nlp_model.dictionary.keys()]

            sorted_distances = sorted(distances)

            return sorted_distances[1:topn+1]    # Se quita el primero porque es el target (cos_distance = 0.0)

    def exit_handler(self):
        print("\tSalvando caché...")
        with open(PATH_CACHE, 'w+') as json_file:
            json.dump(self.usda_cache, json_file)
