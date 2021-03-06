import json
import os
import requests
import nltk
import atexit
import spacy
import pandas as pd
import owlready2 as owl

from constants import PATH_CACHE, REQUIREMENT_LIST, API_KEY, SPACY_TOKENIZER
from neural_network import NeuralNetwork
from fuzzywuzzy import fuzz
from sortedcollections import ValueSortedDict
from scipy.spatial import distance


class IngredientManager:
    def __init__(self, requirements, ing_vocab, vocab, nlp_model, model_type, kg_ing, kg_tag, detection=3):
        self.recipe = None
        self.tokenized_recipe = None
        self.ingredients = None
        self.unwanted = None
        self.replacements = None

        self.requirements = REQUIREMENT_LIST[requirements]
        self.vocab = vocab
        self.nlp_model = nlp_model
        self.model_type = model_type
        self.kg_ing = kg_ing
        self.kg_tag = kg_tag
        self.ing_vocab = ing_vocab
        self.detection = detection

        self.spacy = spacy.load(SPACY_TOKENIZER)
        self.mlpc = NeuralNetwork(ing_vocab, vocab)
        self.onto = owl.get_ontology("http://purl.obolibrary.org/obo/foodon.owl").load()

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

        if self.detection in [1, 2, 3]:
            self.ingredients = self.detect_ingredients()
        elif self.detection in [4]:
            self.ingredients = self.detect_ingredients_onto()
        elif not self.detection == 5:
            self.ingredients = self.detect_ingredients_chunks()

        self.unwanted = self.unwanted_ingredients()
        self.replacements = self.get_replacements()

    def tokenizar_receta_spacy_pro(self, recipe):
        tokens = []
        doc = list(self.spacy(recipe))
        for token in doc:
            # Se filtran signos de puntuaci??n, n??meros y palabras vac??as
            if not token.is_punct and not token.is_digit and not token.is_stop:
                tokens.append(token)
        return tokens

    @staticmethod
    def word_similarity(token, string):
        # Si un ingrediente compuesto contiene un token, la similitud es 1 / el n??mero de palabras del ingrediente
        tok_string = [word for word in nltk.word_tokenize(string) if word.isalnum()]
        if len(tok_string) > 0 and token in string:
            return min(tok_string.count(token), 1) / len(tok_string)
        else:
            return 0

# MODULO 2
    # 1. Tokenizar receta y usar uno de los 3 primeros m??todos
    def detect_ingredients(self):
        detected_ingredients = set()
        for token in self.tokenized_recipe:
            if self.detection == 1:
                if self.is_ingredient_v1(token):
                    detected_ingredients.add(token)
            elif self.detection == 2:
                if self.is_ingredient_v2(token):
                    detected_ingredients.add(token)
            elif self.detection == 3:
                if self.is_ingredient_v3(token):
                    detected_ingredients.add(token)

        return detected_ingredients

    # 2. Tokenizar receta con spacy y usar ontolog??a sobre los sustantivos
    def detect_ingredients_onto(self):
        detected_ingredients = set()
        for token in self.tokenizar_receta_spacy_pro(self.recipe):
            if token.pos_ in ['NOUN'] and self.is_ingredient_v4(token.text):
                detected_ingredients.add(token.text)
        return detected_ingredients

    # 3. Sacar chunks de receta
    def detect_ingredients_chunks(self):
        detected_ingredients = []
        doc = self.spacy(self.recipe)

        for chunk in doc.noun_chunks:
            if self.is_ingredient_chunk(chunk.text):
                detected_ingredients.append(chunk.text)

        return sorted(set(detected_ingredients))

#   Detectores de ingredientes:
#   1. Pertenece al vocabulario
    def is_ingredient_v1(self, token):
        return token in self.ing_vocab

#   2. Usando la distancia de Levenshtein con el vocabulario
    def is_ingredient_v2(self, token):
        for ingredient in self.ing_vocab:
            lev_patial_ratio = fuzz.partial_ratio(token, ingredient)
            lev_ratio = fuzz.ratio(token, ingredient)
            # Comprobar que el termino del vocabulario contiene el token, y si es compuesto, no es muy diferente
            if lev_ratio > 80 and lev_patial_ratio == 100:
                return True
        return False

#   3. Usando una red neuronal
    def is_ingredient_v3(self, token):
        return self.mlpc.predict(token)

#   4. Usando una ontolog??a
    # Para usar este m??todo es recomendable filtrar antes los sustantivos
    def is_ingredient_v4(self, token):
        # Buscar el t??rmino en la ontolog??a
        entity = self.onto.search_one(label=token + "*")

        # Comprobar que lo que hemos encontrado existe, es una entidad y que se parece a lo que queremos
        if entity is None or entity.iri == '' or self.word_similarity(token, entity.label[0]) < 0.5:
            return False

        # Extraer sus ancestros
        ancestors = set()
        for item in entity.ancestors():
            ancestors.update(item.label)
        # Buscar entre sus ancestros, una entidad que lo clasifique como ingrediente
        return 'food product' in ancestors or 'dietary nutritional component' in ancestors

#   Detector de ingredientes en chunks:
    def is_ingredient_chunk(self, chunk):
        for ingredient in self.ing_vocab:
            lev_ratio = fuzz.ratio(chunk, ingredient)
            if lev_ratio > 80:
                # print(chunk, "-", ingredient, "->", lev_ratio)
                return True
        return False

# Evaluaci??n de ingredientes:
    def unwanted_ingredients(self):
        unwanted_ingredients = []

        for ingredient in self.ingredients:
            if not self.passes_requirements(ingredient):
                unwanted_ingredients.append(ingredient)

        return unwanted_ingredients

    # Si por cualquier raz??n, una propiedad del ingrediente no se puede evaluar,
    # se da por buena y el ingrediente pasa como v??lido
    def passes_requirements(self, food):
        nutrients, category, ingredients = self.get_USDA_data(food)

        if 'Dieta vegetariana' in self.requirements:
            if food not in self.kg_tag['vegan']:
                return False

        if 'Dieta vegana' in self.requirements:
            if food not in self.kg_tag['vegetarian']:
                return False

        if 'Dieta hipocal??rica' in self.requirements and nutrients is not None:
            # La energ??a puede aparecer en julios, s??lo la queremos en KCals:
            kcals = nutrients.loc[(nutrients['nutrientName'] == 'Energy') & (nutrients['unitName'] == 'KCAL')]
            if len(kcals) > 0 and kcals['value'].item() > 300.0:
                return False

        if 'Dieta proteica' in self.requirements and nutrients is not None:
            protein = nutrients.loc[nutrients['nutrientName'] == 'Protein']
            kcals = nutrients.loc[(nutrients['nutrientName'] == 'Energy') & (nutrients['unitName'] == 'KCAL')]
            if len(protein) > 0 and len(kcals) > 0:
                v_protein = protein['value'].item()
                v_kcals = kcals['value'].item()
                # Un alimento es alto en prote??nas si ??stas representan m??s del 16% de su aporte cal??rico
                if (4 * v_protein) < (0.16 * v_kcals):
                    return False

        if 'Dieta baja en carbohidratos' in self.requirements and nutrients is not None:
            carbs = nutrients.loc[nutrients['nutrientName'] == 'Carbohydrate, by difference']
            kcals = nutrients.loc[(nutrients['nutrientName'] == 'Energy') & (nutrients['unitName'] == 'KCAL')]
            if len(carbs) > 0 and len(kcals) > 0:
                v_carbs = carbs['value'].item()
                v_kcals = kcals['value'].item()
                # Un alimento es alto en carbs si ??stos representan m??s del 65% de su aporte cal??rico
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
            if 'nut' in category or 'nuts' in category or 'seed' in category or 'seeds' in category:
                return False

        if 'Alergia al marisco' in self.requirements and category is not None:
            if 'seafood' in category:
                return False

        if 'Intolerancia al gluten' in self.requirements and ingredients is not None:
            if 'wheat' in ingredients:
                return False

        if 'Celiaqu??a' in self.requirements and ingredients is not None:
            if 'wheat' in ingredients or 'barley' in ingredients or 'rye' in ingredients:
                return False

        return True

    def query(self, food):
        if food in self.usda_cache:
            return self.usda_cache[food]
        else:
            payload = {'query': food, 'requireAllWords': True}  # 'dataType': 'Survey (FNDDS)'
            response = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search?api_key=' + API_KEY, params=payload)
            if response.status_code != 200 or response.json()['totalHits'] == 0:
                return None
            else:
                self.usda_cache[food] = response.json()
                return response.json()

    # Devuelve la primera fila de la query como dataframe
    def get_info(self, food):
        response = self.query(food)
        if response is None:
            return None

        food_df = pd.json_normalize(response['foods'])
        if len(food_df) > 0:
            # Cogemos la primera fila de la query (asumimos que la m??s parecida a la b??squeda):
            row = food_df.iloc[0]
            # Eliminar columnas sin valor (ya se comprueba si existe columna m??s adelante):
            clean_row = row.replace(["", " "], float("NaN"))
            return clean_row.dropna()
        else:
            return None

    def get_USDA_data(self, food):
        nutrients, category, ingredients = None, None, None
        info = self.get_info(food)

        if info is not None:
            if 'foodNutrients' in info and len(info['foodNutrients']) > 0:
                nutrients = pd.json_normalize(info['foodNutrients'])

            if 'foodCategory' in info and len(info['foodCategory']) > 0:
                food_cat = [token.lower() for token in nltk.word_tokenize(info['foodCategory'])]

            if 'ingredients' in info and len(info['ingredients']) > 0:
                ingredients = [token.lower() for token in nltk.word_tokenize(info['ingredients'])]
        return nutrients, category, ingredients

    def get_total_nutrients(self):
        total_nutrients = None

        for ingredient in self.ingredients:
            nutrients, category, ingredients = self.get_USDA_data(ingredient)

            if total_nutrients is None:
                total_nutrients = nutrients
            else:
                join = pd.concat([total_nutrients, nutrients], join='inner')
                total_nutrients = join.groupby(["nutrientId", "nutrientName", "nutrientNumber", "unitName"],
                                               as_index=False).sum()
        return total_nutrients

# MODULO 3
    def get_replacements(self):
        replacements = dict()

        for ingredient in self.unwanted:
            replacements[ingredient] = self.find_replacement(ingredient)

        return replacements

    def find_replacement(self, ingredient):
        # Los ingredientes son tokens:
        if not self.detection == 5:
            # Comprueba que exista la palabra en el modelo NLP:
            if ingredient in self.nlp_model:
                # Comprueba los 25 ingredientes m??s cercanos:
                for (alternative, similarity) in self.most_similar(ingredient, topn=25):
                    # Comprobar que es un ingrediente y que es v??lido
                    if self.is_ingredient_v1(alternative) and self.passes_requirements(alternative):
                        return alternative

        # Los ingredientes son varias palabras (es necesario Word2Vec):
        elif self.model_type in ['word2vec']:
            ingredient_words = [word.replace(',', '') for word in ingredient.replace('-', ' ').split()]
            for word in ingredient_words:
                # Comprobar que el ingrediente est?? formado por palabras que existen en el modelo:
                if word not in self.nlp_model:
                    return '<None>'

                for (alternative, similarity) in self.nlp_model.most_similar(positive=ingredient_words, topn=25):
                    # alternative s?? que va a ser un token
                    if self.is_ingredient_v1(alternative) and self.passes_requirements(alternative):
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

    def replace_unwanted(self):
        if self.recipe is None:
            raise Exception("No recipe is loaded.")

        if not self.detection == 5:
            new_recipe = []
            for token in nltk.word_tokenize(self.recipe):
                if token in self.unwanted:
                    token = self.replacements[token]
                new_recipe.append(token)

            return nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(new_recipe)

        else:
            new_recipe = self.recipe
            for (key, value) in self.replacements.items():
                new_recipe.replace(key, value)
            return new_recipe.replace(' ,', ',')

    def exit_handler(self):
        print("\tSalvando cach??...")
        with open(PATH_CACHE, 'w+') as json_file:
            json.dump(self.usda_cache, json_file)
