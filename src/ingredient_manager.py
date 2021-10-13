import requests
import nltk

import numpy as np
import pandas as pd



class IngredientManager:
    def __init__(self, recipe, requirements, ing_vocab, wvmodel):
        self.recipe = recipe
        self.wvmodel = wvmodel
        self.requirement_list = np.array(['Dieta vegetariana', 'Dieta vegana', 'Dieta hipocalórica', 'Dieta proteica',
                                          'Dieta baja en carbohidratos', 'Dieta baja en sodio',
                                          'Intolerancia a la lactosa', 'Alergia los frutos secos', 'Alergia al marisco',
                                          'Intolerancia al gluten', 'Celiaquía'])
        self.requirements = self.requirement_list[requirements]
        # set para evitar repeticiones, sorted por comodidad
        self.ingredients = sorted(set([ingredient for ingredient in recipe if ingredient in ing_vocab]))

    def unwanted_ingredients(self):
        unwanted_ingredients = []

        for ingredient in self.ingredients:
            if not self.passes_requirements(ingredient):
                unwanted_ingredients.append(ingredient)

        return unwanted_ingredients

    def passes_requirements(self, food):
        nutrients = self.get_nutrients(food)
        category = self.get_food_category(food)
        ingredients = self.get_ingredients(food)

        if 'Dieta vegetariana' in self.requirements:
            return False

        if 'Dieta vegana' in self.requirements:
            return False

        if 'Dieta hipocalórica' in self.requirements:
            return False

        if 'Dieta proteica' in self.requirements:
            return False

        if 'Dieta baja en carbohidratos' in self.requirements and nutrients is not None:
            carbs = nutrients.loc[nutrients['nutrientName'] == 'Carbohydrate, by difference']
            if carbs['value'] > 150.0:
                return False

        if 'Dieta baja en sodio' in self.requirements and nutrients is not None:
            sodium = nutrients.loc[nutrients['nutrientName'] == 'Sodium, Na']
            if sodium['value'] > 100.0:
                return False

        if 'Intolerancia a la lactosa' in self.requirements:
            return False

        if 'Alergia los frutos secos' in self.requirements:
            return False

        if 'Alergia al marisco' in self.requirements:
            return False

        if 'Intolerancia al gluten' in self.requirements:
            return False

        if 'Celiaquía' in self.requirements:
            return False

        return True

    @staticmethod
    # Devuelve la primera fila de la query como dataframe
    def get_info(ingredient):
        response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search?query=" + ingredient +
                                "&pageSize=10&api_key=dGv22hi1mexUfPPHzeKpENdiVUag9gnFMaEbbKio")
        ingredient_df = pd.json_normalize(response.json()['foods'])
        if len(ingredient_df) > 0:
            return ingredient_df.iloc[0]

    # Devuelve la columna 'Food Nutrients' de la primera fila de la query, en forma de dataframe
    def get_nutrients(self, ingredient):
        info = self.get_info(ingredient)

        if info is None:
            return None

        nutrients = info['foodNutrients']
        return pd.json_normalize(nutrients)

    # Devuelve la columna 'Food Category' de la primera fila de la query, tokenizada
    def get_food_category(self, ingredient):
        info = self.get_info(ingredient)

        if info is None:
            return None

        food_cat = info['foodCategory']
        return nltk.word_tokenize(food_cat)

    # Devuelve la columna 'Ingredients' de la primera fila de la query, tokenizada
    def get_ingredients(self, ingredient):
        info = self.get_info(ingredient)

        if info is None:
            return None

        ingredients = info['ingredients']
        return nltk.word_tokenize(ingredients)
