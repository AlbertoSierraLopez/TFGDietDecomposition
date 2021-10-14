import requests
import nltk

import numpy as np
import pandas as pd


class IngredientManager:
    def __init__(self, recipe, requirements, ing_vocab, wvmodel, kg_tag):
        self.recipe = recipe
        self.wvmodel = wvmodel
        self.kg_tag = kg_tag
        self.requirement_list = np.array(['Dieta vegetariana', 'Dieta vegana', 'Dieta hipocalórica', 'Dieta proteica',
                                          'Dieta baja en carbohidratos', 'Dieta baja en sodio',
                                          'Intolerancia a la lactosa', 'Alergia los frutos secos', 'Alergia al marisco',
                                          'Intolerancia al gluten', 'Celiaquía'])
        self.requirements = self.requirement_list[requirements]
        # set para evitar repeticiones, sorted por comodidad:
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
            if food not in self.kg_tag['vegan']:
                return False

        if 'Dieta vegana' in self.requirements:
            if food not in self.kg_tag['vegetarian']:
                return False

        if 'Dieta hipocalórica' in self.requirements and nutrients is not None:
            kcals = nutrients.loc[nutrients['nutrientName'] == 'Energy']
            if kcals['value'].item() > 200.0:
                return False

        if 'Dieta proteica' in self.requirements and nutrients is not None:
            protein = nutrients.loc[nutrients['nutrientName'] == 'Protein']
            if protein['value'].item() < 15.0:
                return False

        if 'Dieta baja en carbohidratos' in self.requirements and nutrients is not None:
            carbs = nutrients.loc[nutrients['nutrientName'] == 'Carbohydrate, by difference']
            if carbs['value'].item() > 150.0:
                return False

        if 'Dieta baja en sodio' in self.requirements and nutrients is not None:
            sodium = nutrients.loc[nutrients['nutrientName'] == 'Sodium, Na']
            if sodium['value'].item() > 100.0:
                return False

        if 'Intolerancia a la lactosa' in self.requirements and ingredients is not None:
            if 'milk' in ingredients:
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

    @staticmethod
    # Devuelve la primera fila de la query como dataframe
    def get_info(food):
        response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search?query=" + food +
                                "&pageSize=5&api_key=dGv22hi1mexUfPPHzeKpENdiVUag9gnFMaEbbKio")
        food_df = pd.json_normalize(response.json()['foods'])
        if len(food_df) > 0:
            return food_df.iloc[0]

    # Devuelve la columna 'Food Nutrients' de la primera fila de la query, en forma de dataframe
    def get_nutrients(self, food):
        info = self.get_info(food)

        if info is None:
            return None

        nutrients = info['foodNutrients']
        return pd.json_normalize(nutrients)

    # Devuelve la columna 'Food Category' de la primera fila de la query, tokenizada
    def get_food_category(self, food):
        info = self.get_info(food)

        if info is None:
            return None

        food_cat = info['foodCategory']
        return [token.lower() for token in nltk.word_tokenize(food_cat)]        # Devolver en minuscula

    # Devuelve la columna 'Ingredients' de la primera fila de la query, tokenizada
    def get_ingredients(self, food):
        info = self.get_info(food)

        if info is None:
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
