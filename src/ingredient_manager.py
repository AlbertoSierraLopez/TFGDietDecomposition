import requests

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
            info = self.get_nutrients(ingredient)
            # switch(info): case1: case2: case3:...

        return unwanted_ingredients

    # Array de booleanos que señala donde hay elementos del dataframe iguales al ingrediente buscado
    def get_similarity(self, ingredient, dataframe):
        similarities = []
        # Saca el array de similitudes entre el ingrediente y lo que devuelve la query
        # La idea es que si el ingrediente es 'apple'
        # podamos separar de la query donde ponga 'APPLE' y donde ponga otras cosas como 'APPLE, JUICE'
        for word in dataframe['description']:
            if word.lower() in self.wvmodel.wv.key_to_index:
                similarities.append(self.wvmodel.wv.similarity(ingredient, word.lower()))
            else:
                similarities.append(0)
        # Devuelve True en las posiciones de elementos que son el ingrediente
        return np.array(similarities) > 0.9

    # Devuelve el dataframe con la información nutricional de una ingrediente dado, o None
    def get_nutrients(self, ingredient):
        response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search?query=" + ingredient +
                                "&pageSize=10&api_key=dGv22hi1mexUfPPHzeKpENdiVUag9gnFMaEbbKio")
        ingredient_df = pd.json_normalize(response.json()['foods'])
        similarity = self.get_similarity(ingredient, ingredient_df)
        # Si el ingrediente no está en base de datos, None
        if len(ingredient_df) > 0:
            similar_ingredients = ingredient_df.iloc[similarity]
            # Si en base de datos no salen cosas muy parecidas al ingrediente buscado, None
            if len(similar_ingredients) > 0:
                # En caso de que haya varios resultados parecidos a lo que se busca
                # se devuelve la información nutricional más detallada de entre las de todos ellos (la más larga)
                nutrients_df = pd.json_normalize(similar_ingredients['foodNutrients'].iloc[
                                                     np.argmax(similar_ingredients['foodNutrients'].str.len())])
                return nutrients_df     # La info nutricional es una dataframe

    def get_nutrition_dict(self):
        nutrition_dict = dict()

        for ingredient in self.ingredients:
            nutrition_dict[ingredient] = self.get_nutrients(ingredient)

        return nutrition_dict
