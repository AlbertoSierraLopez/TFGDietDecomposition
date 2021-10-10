import numpy as np


class IngredientManager:
    def __init__(self, recipe, requirements, ing_vocab):
        self.recipe = recipe
        self.requirement_list = np.array(['Dieta vegetariana', 'Dieta vegana', 'Dieta hipocalórica', 'Dieta proteica',
                                          'Dieta baja en carbohidratos', 'Dieta baja en sodio', 'Intolerancia a la lactosa',
                                          'Alergia los frutos secos', 'Alergia al marisco', 'Intolerancia al gluten', 'Celiaquía'])
        self.requirements = self.requirement_list[requirements]
        self.ingredients = [ingredient for ingredient in recipe if ingredient in ing_vocab]

    def unwanted_ingredients(self):
        unwanted_ingredients = []

        for ingredient in self.ingredients:
            pass
            # info = #Sacar información del ingrediente

        return unwanted_ingredients
