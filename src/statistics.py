from fuzzywuzzy import fuzz

from constants import TEST_SIZE, DEBUG
from tokenizer import Tokenizer
from data_loader import sentencizer


class Statistics:
    def __init__(self, test, test_tuples, ingredient_manager):
        self.test = test
        self.test_ingredients = test['ingredients'].dropna().apply(sentencizer)
        self.test_tuples = test_tuples
        self.ingredient_manager = ingredient_manager

        self.tokenizer = Tokenizer()

    # Estadísticas sobre detección de ingredientes
    def compute_statistics(self):
        hit_sum = 0
        miss_sum = 0
        if DEBUG:
            print('\t\t', 'Recetas procesadas:')

        for i, (input_recipe, clean_recipe) in enumerate(self.test_tuples):
            self.ingredient_manager.load_recipe(clean_recipe, self.tokenizer.spacy_tokenize_pro_str(input_recipe))
            detected_ingredients = self.ingredient_manager.ingredients
            real_ingredients = self.test_ingredients.iloc[i]

            hits, misses = self.hit_miss_by_coincidence(detected_ingredients, real_ingredients)

            hit_sum += hits
            miss_sum += misses
            if DEBUG:
                print('\t\t', i+1, '/', TEST_SIZE)

        return round(hit_sum / (hit_sum + miss_sum), 4)    # Accuracy

    # Compara el número de ingredientes detectados con el número de ingredientes reales
    @staticmethod
    def hit_miss_by_number(detected, real):
        hits = len(detected)
        misses = abs(len(detected) - len(real))
        return hits, misses

    # Compara los ingredientes detectados con los ingredientes reales
    def hit_miss_by_coincidence(self, detected_ingredients, real_ingredients):
        hits = 0
        misses = 0

        for real_ingredient in real_ingredients:
            if self.coincide(real_ingredient, detected_ingredients):
                hits += 1
            else:
                misses += 1

        return hits, misses

    @staticmethod
    def coincide(real_ingredient, detected_ingredients):
        for detected_ingredient in detected_ingredients:
            if fuzz.partial_ratio(real_ingredient, detected_ingredient) > 85:
                return True

        return False
