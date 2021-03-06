import atexit
import re
import pandas as pd

from fuzzywuzzy import fuzz

from constants import TEST_SIZE, PATH_GOLDEN, PATH_OUTPUT, REQUIREMENT_LIST
from tokenizer import Tokenizer
from data_loader import sentencizer


class Statistics:
    def __init__(self, test, test_tuples, ingredient_manager, chunks=False, debug=False):
        self.test = test
        self.test_ingredients = test['ingredients'].dropna().apply(sentencizer)
        self.test_tuples = test_tuples
        self.ingredient_manager = ingredient_manager
        self.chunks = chunks
        self.debug = debug

        self.tokenizer = Tokenizer()

        self.debug_file = open("../output/coincide_debug.txt", 'w+')

        atexit.register(self.exit_handler)

    # Estadísticas sobre detección de ingredientes
    def compute_statistics(self):
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        if self.debug:
            print('\t\t', 'Recetas procesadas:')

        for i, (input_recipe, clean_recipe) in enumerate(self.test_tuples):
            self.ingredient_manager.load_recipe(clean_recipe, self.tokenizer.spacy_tokenize_pro_str(input_recipe))
            detected_ingredients = self.ingredient_manager.ingredients
            real_ingredients = self.test_ingredients.iloc[i]

            true_positives, false_positives, false_negatives = self.precision_by_coincidence(detected_ingredients,
                                                                                             real_ingredients)

            tp_sum += true_positives
            fp_sum += false_positives
            fn_sum += false_negatives

            if self.debug:
                print('\t\t', i+1, '/', TEST_SIZE)

        if self.debug:
            print("\tTrue Positives:",  tp_sum)
            print("\tFalse Positives:", fp_sum)
            print("\tFalse Negatives:", fn_sum)

        precision = round(tp_sum / (tp_sum + fp_sum), 5)
        recall = round(tp_sum / (tp_sum + fn_sum), 5)
        f1 = round(tp_sum / (tp_sum + (fp_sum + fn_sum) / 2), 5)
        return precision, recall, f1

    # Compara el número de ingredientes detectados con el número de ingredientes reales
    @staticmethod
    def precision_by_number(detected, real):
        hits = len(detected)
        misses = abs(len(detected) - len(real))
        return hits, misses

    # Compara los ingredientes detectados con los ingredientes reales
    def precision_by_coincidence(self, detected_ingredients, real_ingredients):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for detected_ingredient in detected_ingredients:
            if self.match(detected_ingredient, real_ingredients):
                true_positives += 1
            else:
                false_positives += 1

        for real_ingredient in real_ingredients:
            if not self.match(real_ingredient, detected_ingredients):
                false_negatives += 1

        return true_positives, false_positives, false_negatives

    def match(self, target, ingredient_list):
        for ingredient in ingredient_list:
            lev_ratio = fuzz.partial_ratio(ingredient, target)

            if self.debug:
                self.debug_file.write(ingredient + ' / ' + target + ' -> ' + str(lev_ratio) + '\n')

            if lev_ratio >= 80:
                return True

        return False

    def process_golden_standard(self):
        golden_standard = pd.read_csv(PATH_GOLDEN)
        og_requirements = self.ingredient_manager.requirements

        for i in range(len(REQUIREMENT_LIST)):
            file_out = open(PATH_OUTPUT + "/golden_standard/" + REQUIREMENT_LIST[i] + ".txt", 'w+')
            self.ingredient_manager.requirements = REQUIREMENT_LIST[i]

            for j in range(len(golden_standard)):
                row = golden_standard.iloc[j]
                recipe_steps = row['steps']
                recipe = sentencizer(recipe_steps)

                tokenized_recipe = self.tokenizer.spacy_tokenize_pro_str(recipe)
                clean_recipe = ', '.join(recipe)
                clean_recipe = clean_recipe[0].upper() + clean_recipe[1:] + '.\n'

                self.ingredient_manager.load_recipe(clean_recipe, tokenized_recipe)
                new_recipe = self.ingredient_manager.replace_unwanted()

                file_out.write('Recipe ' + str(j) + '.\n' +
                               'Original: ' + clean_recipe +
                               'New: ' + new_recipe + '\n' +
                               'Ingredients: ' + row['ingredients'] + '\n' +
                               'Detected ingredients: ' + str(sorted(self.ingredient_manager.ingredients)) + '\n' +
                               'Replacements: ' + str(sorted(self.ingredient_manager.replacements.items())) + '\n\n')

            file_out.close()
        self.ingredient_manager.requirements = og_requirements

    def exit_handler(self):
        self.debug_file.close()
