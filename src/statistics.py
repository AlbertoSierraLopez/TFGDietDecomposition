from constants import TEST_SIZE, DEBUG
from tokenizer import Tokenizer


class Statistics:
    def __init__(self, test, test_generator, ingredient_manager):
        self.test = test
        self.test_generator = test_generator
        self.ingredient_manager = ingredient_manager

        self.tokenizer = Tokenizer()

    # Estadísticas sobre detección de ingredientes
    def compute_statistics(self):
        true_sum = 0
        total = 0
        print('\t\t', 'Recetas procesadas:')

        for i, (input_recipe, clean_recipe) in enumerate(self.test_generator):
            self.ingredient_manager.load_recipe(clean_recipe, self.tokenizer.nltk_tokenize(input_recipe))
            detected_ingredients = self.ingredient_manager.ingredients

            detected_ingredients = len(detected_ingredients)
            real_ingredients = self.test['n_ingredients'].iloc[i]

            false = abs(detected_ingredients - real_ingredients)
            true = detected_ingredients

            true_sum += true
            total += true + false
            if DEBUG:
                print('\t\t', i+1, '/', TEST_SIZE)

        return round(true_sum / total, 4)    # Accuracy
