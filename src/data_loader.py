import re
import pandas as pd

from constants import TEST_SIZE
from collections import Counter
# from sklearn.model_selection import train_test_split
from IPython.display import display


class DataLoader:
    def __init__(self, path_csv='/datasets/RAW_recipes.csv'):
        self.dataframe = pd.read_csv(path_csv)  # 231.637 recetas

        # self.train, self.test = train_test_split(self.dataframe, test_size=0.001, random_state=1)
        self.train, self.test = self.train_test_split(self.dataframe, TEST_SIZE)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 680)

    @staticmethod
    def train_test_split(dataframe, n):
        train = dataframe.iloc[n:, :]
        test = dataframe.iloc[:n, :]    # test <- n primeros
        # Salvar en /datasets
        train.to_csv("../datasets/train_recipes.csv")
        test.to_csv("../datasets/test_recipes.csv")
        return train, test

    def display_train(self):
        display(self.train)

    def display_test(self):
        display(self.test)

    @staticmethod
    def sentencizer(string):
        # Hace falta quitar los símbolos [ ] y ' ' del principio y luego dividir por ', ' o por ", "
        return re.split('[\"\'], [\"\']', string[2:-2])

    def get_column(self, column):
        df_column = self.train[column.lower()].dropna()

        if column.lower() in ['tags', 'steps', 'ingredients']:
            df_column = df_column.apply(self.sentencizer)

        return df_column

    def get_list(self, column):
        element_list = []

        if column.lower() in ['tags', 'steps', 'ingredients']:
            for recipe in self.get_column(column.lower()):
                for element in recipe:
                    element_list.append(element)
        else:
            element_list = self.get_column(column.lower()).to_list()

        return element_list

    def get_column_counter(self, column):
        element_list = self.get_list(column.lower())

        return Counter(element_list)

    def test_recipes_generator(self):
        test_recipes = [self.sentencizer(steps) for steps in self.test['steps']]
        for recipe in test_recipes:
            clean_recipe = ', '.join(recipe)
            clean_recipe = clean_recipe[0].upper() + clean_recipe[1:] + '.\n'
            yield recipe, clean_recipe
