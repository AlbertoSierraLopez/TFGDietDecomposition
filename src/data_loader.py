import re
from collections import Counter

import pandas as pd
from IPython.display import display


class DataLoader:
    def __init__(self, path_csv='/datasets/RAW_recipes.csv', cap=None):
        self.dataframe = pd.read_csv(path_csv)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 680)

        if cap is not None:
            # Me quedo con todas las filas hasta cap y con todas las columnas
            self.dataframe = self.dataframe.iloc[:cap, :]

    def display_dataframe(self):
        display(self.dataframe)

    @staticmethod
    def sentencizer(string):
        # Hace falta quitar los símbolos [ ] y ' ' del principio y luego dividir por ', ' o por ", "
        return re.split('[\"\'], [\"\']', string[2:-2])

    def get_column(self, column):
        df_column = self.dataframe[column.lower()].dropna()

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

    def get_counter(self, column):
        element_list = self.get_list(column.lower())

        return Counter(element_list)
