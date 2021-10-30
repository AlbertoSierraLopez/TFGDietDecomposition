from collections import Counter

from constants import PATH_TOKENS, PATH_IMP_ING, PATH_ING_VOCAB, TORCH_TOKENIZER, SPACY_TOKENIZER
import nltk
import spacy
import json
import os
import pickle
import pandas as pd
from torchtext.data import get_tokenizer


class Tokenizer:
    def __init__(self):
        # nltk.download('punkt')
        self.torch = get_tokenizer(TORCH_TOKENIZER)
        self.spacy = spacy.load(SPACY_TOKENIZER)

        self.ing_vocab = self.get_ing_vocab()

    @staticmethod
    def nltk_tokenize(sentences):
        tokens = []
        for sentence in sentences:
            for token in nltk.word_tokenize(sentence):
                if token.isalpha() or token in ['.']:
                    tokens.append(token.lower())
                elif token.isdigit():
                    tokens.append('<number>')
        return tokens

    def torch_tokenize(self, sentences):
        tokens = []
        for sentence in sentences:
            for token in self.torch(sentence):
                if token.isalpha() or token in ['.']:
                    tokens.append(token.lower())
                elif token.isdigit():
                    tokens.append('<number>')
        return tokens

    def spacy_tokenize(self, sentences):
        tokens = []
        for sentence in sentences:
            for token in self.spacy(sentence):
                if token.is_alpha or token.text in ['.']:
                    tokens.append(token.lower_)
                elif token.is_digit:
                    tokens.append('<number>')
        return tokens

    # Más costoso. Devuelve el type Token, no str
    def spacy_tokenize_pro(self, sentences):
        tokens = []
        for doc in list(self.spacy.pipe(sentences)):
            for token in doc:
                # Se filtran signos de puntuación, números y palabras vacías
                if not token.is_punct and not token.is_digit and not token.is_stop:
                    # Se guarda el Token
                    tokens.append(token)
        return tokens

    def get_tokenized_recipes(self, recipes):
        if os.path.exists(PATH_TOKENS):
            print("\tCargando tokens guardados...")
            pickle_in = open(PATH_TOKENS, "rb")
            tokenized_recipes = pickle.load(pickle_in)
            pickle_in.close()
        else:
            print("\tCreando tokens nuevos...")
            tokenized_recipes = [self.nltk_tokenize(recipe) for recipe in recipes]
            pickle_out = open(PATH_TOKENS, "wb")
            pickle.dump(tokenized_recipes, pickle_out)
            pickle_out.close()

        return tokenized_recipes

    @staticmethod
    def get_vocab(recipes):
        word_list = [token for recipe in recipes for token in recipe]
        word_counter = Counter(word_list)

        vocab = sorted(set([token for token in word_list if word_counter[token] >= 45]))

        word2ind = dict(zip(vocab, range(0, len(vocab))))
        ind2word = dict(zip(range(0, len(vocab)), vocab))

        return vocab, word2ind, ind2word

    @staticmethod
    def get_ing_vocab():
        with open(PATH_ING_VOCAB, encoding="utf8") as json_file:
            off_json = json.load(json_file)     # OFF = OpenFoodFacts

        off = pd.json_normalize(off_json['tags'])

        # Hay que hacer un pelín de limpieza en los datos, minúsculas y quitar guiones y números
        return sorted([ingredient.lower().replace('-', ' ') for ingredient in off['name'] if len(ingredient) > 0 and
                       ingredient[0] not in ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']])

    def get_improved_ingredients(self, recipe_ingredients):
        if os.path.exists(PATH_IMP_ING):
            print("\tCargando ingredientes mejorados...")
            pickle_in = open(PATH_IMP_ING, "rb")
            improved_ingredients = pickle.load(pickle_in)
            pickle_in.close()
        else:
            print("\tMejorando ingredientes nuevos...")
            improved_ingredients = [self.improve_ingredients(recipe) for recipe in recipe_ingredients]
            pickle_out = open(PATH_IMP_ING, "wb")
            pickle.dump(PATH_IMP_ING, pickle_out)
            pickle_out.close()

        return improved_ingredients

    def improve_ingredients(self, ingredients):
        improved_ingredients = set()

        for token in self.nltk_tokenize(ingredients):
            if token in self.ing_vocab:
                improved_ingredients.add(token)

        return sorted(improved_ingredients)
