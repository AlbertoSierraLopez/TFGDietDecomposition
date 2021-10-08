from collections import Counter

import nltk
import spacy
from torchtext.data import get_tokenizer


class Tokenizer:
    def __init__(self):
        # nltk.download('punkt')
        self.torch = get_tokenizer("basic_english")
        self.spacy = spacy.load(r'C:/Users/Aussar/AppData/Local/Programs/Python/Python38/Lib/site-packages/en_core_web_sm/en_core_web_sm-3.1.0')

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

    # Más costoso
    def spacy_tokenize_pro(self, sentences):
        tokens = []
        for doc in list(self.spacy.pipe(sentences)):
            for token in doc:
                # Se filtran signos de puntuación, números y palabras vacías
                if not token.is_punct and not token.is_digit and not token.is_stop:
                    # Se guarda el lema
                    tokens.append(token.lemma_)
        return tokens

    def get_vocab(self, recipes):
        word_list = [token for recipe in recipes for token in recipe]
        word_counter = Counter(word_list)

        vocab = sorted(set([token for token in word_list if word_counter[token] >= 45]))

        word2ind = dict(zip(vocab, range(0, len(vocab))))
        ind2word = dict(zip(range(0, len(vocab)), vocab))

        return vocab, word2ind, ind2word
