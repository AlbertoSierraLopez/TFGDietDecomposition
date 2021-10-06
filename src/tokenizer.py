import nltk
from torchtext.data import get_tokenizer
import spacy


class Tokenizer:
    def __init__(self):
        nltk.download('punkt')
        self.torch = get_tokenizer("basic_english")
        self.spacy = spacy.load("en_core_web_sm")

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
                if token.isalpha() or token in ['.']:
                    tokens.append(token.lower())
                elif token.isdigit():
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
