import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


class KnowledgeManager:
    def __init__(self, ingredients, tags=None):
        self.KG_ing = None      # Grafo con pesos
        self.KG_tags = None     # Grafo dirigido

        self.relations_ing = []         # Relaciones ingrediente-ingrediente
        self.relations_tags = set()     # Relaciones etiqueta-ingrediente

        self.build_kg_ingredients(ingredients)
        if tags is not None:
            self.build_kg_tags(tags, ingredients)

    def build_kg_ingredients(self, ingredients):
        for recipe in ingredients:
            for main_ingredient in recipe:
                # Al añadir primero el min y después el max, nos aseguramos de que siempre se inserta la misma tupla,
                # da igual el orden en el que nos encontramos los ingredientes. Esto también significa que por cada
                # pareja en una receta se mete la tupla dos veces, por lo que el contador debe dividirse por 2
                self.relations_ing += [(min(main_ingredient, ingredient), max(main_ingredient, ingredient))
                                       for ingredient in ingredients if ingredient != main_ingredient]

        relation_counter = Counter(self.relations_ing)
        for key in relation_counter.keys():
            # Como en el diccionario se meten las relaciones (1, 2) y (2, 1), hay que dividir a la mitad
            relation_counter[key] = int(relation_counter[key] / 2)

        # Grafo no-dirigido porque las relaciones son en los dos sentidos (ingrediente-ingrediente)
        self.KG_ing = nx.Graph()

        for relation in list(relation_counter.keys()):
            self.KG_ing.add_edge(relation[0], relation[1], weight=relation_counter[relation])

    def build_kg_tags(self, tags, ingredients):

        for i in range(len(tags)):
            self.relations_tags.update([(tag, ingredient) for ingredient in ingredients[i] for tag in tags[i]])

            self.KG_tags = nx.DiGraph(list(self.relations_tags))

    def display(self, ing=0, tag=0):
        if ing != 0:
            plt.figure(figsize=(18, 12))
            nx.draw(self.KG_ing, with_labels=True)

        if tag != 0:
            plt.figure(figsize=(18, 12))
            nx.draw(self.KG_tags, with_labels=True)
