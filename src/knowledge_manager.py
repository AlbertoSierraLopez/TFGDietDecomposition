from collections import Counter

from constants import PATH_KG_ING, PATH_KG_TAG
import matplotlib.pyplot as plt
import networkx as nx
import os


class KnowledgeManager:
    def __init__(self, ingredients, tags):  # Tanto ingredients como tags son List
        self.KG_ing = None  # Grafo con pesos
        self.KG_tag = None  # Grafo dirigido

        self.relations_ing = []     # Relaciones ingrediente-ingrediente
        self.relations_tags = []    # Relaciones etiqueta-ingrediente

        if os.path.exists(PATH_KG_ING):
            self.KG_ing = nx.read_gpickle(PATH_KG_ING)
        else:
            self.build_kg_ingredients(ingredients)
            nx.write_gpickle(self.KG_ing, PATH_KG_ING)

        if os.path.exists(PATH_KG_TAG):
            self.KG_tag = nx.read_gpickle(PATH_KG_TAG)
        else:
            self.build_kg_tag(tags, ingredients)
            nx.write_gpickle(self.KG_tag, PATH_KG_TAG)

    def build_kg_ingredients(self, ingredients):
        for recipe in ingredients:
            for main_ingredient in recipe:
                # Al añadir primero el min y después el max, nos aseguramos de que siempre se inserta la misma tupla,
                # da igual el orden en el que nos encontramos los ingredientes. Esto también significa que por cada
                # pareja en una receta se mete la tupla dos veces, por lo que el contador debe dividirse por 2
                self.relations_ing += [(min(main_ingredient, ingredient), max(main_ingredient, ingredient))
                                       for ingredient in recipe if ingredient != main_ingredient]

        relation_counter = Counter(self.relations_ing)
        for key in relation_counter.keys():
            # Como en el diccionario se meten las relaciones (1, 2) y (2, 1), hay que dividir a la mitad
            relation_counter[key] = int(relation_counter[key] / 2)

        # Grafo no-dirigido porque las relaciones son en los dos sentidos (ingrediente-ingrediente)
        self.KG_ing = nx.Graph()

        for relation in list(relation_counter.keys()):
            self.KG_ing.add_edge(relation[0], relation[1], weight=relation_counter[relation])

    def build_kg_tag(self, tags, ingredients):
        # Construir lista de relaciones:
        for i in range(len(tags)):
            self.relations_tags += ([(tag, ingredient) for ingredient in ingredients[i] for tag in tags[i]])

        # Mirar el número que se repiten las relaciones y eliminar aquellas extraordinarias (<25 veces):
        rtags_count = Counter(self.relations_tags)
        self.relations_tags = [relation for relation in rtags_count.keys() if rtags_count[relation] >= 25]

        # Construir el grafo:
        self.KG_tag = nx.DiGraph(self.relations_tags)

    def display(self, ing=0, tag=0):
        if ing != 0:
            plt.figure(figsize=(18, 12))
            nx.draw(self.KG_ing, with_labels=True)
            plt.show()

        if (self.KG_tag is not None) and (tag != 0):
            plt.figure(figsize=(18, 12))
            nx.draw(self.KG_tag, with_labels=True)
            plt.show()

    def top_edges(self, ingredient, n=10):
        edges = [list(edge) for edge in self.KG_ing.edges(data=True)]

        ingredient_edges = [edge for edge in edges if edge[0] == ingredient or edge[1] == ingredient]

        return sorted(ingredient_edges, key=lambda y: y[2].get('weight', 1), reverse=True)[:n]
