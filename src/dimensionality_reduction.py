from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionalityReduction:
    def __init__(self, dim=2):
        self.PCA = PCA(n_components=dim)
        self.TSNE = TSNE(perplexity=6, n_components=dim, init='random', n_iter=500)

    # Reducir dimensionalidad PCA
    def pca_reduce(self, vectors):
        return self.PCA.fit_transform(vectors)

    # Para mostrar los embeddings en un plano cartesiano
    def tsne_reduce(self, vectors):
        return self.TSNE.fit_transform(vectors)
