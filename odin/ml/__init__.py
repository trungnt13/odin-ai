from odin.ml.base import evaluate
from odin.ml.cluster import fast_kmeans, fast_knn
from odin.ml.decompositions import *
from odin.ml.fast_tsne import fast_tsne
from odin.ml.fast_umap import fast_umap
from odin.ml.gmm_classifier import GMMclassifier
from odin.ml.gmm_embedding import ProbabilisticEmbedding
from odin.ml.gmm_tmat import GMM, Tmatrix
from odin.ml.ivector import Ivector
from odin.ml.plda import PLDA
from odin.ml.scoring import (Scorer, VectorNormalizer, compute_class_avg,
                             compute_wccn, compute_within_cov)


def dimension_reduce(*X, algo, n_components=2, random_state=1234, **kwargs):
  algo = str(algo).strip().lower()
  if 'pca' in algo:
    outputs = fast_pca(*X,
                       n_components=n_components,
                       random_state=random_state,
                       **kwargs)
  elif 'umap' in algo:
    outputs = fast_umap(*X,
                        n_components=n_components,
                        random_state=random_state,
                        **kwargs)
  elif 'tsne' in algo:
    outputs = fast_tsne(*X,
                        n_components=n_components,
                        random_state=random_state,
                        **kwargs)
  elif 'knn' in algo:
    model = fast_knn(X[0], n_neighbors=n_components, **kwargs)
    outputs = [model.kneighbors(x) for x in X]
  elif 'kmean' in algo:
    model = fast_kmeans(X[0],
                        n_clusters=n_components,
                        random_state=random_state,
                        **kwargs)
    outputs = [model.transform(x) for x in X]
  else:
    raise ValueError(
        "No support for dimension reduction algorithm with name: '%s'" % algo)
  return outputs
