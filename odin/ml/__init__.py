from odin.ml.base import evaluate
from odin.ml.cluster import fast_dbscan, fast_kmeans, fast_knn
from odin.ml.decompositions import *
from odin.ml.fast_tsne import fast_tsne
from odin.ml.fast_umap import fast_umap
from odin.ml.gmm_thresholding import GMMThreshold
from odin.ml.gmm_classifier import GMMclassifier
from odin.ml.gmm_embedding import ProbabilisticEmbedding
from odin.ml.gmm_tmat import GMM, Tmatrix
from odin.ml.ivector import Ivector
from odin.ml.plda import PLDA
from odin.ml.scoring import (Scorer, VectorNormalizer, compute_class_avg,
                             compute_wccn, compute_within_cov)
from odin.ml.neural_nlp import *

def clustering(X,
               algo,
               n_clusters=8,
               force_sklearn=False,
               random_state=1234,
               **kwargs):
  algo = str(algo).strip().lower()
  if 'kmean' in algo:
    return fast_kmeans(X,
                       n_clusters=n_clusters,
                       random_state=random_state,
                       force_sklearn=force_sklearn,
                       **kwargs)
  elif 'knn' in algo:
    return fast_knn(X,
                    n_clusters=n_clusters,
                    random_state=random_state,
                    force_sklearn=force_sklearn,
                    **kwargs)
  elif 'dbscan' in algo:
    return fast_dbscan(X,
                       n_clusters=n_clusters,
                       random_state=random_state,
                       force_sklearn=force_sklearn,
                       **kwargs)
  raise ValueError("No support for clustering algorithm with name: '%s'" % algo)


def dimension_reduce(*X,
                     algo,
                     n_components=2,
                     return_model=False,
                     random_state=1234,
                     **kwargs):
  r""" Unified interface for dimension reduction algorithms

  Arguments:
    X : the first array will be use for training, all inputs will be
      transformed by the same model afterward.
    algo : {'pca', 'umap', 'tsne', 'knn', 'kmean}
    n_components : an Integer or None (all dimensions remained)
    return_model : a Boolean. If `True`, return both transformed array and
      trained models, otherwise, only return the array.
    random_state : an Integer or `numpy.random.RandomState`
    kwargs : specialized arguments for each algorithm

  Returns:
    transformed array and trained model (if `return_model=True`)
  """
  algo = str(algo).strip().lower()
  if 'pca' in algo:
    outputs = fast_pca(*X,
                       n_components=n_components,
                       random_state=random_state,
                       return_model=return_model,
                       **kwargs)
  elif 'umap' in algo:
    outputs = fast_umap(*X,
                        n_components=n_components,
                        random_state=random_state,
                        return_model=return_model,
                        **kwargs)
  elif 'tsne' in algo:
    outputs = fast_tsne(*X,
                        n_components=n_components,
                        random_state=random_state,
                        return_model=return_model,
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
