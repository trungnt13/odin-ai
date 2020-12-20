import inspect
from typing import Optional, Union

import numpy as np
from odin.ml.base import evaluate
from odin.ml.cluster import fast_dbscan, fast_kmeans, fast_knn
from odin.ml.tree import *
from odin.ml.decompositions import *
from odin.ml.fast_lda_topics import fast_lda_topics, get_topics_string
from odin.ml.linear_model import *
from odin.ml.fast_tsne import fast_tsne
from odin.ml.fast_umap import fast_umap
from odin.ml.gmm_classifier import GMMclassifier
from odin.ml.gmm_embedding import ProbabilisticEmbedding
from odin.ml.gmm_thresholding import GMMThreshold
from odin.ml.gmm_tmat import GMM, Tmatrix
from odin.ml.ivector import Ivector
from odin.ml.neural_nlp import *
from odin.ml.plda import PLDA
from odin.ml.scoring import (Scorer, VectorNormalizer, compute_class_avg,
                             compute_wccn, compute_within_cov)
from odin.utils import get_function_arguments
from sklearn.base import ClassifierMixin
from typing_extensions import Literal
from enum import IntFlag, auto


# ===========================================================================
# Helpers functions
# ===========================================================================
def linear_classifier(X: np.ndarray,
                      y: np.ndarray,
                      algo: Literal['svm', 'lda', 'knn', 'tree', 'logistic',
                                    'gbt'],
                      seed: int = 1,
                      **kwargs) -> ClassifierMixin:
  """Train a linear classifier

  Parameters
  ----------
  X : np.ndarray
    input data
  y : np.ndarray
    target data
  algo : 'svm', 'lda', 'knn', 'tree', 'logistic', 'gbt'
    classifier algorithm
  seed : int, optional
      seed for random state, by default 1

  Returns
  -------
  ClassifierMixin
      the trained classifier

  Raises
  ------
  ValueError
      Unknown classifier algorithm
  """
  max_iter = 1000
  if algo == 'svm':
    from sklearn.svm import LinearSVC
    max_iter = 3000
    model = LinearSVC
  elif algo == 'lda':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis
  elif algo == 'knn':
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier
  elif algo == 'tree':
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier
  elif algo == 'logistic':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression
  elif algo == 'gbt':
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier
  else:
    raise ValueError(f"No support for linear classifier with name='{algo}'")
  ## select the right kwargs
  f_init = model.__init__
  if hasattr(f_init, '__wrapped__'):
    f_init = f_init.__wrapped__
  args = inspect.getfullargspec(f_init)
  args = set(
      list(args.args) +
      (list(args.defaults) if args.defaults is not None else []) +
      list(args.kwonlyargs))
  ## update the kwargs
  kwargs.update(random_state=seed)
  kwargs.setdefault('max_iter', max_iter)
  kwargs = {i: j for i, j in kwargs.items() if i in args}
  model = model(**kwargs)
  ## fit the model
  model.fit(X, y)
  return model


def clustering(X,
               algo,
               n_clusters=8,
               random_state=1,
               framework='auto',
               **kwargs):
  algo = str(algo).strip().lower()
  if 'kmean' in algo:
    return fast_kmeans(X,
                       n_clusters=n_clusters,
                       random_state=random_state,
                       framework=framework,
                       **kwargs)
  elif 'knn' in algo:
    return fast_knn(X,
                    n_clusters=n_clusters,
                    random_state=random_state,
                    framework=framework,
                    **kwargs)
  elif 'dbscan' in algo:
    return fast_dbscan(X,
                       n_clusters=n_clusters,
                       random_state=random_state,
                       framework=framework,
                       **kwargs)
  raise ValueError("No support for clustering algorithm with name: '%s'" % algo)


def dimension_reduce(*X,
                     algo: Literal['pca', 'umap', 'tsne', 'knn',
                                   'kmean'] = 'pca',
                     n_components: int = 2,
                     max_samples: Optional[int] = None,
                     return_model: bool = False,
                     random_state: int = 1,
                     **kwargs) -> np.ndarray:
  """Applying dimension reduction algorithm on a list of array

  Parameters
  ----------
  algo :  {'pca', 'umap', 'tsne', 'knn', 'kmean'}, optional
      the algorithm, by default pca
  n_components : int, optional
      number of components or cluster, by default 2
  return_model : bool, optional
      If `True`, return both transformed array and trained models,
      otherwise, only return the array., by default False
  random_state : int, optional
      seed for random state, by default 1
  kwargs : dict
      specialized arguments for each algorithm

  Returns
  -------
  np.ndarray
      the transformed array and trained model (if `return_model=True`)

  Raises
  ------
  ValueError
      Invalid algorithm
  """
  algo = str(algo).strip().lower()
  X_train = X
  if 'pca' in algo:
    fn = fast_pca
  elif 'umap' in algo:
    fn = fast_umap
  elif 'tsne' in algo:
    fn = fast_tsne
  elif 'knn' in algo:
    fn = fast_knn
    X_train = X[0]
  elif 'kmean' in algo:
    fn = fast_kmeans
    X_train = X[0]
  else:
    raise ValueError(
        f"No support for dimension reduction algorithm with name: '{algo}'")
  ## prepare k
  kw = dict(max_samples=max_samples,
            return_model=return_model,
            random_state=random_state)
  if algo == 'knn':
    kw['n_neighbors'] = n_components
  else:
    kw['n_components'] = n_components
  kw.update(kwargs)
  args = set(get_function_arguments(fn))
  kw = {k: v for k, v in kw.items() if k in args}
  ## train and predict
  outputs = fn(X_train, **kw)
  if algo == 'knn':
    outputs = [outputs.kneighbors(x) for x in X]
    if len(X) == 1:
      outputs = outputs[0]
  elif 'kmean' in algo:
    outputs = [outputs.transform(x) for x in X]
    if len(X) == 1:
      outputs = outputs[0]
  return outputs


class DimReduce(IntFlag):
  """Applying dimension reduction algorithm on a list of array

  Parameters
  ----------
  n_components : int, optional
      number of components or cluster, by default 2
  return_model : bool, optional
      If `True`, return both transformed array and trained models,
      otherwise, only return the array., by default False
  random_state : int, optional
      seed for random state, by default 1
  kwargs : dict
      specialized arguments for each algorithm

  Returns
  -------
  np.ndarray
      the transformed array and trained model (if `return_model=True`)
  """
  PCA = auto()
  UMAP = auto()
  TSNE = auto()
  KNN = auto()
  KMEANS = auto()

  def __iter__(self):
    for method in DimReduce:
      if method in self:
        yield method

  def __len__(self):
    return len(list(iter(self)))

  @property
  def is_single(self) -> bool:
    return len(self) == 1

  def __call__(self,
               *X,
               n_components: int = 2,
               max_samples: Optional[int] = None,
               return_model: bool = False,
               random_state: int = 1,
               **kwargs) -> np.ndarray:
    if len(self) > 1:
      return [
          method(*X,
                 n_components=n_components,
                 max_samples=max_samples,
                 return_model=return_model,
                 random_state=random_state,
                 **kwargs) for method in self
      ]
    return dimension_reduce(*X,
                            algo=self.name.lower(),
                            n_components=n_components,
                            max_samples=max_samples,
                            return_model=return_model,
                            random_state=random_state,
                            **kwargs)
