from __future__ import absolute_import, division, print_function

import types
from numbers import Number
from warnings import warn
from typing import Optional, Union
from typing_extensions import Literal

import numpy as np
from scipy import sparse, stats
from scipy.sparse import csr_matrix

__all__ = ['fast_kmeans', 'fast_knn', 'fast_dbscan']


# ===========================================================================
# Helper
# ===========================================================================
def _check_cuml(framework):
  r""" Check if RAPIDS - cuml is installed

  Return True if `cuml` available, otherwise, print warning message how to
  install `cuml`
  """
  if framework == 'sklearn':
    return False
  try:
    import cuml
    return True
  except ImportError:
    warn("conda install "
         "-c rapidsai -c nvidia -c conda-forge -c defaults "
         "cuml=0.17 cudatoolkit=10.1")
  return False


def nn_kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
  if n_neighbors is None:
    n_neighbors = self.n_neighbors
  # check the input only in self.kneighbors
  # construct CSR matrix representation of the k-NN graph
  A_data, A_ind = self.kneighbors(X, n_neighbors)
  n_queries = A_ind.shape[0]
  n_samples_fit = self.n_samples_fit_
  n_nonzero = n_queries * n_neighbors
  A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)
  # prepare the return
  returns = []
  if mode in ('distance', 'both'):
    returns.append(
        csr_matrix((np.ravel(A_data), A_ind.ravel(), A_indptr),
                   shape=(n_queries, n_samples_fit)))
  if mode in ('connectivity', 'both'):
    returns.append(
        csr_matrix((np.ones(n_queries * n_neighbors), A_ind.ravel(), A_indptr),
                   shape=(n_queries, n_samples_fit)))
  if len(returns) == 0:
    raise ValueError('Unsupported mode, must be one of "connectivity" '
                     'or "distance" but got "%s" instead' % mode)
  return tuple(returns) if len(returns) > 1 else returns[0]


def nn_transform(self, X):
  mode = self._graph_mode
  add_one = mode == 'distance'
  return self.kneighbors_graph(X,
                               mode=mode,
                               n_neighbors=self.n_neighbors + add_one)


def nn_fit_transform(self, X, y=None):
  return self.fit(X).transform(X)


def nn_predict(self, X):
  from sklearn.utils.validation import check_array
  from sklearn.neighbors import NearestNeighbors
  ## prepare inputs
  cluster_mode = self._cluster_mode
  n_clusters = self._n_clusters
  n_neighbors = self.n_neighbors
  random_state = self._random_state
  X = check_array(X, accept_sparse='csr')
  # transform in to neighbor of neighbor space
  distances = self.kneighbors_graph(X, mode='distance')
  nn = NearestNeighbors(n_neighbors=n_neighbors).fit(distances)
  nn.kneighbors_graph = types.MethodType(nn_kneighbors_graph, nn)
  distances, connectivity = nn.kneighbors_graph(mode='both')
  ## classifying by vote
  if cluster_mode == 'spectral':
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(n_clusters=n_clusters,
                              random_state=random_state,
                              n_init=max(10, n_clusters * 2),
                              affinity='precomputed_nearest_neighbors',
                              n_neighbors=n_neighbors).fit_predict(connectivity)
  ## isomap
  elif cluster_mode == 'isomap':
    from sklearn.manifold import Isomap
    from sklearn.cluster import KMeans
    return KMeans(
        n_clusters=n_clusters,
        n_init=max(10, 2 * n_clusters),
        random_state=1234).fit_predict(
            Isomap(n_neighbors=self.n_neighbors - 1,
                   metric='precomputed',
                   n_components=self.n_neighbors).fit_transform(distances))
  ## dbscan
  elif cluster_mode == 'dbscan':
    from sklearn.cluster import DBSCAN
    return DBSCAN(n_jobs=None, metric='precomputed').fit_predict(distances)
  ## kmeans
  elif cluster_mode == 'kmeans':
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_clusters,
                  n_init=max(10, 2 * n_clusters),
                  random_state=random_state).fit_predict(distances)
  ## error
  else:
    raise ValueError("No support for nearest neighbors clustering mode: '%s'" %
                     cluster_mode)


def dbscan_predict(self, X=None):
  if X is not None and id(X) != self._fitid:
    warn("DBSCAN cannot predict on new data, only fitted data")
  y = self.labels_
  if hasattr(y, 'to_array'):
    y = y.to_array()
  return y


# ===========================================================================
# Main method
# ===========================================================================
def fast_kmeans(X,
                *,
                n_clusters: int = 8,
                max_iter: int = 300,
                tol: float = 0.0001,
                n_init: int = 10,
                random_state: int = 1,
                init: Literal['scalable-kmeans++', 'k-means||',
                              'random'] = 'scalable-k-means++',
                oversampling_factor: float = 2.0,
                max_samples_per_batch: int = 32768,
                framework: Literal['auto', 'cuml', 'sklearn'] = 'auto'):
  """KMeans clustering

  Parameters
  ----------
  n_clusters : int (default = 8)
      The number of centroids or clusters you want.
  max_iter : int (default = 300)
      The more iterations of EM, the more accurate, but slower.
  tol : float64 (default = 1e-4)
      Stopping criterion when centroid means do not change much.
  random_state : int (default = 1)
      If you want results to be the same when you restart Python, select a
      state.
  init : {'scalable-kmeans++', 'k-means||' , 'random' or an ndarray}
         (default = 'scalable-k-means++')
      'scalable-k-means++' or 'k-means||': Uses fast and stable scalable
      kmeans++ intialization.
      'random': Choose 'n_cluster' observations (rows) at random from data
      for the initial centroids.
      If an ndarray is passed, it should be of
      shape (n_clusters, n_features) and gives the initial centers.
  max_samples_per_batch : int maximum number of samples to use for each batch
                              of the pairwise distance computation.
  oversampling_factor : int (default = 2) The amount of points to sample
      in scalable k-means++ initialization for potential centroids.
      Increasing this value can lead to better initial centroids at the
      cost of memory. The total number of centroids sampled in scalable
      k-means++ is oversampling_factor * n_clusters * 8.
  max_samples_per_batch : int (default = 32768) The number of data
      samples to use for batches of the pairwise distance computation.
      This computation is done throughout both fit predict. The default
      should suit most cases. The total number of elements in the batched
      pairwise distance computation is max_samples_per_batch * n_clusters.
      It might become necessary to lower this number when n_clusters
      becomes prohibitively large.
  """
  kwargs = dict(locals())
  X = kwargs.pop('X')
  kwargs.pop('framework')
  ## fine-tuning the kwargs
  cuml = _check_cuml(framework)
  if cuml:
    from cuml.cluster import KMeans
    kwargs.pop('n_init')
  else:
    from sklearn.cluster import MiniBatchKMeans
    kwargs.pop('oversampling_factor')
    kwargs.pop('max_samples_per_batch')
    if kwargs['init'] in ('scalable-k-means++', 'k-means||'):
      kwargs['init'] = 'k-means++'
  ## fitting
  if not cuml:
    from odin.utils import batching
    kmean = MiniBatchKMeans(**kwargs)
    for s, e in batching(int(max_samples_per_batch),
                         n=X.shape[0],
                         seed=random_state):
      kmean.partial_fit(X[s:e])
  else:
    kmean = KMeans(verbose=False, **kwargs)
    kmean.fit(X)
  return kmean


def fast_knn(X,
             n_clusters: int = 5,
             n_neighbors: Optional[int] = None,
             graph_mode='distance',
             cluster_mode='spectral',
             algorithm='brute',
             n_jobs=1,
             random_state=1,
             framework: Literal['auto', 'cuml', 'sklearn'] = 'auto'):
  r"""
  Parameters
  ----------
  X : `ndarray` or tuple of (X, y)
  n_neighbors: int (default = 5)
    The top K closest datapoints you want the algorithm to return.
    Currently, this value must be < 1024.
  graph_mode : {'distance', 'connectivity'}, default='distance'
    This mode decides which values `kneighbors_graph` will return:
      - 'connectivity' : will return the connectivity matrix with ones and
        zeros (for 'SpectralClustering').
      - 'distance' : will return the distances between neighbors according
        to the given metric (for 'DBSCAN').
  cluster_mode: {'vote', 'spectral', 'isomap'}, default='vote'
      This mode decides how to generate cluster prediction from the
      neighbors graph:
      - 'dbscan' :
      - 'spectral' :
      - 'isomap' :
      - 'kmeans' :
  algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
      Algorithm used to compute the nearest neighbors:
      - 'ball_tree' will use :class:`BallTree`
      - 'kd_tree' will use :class:`KDTree`
      - 'brute' will use a brute-force search.
      - 'auto' will attempt to decide the most appropriate algorithm
        based on the values passed to :meth:`fit` method.
      Note: fitting on sparse input will override the setting of
      this parameter, using brute force.
  """
  kwargs = dict(locals())
  X = kwargs.pop('X')
  framework = kwargs.pop('framework')
  random_state = kwargs.pop('random_state')
  n_clusters = int(kwargs.pop('n_clusters'))
  if n_neighbors is None:
    kwargs['n_neighbors'] = n_clusters
    n_neighbors = n_clusters
  ## graph mode
  graph_mode = str(kwargs.pop('graph_mode')).strip().lower()
  assert graph_mode in ('distance', 'connectivity')
  ## cluster mode
  cluster_mode = str(kwargs.pop('cluster_mode')).strip().lower()
  ## fine-tuning the kwargs
  use_cuml = _check_cuml(framework)
  if use_cuml:
    from cuml.neighbors import NearestNeighbors
    kwargs['n_gpus'] = kwargs['n_jobs']
    kwargs.pop('n_jobs')
    kwargs.pop('algorithm')
  else:
    from sklearn.neighbors import NearestNeighbors
  ## fitting
  knn = NearestNeighbors(**kwargs)
  knn.fit(X)
  knn._fitid = id(X)
  ## Transform mode
  knn._random_state = random_state
  knn._n_clusters = n_clusters
  knn._graph_mode = graph_mode
  knn._cluster_mode = cluster_mode
  if use_cuml:
    knn.n_samples_fit_ = X.shape[0]
  knn.kneighbors_graph = types.MethodType(nn_kneighbors_graph, knn)
  knn.transform = types.MethodType(nn_transform, knn)
  knn.fit_transform = types.MethodType(nn_fit_transform, knn)
  knn.predict = types.MethodType(nn_predict, knn)
  return knn


def fast_dbscan(X,
                eps=0.5,
                min_samples=5,
                n_clusters=None,
                metric='euclidean',
                algorithm='brute',
                random_state: int = 1,
                framework: Literal['auto', 'cuml', 'sklearn'] = 'auto'):
  r""" DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    DBSCAN is a very powerful if the datapoints tend to congregate in
    larger groups.

    Arguments:
      eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.
      min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
      algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='brute'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

  """
  kwargs = dict(locals())
  X = kwargs.pop('X')
  framework = kwargs.pop('framework')
  n_cluster = kwargs.pop('n_clusters')
  random_state = kwargs.pop('random_state')
  ## fine-tuning the kwargs
  if _check_cuml(framework):
    from cuml.cluster import DBSCAN
    kwargs.pop('algorithm')
    kwargs.pop('metric')
  else:
    from sklearn.cluster import DBSCAN
  ## fitting
  dbscan = DBSCAN(**kwargs)
  dbscan.fit(X)
  dbscan._fitid = id(X)
  dbscan.predict = types.MethodType(dbscan_predict, dbscan)
  return dbscan
