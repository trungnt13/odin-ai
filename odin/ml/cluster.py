from __future__ import absolute_import, division, print_function


def fast_kmeans(X,
                n_clusters=8,
                max_iter=300,
                tol=0.0001,
                verbose=0,
                random_state=8,
                init='scalable-k-means++',
                oversampling_factor=2.0,
                max_samples_per_batch=32768,
                n_jobs=None,
                force_sklearn=False):
  r"""
  Arguments:
    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float64 (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : boolean (default = 0)
        If True, prints diagnositc information.
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
  kwargs.pop('force_sklearn')
  use_cuda = False
  if not force_sklearn:
    try:
      from cuml.cluster import KMeans
      use_cuda = True
    except ImportError:
      pass
  if not use_cuda:
    from sklearn.cluster import KMeans
  # ====== fine-tuning the kwargs ====== #
  if use_cuda:
    kwargs.pop('n_jobs')
  else:
    kwargs.pop('oversampling_factor')
    kwargs.pop('max_samples_per_batch')
    if kwargs['init'] in ('scalable-k-means++', 'k-means||'):
      kwargs['init'] = 'k-means++'
  kmean = KMeans(**kwargs)
  kmean.fit(X)
  return kmean


def fast_knn(X, n_neighbors=5, verbose=False, n_jobs=1, force_sklearn=False):
  r"""
  Arguments:
    n_neighbors: int (default = 5)
      The top K closest datapoints you want the algorithm to return.
      Currently, this value must be < 1024.
  """
  kwargs = dict(locals())
  X = kwargs.pop('X')
  force_sklearn = kwargs.pop('force_sklearn')
  use_cuda = False
  if not force_sklearn:
    try:
      from cuml.neighbors import NearestNeighbors
      use_cuda = True
    except ImportError:
      pass
  if not use_cuda:
    from sklearn.neighbors import NearestNeighbors
  # ====== fine-tuning the kwargs ====== #
  if use_cuda:
    kwargs['n_gpus'] = kwargs['n_jobs']
    kwargs.pop('n_jobs')
  else:
    kwargs.pop('verbose')
  knn = NearestNeighbors(**kwargs)
  knn.fit(X)
  return knn
