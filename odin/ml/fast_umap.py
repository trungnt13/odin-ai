from __future__ import absolute_import, division, print_function

import numpy as np


def fast_umap(*X,
              n_components=2,
              n_neighbors=12,
              n_samples=None,
              metric="euclidean",
              n_epochs=None,
              learning_rate=1.0,
              init="spectral",
              min_dist=0.1,
              spread=1.0,
              set_op_mix_ratio=1.0,
              local_connectivity=1.0,
              repulsion_strength=1.0,
              negative_sample_rate=5,
              transform_queue_size=4.0,
              a=None,
              b=None,
              random_state=1,
              metric_kwds=None,
              angular_rp_forest=False,
              target_n_neighbors=-1,
              target_metric="categorical",
              target_metric_kwds=None,
              target_weight=0.5,
              transform_seed=42,
              return_model=False,
              verbose=False):
  r"""Uniform Manifold Approximation and Projection

  Finds a low dimensional embedding of the data that approximates
  an underlying manifold.

  Parameters
  ----------
  n_neighbors: float (optional, default 15)
      The size of local neighborhood (in terms of number of neighboring
      sample points) used for manifold approximation. Larger values
      result in more global views of the manifold, while smaller
      values result in more local data being preserved. In general
      values should be in the range 2 to 100.
      Note: try to reduce `n_neighbors` for big dataset
  n_components: int (optional, default 2)
      The dimension of the space to embed into. This defaults to 2 to
      provide easy visualization, but can reasonably be set to any
      integer value in the range 2 to 100.
  metric: string or function (optional, default 'euclidean')
      The metric to use to compute distances in high dimensional space.
      If a string is passed it must match a valid predefined metric. If
      a general metric is required a function that takes two 1d arrays and
      returns a float can be provided. For performance purposes it is
      required that this be a numba jit'd function. Valid string metrics
      include:
          * euclidean
          * manhattan
          * chebyshev
          * minkowski
          * canberra
          * braycurtis
          * mahalanobis
          * wminkowski
          * seuclidean
          * cosine
          * correlation
          * haversine
          * hamming
          * jaccard
          * dice
          * russelrao
          * kulsinski
          * rogerstanimoto
          * sokalmichener
          * sokalsneath
          * yule
      Metrics that take arguments (such as minkowski, mahalanobis etc.)
      can have arguments passed via the metric_kwds dictionary. At this
      time care must be taken and dictionary elements must be ordered
      appropriately; this will hopefully be fixed in the future.
  n_epochs: int (optional, default None)
      The number of training epochs to be used in optimizing the
      low dimensional embedding. Larger values result in more accurate
      embeddings. If None is specified a value will be selected based on
      the size of the input dataset (200 for large datasets, 500 for small).
  learning_rate: float (optional, default 1.0)
      The initial learning rate for the embedding optimization.
  init: string (optional, default 'spectral')
      How to initialize the low dimensional embedding. Options are:
          * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
          * 'random': assign initial embedding positions at random.
          * A numpy array of initial embedding positions.
  min_dist: float (optional, default 0.1)
      The effective minimum distance between embedded points. Smaller values
      will result in a more clustered/clumped embedding where nearby points
      on the manifold are drawn closer together, while larger values will
      result on a more even dispersal of points. The value should be set
      relative to the ``spread`` value, which determines the scale at which
      embedded points will be spread out.
  spread: float (optional, default 1.0)
      The effective scale of embedded points. In combination with ``min_dist``
      this determines how clustered/clumped the embedded points are.
  set_op_mix_ratio: float (optional, default 1.0)
      Interpolate between (fuzzy) union and intersection as the set operation
      used to combine local fuzzy simplicial sets to obtain a global fuzzy
      simplicial sets. Both fuzzy set operations use the product t-norm.
      The value of this parameter should be between 0.0 and 1.0; a value of
      1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
      intersection.
  local_connectivity: int (optional, default 1)
      The local connectivity required -- i.e. the number of nearest
      neighbors that should be assumed to be connected at a local level.
      The higher this value the more connected the manifold becomes
      locally. In practice this should be not more than the local intrinsic
      dimension of the manifold.
  repulsion_strength: float (optional, default 1.0)
      Weighting applied to negative samples in low dimensional embedding
      optimization. Values higher than one will result in greater weight
      being given to negative samples.
  negative_sample_rate: int (optional, default 5)
      The number of negative samples to select per positive sample
      in the optimization process. Increasing this value will result
      in greater repulsive force being applied, greater optimization
      cost, but slightly more accuracy.
  transform_queue_size: float (optional, default 4.0)
      For transform operations (embedding new points using a trained model_
      this will control how aggressively to search for nearest neighbors.
      Larger values will result in slower performance but more accurate
      nearest neighbor evaluation.
  a: float (optional, default None)
      More specific parameters controlling the embedding. If None these
      values are set automatically as determined by ``min_dist`` and
      ``spread``.
  b: float (optional, default None)
      More specific parameters controlling the embedding. If None these
      values are set automatically as determined by ``min_dist`` and
      ``spread``.
  random_state: int, RandomState instance or None, optional (default: None)
      If int, random_state is the seed used by the random number generator;
      If RandomState instance, random_state is the random number generator;
      If None, the random number generator is the RandomState instance used
      by `np.random`.
  metric_kwds: dict (optional, default None)
      Arguments to pass on to the metric, such as the ``p`` value for
      Minkowski distance. If None then no arguments are passed on.
  angular_rp_forest: bool (optional, default False)
      Whether to use an angular random projection forest to initialise
      the approximate nearest neighbor search. This can be faster, but is
      mostly on useful for metric that use an angular style distance such
      as cosine, correlation etc. In the case of those metrics angular forests
      will be chosen automatically.
  target_n_neighbors: int (optional, default -1)
      The number of nearest neighbors to use to construct the target simplcial
      set. If set to -1 use the ``n_neighbors`` value.
  target_metric: string or callable (optional, default 'categorical')
      The metric used to measure distance for a target array is using supervised
      dimension reduction. By default this is 'categorical' which will measure
      distance in terms of whether categories match or are different. Furthermore,
      if semi-supervised is required target values of -1 will be trated as
      unlabelled under the 'categorical' metric. If the target array takes
      continuous values (e.g. for a regression problem) then metric of 'l1'
      or 'l2' is probably more appropriate.
  target_metric_kwds: dict (optional, default None)
      Keyword argument to pass to the target metric when performing
      supervised dimension reduction. If None then no arguments are passed on.
  target_weight: float (optional, default 0.5)
      weighting factor between data topology and target topology. A value of
      0.0 weights entirely on data, a value of 1.0 weights entirely on target.
      The default of 0.5 balances the weighting equally between data and target.
  transform_seed: int (optional, default 42)
      Random seed used for the stochastic aspects of the transform operation.
      This ensures consistency in transform operations.
  verbose: bool (optional, default False)
      Controls verbosity of logging.
  """
  # ====== kwarg for creating UMAP class ====== #
  kwargs = dict(locals())
  del kwargs['X']
  n_samples = kwargs.pop('n_samples', None)
  return_model = kwargs.pop('return_model', False)
  # check X
  if isinstance(X[0], (tuple, list)):
    X = X[0]
  if not all(isinstance(x, np.ndarray) for x in X):
    raise ValueError("`X` can only be list of numpy.ndarray or numpy.ndarray")
  # ====== downsampling ====== #
  if n_samples is not None:
    n_samples = int(n_samples)
    assert n_samples > 0
    new_X = []
    rand = random_state if isinstance(random_state, np.random.RandomState) else \
    np.random.RandomState(seed=random_state)
    for x in X:
      if x.shape[0] > n_samples:
        ids = rand.permutation(x.shape[0])[:n_samples]
        x = x[ids]
      new_X.append(x)
    X = new_X
  # ====== train UMAP ====== #
  msg = '`pip install umap-learn` or `conda install -c conda-forge umap-learn`'
  try:
    import umap
  except ImportError:
    raise ImportError(msg)
  umap = umap.UMAP(**kwargs)
  umap.fit(X[0])
  results = [umap.transform(x) for x in X]
  if return_model:
    return results[0] if len(results) == 1 else results, umap
  return results[0] if len(results) == 1 else results
