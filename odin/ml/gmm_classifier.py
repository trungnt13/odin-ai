from numbers import Number

import numpy as np
import scipy as sp

from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import softmax

from odin.utils import as_tuple
from odin.ml.base import BaseEstimator, ClassifierMixin, Evaluable
from odin.ml.scoring import VectorNormalizer

class GMMclassifier(BaseEstimator, ClassifierMixin, Evaluable):
  """ GMMclassifier
  Parameters
  ----------
  strategy : str
    'ova' - one-vs-all, for each class represented as a GMM
    'all' - use a single GMM for all classes
  covariance_type : {'full', 'tied', 'diag', 'spherical'},
        defaults to 'full'.
    String describing the type of covariance parameters to use.
    Must be one of::
        'full' (each component has its own general covariance matrix),
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance).
  max_iter : int, defaults to 100.
    The number of EM iterations to perform.
  n_init : int, defaults to 1.
    The number of initializations to perform. The best results are kept.
  init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
    The method used to initialize the weights, the means and the
    precisions.
    Must be one of::
        'kmeans' : responsibilities are initialized using kmeans.
        'random' : responsibilities are initialized randomly.
  n_components : {int, list of int}
    only used in case `strategy='ova'`, number of Gaussian components
    for each class
  """

  def __init__(self, strategy="ova", covariance_type='full',
               max_iter=100, n_init=1,
               init_params='kmeans', n_components=1,
               centering=False, wccn=False, unit_length=False,
               lda=False, concat=False, labels=None, seed=5218):
    super(GMMclassifier, self).__init__()
    self._strategy = str(strategy)
    self._n_components = int(n_components)
    self._covariance_type = str(covariance_type)
    self._max_iter = int(max_iter)
    self._n_init = int(n_init)
    self._init_params = str(init_params)
    self._seed = 5218 if not isinstance(seed, Number) else int(seed)
    # ====== default attribute ====== #
    self._labels = labels
    self._feat_dim = None
    self._gmm = None
    self._normalizer = VectorNormalizer(
        centering=centering, wccn=wccn, unit_length=unit_length,
        lda=lda, concat=concat)

  # ==================== Pickling ==================== #
  def __getstate__(self):
    if not self.is_fitted:
      raise RuntimeError("The GMMclassifier have not been fitted, "
                         "nothing to pickle!")
    return (self._strategy, self._n_components, self._covariance_type,
            self._max_iter, self._n_init, self._init_params,
            self._labels, self._feat_dim, self._gmm,
            self._normalizer)

  def __setstate__(self, states):
    (self._strategy, self._n_components, self._covariance_type,
     self._max_iter, self._n_init, self._init_params,
     self._labels, self._feat_dim, self._gmm,
     self._normalizer) = states

  # ==================== Properties ==================== #
  @property
  def is_fitted(self):
    return self._gmm is not None

  @property
  def feat_dim(self):
    return self._feat_dim

  @property
  def labels(self):
    return self._labels

  @property
  def nb_classes(self):
    return len(self.labels)

  # ==================== Helpers ==================== #
  def initialize(self, X, y=None):
    if isinstance(X, (tuple, list)):
      X = np.array(X)
    elif not isinstance(X, np.ndarray):
      X = X[:]
    if isinstance(y, (tuple, list)):
      y = np.array(y)
    elif y is not None and not isinstance(y, np.ndarray):
      y = y[:]
    # ====== check dimensions ====== #
    feat_dim = X.shape[1]
    if self._feat_dim is None:
      self._feat_dim = feat_dim
    if y is not None:
      classes = np.unique(y)
      if self._labels is None:
        self._labels = classes
    else:
      classes = self.labels
    # ====== exception ====== #
    if self.feat_dim != feat_dim:
      raise ValueError("Initialized with `feat_dim`=%d, given data with %d "
                       "dimensions" % (self.feat_dim, feat_dim))
    if self.nb_classes != len(classes):
      raise ValueError("Initialized with `nb_classes`=%d, given data with %d "
                       "classes" % (self.nb_classes, len(classes)))
    # ====== normalizing ====== #
    if not self._normalizer.is_fitted:
      self._normalizer.fit(X, y)
    X = self._normalizer.transform(X)
    # ====== initialize GMMs ====== #
    if self._gmm is None:
      if self._strategy == 'ova':
        self._gmm = []
        rand = np.random.RandomState(seed=self._seed)
        for n_components in as_tuple(self._n_components, t=int, N=self.nb_classes):
          gmm = GaussianMixture(n_components=n_components,
            covariance_type=self._covariance_type, max_iter=self._max_iter,
            n_init=self._n_init, init_params=self._init_params,
            random_state=rand.randint(0, 10e8))
          self._gmm.append(gmm)
      elif self._strategy == 'all':
        gmm = GaussianMixture(n_components=self.nb_classes,
            covariance_type=self._covariance_type, max_iter=self._max_iter,
            n_init=self._n_init, init_params=self._init_params,
            means_init=np.array([X[y == clz].mean(axis=0) for clz in np.unique(y)]),
            random_state=self._seed)
        self._gmm = gmm
      else:
        raise ValueError("No support for `strategy`=%s" % self._strategy)
    # ====== return ====== #
    if y is None:
      return X
    return X, y

  # ==================== Sklearn ==================== #
  def fit(self, X, y):
    X, y = self.initialize(X, y)
    classes = np.unique(y)
    if self._strategy == 'ova':
      for i, (clz, gmm) in enumerate(zip(classes, self._gmm)):
        X_cls = X[y == clz]
        gmm.fit(X_cls)
    elif self._strategy == 'all':
      self._gmm.fit(X)

  def score_samples(self, X):
    X = self.initialize(X)
    if self._strategy == 'ova':
      scores = np.concatenate([gmm.score_samples(X)[:, None]
                               for k, gmm in enumerate(self._gmm)],
                              axis=-1)
    elif self._strategy == 'all':
      scores = self._gmm.predict_proba(X)
    return scores

  def predict(self, X):
    return np.argmax(self.score_samples(X), axis=-1)

  def predict_proba(self, X):
    if self._strategy == 'ova':
      scores = self.score_samples(X)
      smin = np.min(scores, axis=-1, keepdims=True)
      smax = np.max(scores, axis=-1, keepdims=True)
      scores = (scores - smin) / (smax - smin)
      proba = softmax(scores)
    elif self._strategy == 'all':
      proba = self.score_samples(X)
    return proba

  def predict_log_proba(self, X):
    if self._strategy == 'ova':
      return self.score_samples(X)
    elif self._strategy == 'all':
      return np.log(self.predict_proba(X))
