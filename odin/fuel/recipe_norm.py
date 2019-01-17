from __future__ import print_function, division, absolute_import


import numpy as np

from odin.fuel.recipe_base import FeederRecipe
from odin.utils import (axis_normalize, as_tuple)
from odin.preprocessing.signal import delta

# ===========================================================================
# Features preprocessing
# ===========================================================================
class Pooling(FeederRecipe):
  """docstring for Pooling"""

  def __init__(self, size=2, pool_func=np.mean, data_idx=0):
    super(Pooling, self).__init__()
    self.size = size
    self.pool_func = pool_func
    self.data_idx = data_idx

  def process(self, name, X):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    X_pooled = []
    for i, x in enumerate(X):
      if i in data_idx:
        shape = x.shape
        x = x[:, 2:-2]
        x = x.reshape(shape[0], -1, 2)
        x = self.pool_func(x, axis=-1)
        x = x.reshape(shape[0], -1)
      X_pooled.append(x)
    return name, X_pooled

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(shapes),
                              return_tuple=True)
    shapes = [(tuple(shp[:-1] + (shp[-1] // self.size - 2,)), ids)
              if i in data_idx else (shp, ids)
              for i, (shp, ids) in enumerate(shapes)]
    return shapes


class Normalization(FeederRecipe):
  """ Normalization

  Parameters
  ----------
  local_normalize: str or None
      default (True, or 'normal'), normalize the data to have mean=0, std=1
      'sigmoid', normalize by the min and max value to have all element in [0, 1]
      'tanh', normalize to have all element in [-1, 1].
      'mean', only subtract the mean
  axis: int, list of int
      normalizing axis for `local_normalize`
  data_idx: int, or list of int
      In case multiple Data is given, only normalize in the given indices.

  Note
  ----
  Global normalization by given `mean` and `std` is performed
  before `local_normalize`.
  All computation are performed in float32, hence, the return dtype
  is always float32.
  """

  def __init__(self, mean=None, std=None, local_normalize=None,
               axis=0, data_idx=0):
    super(Normalization, self).__init__()
    # mean
    if mean is not None:
      mean = mean[:].astype('float32')
    # std
    if std is not None:
      std = std[:].astype('float32')
    self.mean = mean
    self.std = std
    self.local_normalize = str(local_normalize).lower()
    if self.local_normalize not in ('none', 'false', 'tanh', 'sigmoid',
                                    'normal', 'true'):
      raise ValueError("Not support for local_normalize=%s, you must specify "
                      "one of the following mode: none, false, tanh, sigmoid, "
                      "normal (or true)." % self.local_normalize)
    self.axis = axis
    self.data_idx = data_idx

  def process(self, name, X):
    X_normlized = []
    data_idx = axis_normalize(axis=self.data_idx, ndim=len(X),
                              return_tuple=True)
    for i, x in enumerate(X):
      if i in data_idx:
        x = x.astype('float32')
        # ====== global normalization ====== #
        if self.mean is not None and self.std is not None:
          x = (x - self.mean) / (self.std + 1e-20)
        # ====== perform local normalization ====== #
        if 'normal' in self.local_normalize or 'true' in self.local_normalize:
          x = ((x - x.mean(self.axis, keepdims=True)) /
               (x.std(self.axis, keepdims=True) + 1e-20))
        elif 'sigmoid' in self.local_normalize:
          min_, max_ = np.min(x), np.max(x)
          x = (x - min_) / (max_ - min_)
        elif 'tanh' in self.local_normalize:
          min_, max_ = np.min(x), np.max(x)
          x = 2 * (x - min_) / (max_ - min_) - 1
        elif 'mean' in self.local_normalize:
          x -= x.mean(0)
      X_normlized.append(x)
    return name, X_normlized


class PCAtransform(FeederRecipe):
  """ FeatureScaling
  Scaling data into range [0, 1]

  Parameters
  ----------
  pca: odin.ml.MiniBatchPCA
      instance of MiniBatchPCA
  nb_components: int, or float (0.0 - 1.0)
      number of components, or percentage of explained variance
      in case of float.
  """

  def __init__(self, pca, nb_components=0.9, whiten=False,
               data_idx=0):
    super(PCAtransform, self).__init__()
    self._pca = pca
    self.whiten = whiten
    # specified percentage of explained variance
    if nb_components < 1.:
      _ = np.cumsum(pca.explained_variance_ratio_)
      nb_components = (_ > nb_components).nonzero()[0][0] + 1
    # specified the number of components
    else:
      nb_components = int(nb_components)
    self.nb_components = nb_components
    self.data_idx = data_idx

  def process(self, name, X):
    # update the whiten
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    pca_whiten = self._pca.whiten
    self._pca.whiten = self.whiten
    X = [self._pca.transform(x, n_components=self.nb_components)
         if i in data_idx else x
         for i, x in enumerate(X)]
    # reset the white value
    self._pca.whiten = pca_whiten
    return name, X

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(shapes),
                              return_tuple=True)
    shapes = [(shp[:-1] + (self.nb_components,), ids)
              if i in data_idx else (shp, ids)
              for i, (shp, ids) in enumerate(shapes)]
    return shapes


class FeatureScaling(FeederRecipe):
  """ FeatureScaling
  Scaling data into range [0, 1]
  """

  def __init__(self, data_idx=None):
    super(FeatureScaling, self).__init__()
    self.data_idx = data_idx

  def process(self, name, X):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    # ====== scaling features to [0, 1] ====== #
    X_new = []
    for i, x in enumerate(X):
      if i in data_idx:
        x = x.astype('float32')
        min_ = x.min(); max_ = x.max()
        x = (x - min_) / (max_ - min_)
        X_new.append(x)
    return name, X_new


class Whitening(FeederRecipe):
  """ Whitening
  TODO
  """

  def __init__(self):
    super(Whitening, self).__init__()


class ComputeDelta(FeederRecipe):
  """Compute delta features: local estimate of the derivative
  of the input data along the selected axis.

  Parameters
  ----------
  delta: int > 0 [scalar]
      the order of the difference operator.
      1 for first derivative, 2 for second, etc.
  axis: int [scalar]
      the axis along which to compute deltas.
      Default is -1 (columns).
  keep_original: bool
      if False, ignore original data and only return concatenated deltas
      features
  data_idx: int, list of int, None
      if None, calculate data for all data processed by this recipes
      if int or list of int, calculte delta for the given set of data.

  Returns
  -------
  delta_data   : list(np.ndarray) [shape=(d, t) or (d, t + window)]
      delta matrix of `data`.
      return list of deltas

  """

  def __init__(self, delta=1, axis=-1, keep_original=True,
               data_idx=0):
    super(ComputeDelta, self).__init__()
    delta = int(delta)
    if delta < 0:
      raise ValueError("delta must >= 0")
    self.delta = delta
    self.axis = axis
    self.keep_original = keep_original
    self.data_idx = data_idx

  def process(self, name, X):
    if self.delta > 0:
      data_idx = axis_normalize(axis=self.data_idx,
                                ndim=len(X),
                                return_tuple=True)
      X = [x if i not in data_idx else
           np.concatenate(
               ([x] if self.keep_original else []) +
               delta(x, order=self.delta, axis=self.axis),
               axis=self.axis)
           for i, x in enumerate(X)]
    return name, X

  def shape_transform(self, shapes):
    if self.delta > 0:
      data_idx = axis_normalize(axis=self.data_idx,
                                ndim=len(shapes),
                                return_tuple=True)
      n = (self.delta + 1) if self.keep_original else self.delta
      axis = self.axis
      shapes = [(shp, ids) if i not in data_idx else
                (shp[:axis] + (shp[axis] * n,) + shp[axis:], ids)
                for i, (shp, ids) in enumerate(shapes)]
    return shapes
