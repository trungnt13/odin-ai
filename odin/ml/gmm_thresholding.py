from numbers import Number

import numpy as np
from six import string_types
from sklearn.mixture import GaussianMixture

from odin.ml.base import BaseEstimator, TransformerMixin
from odin.utils import as_tuple


class GMMThreshold(BaseEstimator, TransformerMixin):

  def __init__(self, independent=True, n_components='auto', random_state=1):
    super().__init__()
    if isinstance(random_state, np.random.RandomState):
      self.randome_state = random_state
    else:
      self.randome_state = np.random.RandomState(seed=random_state)
    self.independent = bool(independent)
    self.n_components = n_components

  def fit(self, X, y=None):
    # TODO
    pass
