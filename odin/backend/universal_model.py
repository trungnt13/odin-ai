from __future__ import absolute_import, division, print_function

from sklearn.base import BaseEstimator


class Model(BaseEstimator):

  def fit(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    pass
