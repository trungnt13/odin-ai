from __future__ import print_function, division, absolute_import

from sklearn.base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator

class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
  """docstring for LogisticRegression"""

  def __init__(self, arg):
    super(LogisticRegression, self).__init__()
    self.arg = arg
