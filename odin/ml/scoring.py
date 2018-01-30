from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

from .base import BaseEstimator, TransformerMixin

class CosineScoring(BaseEstimator, TransformerMixin):
  """ CosineScoring """

  def __init__(self, arg):
    super(CosineScoring, self).__init__()
    self.arg = arg


class PLDA(BaseEstimator, TransformerMixin):
  """ PLDA """

  def __init__(self, arg):
    super(PLDA, self).__init__()
    self.arg = arg
