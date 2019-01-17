# -*- coding: utf-8 -*-
"""This code is collections of sequence processing toolkits
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from odin.preprocessing.base import Extractor

class _SequenceExtractor(Extractor):
  pass

class MaxLength(_SequenceExtractor):
  """ Sequences longer than this will be filtered out. """

  def __init__(self, max_len=5218,
               input_name=None):
    super(MaxLength, self).__init__()
    self.max_len = int(max_len)
    self.input_name = input_name

  def _transform(self, X):
    pass

class IndexShift(object):
  """ IndexShift """

  def __init__(self, start_index=None, end_index=None, index_from=None):
    super(IndexShift, self).__init__()

class SkipFrequent(_SequenceExtractor):

  def __init__(self, new):
    pass

class OOVindex(_SequenceExtractor):
  """ Out-of-vocabulary processing
  Any index that is: < lower or > upper will be replaced
  by given `oov_index`

  Parameters
  ----------
  oov_index : scalar
    pass
  lower : {scalar or None}
    if None, use `min` value of all given sequences
  upper : {scalar or None}
    if None, use `max` value of all given sequences
  input_name : {list of string, None}
    pass
  """

  def __init__(self, oov_index,
               lower=None, upper=None,
               input_name=None):
    super(OOVindex, self).__init__()
    self.oov_index = int(oov_index)
