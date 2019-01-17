from __future__ import print_function, division, absolute_import
import os
import math
import types
import inspect
import warnings
from collections import Counter, Mapping
from six.moves import cPickle
from six.moves import zip, zip_longest, range

import numpy as np

from odin.utils import (one_hot, is_string, ctext, is_number,
                        is_primitives, as_tuple, flatten_list,
                        is_pickleable)
from odin.fuel.utils import MmapDict
from odin.fuel.recipe_base import FeederRecipe
from odin.fuel.recipe_shape import *
from odin.fuel.recipe_norm import *


# ===========================================================================
# Loader
# ===========================================================================
class TransLoader(FeederRecipe):
  """ Load transcription from dictionary using name specifed in
  indices

  map_func:
      inputs: name, X
      outputs: name, X, loaded_transcription
  reduce_func:
      inputs: same
      outputs: same

  Parameters
  ----------
  transcription: dict
      if path to a file is specified, the file must specified
      <name> -> [frame1, frame2, ...]
      if list is given, the list must contain the same information
      if dictionary is given, the dict must repesent the same mapping
      above

  """

  def __init__(self, transcription, dtype, delimiter=' ', label_dict=None,
               ignore_not_found=True):
    super(TransLoader, self).__init__()
    raise NotImplementedError
    self.ignore_not_found = ignore_not_found
    # ====== transcription ====== #
    share_dict = None
    if isinstance(transcription, (list, tuple)):
      share_dict = {i: j for i, j in transcription}
    elif isinstance(transcription, Mapping):
      share_dict = transcription
    elif isinstance(transcription, str) and os.path.isfile(transcription):
      share_dict = MmapDict(transcription)
    else:
      raise Exception('Cannot understand given transcription information.')
    self._transcription = share_dict
    # ====== datatype and delimiter ====== #
    # NO 64bit data type
    self.dtype = str(np.dtype(dtype)).replace('64', '32')
    self.delimiter = delimiter
    # ====== label dict if available ====== #
    if label_dict is None:
      label_func = lambda x: x
    elif isinstance(label_dict, Mapping):
      label_func = lambda x: label_dict[x]
    elif hasattr(label_dict, '__call__'):
      label_func = label_dict
    else:
      raise ValueError('label_dict must be a dictionary, function or None.')
    self.label_dict = label_func

  def process(self, name, X):
    if name not in self._transcription and self.ignore_not_found:
      return None
    trans = self._transcription[name]
    # ====== parse string using delimiter ====== #
    if isinstance(trans, str):
      trans = [self.label_dict(i)
               for i in trans.split(self.delimiter)
               if len(i) > 0]
    else:
      trans = [self.label_dict(i) for i in trans]
    trans = np.asarray(trans, dtype=self.dtype)
    # append to trans list
    X.append(trans)
    return name, X

  def shape_transform(self, shapes):
    pass


# ===========================================================================
# Basic recipes
# ===========================================================================
class Filter(FeederRecipe):

  """
  Parameters
  ----------
  filter_func: function(name, X)
      return True if the given data is accepted for further processing
      otherwise False

  """

  def __init__(self, filter_func):
    super(Filter, self).__init__()
    if not hasattr(filter_func, '__call__'):
      raise ValueError('"filter_func" must be call-able.')
    if not is_pickleable(filter_func):
      raise ValueError('"filter_func" must be pickle-able.')
    self.filter_func = filter_func

  def process(self, name, X):
    if self.filter_func(name, X):
      return name, X
    return None


# ===========================================================================
# Label processing
# ===========================================================================
class LabelOneHot(FeederRecipe):

  def __init__(self, nb_classes, data_idx=()):
    super(LabelOneHot, self).__init__()
    self.nb_classes = int(nb_classes)
    self.data_idx = data_idx

  def process(self, name, X):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    X_new = []
    for idx, x in enumerate(X):
      # transform into one-label y
      if idx in data_idx:
        x = np.array(x, dtype='int32')
        x = one_hot(x, nb_classes=self.nb_classes)
      X_new.append(x)
    return name, X_new

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(shapes),
                              return_tuple=True)
    return [((shp[0], self.nb_classes), ids) if idx in data_idx else
            (shp, ids)
            for idx, (shp, ids) in enumerate(shapes)]


class Name2Label(FeederRecipe):
  """ This function convert the name (in indices) to transcription
  for given data

  Parameters
  ----------
  converter_func: callbale (1 input arguments)
      for example, lambda name: 1 if 'true' in name else 0
      the return label then is duplicated for all data points in 1 file.
      (e.g. X.shape = (1208, 13), then, transcription=[ret] * 1208)
  ref_idx: int
      the new label will be duplicated based on the length of data
      at given idx

  Example
  -------
  >>> cluster_idx = ['spa-car', 'por-brz', 'spa-lac', 'spa-eur']
  >>> feeder = F.Feeder(ds['mfcc'], ds.path, ncpu=1, buffer_size=12)
  >>> feeder.set_batch(256, seed=None, shuffle_level=0)
  >>> feeder.set_recipes([
  >>>     F.recipes.NameToTranscription(lambda x: cluster_idx.index(x),
  >>>     F.recipes.CreateBatch()
  >>> ])

  """

  def __init__(self, converter_func, dtype=None, ref_idx=0):
    super(Name2Label, self).__init__()
    if inspect.isfunction(converter_func):
      converter_func = converter_func
    if not hasattr(converter_func, '__call__'):
      raise ValueError('"converter_func" must be call-able.')
    self.converter_func = converter_func
    self.ref_idx = int(ref_idx)
    self.dtype = dtype

  def process(self, name, X):
    # X: is a list of ndarray
    ref_idx = axis_normalize(axis=self.ref_idx, ndim=len(X),
                             return_tuple=False)
    y = self.converter_func(name)
    y = np.full(shape=(X[ref_idx].shape[0],),
                fill_value=y,
                dtype=X[ref_idx].dtype if self.dtype is None else self.dtype)
    X.append(y)
    return name, X

  def shape_transform(self, shapes):
    ref_idx = axis_normalize(axis=self.ref_idx, ndim=len(shapes),
                             return_tuple=False)
    ref_shp, ref_ids = shapes[ref_idx]
    ids = list(ref_ids)
    n = int(ref_shp[0])
    shapes.append(((n,), ids))
    return shapes
