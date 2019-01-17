from __future__ import print_function, division, absolute_import

import warnings
from collections import Counter, Mapping

import numpy as np

from odin.fuel.recipe_base import FeederRecipe
from odin.preprocessing.signal import (segment_axis, stack_frames,
                                       mvn as _mvn, wmvn as _wmvn)
from odin.utils import (axis_normalize, is_pickleable, as_tuple, is_number,
                        is_string)

# ===========================================================================
# Helper
# ===========================================================================
def _get_data_label_idx(data_idx, label_idx, ndim):
  data_idx = axis_normalize(axis=data_idx, ndim=ndim, return_tuple=True)
  label_idx = axis_normalize(axis=label_idx, ndim=ndim, return_tuple=True)
  # exclude all index in label_idx
  data_idx = [i for i in data_idx if i not in label_idx]
  return data_idx, label_idx


def _check_label_mode(mode):
  if is_number(mode):
    return np.clip(float(mode), 0., 1.)
  if is_string(mode):
    mode = mode.lower()
    if mode == 'mid':
      mode = 'middle'
    if mode not in ('common', 'last', 'first', 'middle'):
      raise ValueError(
          "`label_mode` can be: 'common', 'last', 'first', 'middle'")
    return mode
  raise ValueError("No support for `label_mode`=%s" % str(mode))


def _apply_label_mode(y, mode):
  # This applying the label transform to 1-st axis
  if is_number(mode):
    n = y.shape[1]
    n = int(float(mode) * n)
    return y[:, n]
  if mode == 'common':
    raise NotImplementedError
  if mode == 'last':
    return y[:, -1]
  elif mode == 'first':
    return y[:, 0]
  elif mode == 'middle':
    n = y.shape[1]
    if n % 2 == 0:
      n //= 2
    else:
      n = n // 2 + 1
    return y[:, n]
  raise NotImplementedError("No support for label mode: '%s'" % mode)


# ===========================================================================
# Shape manipulation
# ===========================================================================
class Indexing(FeederRecipe):

  """ Indexing

  Parameters
  ----------
  idx: any object support `__getitem__`
      pass
  threshold: None, float, call-able
      pass
  mvn: bool
      if True, applying mean-variance normalization
  data_idx: int, list of int
      any Data at given index will be indexed by `idx`
  label_idx: int, list of int
      any Data at given index won't be applied normalization,
      since normalization on integer label can make everything
      go zero!
  """

  def __init__(self, idx, threshold=None,
               mvn=False, varnorm=True,
               data_idx=None, label_idx=()):
    super(Indexing, self).__init__()
    if not hasattr(idx, '__getitem__'):
      raise ValueError("`sad` must has attribute __getitem__ which takes "
          "file name as input and return array of index, same length as data.")
    if threshold is not None and \
    not hasattr(threshold, '__call__') and \
    not is_number(threshold):
      raise ValueError("`threshold` can be None, call-able, or number.")
    self.idx = idx
    self.threshold = threshold
    self.data_idx = data_idx
    self.label_idx = label_idx
    # ====== for normalization ====== #
    self.mvn = bool(mvn)
    self.varnorm = bool(varnorm)

  def _get_index(self, name):
    index = self.idx[name]
    if self.threshold is None:
      index
    elif hasattr(self.threshold, '__call__'):
      index = self.threshold(index)
    elif is_number(self.threshold):
      index = index >= float(self.threshold)
    if index.dtype != np.bool:
      index = index.astype('bool')
    return index

  def process(self, name, X):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    label_idx = axis_normalize(axis=self.label_idx,
                               ndim=len(X),
                               return_tuple=True)
    index = self._get_index(name)
    # ====== indexing ====== #
    X_new = []
    for i, x in enumerate(X):
      if i in data_idx:
        x = x[index]
        # if NOT label, normalization
        if self.mvn and i not in label_idx:
          x = _mvn(x, varnorm=self.varnorm)
      X_new.append(x)
    return name, X_new

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(shapes),
                              return_tuple=True)
    shapes_new = []
    for i, (shp, ids) in enumerate(shapes):
      if i in data_idx:
        ids_new = []
        n_total = 0
        # ====== update the indices ====== #
        for name, _ in ids:
          # this take a lot of time, but
          # we only calculate new shapes once.
          index = self._get_index(name)
          n = np.sum(index)
          n_total += n
          ids_new.append((name, n))
        # ====== update the shape ====== #
        ids = ids_new
        shp = (n_total,) + shp[1:]
      shapes_new.append((shp, ids))
    return shapes_new


class Slice(FeederRecipe):
  """ Slice
  Parameters
  ----------
  indices: int, slice, list of int(or slice)
      for example: [slice(0, 12), slice(20, 38)] will becomes
      x = np.hstack([x[0:12], x[20:38]])
  axis: int
      the axis will be applied given indices
  data_idx: int, list of int, None
      in case Feeders is given multiple Data, target_data is
      the index of Data that will be applied given indices.
      if None is given, the Slice is applied to all Data

  Note
  ----
  Docs for python `slice` as reference: `slice(stop)`,
  `slice(start, stop[, step])`. Create a slice object.
  This is used for extended slicing (e.g. a[0:10:2]).
  """

  def __init__(self, slices, axis, data_idx=None):
    super(Slice, self).__init__()
    # ====== validate axis ====== #
    if not is_number(axis):
      raise ValueError('axis for Slice must be an integer.')
    self.axis = int(axis)
    # ====== validate indices ====== #
    if is_number(slices):
      slices = slice(int(slices), int(slices + 1))
    elif isinstance(slices, (tuple, list)):
      slices = [i if isinstance(i, slice) else slice(int(i), int(i + 1))
                for i in slices
                if isinstance(i, slice) or is_number(i)]
    elif not isinstance(slices, slice):
      raise ValueError('indices must be int, slice, or list of int and slice.')
    self.slices = slices
    # ====== validate target_data ====== #
    self.data_idx = data_idx

  def process(self, name, X):
    X_new = []
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    for _, x in enumerate(X):
      # apply the indices if _ in target_data
      if _ in data_idx:
        ndim = x.ndim
        axis = self.axis % ndim
        # just one index given
        if isinstance(self.slices, (slice, int)):
          indices = tuple([slice(None) if i != axis else self.slices
                           for i in range(ndim)])
          x = x[indices]
        # multiple indices are given
        else:
          indices = []
          for idx in self.slices:
            indices.append(tuple([slice(None) if i != axis else idx
                                  for i in range(ndim)]))
          x = np.concatenate([x[i] for i in indices], axis=self.axis)
        # check if array still contigous
        x = np.ascontiguousarray(x)
      X_new.append(x)
    return name, X_new

  def _from_indices(self, n):
    """ This function estimates number of sample given indices """
    # slice indices
    indices = (self.slices,) if isinstance(self.slices, slice) \
        else self.slices
    count = 0
    for idx in indices:
      idx = idx.indices(n)
      count += idx[1] - idx[0]
    return count

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(shapes),
                              return_tuple=True)
    new_shapes = []
    # ====== check if first dimension is sliced ====== #
    for idx, (shp, ids) in enumerate(shapes):
      if idx in data_idx:
        if self.axis == 0:
          ids = [(name, self._from_indices(length))
                 for name, length in ids]
          n = sum(i[1] for i in ids)
          shp = (n,) + shp[1:]
        else:
          axis = self.axis % len(shp) # axis in case if negative
          # int indices, just 1
          n = self._from_indices(shp[axis])
          shp = tuple([j if i != axis else n
                       for i, j in enumerate(shp)])
      new_shapes.append((shp, ids))
    return new_shapes


class HStack(FeederRecipe):
  """Horizontal Stacking
  merge a list of np.ndarray horizontally (i.e. the last axis)
  into 1 np.array

  Note
  ----
  The new value will be `prepended` to the data list
  """

  def __init__(self, data_idx=None):
    super(HStack, self).__init__()
    self.data_idx = data_idx

  def process(self, name, X):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    if len(X) > 1 and len(data_idx) > 1:
      X_old = [x for i, x in enumerate(X) if i not in data_idx]
      X_new = [x for i, x in enumerate(X) if i in data_idx]
      X = [np.hstack(X_new)] + X_old
    return name, X

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(shapes),
                              return_tuple=True)
    # just 1 shape, nothing to merge
    if len(shapes) <= 1 or len(data_idx) <= 1:
      return shapes
    # merge
    old_shapes = []
    new_shapes = []
    for idx, (shp, ids) in enumerate(shapes):
      if idx in data_idx:
        new_shapes.append((shp, ids))
      else:
        old_shapes.append((shp, ids))
    # ====== horizontal stacking ====== #
    shape, ids = new_shapes[0]
    new_shapes = (
        shape[:-1] + (sum(shp[-1] for shp, _ in new_shapes),),
        ids
    )
    return [new_shapes] + old_shapes


class ExpandDims(FeederRecipe):
  """ ExpandDims """

  def __init__(self, axis, data_idx=0):
    super(ExpandDims, self).__init__()
    self.axis = int(axis)
    self.data_idx = data_idx

  def process(self, name, X):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(X),
                              return_tuple=True)
    X = [np.expand_dims(x, axis=self.axis)
         if i in data_idx else x
         for i, x in enumerate(X)]
    return name, X

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx,
                              ndim=len(shapes),
                              return_tuple=True)
    new_shapes = []
    for idx, (shp, ids) in enumerate(shapes):
      if idx in data_idx:
        shp = list(shp)
        axis = self.axis if self.axis >= 0 else \
            (len(shp) + 1 - self.axis)
        shp.insert(axis, 1)
        shp = tuple(shp)
      new_shapes.append((shp, ids))
    return new_shapes


# ===========================================================================
# Feature grouping
# ===========================================================================
class Stacking(FeederRecipe):
  """
  Parameters
  ----------
  left_context: int
      pass
  right_context: int
      pass
  shift: int, None
      if None, shift = right_context
      else amount of frames will be shifted
  keep_length: bool
      if True, padding zeros to begin and end of `X` to
      make the output array has the same length as original
      array.
  data_idx: int, list of int, or None
      list of all Features indices will be applied
  label_mode: string
      'common': most common label in the sequence of label
      'last': last seen label in the sequence
      'first': first seen label in the sequence
      'middle' or 'mid': middle of the sequence.
  label_idx: int, list of int, None, or empty list, tuple
      which data is specified as label will be treated differently
      based on label_mode

  NOTE
  ----
  You must be carefull applying stacking on label before one-hot
  encoded it.
  """

  def __init__(self, left_context=10, right_context=10,
               shift=1, keep_length=False,
               data_idx=None, label_mode='middle', label_idx=()):
    super(Stacking, self).__init__()
    self.left_context = int(left_context)
    self.right_context = int(right_context)
    self.shift = self.left_context if shift is None else int(shift)
    self.data_idx = data_idx
    self.label_mode = _check_label_mode(label_mode)
    self.label_idx = label_idx
    self.keep_length = bool(keep_length)

  @property
  def frame_length(self):
    return self.left_context + 1 + self.right_context

  def process(self, name, X):
    # not enough data points for stacking
    if X[0].shape[0] < self.frame_length:
      return None
    data_idx, label_idx = _get_data_label_idx(
        self.data_idx, self.label_idx, len(X))
    # ====== stacking  ====== #
    X_new = []
    for idx, x in enumerate(X):
      if idx in data_idx:
        if x.ndim == 1:
          x = np.expand_dims(x, axis=-1)
        x = stack_frames(x, frame_length=self.frame_length,
                         step_length=self.shift,
                         keep_length=self.keep_length)
      elif idx in label_idx:
        if not self.keep_length:
          x = segment_axis(x, frame_length=self.frame_length,
                           step_length=self.shift, axis=0,
                           end='cut')
          x = _apply_label_mode(x, self.label_mode)
        else:
          raise NotImplementedError # TODO
      X_new.append(x)
    return name, X_new

  def shape_transform(self, shapes):
    data_idx, label_idx = _get_data_label_idx(
        self.data_idx, self.label_idx, len(shapes))
    # ====== update the shape and indices ====== #
    new_shapes = []
    for idx, (shp, ids) in enumerate(shapes):
      if idx in data_idx or idx in label_idx:
        # calculate new number of samples
        n = 0; ids_new = []
        for name, n_samples in ids:
          n_samples = 1 + (n_samples - self.frame_length) // self.shift
          ids_new.append((name, n_samples))
          n += n_samples
        # for label_idx, number of features kept as original
        if idx in label_idx:
          shp = (n,) + shp[1:]
        # for data_idx, only apply for 2D
        elif idx in data_idx:
          n_features = shp[1] * self.frame_length if len(shp) == 2 \
              else self.frame_length # 1D case
          shp = (n, n_features)
      new_shapes.append((shp, ids_new))
    # ====== do the shape infer ====== #
    return new_shapes

class StackingSequence(FeederRecipe):
  """ Using `stack_frames` method to create data sequence
  """

  def __init__(self, length, data_idx=None):
    super(StackingSequence, self).__init__()
    self.length = int(length)
    self.data_idx = data_idx

  def process(self, name, X):
    data_idx = axis_normalize(axis=self.data_idx, ndim=len(X), return_tuple=True)
    # ====== stacking  ====== #
    X_new = []
    for idx, x in enumerate(X):
      # stack the data
      if idx in data_idx:
        if x.ndim == 1:
          x = np.expand_dims(x, axis=-1)
        feat_shape = x.shape[1:]
        x = stack_frames(x, frame_length=self.length,
                         step_length=1, keep_length=True, make_contigous=True)
        x = np.reshape(x, newshape=(-1, self.length) + feat_shape)
      X_new.append(x)
    return name, X_new

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx, ndim=len(shapes), return_tuple=True)
    # ====== update the shape and indices ====== #
    new_shapes = []
    for idx, (shp, ids) in enumerate(shapes):
      if idx in data_idx:
        n_samples = shp[0]
        shp = (n_samples, self.length) + shp[1:]
      new_shapes.append((shp, ids))
    # ====== do the shape infer ====== #
    return new_shapes

class Sequencing(FeederRecipe):
  """Generate a new array that chops the given array along the given axis
  into overlapping frames.

  This method has been implemented by Anne Archibald,
  as part of the talk box toolkit
  example::

      segment_axis(arange(10), 4, 2)
      array([[0, 1, 2, 3],
         ( [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

  Parameters
  ----------
  frame_length: int
      the length of each frame
  step_length: int
      the number of array elements by which the frames should overlap
  axis: int
      the axis to operate on; if None, act on the flattened array
  end: str
      what to do with the last frame, if the array is not evenly
          divisible into pieces. Options are:
          - 'cut'    Simply discard the extra values
          - 'wrap'   Copy values from the beginning of the array
          - 'pad'    Pad with a constant value
          - 'ignore' Pad any samples with length <= frame_length, otherwise, ignore
          - 'mix'   'cut' when >= frame_length, otherwise, 'pad'
  pad_value: Number
      the value to use for end='pad'
  pad_mode: 'pre', 'post'
      if "pre", padding or wrapping at the beginning of the array.
      if "post", padding or wrapping at the ending of the array.
  data_idx: int, list of int, None, or empty list, tuple
      list of index of all data will be applied

  Return
  ------
  a ndarray
  The array is not copied unless necessary (either because it is unevenly
  strided and being flattened or because end is set to 'pad' or 'wrap').

  Note
  ----
  It is suggested the use end='cut', and choose relevant short
  `frame_length`, using `pad` or `wrap` will significant increase
  amount of memory usage.

  """

  def __init__(self, frame_length=256, step_length=None,
               end='cut', pad_value=0., pad_mode='post',
               data_idx=None):
    super(Sequencing, self).__init__()
    frame_length = int(frame_length)
    step_length = frame_length // 2 if step_length is None else int(step_length)
    if step_length > frame_length:
      raise ValueError("step_length=%d must be smaller than frame_length=%d"
                       % (step_length, frame_length))
    self.frame_length = frame_length
    self.step_length = step_length
    # ====== check mode ====== #
    end = str(end).lower()
    if end not in ('cut', 'pad', 'wrap', 'ignore', 'mix'):
      raise ValueError(
          "`end` mode support included: 'cut', 'pad', 'wrap', 'ignore', 'mix'")
    self.end = end
    self.pad_value = pad_value
    self.pad_mode = str(pad_mode)
    # ====== transform function ====== #
    # specific index
    self.data_idx = data_idx

  def process(self, name, X):
    # ====== not enough data points for sequencing ====== #
    if self.end == 'cut' and \
    any(x.shape[0] < self.frame_length for x in X):
      return None
    if self.end == 'ignore' and \
    any(x.shape[0] > self.frame_length for x in X):
      return None
    end = self.end
    if end == 'ignore':
      end = 'pad'
    # ====== preprocessing data-idx, label-idx ====== #
    data_idx = axis_normalize(axis=self.data_idx, ndim=len(X),
                              return_tuple=True)
    # ====== segments X ====== #
    X_new = []
    for idx, x in enumerate(X):
      ## for data
      if idx in data_idx:
        if end == 'mix':
          x = segment_axis(a=x,
                           frame_length=self.frame_length,
                           step_length=self.step_length, axis=0,
                           end='cut' if x.shape[0] >= self.frame_length else 'pad',
                           pad_value=self.pad_value, pad_mode=self.pad_mode)
        else:
          x = segment_axis(a=x,
                           frame_length=self.frame_length,
                           step_length=self.step_length, axis=0,
                           end=end, pad_value=self.pad_value,
                           pad_mode=self.pad_mode)
      ## for all
      X_new.append(x)
    return name, X_new

  def shape_transform(self, shapes):
    data_idx = axis_normalize(axis=self.data_idx, ndim=len(shapes),
                              return_tuple=True)
    # ====== update the indices ====== #
    new_shapes = []
    for idx, (shp, ids) in enumerate(shapes):
      if idx in data_idx:
        # transoform the indices
        n = 0; ids_new = []
        for name, n_samples in ids:
          ## MODE = cut
          if self.end == 'cut':
            if n_samples < self.frame_length:
              n_samples = 0
            else:
              n_samples = 1 + np.floor(
              (n_samples - self.frame_length) / self.step_length)
          ## MODE = ignore and pad
          elif self.end == 'ignore':
            if n_samples > self.frame_length:
              n_samples = 0
            else:
              n_samples = 1
          ## MODE = mix
          elif self.end == 'mix':
            if n_samples < self.frame_length:
              n_samples = 1
            else:
              n_samples = 1 + np.floor(
              (n_samples - self.frame_length) / self.step_length)
          ## MODE = pad or wrap
          else:
            if n_samples < self.frame_length:
              n_samples = 1
            else:
              n_samples = 1 + np.ceil(
              (n_samples - self.frame_length) / self.step_length)
          # make sure everything is integer
          n_samples = int(n_samples)
          if n_samples > 0:
            ids_new.append((name, n_samples))
          n += n_samples
        # transform the shape for data
        if idx in data_idx:
          feat_shape = (shp[-1],) if len(shp) >= 2 else ()
          mid_shape = tuple(shp[1:-1])
          shp = (n, self.frame_length,) + mid_shape + feat_shape
      # end
      new_shapes.append((shp, ids))
    return new_shapes
