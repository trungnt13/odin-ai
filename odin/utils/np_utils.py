# ===========================================================================
# This module is created based on the code from 2 libraries: Lasagne and keras
# Original work Copyright (c) 2014-2015 keras contributors
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import division, absolute_import, print_function

import math
import marshal

import numpy as np
import scipy as sp

from six import string_types

__all__ = [
    'array2bytes',
    'bytes2array',
    'one_hot',
    'unique_labels',
    'label_splitter'
]


# ===========================================================================
# Serialization
# ===========================================================================
idx_2_dt = {b'0': 'float32', b'1': 'float64',
            b'2': 'int32', b'3': 'int64',
            b'4': 'bool',
            b'5': 'float16', b'6': 'int16',
            b'7': 'complex64', b'8': 'complex128'}
dt_2_idx = {'float32': b'0', 'float64': b'1',
            'int32': b'2', 'int64': b'3',
            'bool': b'4',
            'float16': b'5', 'int16': b'6',
            'complex64': b'7', 'complex128': b'8'}

# support to 12 Dimension
nd_2_idx = {0: b'0', 1: b'1', 2: b'2', 3: b'3', 4: b'4',
            5: b'5', 6: b'6', 7: b'7', 8: b'8', 9: b'9',
            10: b'10', 11: b'11', 12: b'12'}


def array2bytes(a):
  """ Fastest way to convert `numpy.ndarray` and all its
  metadata to bytes array.
  """
  shape = marshal.dumps(a.shape, 0)
  array = a.tobytes() + shape + dt_2_idx[a.dtype.name] + nd_2_idx[a.ndim]
  return array


def bytes2array(b):
  """ Deserialize result from `array2bytes` back to `numpy.ndarray` """
  ndim = int(b[-1:])
  dtype = idx_2_dt[b[-2:-1]]
  i = -((ndim + 1) * 5) - 2
  shape = marshal.loads(b[i:-2])
  return np.frombuffer(b[:i], dtype=dtype).reshape(shape)


# ===========================================================================
# Helper
# ===========================================================================
class _LabelsIndexing(object):
  """ LabelsIndexing

  Parameters
  ----------
  key_func: callabe
      a function transform each element of `y` into unique ID
      for labeling.
  fast_index: dict
      mapping from label -> index
  sorted_labels: list
      list of all labels, sorted for unique order
  """

  def __init__(self, key_func, fast_index, sorted_labels):
    super(_LabelsIndexing, self).__init__()
    self._key_func = key_func
    self._fast_index = fast_index
    self._sorted_labels = sorted_labels

  def __call__(self, x):
    x = self._key_func(x)
    if x in self._fast_index:
      return self._fast_index[x]
    raise ValueError("Cannot find key: '%s' in %s" %
                     (str(x), str(self._sorted_labels)))


# ===========================================================================
# Main
# ===========================================================================
def one_hot(y, nb_classes=None, dtype='float32'):
  '''Convert class vector (integers from 0 to nb_classes)
  to binary class matrix, for use with categorical_crossentropy

  Note
  ----
  if any class index in y is smaller than 0, then all of its one-hot
  values is 0.
  '''
  if 'int' not in str(y.dtype):
    y = y.astype('int32')
  if nb_classes is None:
    nb_classes = np.max(y) + 1
  else:
    nb_classes = int(nb_classes)
  return np.eye(nb_classes, dtype=dtype)[y]

def unique_labels(y, key_func=None, return_labels=False):
  """
  Parameters
  ----------
  y: list, tuple, `numpy.ndarray`
      list of object that is label or contain label information.
  key_func: callabe
      a function transform each element of `y` into unique ID for labeling.
  return_labels: bool
      if True, return the ordered labels.

  Returns
  -------
  (call-able, tuple):
      function that transform any object into unique label index
      (optional) list of ordered labels.
  """
  if not isinstance(y, (list, tuple, np.ndarray)):
    raise ValueError("`y` must be iterable (list, tuple, or numpy.ndarray).")
  # ====== Get an unique order of y ====== #
  if key_func is None or not hasattr(key_func, '__call__'):
    key_func = lambda _: str(_)
  sorted_labels = list(sorted(set(key_func(i) for i in y)))
  fast_index = {j: i for i, j in enumerate(sorted_labels)}
  # ====== create label indexing object ====== #
  labels_indexing = _LabelsIndexing(key_func,
                                   fast_index,
                                   sorted_labels)
  if return_labels:
    return labels_indexing, tuple(sorted_labels)
  return labels_indexing


# ===========================================================================
# Label splitter
# ===========================================================================
_CACHE_SPLITTER = {}


class _label_split_helper(object):
  def __init__(self, pos, delimiter):
    super(_label_split_helper, self).__init__()
    self.pos = pos
    self.delimiter = delimiter

  def __call__(self, x):
    if isinstance(x, string_types):
      return x.split(self.delimiter)[self.pos]
    elif isinstance(x, (tuple, list, np.ndarray)):
      for i in x:
        if isinstance(i, string_types):
          return i.split(self.delimiter)[self.pos]
    else:
      raise RuntimeError("Unsupport type=%s for label splitter" %
          str(type(x)))


def label_splitter(pos, delimiter='/'):
  pos = int(pos)
  delimiter = str(delimiter)
  splitter_id = str(pos) + delimiter
  if splitter_id not in _CACHE_SPLITTER:
    splitter = _label_split_helper(pos, delimiter)
    _CACHE_SPLITTER[splitter_id] = splitter
  return _CACHE_SPLITTER[splitter_id]


# def replace(array, value, new_value):
#     if value is None:
#         return np.where(array == np.array(None), new_value, array)
#     return np.where(array == value, new_value, array)


# def is_ndarray(x):
#     return isinstance(x, np.ndarray)


# def masked_output(X, X_mask):
#     '''
#     Example
#     -------
#         X: [[1,2,3,0,0],
#             [4,5,0,0,0]]
#         X_mask: [[1,2,3,0,0],
#                  [4,5,0,0,0]]
#         return: [[1,2,3],[4,5]]
#     '''
#     res = []
#     for x, mask in zip(X, X_mask):
#         x = x[np.nonzero(mask)]
#         res.append(x.tolist())
#     return res


# def split_chunks(a, maxlen, overlap):
#     '''
#     Example
#     -------
#     >>> print(split_chunks(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 5, 1))
#     >>> [[1, 2, 3, 4, 5],
#          [4, 5, 6, 7, 8]]
#     '''
#     chunks = []
#     nchunks = int((max(a.shape) - maxlen) / (maxlen - overlap)) + 1
#     for i in xrange(nchunks):
#         start = i * (maxlen - overlap)
#         chunks.append(a[start: start + maxlen])

#     # ====== Some spare frames at the end ====== #
#     wasted = max(a.shape) - start - maxlen
#     if wasted >= (maxlen - overlap) / 2:
#         chunks.append(a[-maxlen:])
#     return chunks


# def ordered_set(seq):
#     seen = {}
#     result = []
#     for marker in seq:
#         if marker in seen: continue
#         seen[marker] = 1
#         result.append(marker)
#     return np.asarray(result)


# def shrink_labels(labels, maxdist=1):
#     '''
#     Example
#     -------
#     >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 1))
#     >>> [0, 1, 0, 1, 0, 4, 5, 4, 6, 0]
#     >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 2))
#     >>> [0, 1, 0, 4, 6, 0]
#     Notes
#     -----
#     Different from ordered_set, the resulted array still contain duplicate
#     if they a far away each other.
#     '''
#     maxdist = max(1, maxdist)
#     out = []
#     l = len(labels)
#     i = 0
#     while i < l:
#         out.append(labels[i])
#         last_val = labels[i]
#         dist = min(maxdist, l - i - 1)
#         j = 1
#         while (i + j < l and labels[i + j] == last_val) or (j < dist):
#             j += 1
#         i += j
#     return out


# def roll_sequences(sequences, maxlen, step, outlen, end='ignore'):
#     ''' Rolling sequences for generative RNN, for every sequence
#     of length=`maxlen` generate a small sequenc length=`outlen`, then move
#     the sequence by a number of `step` to get the next pair of (input, output)

#     Parameters
#     ----------
#     end : 'ignore', 'pad'(int)
#         ignore: just ignore the border of sequences
#         pad(int): pad given value
#     '''
#     if end == 'ignore':
#         pass
#     elif end == 'pad' or isinstance(end, (float, int, long)):
#         if end == 'pad':
#             end = 0.
#         end = np.cast[sequences.dtype](end)
#         # number of step
#         pad = (sequences.shape[0] - (maxlen + outlen)) / step
#         # ceil then multiply back => desire size for full sequence
#         pad = math.ceil(pad) * step + (maxlen + outlen)
#         pad = int(pad - sequences.shape[0])
#         if pad > 0:
#             pad = np.zeros((pad,) + sequences.shape[1:]) + pad
#             sequences = np.concatenate((sequences, pad), axis = 0)
#     # reupdate n value
#     n = int(math.ceil((sequences.shape[0] - (maxlen + outlen) + 1) / step))
#     rvalX = []
#     rvaly = []
#     for i in range(n):
#         start = i * step
#         end = start + maxlen
#         end_out = end + outlen
#         rvalX.append(sequences[start:end])
#         rvaly.append(sequences[end:end_out])
#     return np.asarray(rvalX), np.asarray(rvaly)
