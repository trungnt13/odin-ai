# ===========================================================================
# Class handle extreme large numpy ndarray
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import re
import marshal
from math import ceil
from six import add_metaclass
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod, abstractproperty
from six.moves import range, zip, zip_longest
from collections import OrderedDict, defaultdict

import numpy as np
from numpy import ndarray

from odin.utils.decorators import autoattr
from odin.utils import (struct, as_tuple, cache_memory, is_string,
                        is_number, is_primitives, axis_normalize)

__all__ = [
  'as_data',
  'open_hdf5',
  'close_all_hdf5',
  'get_all_hdf_dataset',

  'Data',
  'DataConcat',
  'DataGroup',
  'DataCopy',
  'NdarrayData',
  'MmapData',
  'Hdf5Data',
  'DataIterator',
]

# maximum batch size in bytes when performing apply
MAX_BUFFER_SIZE = 100 * 1024 * 1024 # 100 MB


# ===========================================================================
# Helper function
# ===========================================================================
def as_data(x, copy=False):
  """ make sure x is instance Data
  Parameters
  ----------
  copy : bool
    if True, copy the Data object to a separated instance
    to prevent further unexpected change to the configuration
  """
  if isinstance(x, Data):
    if copy:
      if 'odin.fuel.feeder.Feeder' in str(type(x)):
        raise ValueError("Cannot copy `Feeder` data type.")
      return DataCopy(x)
    else:
      return x
  if isinstance(x, np.ndarray):
    return NdarrayData(x)
  if isinstance(x, (tuple, list)):
    if is_primitives(x[0], inc_ndarray=False): # given value for array
      return NdarrayData(np.asarray(x))
    return DataGroup(data=x)
  raise ValueError('Cannot create Data object from given object:{}'.format(x))


def _get_chunk_size(shape, size):
  if isinstance(size, int):
    return (2**int(np.ceil(np.log2(size))),) + shape[1:]
  elif size is None:
    return False
  return True


def _validate_operate_axis(axis):
  ''' as we iterate over first dimension, it is prerequisite to
  have 0 in the axis of operator
  '''
  if not isinstance(axis, (tuple, list)):
    axis = [axis]
  axis = tuple(int(i) for i in axis)
  if 0 not in axis:
    raise ValueError('Expect 0 in the operating axis because we always'
                     ' iterate data over the first dimension.')
  return axis


# x can be percantage or number of samples
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)


# ===========================================================================
# Data
# ===========================================================================
@add_metaclass(ABCMeta)
class Data(object):

  """ Note for overriding `Data` class:
  `_data` will always be a tuple.
  """

  def __init__(self, data, read_only):
    # batch information
    self._batch_size = 256
    self._start = 0.
    self._end = 1.
    self._seed = None
    self._shuffle_level = 0
    # ====== main data ====== #
    # object that have shape, dtype ...
    self._data = as_tuple(data)
    if isinstance(data, (tuple, list)):
      self._is_data_list = True
    else:
      self._is_data_list = False
    self._read_only = bool(read_only)
    # ====== special flags ====== #
    # to detect if cPickle called with protocol >= 2
    self._new_args_called = False
    # flag show that array valued changed
    self._status = 0

  def resize(self, new_length):
    raise NotImplementedError

  # ==================== abstract ==================== #
  @abstractmethod
  def _restore_data(self, info):
    raise NotImplementedError

  @abstractproperty
  def data_info(self):
    """This property return info for restore data object after
    pickling.

    NOTE
    ----
    The return values from this function will be introduce as
    input argument to `_restore_data`
    """
    raise NotImplementedError

  # ==================== basic ==================== #
  def flush(self):
    pass

  def close(self):
    pass

  # ==================== basic properties ==================== #
  @property
  def is_data_list(self):
    """ Return whether this Data was created as a list of given Data
    or just a single Data.
    This will affects the return result of `shape` (i.e.
    whether it should be a list of shape tuple, or just single shape)
    """
    return self._is_data_list

  @property
  def read_only(self):
    return self._read_only

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def data(self):
    return self._data

  @property
  def array(self):
    """Convert any data manipulated by this object to
    numpy.ndarray"""
    a = tuple([dat[:] for dat in self._data])
    return a if self.is_data_list else a[0]

  def tolist(self):
    a = tuple([dat.tolist() for dat in self._data])
    return a if self.is_data_list else a[0]

  # ==================== For pickling ==================== #
  @property
  def new_args(self):
    """ This positional arguments will be presented during
    unpickling. """
    return ()

  def __getstate__(self):
    if not self._new_args_called:
      raise RuntimeError(
          "You must use argument `protocol=cPickle.HIGHEST_PROTOCOL` "
          "when using `pickle` or `cPickle` to be able pickling Data.")
    self._new_args_called = False
    return (self._batch_size, self._start, self._end,
            self._seed, self._shuffle_level,
            self._is_data_list, self.data_info,
            self._read_only)

  def __setstate__(self, states):
    (self._batch_size, self._start, self._end,
        self._seed, self._shuffle_level,
        self._is_data_list, data_info,
        self._read_only) = states
    self._status = 0
    self._new_args_called = False
    # ====== restore data ====== #
    self._restore_data(data_info)
    if not hasattr(self, '_data') or self._data is None:
      raise RuntimeError("The `_data` attribute is None, and have not "
                         "been restored after pickling, you must properly "
                         "implement function `restore_data` for class '%s'"
                         % self.__class__.__name__)

  def __getnewargs__(self):
    self._new_args_called = True
    return as_tuple(self.new_args)

  # ==================== properties ==================== #
  @property
  def ndim(self):
    if self.is_data_list:
      return tuple([len(i) for i in self.shape])
    return len(self.shape)

  @property
  def shape(self):
    s = tuple([i.shape for i in self._data])
    return s if self.is_data_list else s[0]

  @property
  def iter_len(self):
    length = self.__len__()
    start = _apply_approx(length, self._start)
    end = _apply_approx(length, self._end)
    return end - start

  def __len__(self):
    """ len always return 1 number """
    shape = self.shape
    if is_number(shape[0]):
      return shape[0]
    return self.shape[0][0]

  @property
  def T(self):
    x = tuple([i.T for i in self._data])
    return x if self.is_data_list else x[0]

  @property
  def dtype(self):
    x = tuple([np.dtype(i.dtype) for i in self._data])
    return x if self.is_data_list else x[0]

  @contextmanager
  def set_batch_context(self, batch_size=None, seed=-1, start=None, end=None,
                        shuffle_level=None):
    _batch_size = self._batch_size
    _seed = self._seed
    _start = self._start
    _end = self._end
    _shuffle_level = self._shuffle_level
    # temporary batch configuration
    self.set_batch(batch_size=_batch_size if batch_size is None else batch_size,
        seed=_seed if seed == -1 else seed,
        start=_start if start is None else start,
        end=_end if end is None else end,
        shuffle_level=_shuffle_level if shuffle_level is None else shuffle_level)
    yield self
    # reset batch to original batch configuration
    self.set_batch(batch_size=_batch_size, seed=_seed,
                   start=_start, end=_end,
                   shuffle_level=_shuffle_level)

  def set_batch(self, batch_size=None, seed=-1, start=None, end=None,
                shuffle_level=None):
    """
    Parameters
    ----------
    batch_size: int
        size of each batch return when iterate this Data
    seed: None, int
        if None, no shuffling is performed while iterating,
        if < 0, do not change the current seed
        if >= 0, enable randomization with given seed
    start: int, float
        if int, start indicates the index of starting data points to
        iterate. If float, start is the percentage of data to start.
    end: int, float
        ending point of the interation
    shuffle_level: int
        0: only shuffle the order of each batch
        1: shuffle the order of batches and inside each batch as well.
        2: includes level 0 and 1, and custom shuffling (strongest form)
    """
    if isinstance(batch_size, int) and batch_size > 0:
      self._batch_size = batch_size
    if seed is None or seed >= 0:
      self._seed = seed
    if start is not None and start > 0. - 1e-12:
      self._start = start
    if end is not None and end > 0. - 1e-12:
      self._end = end
    if shuffle_level is not None:
      self._shuffle_level = min(max(int(shuffle_level), 0), 2)
    return self

  # ==================== Slicing methods ==================== #
  def __getitem__(self, key):
    x = tuple([dat.__getitem__(key) for dat in self.data])
    return x if self.is_data_list else x[0]

  @autoattr(_status=lambda x: x + 1)
  def __setitem__(self, x, y):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode.")
    # perform
    for dat in self._data:
      dat.__setitem__(x, y)
    return self

  # ==================== iteration ==================== #
  def __iter__(self):
    def create_iteration():
      batch_size = int(self._batch_size)
      length = self.__len__()
      seed = self._seed
      shuffle_level = int(self._shuffle_level)
      self._seed = None
      # custom batch_size
      start = _apply_approx(length, self._start)
      end = _apply_approx(length, self._end)
      if not (0 <= start < end <= length):
        raise ValueError('`start`={} or `end`={} excess `data_size`={}'
                         ''.format(start, end, length))
      # ====== create batch ====== #
      idx = list(range(start, end, batch_size))
      if idx[-1] < end:
        idx.append(end)
      idx = list(zip(idx, idx[1:]))
      # ====== shuffling the batch ====== #
      if seed is not None:
        rand = np.random.RandomState(seed=seed)
        rand.shuffle(idx)
      else:
        rand = None
      # this dummy return to make everything initialized
      yield None
      # ====== start the iteration ====== #
      for start, end in idx:
        # [i[start:end] for i in self._data]
        x = self.__getitem__(slice(start, end))
        # shuffle with higher level
        if shuffle_level > 0 and rand is not None:
          permu = rand.permutation(end - start)
          x = tuple([i[permu] for i in x]) \
              if isinstance(x, (tuple, list)) else x[permu]
        yield x
    # ====== create, init, and return the iteration ====== #
    it = create_iteration()
    next(it)
    return it

  # ==================== Strings ==================== #
  def __str__(self):
    s = "<%s: " % self.__class__.__name__
    for dat in self.data:
      s += "(%s dtype:%s shape:%s)" % \
          (dat.__class__.__name__, dat.dtype.name, dat.shape)
    s += '>'
    return s

  def __repr__(self):
    return self.__str__()

  # ==================== manipulation ==================== #
  @autoattr(_status=lambda x: x + 1)
  def append(self, *arrays):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode")
    accepted_arrays = []
    add_size = 0
    # ====== check if shape[1:] matching ====== #
    for a, d in zip(arrays, self._data):
      if hasattr(a, 'shape'):
        if a.shape[1:] == d.shape[1:]:
          accepted_arrays.append(a)
          add_size += a.shape[0]
      else:
        accepted_arrays.append(None)
    # ====== resize ====== #
    old_size = self.__len__()
    # special case, Mmap is init with temporary size = 1 (all zeros),
    # NOTE: risky to calculate sum of big array here
    if old_size == 1 and \
    sum(np.sum(np.abs(d[:])) for d in self._data) == 0.:
      old_size = 0
    # resize and append data
    self.resize(old_size + add_size) # resize only once will be faster
    # ====== update values ====== #
    for a, d in zip(accepted_arrays, self._data):
      if a is not None:
        d[old_size:old_size + a.shape[0]] = a
    return self

  # ==================== high-level operators ==================== #
  @cache_memory('_status')
  def apply(self, f, f_merge=lambda x: x):
    """ Iteratively execute a list of function `f` on the Data.

    BigData instance store large dataset that need to be iterate
    over to perform any operators.

    Parameters
    ----------
    f: call-able
        a function applied on each returned batch
    f_merge: call-able
        a function applied on all batch results
    """
    # ====== validate arguments ====== #
    if not hasattr(f, '__call__'):
      raise ValueError("`f` must be call-able function.")
    if not hasattr(f_merge, '__call__'):
      raise ValueError("`f_merge` must be call-able function.")
    # ====== Processing ====== #
    results = []
    old_batch_size = self._batch_size
    old_seed = self._seed
    old_start = self._start
    old_end = self._end
    bs = MAX_BUFFER_SIZE / sum(np.prod(i.shape[1:]) * i.dtype.itemsize
                               for i in self._data)
    # less than million data points, not a big deal
    for X in self.set_batch(batch_size=bs, start=0., end=1., seed=None):
      results.append(f(X))
    self.set_batch(batch_size=old_batch_size, start=old_start, end=old_end,
                   seed=old_seed)
    # ====== reset and return ====== #
    return f_merge(results)

  @cache_memory('_status')
  def sum(self, axis=None):
    if axis is None or 0 in as_tuple(axis, t=int):
      y = self.apply(
          f=lambda x: [i.sum(axis=axis)
                       for i in as_tuple(x)],
          f_merge=lambda x: [sum(j[i] for j in x)
                             for i in range(len(x[0]))])
    else:
      y = [i.sum(axis=axis)
           for i in self._data]
    return y if self.is_data_list else y[0]

  @cache_memory('_status')
  def cumsum(self, axis=None):
    x = [i.cumsum(axis) for i in self._data]
    return x if self.is_data_list else x[0]

  @cache_memory('_status')
  def sum2(self, axis=None):
    if axis is None or 0 in as_tuple(axis, t=int):
      y = self.apply(
          f=lambda x: [i.__pow__(2).sum(axis=axis)
                       for i in as_tuple(x)],
          f_merge=lambda x: [sum(j[i] for j in x)
                             for i in range(len(x[0]))])
    else:
      y = [i.__pow__(2).sum(axis=axis)
           for i in self._data]
    return y if self.is_data_list else y[0]

  @cache_memory('_status')
  def pow(self, y):
    x = [i.__pow__(y) for i in self._data]
    return x if self.is_data_list else x[0]

  @cache_memory('_status')
  def min(self, axis=None):
    if axis is None or 0 in as_tuple(axis, t=int):
      y = self.apply(
          f=lambda x: [i.min(axis=axis)
                       for i in as_tuple(x)],
          f_merge=lambda x: [np.min([j[i] for j in x], axis=0)
                             for i in range(len(x[0]))])
    else:
      y = [i.min(axis=axis)
           for i in self._data]
    return y if self.is_data_list else y[0]

  @cache_memory('_status')
  def argmin(self, axis=None):
    x = [i.argmin(axis) for i in self._data]
    return x if self.is_data_list else x[0]

  @cache_memory('_status')
  def max(self, axis=None):
    if axis is None or 0 in as_tuple(axis, t=int):
      y = self.apply(
          f=lambda x: [i.max(axis=axis)
                       for i in as_tuple(x)],
          f_merge=lambda x: [np.max([j[i] for j in x], axis=0)
                             for i in range(len(x[0]))])
    else:
      y = [i.max(axis=axis)
           for i in self._data]
    return y if self.is_data_list else y[0]

  @cache_memory('_status')
  def argmax(self, axis=None):
    x = [i.argmax(axis) for i in self._data]
    return x if self.is_data_list else x[0]

  # ==================== probs and stats ==================== #
  @cache_memory('_status')
  def mean(self, axis=None):
    sum1 = as_tuple(self.sum(axis))
    results = []
    for s1, dat in zip(sum1, self._data):
      shape = dat.shape
      if axis is None:
        n = np.prod(shape)
      else:
        n = np.prod([shape[i] for i in as_tuple(axis, t=int)])
      results.append(s1 / n)
    return results if self.is_data_list else results[0]

  @cache_memory('_status')
  def var(self, axis=None):
    sum1 = as_tuple(self.sum(axis))
    sum2 = as_tuple(self.sum2(axis))

    results = []
    for s1, s2, dat in zip(sum1, sum2, self._data):
      shape = dat.shape
      if axis is None:
        n = np.prod(shape)
      else:
        n = np.prod([shape[i] for i in as_tuple(axis, t=int)])
      results.append((s2 - np.power(s1, 2) / n) / n)
    return results if self.is_data_list else results[0]

  @cache_memory('_status')
  def std(self, axis=None):
    s = [np.sqrt(v) for v in as_tuple(self.var(axis))]
    return s if self.is_data_list else s[0]

  # ==================== low-level operator ==================== #
  def __add__(self, y):
    x = [i.__add__(y) for i in self._data]
    return x if self.is_data_list else x[0]

  def __sub__(self, y):
    x = [i.__sub__(y) for i in self._data]
    return x if self.is_data_list else x[0]

  def __mul__(self, y):
    x = [i.__mul__(y) for i in self._data]
    return x if self.is_data_list else x[0]

  def __div__(self, y):
    x = [i.__div__(y) for i in self._data]
    return x if self.is_data_list else x[0]

  def __floordiv__(self, y):
    x = [i.__floordiv__(y) for i in self._data]
    return x if self.is_data_list else x[0]

  def __pow__(self, y):
    x = [i.__pow__(y) for i in self._data]
    return x if self.is_data_list else x[0]

  def __neg__(self):
    x = [i.__neg__() for i in self._data]
    return x if self.is_data_list else x[0]

  def __pos__(self):
    x = [i.__pos__() for i in self._data]
    return x if self.is_data_list else x[0]

  # ==================== update function ==================== #
  @autoattr(_status=lambda x: x + 1)
  def __iadd__(self, y):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode")
    for i in self._data:
      i.__iadd__(y)

  @autoattr(_status=lambda x: x + 1)
  def __isub__(self, y):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode")
    for i in self._data:
      i.__isub__(y)

  @autoattr(_status=lambda x: x + 1)
  def __imul__(self, y):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode")
    for i in self._data:
      i.__imul__(y)

  @autoattr(_status=lambda x: x + 1)
  def __idiv__(self, y):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode")
    for i in self._data:
      i.__idiv__(y)

  @autoattr(_status=lambda x: x + 1)
  def __ifloordiv__(self, y):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode")
    for i in self._data:
      i.__ifloordiv__(y)

  @autoattr(_status=lambda x: x + 1)
  def __ipow__(self, y):
    if self.read_only:
      raise RuntimeError("This Data is set in read-only mode")
    for i in self._data:
      i.__ipow__(y)

# ===========================================================================
# Utils
# ===========================================================================
class DataConcat(Data):
  """ Concatenate two Data while iterating over them
  """

  def __init__(self, data, axis=-1):
    data = as_tuple(data)
    if len(data) < 2:
      raise ValueError("2 or more Data must be given to `DataConcat`")
    if axis == 0:
      raise ValueError("Cannot concatenate axis=0")
    if len(set(d.ndim for d in data)) > 2:
      raise ValueError("All Data must have the same number of dimension (i.e. `ndim`)")
    if len(set(d.shape[0] for d in data)) > 2:
      raise ValueError("All Data must have the same length (i.e. first dimension)")
    super(DataConcat, self).__init__(data, read_only=True)
    self._is_data_list = False
    self._axis = axis_normalize(int(axis), ndim=data[0].ndim)

  @property
  def data_info(self):
    return self._data, self._axis

  def _restore_data(self, info):
    self._data, self._axis = info

  @property
  def array(self):
    """Convert any data manipulated by this object to
    numpy.ndarray"""
    return np.concatenate([dat[:] for dat in self._data],
                          axis=self._axis)

  def tolist(self):
    return self.array.tolist()

  @property
  def shape(self):
    all_shapes = [d.shape for d in self.data]
    new_shape = [all_shapes[0][0]]
    ndim = len(all_shapes[0])
    for i in range(1, ndim):
      if i == self._axis:
        new_shape.append(sum(s[i] for s in all_shapes))
      else:
        new_shape.append(all_shapes[0][i])
    return tuple(new_shape)

  def __getitem__(self, key):
    return np.concatenate([d.__getitem__(key) for d in self.data],
                          axis=self._axis)

class DataGroup(Data):
  """ Group serveral Data together and return each Data
  separately during iteration

  This class is always considered as list of Data
  """

  def __init__(self, data):
    for d in as_tuple(data):
      if not isinstance(d, (np.ndarray, Data)):
        raise ValueError('`data` can only be instance of numpy.ndarray or '
                         'odin.fuel.data.Data, but given type: %s' % str(type(d)))
    data = [NdarrayData(d) if isinstance(d, np.ndarray) else d
            for d in as_tuple(data)]
    if len(set(len(d) for d in data)) > 1:
      raise ValueError("All data in given data list must have the same length, "
                       "but given: %s" % str([len(d) for d in data]))
    super(DataGroup, self).__init__(data, read_only=True)

  @property
  def data_info(self):
    return self._data

  def _restore_data(self, info):
    self._data = info

class DataCopy(Data):
  """ Simple copy that contain original version of Data
  """

  def __init__(self, data):
    if not isinstance(data, Data):
      raise ValueError("`data` must be instance of odin.fuel.data.Data but "
                       "given type: %s" % str(type(data)))
    # ====== special case DataGroup ====== #
    if isinstance(data, DataGroup):
      super(DataCopy, self).__init__(data._data, read_only=True)
    else:
      super(DataCopy, self).__init__(data, read_only=True)
    # ====== copy information ====== #
    self._is_data_list = data._is_data_list
    self._batch_size = data._batch_size
    self._start = data._start
    self._end = data._end
    self._seed = data._seed
    self._shuffle_level = data._shuffle_level

  @property
  def data_info(self):
    return self._data

  def _restore_data(self, info):
    self._data = info

class NdarrayData(Data):
  """ Simple wrapper for `numpy.ndarray` """

  def __init__(self, array):
    if not isinstance(array, np.ndarray):
      raise ValueError('array must be instance of numpy ndarray')
    super(NdarrayData, self).__init__(array, read_only=False)

  @property
  def data_info(self):
    return self._data

  def _restore_data(self, info):
    self._data = info

  # ==================== abstract ==================== #
  def resize(self, new_length):
    shape = self._data[0].shape
    new_shape = (new_length,) + shape[1:]
    return self._data[0].resize(new_shape)

# ===========================================================================
# Memmap Data object
# ===========================================================================
MAX_OPEN_MMAP = 120

def _aligned_memmap_offset(dtype):
  header_size = len(MmapData.HEADER) + 8 + MmapData.MAXIMUM_HEADER_SIZE
  type_size = np.dtype(dtype).itemsize
  n = np.ceil(header_size / type_size)
  return int(n * type_size)

class MmapData(Data):

  """Create a memory-map to an array stored in a *binary* file on disk.

  Memory-mapped files are used for accessing small segments of large files
  on disk, without reading the entire file into memory.  Numpy's
  memmap's are array-like objects.  This differs from Python's ``mmap``
  module, which uses file-like objects.

  Parameters
  ----------
  path : str
      The file name or file object to be used as the array data buffer.
  dtype : data-type, optional
      The data-type used to interpret the file contents.
      Default is `uint8`.
  shape : tuple, optional
      The desired shape of the array. If ``mode == 'r'`` and the number
      of remaining bytes after `offset` is not a multiple of the byte-size
      of `dtype`, you must specify `shape`. By default, the returned array
      will be 1-D with the number of elements determined by file size
      and data-type.

  Note
  ----
  This class always read MmapData with mode=r+
  """
  _INSTANCES = OrderedDict()
  HEADER = b'mmapdata'
  MAXIMUM_HEADER_SIZE = 486

  @staticmethod
  def read_header(path, read_only, return_file):
    """ return: dtype, shape
    Necessary information to create numpy.memmap
    """
    f = open(path, mode='rb' if read_only else 'rb+')
    # ====== check header signature ====== #
    try:
      if f.read(len(MmapData.HEADER)) != MmapData.HEADER:
        raise Exception
    except Exception as e:
      f.close()
      raise Exception('Invalid header for MmapData.')
    # ====== 8 bytes for size of info ====== #
    try:
      size = int(f.read(8))
      dtype, shape = marshal.loads(f.read(size))
    except Exception as e:
      f.close()
      raise Exception('Error reading memmap data file: %s' % str(e))
    # ====== return file object ====== #
    if return_file:
      return dtype, shape, f
    f.close()
    return dtype, shape

  def __new__(clazz, *args, **kwargs):
    path = kwargs.get('path', None)
    if path is None:
      path = args[0]
    if not is_string(path):
      raise ValueError("`path` for MmapData must be string, but given "
                       "object with type: %s" % type(path))
    path = os.path.abspath(path)
    # Found old instance
    if path in MmapData._INSTANCES:
      return MmapData._INSTANCES[path]
    # new MmapData
    # ====== increase memmap count ====== #
    if len(MmapData._INSTANCES) + 1 > MAX_OPEN_MMAP:
      raise ValueError('Only allowed to open maximum of {} memmap file'.format(MAX_OPEN_MMAP))
    # ====== create new instance ====== #
    new_instance = super(MmapData, clazz).__new__(clazz)
    MmapData._INSTANCES[path] = new_instance
    return new_instance

  def __init__(self, path, dtype='float32', shape=None,
               read_only=False):
    # validate path
    path = os.path.abspath(path)
    # ====== check shape info ====== #
    if shape is not None:
      if not isinstance(shape, (tuple, list, np.ndarray)):
        shape = (shape,)
      shape = tuple([0 if i is None or i < 0 else i for i in shape])
    # ====== read exist file ====== #
    if os.path.exists(path):
      dtype, shape, f = MmapData.read_header(path,
                                             read_only=read_only,
                                             return_file=True)
    # ====== create new file ====== #
    else:
      if dtype is None or shape is None:
        raise Exception("First created this MmapData, `dtype` and "
                        "`shape` must NOT be None.")
      f = open(path, 'wb+')
      f.write(MmapData.HEADER)
      dtype = str(np.dtype(dtype))
      if isinstance(shape, np.ndarray):
        shape = shape.tolist()
      if not isinstance(shape, (tuple, list)):
        shape = (shape,)
      _ = marshal.dumps([dtype, shape])
      size = len(_)
      if size > MmapData.MAXIMUM_HEADER_SIZE:
        raise Exception('The size of header excess maximum allowed size '
                        '(%d bytes).' % MmapData.MAXIMUM_HEADER_SIZE)
      size = '%8d' % size
      f.write(size.encode())
      f.write(_)
    # ====== assign attributes ====== #
    self._file = f
    self._path = path
    data = np.memmap(f, dtype=dtype, shape=shape,
                     mode = 'r' if read_only else 'r+',
                     offset=_aligned_memmap_offset(dtype))
    # finally call super initialize
    super(MmapData, self).__init__(data=data, read_only=read_only)

  @property
  def data_info(self):
    return self._path

  @property
  def path(self):
    return self._path

  @property
  def new_args(self):
    return (self._path,)

  def _restore_data(self, info):
    # info here is the path
    dtype, shape, f = MmapData.read_header(info, mode='r+',
                                           return_file=True)
    self._file = f
    self._data = np.memmap(f, dtype=dtype, shape=shape, mode='r+',
                           offset=_aligned_memmap_offset(dtype))
    self._path = info

  def flush(self):
    if self.read_only:
      return
    self._data[0].flush()

  def close(self):
    # Check if exist global instance
    if self.data_info in MmapData._INSTANCES:
      del MmapData._INSTANCES[self.data_info]
      # flush in read-write mode
      if not self.read_only:
        self.flush()
      # close mmap and file
      self._data[0]._mmap.close()
      del self._data
      self._file.close()

  # ==================== properties ==================== #
  def __str__(self):
    return '<MMAP "%s": %s "%s">' % \
    (self.data_info, self.shape, self.dtype)

  # ==================== Save ==================== #
  def resize(self, new_length):
    if self.read_only:
      return
    # ====== local files ====== #
    f = self._file
    mmap = self._data[0]
    old_length = mmap.shape[0]
    # ====== check new shape ====== #
    if new_length < old_length:
      raise ValueError('Only support extend memmap, and do not shrink the memory')
    # nothing to resize
    elif new_length == old_length:
      return self
    # ====== flush previous changes ====== #
    # resize by create new memmap and also rename old file
    shape = (new_length,) + mmap.shape[1:]
    dtype = mmap.dtype.name
    # rewrite the header
    f.seek(len(MmapData.HEADER))
    meta = marshal.dumps([dtype, shape])
    size = '%8d' % len(meta)
    f.write(size.encode(encoding='utf-8'))
    f.write(meta)
    f.flush()
    # extend the memmap
    mmap._mmap.close()
    del self._data
    mmap = np.memmap(self.data_info, dtype=dtype, shape=shape,
                     mode='r+',
                     offset=_aligned_memmap_offset(dtype))
    self._data = (mmap,)
    return self

# ===========================================================================
# Hdf5 Data object
# ===========================================================================
try:
  import h5py
except:
  pass


def get_all_hdf_dataset(hdf, fileter_func=None, path='/'):
  res = []
  # init queue
  q = queue()
  for i in hdf[path].keys():
    q.put(i)
  # get list of all file
  while not q.empty():
    p = q.pop()
    if 'Dataset' in str(type(hdf[p])):
      if fileter_func is not None and not fileter_func(p):
        continue
      res.append(p)
    elif 'Group' in str(type(hdf[p])):
      for i in hdf[p].keys():
        q.put(p + '/' + i)
  return res


_HDF5 = {}


def open_hdf5(path, read_only=False):
  '''
  Parameters
  ----------
  mode : one of the following options
      +------------------------------------------------------------+
      |r        | Readonly, file must exist                        |
      +------------------------------------------------------------+
      |r+       | Read/write, file must exist                      |
      +------------------------------------------------------------+
      |w        | Create file, truncate if exists                  |
      +------------------------------------------------------------+
      |w- or x  | Create file, fail if exists                      |
      +------------------------------------------------------------+
      |a        | Read/write if exists, create otherwise (default) |
      +------------------------------------------------------------+

  check : bool
      if enable, only return openned files, otherwise, None

  Note
  ----
  If given file already open in read mode, mode = 'w' will cause error
  (this is good error and you should avoid this situation)

  '''
  key = os.path.abspath(path)
  mode = 'r' if read_only else 'a'

  if key in _HDF5:
    f = _HDF5[key]
    if 'Closed' in str(f):
      f = h5py.File(path, mode=mode)
  else:
    f = h5py.File(path, mode=mode)
    _HDF5[key] = f
  return f


def close_all_hdf5():
  import gc
  for obj in gc.get_objects():   # Browse through ALL objects
    if isinstance(obj, h5py.File):   # Just HDF5 files
      try:
        obj.close()
      except:
        pass # Was already closed


class Hdf5Data(Data):

  def __init__(self, dataset, hdf=None, dtype=None, shape=None):
    super(Hdf5Data, self).__init__()
    raise Exception("Hdf5Data is under-maintanance!")
    # default chunks size is 32 (reduce complexity of the works)
    self._chunk_size = 32
    if isinstance(hdf, str):
      hdf = open_hdf5(hdf)
    if hdf is None and not isinstance(dataset, h5py.Dataset):
      raise ValueError('Cannot initialize dataset without hdf file')

    if isinstance(dataset, h5py.Dataset):
      self._data = dataset
      self._hdf = dataset.file
    else:
      if dataset not in hdf: # not created dataset
        if dtype is None or shape is None:
          raise ValueError('dtype and shape must be specified if '
                           'dataset has not created in hdf5 file.')
        shape = tuple([0 if i is None else i for i in shape])
        hdf.create_dataset(dataset, dtype=dtype,
            chunks=_get_chunk_size(shape, self._chunk_size),
            shape=shape, maxshape=(None, ) + shape[1:])

      self._data = hdf[dataset]
      if shape is not None and self._data[0].shape[1:] != shape[1:]:
        raise ValueError('Shape mismatch between predefined dataset '
                         'and given shape, {} != {}'
                         ''.format(shape, self._data[0].shape))
      self._hdf = hdf

  # ==================== properties ==================== #
  @property
  def path(self):
    return self._hdf.filename

  @property
  def name(self):
    _ = self._data[0].name
    if _[0] == '/':
      _ = _[1:]
    return _

  @property
  def hdf5(self):
    return self._hdf

  # ==================== Save ==================== #
  def resize(self, shape):
    if self._hdf.mode == 'r':
      return

    if not isinstance(shape, (tuple, list)):
      shape = (shape,)
    if any(i != j for i, j in zip(shape[1:], self._data[0].shape[1:])):
      raise ValueError('Resize only support the first dimension, but '
                       '{} != {}'.format(shape[1:], self._data[0].shape[1:]))
    if shape[0] < self._data[0].shape[0]:
      raise ValueError('Only support extend memmap, and do not shrink the memory')
    elif shape[0] == self._data[0].shape[0]:
      return self

    self._data[0].resize(shape[0], axis=0)
    return self

  def flush(self):
    try:
      if self._hdf.mode == 'r':
        return
      self._hdf.flush()
    except Exception:
      pass

  def close(self):
    try:
      self._hdf.close()
    except Exception:
      pass


# ===========================================================================
# data iterator
# ===========================================================================
def _approximate_continuos_by_discrete(distribution):
  '''original distribution: [ 0.47619048  0.38095238  0.14285714]
     best approximated: [ 5.  4.  2.]
  '''
  if len(distribution) == 1:
    return distribution

  inv_distribution = 1 - distribution
  x = np.round(1 / inv_distribution)
  x = np.where(distribution == 0, 0, x)
  return x.astype(int)


class DataIterator(Data):

  ''' Vertically merge several data object for iteration
  '''

  def __init__(self, data):
    super(DataIterator, self).__init__()
    if not isinstance(data, (tuple, list)):
      data = (data,)
    # ====== validate args ====== #
    if any(not isinstance(i, Data) for i in data):
      raise ValueError('data must be instance of MmapData or Hdf5Data, '
                       'but given data have types: {}'
                       ''.format(map(lambda x: str(type(x)).split("'")[1],
                                    data)))
    shape = data[0].shape[1:]
    if any(i.shape[1:] != shape for i in data):
      raise ValueError('all data must have the same trial dimension, but'
                       'given shape of all data as following: {}'
                       ''.format([i.shape for i in data]))
    # ====== defaults parameters ====== #
    self._data = data
    self._sequential = False
    self._distribution = [1.] * len(data)

  # ==================== properties ==================== #
  @property
  def shape(self):
    return (len(self),) + self._data[0].shape[1:]

  @property
  def array(self):
    start = self._start
    end = self._end
    idx = [(_apply_approx(i.shape[0], start), _apply_approx(i.shape[0], end))
           for i in self._data]
    idx = [(j[0], int(round(j[0] + i * (j[1] - j[0]))))
           for i, j in zip(self._distribution, idx)]
    return np.vstack([i[j[0]:j[1]] for i, j in zip(self._data, idx)])

  def __len__(self):
    start = self._start
    end = self._end
    return sum(round(i * (_apply_approx(j.shape[0], end) - _apply_approx(j.shape[0], start)))
               for i, j in zip(self._distribution, self._data))

  @property
  def distribution(self):
    return self._distribution

  def __str__(self):
    s = ['====== Iterator: ======']
    # ====== Find longest string ====== #
    longest_name = 0
    longest_shape = 0
    for d in self._data:
      name = d.name
      dtype = d.dtype
      shape = d.shape
      longest_name = max(len(name), longest_name)
      longest_shape = max(len(str(shape)), longest_shape)
    # ====== return print string ====== #
    format_str = ('Name:%-' + str(longest_name) + 's  '
                  'dtype:%-7s  '
                  'shape:%-' + str(longest_shape) + 's  ')
    for d in self._data:
      name = d.name
      dtype = d.dtype
      shape = d.shape
      s.append(format_str % (name, dtype, shape))
    # ====== batch configuration ====== #
    s.append('Batch: %d' % self._batch_size)
    s.append('Sequential: %r' % self._sequential)
    s.append('Distibution: %s' % str(self._distribution))
    s.append('Seed: %s' % str(self._seed))
    s.append('Range: [%.2f, %.2f]' % (self._start, self._end))
    return '\n'.join(s)

  def __repr__(self):
    return self.__str__()

  # ==================== batch configuration ==================== #
  def set_mode(self, distribution=None, sequential=None):
    '''
    Parameters
    ----------
    distribution : str, list or float
        'up', 'over': over-sampling all Data
        'down', 'under': under-sampling all Data
        list: percentage of each Data in the iterator will be iterated
        float: the same percentage for all Data
    sequential : bool
        if True, read each Data one-by-one, otherwise, mix all Data

    '''
    if sequential is not None:
      self._sequential = sequential
    if distribution is not None:
      # upsampling or downsampling
      if isinstance(distribution, str):
        distribution = distribution.lower()
        if 'up' in distribution or 'over' in distribution:
          n = max(i.shape[0] for i in self._data)
        elif 'down' in distribution or 'under' in distribution:
          n = min(i.shape[0] for i in self._data)
        else:
          raise ValueError("Only upsampling (keyword: up, over) "
                           "or undersampling (keyword: down, under) "
                           "are supported.")
        self._distribution = [n / i.shape[0] for i in self._data]
      # real values distribution
      elif isinstance(distribution, (tuple, list)):
        if len(distribution) != len(self._data):
          raise ValueError('length of given distribution must equal '
                           'to number of data in the iterator, but '
                           'len_data={} != len_distribution={}'
                           ''.format(len(self._data), len(self._distribution)))
        self._distribution = distribution
      # all the same value
      elif isinstance(distribution, float):
        self._distribution = [distribution] * len(self._data)
    return self

  # ==================== main logic of batch iterator ==================== #
  def __iter__(self):
    def create_iteration():
      seed = self._seed; self._seed = None
      if seed is not None:
        rng = np.random.RandomState(seed)
      else: # deterministic RandomState
        rng = struct()
        rng.randint = lambda x: None
        rng.permutation = lambda x: slice(None, None)
      # ====== easy access many private variables ====== #
      sequential = self._sequential
      start, end = self._start, self._end
      batch_size = self._batch_size
      distribution = np.asarray(self._distribution)
      # shuffle order of data (good for sequential mode)
      idx = rng.permutation(len(self._data))
      data = self._data[idx] if isinstance(idx, slice) else [self._data[i] for i in idx]
      distribution = distribution[idx]
      shape = [i.shape[0] for i in data]
      # ====== prepare distribution information ====== #
      # number of sample should be traversed
      n = np.asarray([i * (_apply_approx(j, end) - _apply_approx(j, start))
                      for i, j in zip(distribution, shape)])
      n = np.round(n).astype(int)
      # normalize the distribution (base on new sample n of each data)
      distribution = n / n.sum()
      distribution = _approximate_continuos_by_discrete(distribution)
      # somehow heuristic, rescale distribution to get more benifit from cache
      if distribution.sum() <= len(data):
        distribution = distribution * 3
      # distribution now the actual batch size of each data
      distribution = (batch_size * distribution).astype(int)
      assert distribution.sum() % batch_size == 0, 'wrong distribution size!'
      # predefined (start,end) pair of each batch (e.g (0,256), (256,512))
      idx = list(range(0, batch_size + distribution.sum(), batch_size))
      idx = list(zip(idx, idx[1:]))
      # Dummy return to initialize everything
      yield None
      #####################################
      # 1. optimized parallel code.
      if not sequential:
        # first iterators
        it = [iter(dat.set_batch(bs, seed=rng.randint(10e8),
                                 start=start, end=end,
                                 shuffle_level=self._shuffle_level))
              for bs, dat in zip(distribution, data)]
        # iterator
        while sum(n) > 0:
          batch = []
          for i, x in enumerate(it):
            if n[i] <= 0:
              continue
            try:
              x = next(x)[:n[i]]
              n[i] -= x.shape[0]
              batch.append(x)
            except StopIteration: # one iterator stopped
              it[i] = iter(data[i].set_batch(distribution[i],
                  seed=rng.randint(10e8), start=start, end=end,
                  shuffle_level=self._shuffle_level))
              x = next(it[i])[:n[i]]
              n[i] -= x.shape[0]
              batch.append(x)
          # got final batch
          batch = np.vstack(batch)
          # no idea why random permutation is much faster than shuffle
          if self._shuffle_level > 0:
            batch = batch[rng.permutation(batch.shape[0])]
          # return the iterations
          for i, j in idx[:int(ceil(batch.shape[0] / batch_size))]:
            yield batch[i:j]
      #####################################
      # 2. optimized sequential code.
      else:
        # first iterators
        batch_size = distribution.sum()
        it = [iter(dat.set_batch(batch_size, seed=rng.randint(10e8),
                                 start=start, end=end,
                                 shuffle_level=self._shuffle_level))
              for dat in data]
        current_data = 0
        # iterator
        while sum(n) > 0:
          if n[current_data] <= 0:
            current_data += 1
          try:
            x = next(it[current_data])[:n[current_data]]
            n[current_data] -= x.shape[0]
          except StopIteration: # one iterator stopped
            it[current_data] = iter(data[current_data].set_batch(batch_size, seed=rng.randint(10e8),
                                start=start, end=end,
                                shuffle_level=self._shuffle_level))
            x = next(it[current_data])[:n[current_data]]
            n[current_data] -= x.shape[0]
          # shuffle x
          if self._shuffle_level > 0:
            x = x[rng.permutation(x.shape[0])]
          for i, j in idx[:int(ceil(x.shape[0] / self._batch_size))]:
            yield x[i:j]
    # ====== create and return the iteration ====== #
    it = create_iteration()
    next(it)
    return it

  # ==================== Slicing methods ==================== #
  def __getitem__(self, y):
    start = self._start
    end = self._end
    idx = [(_apply_approx(i.shape[0], start), _apply_approx(i.shape[0], end))
           for i in self._data]
    idx = [(j[0], int(round(j[0] + i * (j[1] - j[0]))))
           for i, j in zip(self._distribution, idx)]
    size = np.cumsum([i[1] - i[0] for i in idx])
    if isinstance(y, int):
      idx = _get_closest_id(size, y)
      return self._data[idx][y]
    elif isinstance(y, slice):
      return self.array[y]
    else:
      raise ValueError('No support for indices type={}'.format(type(y)))


def _get_closest_id(size, y):
  idx = 0
  for i, j in enumerate(size):
    if y >= j:
      idx = i + 1
  return idx
