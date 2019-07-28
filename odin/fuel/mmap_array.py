from __future__ import print_function, division, absolute_import

import os
import marshal
from six import string_types
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterable, Text, Tuple, Union, List

import numpy as np

__all__ = [
  'get_total_opened_mmap',
  'read_mmaparray_header',
  'MmapArrayWriter',
  'MmapArray',
]

MAX_OPEN_MMAP = 120
_INSTANCES_WRITER = OrderedDict()
_HEADER = b'mmapdata'
_MAXIMUM_HEADER_SIZE = 486

# ===========================================================================
# Helper
# ===========================================================================
def get_total_opened_mmap():
  return len(_INSTANCES_WRITER)

def read_mmaparray_header(path):
  """
  Parameters
  ----------

  Return
  ------
  dtype, shape
    Necessary information to create numpy.memmap
  """
  with open(path, mode='rb') as f:
    # ====== check header signature ====== #
    try:
      if f.read(len(_HEADER)) != _HEADER:
        raise Exception
    except Exception as e:
      raise Exception('Invalid header for MmapData.')
    # ====== 8 bytes for size of info ====== #
    try:
      size = int(f.read(8))
      dtype, shape = marshal.loads(f.read(size))
    except Exception as e:
      f.close()
      raise Exception('Error reading memmap data file: %s' % str(e))
    # ====== return file object ====== #
    return dtype, shape

def _aligned_memmap_offset(dtype):
  header_size = len(_HEADER) + 8 + _MAXIMUM_HEADER_SIZE
  type_size = np.dtype(dtype).itemsize
  n = np.ceil(header_size / type_size)
  return int(n * type_size)

# ===========================================================================
# Writing new memory-mapped array
# ===========================================================================
class MmapArrayWriter(object):
  """
  Parameters
  ----------
  path : str
    path to a file for writing memory-mapped data

  shape : tuple
    pass

  dtype : numpy.dtype
    data type

  remove_exist : boolean (default=False)
    if file at given path exists, remove it
  """
  def __new__(cls, path, *args, **kwargs):
    path = os.path.abspath(path)
    # Found old instance
    if path in _INSTANCES_WRITER:
      obj = _INSTANCES_WRITER[path]
      if not obj.is_closed:
        return obj
    # ====== increase memmap count ====== #
    if get_total_opened_mmap() > MAX_OPEN_MMAP:
      raise RuntimeError(
        "Only allowed to open maximum of %d memmap file" % MAX_OPEN_MMAP)
    # ====== create new instance ====== #
    new_instance = super(MmapArrayWriter, cls).__new__(cls)
    _INSTANCES_WRITER[path] = new_instance
    return new_instance

  def __init__(self, path: Text, shape: Tuple,
               dtype: Union[Text, np.dtype] = 'float32',
               remove_exist: bool = False):
    super(MmapArrayWriter, self).__init__()
    assert isinstance(path, string_types), "path must be string or text."
    # validate path
    path = os.path.abspath(path)
    # ====== remove exist ====== #
    if remove_exist and os.path.exists(path):
      if os.path.isfile(path):
        os.remove(path)
      else:
        raise RuntimeError(
          "Give path at '%s' is a folder, cannot remove!" % path)
    # ====== check shape info ====== #
    if not isinstance(shape, Iterable):
      shape = (shape,)
    shape = tuple([0 if i is None or i < 0 else int(i)
                   for i in shape])
    # ====== read exist file ====== #
    if os.path.exists(path):
      dtype, shape = read_mmaparray_header(path)
      f = open(path, 'rb+')
    # ====== create new file ====== #
    else:
      if dtype is None or shape is None:
        raise Exception("First created this MmapData, `dtype` and "
                        "`shape` must NOT be None.")
      f = open(path, 'wb+')
      f.write(_HEADER)
      dtype = str(np.dtype(dtype))
      if isinstance(shape, np.ndarray):
        shape = shape.tolist()
      if not isinstance(shape, (tuple, list)):
        shape = (shape,)
      _ = marshal.dumps([dtype, shape])
      size = len(_)
      if size > _MAXIMUM_HEADER_SIZE:
        raise Exception('The size of header excess maximum allowed size '
                        '(%d bytes).' % _MAXIMUM_HEADER_SIZE)
      size = '%8d' % size
      f.write(size.encode())
      f.write(_)
    # ====== assign attributes ====== #
    self._file = f
    self._path = path
    data = np.memmap(f, dtype=dtype, shape=shape,
                     mode='r+', offset=_aligned_memmap_offset(dtype))
    self._data = data
    self._is_closed = False

  @property
  def shape(self):
    return self._data.shape

  @property
  def path(self):
    return self._path

  @property
  def is_closed(self):
    return self._is_closed

  def flush(self):
    self._data.flush()

  def close(self):
    if self.is_closed:
      return
    self._is_closed = True
    if self.path in _INSTANCES_WRITER:
      del _INSTANCES_WRITER[self.path]
    # flush in read-write mode
    self.flush()
    # close mmap and file
    self._data._mmap.close()
    del self._data
    self._file.close()

  def _resize(self, new_length):
    # ====== local files ====== #
    f = self._file
    old_length = self._data.shape[0]
    # ====== check new shape ====== #
    if new_length < old_length:
      raise ValueError(
        'Only support extending memmap, and cannot shrink the memory.')
    # nothing to resize
    elif new_length == old_length:
      return self
    # ====== flush previous changes ====== #
    # resize by create new memmap and also rename old file
    shape = (new_length,) + self._data.shape[1:]
    dtype = self._data.dtype.name
    # rewrite the header, update metadata
    f.seek(len(_HEADER))
    meta = marshal.dumps([dtype, shape])
    size = '%8d' % len(meta)
    f.write(size.encode(encoding='utf-8'))
    f.write(meta)
    f.flush()
    # close old data
    self._data._mmap.close()
    del self._data
    # extend the memmap, by open numpy.memmap with bigger shape
    f = open(self.path, 'rb+')
    self._file = f
    mmap = np.memmap(f, dtype=dtype, shape=shape,
                     mode='r+', offset=_aligned_memmap_offset(dtype))
    self._data = mmap
    return self

  def write(self, arrays: Iterable):
    """ Extending the memory-mapped data and copy the array
    into extended area. """
    if self.is_closed:
      raise RuntimeError("The MmapArrayWriter is closed!")
    # only get arrays matched the shape
    add_size = 0
    if not isinstance(arrays, Iterable) or isinstance(arrays, np.ndarray):
      arrays = (arrays,)
    accepted_arrays = []
    # ====== check if shape[1:] matching ====== #
    for a in arrays:
      if a.shape[1:] == self._data.shape[1:]:
        accepted_arrays.append(a)
        add_size += a.shape[0]
    # no new array to append
    if len(accepted_arrays) == 0:
      raise RuntimeError(
        "No appropriate array found for writing, given: %s, "
        "; but require array with shape: %s" %
        (','.join([str(i.shape) for i in arrays]), self.shape))
    # ====== resize ====== #
    old_size = self._data.shape[0]
    # special case, Mmap is init with temporary size = 1 (all zeros),
    # NOTE: risky to calculate sum of big array here
    if old_size == 1 and \
    sum(np.sum(np.abs(d[:])) for d in self._data) == 0.:
      old_size = 0
    # resize and append data
    # resize only once will be faster
    self._resize(old_size + add_size)
    # ====== update values ====== #
    data = self._data
    for a in accepted_arrays:
      data[old_size:old_size + a.shape[0]] = a
      old_size += a.shape[0]
    return self

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()

  def __del__(self):
    self.close()

# ===========================================================================
# Reading the array
# ===========================================================================
class MmapArray(np.memmap):
  """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  NumPy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.

    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.

    Delete the memmap instance to close the memmap file.

    Parameters
    ----------
    path : str, file-like object, or pathlib.Path instance
        The file name or file object to be used as the array data buffer.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+
        Default is 'r+'.
  """
  def __new__(subtype, path, mode='r+'):
    path = os.path.abspath(path)
    if not os.path.exists(path) and os.path.isfile(path):
      raise ValueError('path must be existed file created by MmapArrayWriter.')
    dtype, shape = read_mmaparray_header(path)
    offset = _aligned_memmap_offset(dtype)
    new_array = super(MmapArray, subtype).__new__(
      subtype=subtype, filename=path, dtype=dtype, mode=mode,
      offset=offset, shape=shape)
    new_array._path = path
    return new_array

  @property
  def path(self):
    return self._path
