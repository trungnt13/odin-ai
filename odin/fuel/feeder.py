"""
MIT License
===========

Copyright (c) 2012 TrungNT (email: [name]@imito.ai)

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function, division, absolute_import

import os
from six.moves import zip, zip_longest, range, cPickle
from collections import Mapping

import numpy as np

from odin.utils import (segment_list, one_hot, is_string, Progbar, batching,
                        as_tuple, ctext, is_number, is_primitives,
                        defaultdictkey)
from odin.utils.mpi import MPI, async
from odin.fuel.data import Data, as_data
from odin.fuel.recipe_base import RecipeList

# ===========================================================================
# Helper for grouping
# ===========================================================================
def _to_numpy_array(self, x):
  if not is_string(x[0]) and len(set(i.shape[1:] for i in x)) == 1:
    return np.concatenate(x, axis=0)
  return np.array(x)


def _batch_grouping(batch, batch_size, rng, batch_filter):
  """ batch: contains
      [
          (name, [list of data]),
          (name, [list of data]),
          (name, [list of data]),
          ...
      ]

  Note
  ----
  We assume the shape[0] (or length) of all "data" and "others" are
  the same
  """
  if len(batch) == 0:
    yield None
  else:
    # create batch of indices for each file (indices is the start
    # index of each batch)
    indices = [list(range(0, X[0].shape[0], batch_size))
               for name, X in batch]
    # shuffle if possible
    if rng is not None:
      [rng.shuffle(i) for i in indices]
    # ====== create batch of data ====== #
    for idx in zip_longest(*indices):
      ret = []
      for start, (name, X) in zip(idx, batch):
        # skip if the one data that is not enough
        if start is None: continue
        # pick data from each given input
        end = start + batch_size
        _ = [x[start:end] for x in X]
        ret.append(_)
      ret = [np.concatenate(x, axis=0) for x in zip(*ret)]
      # shuffle 1 more time
      N = list(set([r.shape[0] for r in ret]))
      if len(N) > 1:
        raise ValueError("The shape[0] of Data is different, found "
                         "%d different length: %s" % (len(N), str(N)))
      N = N[0]
      if rng is not None:
        permutation = rng.permutation(N)
        ret = [r[permutation] for r in ret]
      # return the batches
      for start in range(0, N, batch_size):
        end = start + batch_size
        _ = batch_filter([x[start:end] for x in ret])
        # always return tuple or list
        if _ is not None:
          yield _ if isinstance(_, (tuple, list)) else (ret,)

def _file_grouping(batch, batch_size, rng, batch_filter):
  """ Return: [(name, index, data1, data2, ...), ...]
      NOTE: each element in batch is one file
  """
  # ====== shuffle the file ====== #
  if rng is not None:
    rng.shuffle(batch)
  # ====== return batched files with index for ordering ====== #
  for name, X in batch:
    n = X[0].shape[0]
    ret = list(X)
    for i, (start, end) in enumerate(batching(n=n, batch_size=batch_size)):
      r = [name, i] + [j[start:end] for j in ret]
      yield tuple(batch_filter(r))

# ===========================================================================
# IndexedData
# ===========================================================================
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)
_indices_dtype = [('name', 'object'), ('start', 'i4'), ('end', 'i4')]

def _preprocessing_indices(indices):
  """ Three different kind of indices:
  * file: load from csv file with ' ' (i.e. space as separator)
  * array: numpy array
  * nosql: instance of NoSQL
  """
  # store information for reloading the indices
  indices_info = None
  # indices always sorted in [(name, start, end), ...]
  if isinstance(indices, str) and os.path.isfile(indices):
    indices = {str(name): (start, end) for name, start, end in
               np.genfromtxt(indices, dtype=_indices_dtype, delimiter=' ')}
    indices_info = ('file', str(indices))
  # list or tuple form: (name, (start, end)) or dictionary
  else:
    if isinstance(indices, (tuple, list, np.ndarray)):
      if len(indices[0]) == 2:
        indices = {name: (int(start), int(end))
                   for name, (start, end) in indices}
      else:
        indices = {name: (int(start), int(end))
                   for name, start, end in indices}
    # dictionary: name -> (start, end) or (name, start, end)
    elif isinstance(indices, Mapping):
      pass
    else:
      raise ValueError('Unsupport `indices` type: "%s".' % type(indices))
    indices_info = ('mapping', indices)
  return indices, indices_info

class IndexedData(Data):
  """

  Parameters
  ----------
  data : {Data, list of Data}
    list of Data will be manipulated by this descriptor
    NOTE: all Data must have the same length
  indices : Mapping
    mapping from `name`->(start, end)

  """

  def __init__(self, data, indices):
    super(IndexedData, self).__init__(data=data, read_only=True)
    # ====== states variables ====== #
    self._length = None
    # if True return name during __iter__
    self._return_name = False
    # ====== load indices ====== #
    self._indices_loader = async(_preprocessing_indices)(indices)
    self._indices_info = None
    self._indices = None # dictionary: name -> (start, end)
    # ====== Load data ====== #
    # check all data have the same shape[0]
    length = len(self.data[0])
    if any(d.shape[0] != length for d in self.data):
      raise ValueError('All Data must have the same length '
                       '(i.e. shape[0]), the given data have '
                       'shape: %s' % str([d.shape for d in data]))

  # ==================== Properties ==================== #
  @property
  def indices_info(self):
    if self._indices_info is None:
      self._indices, self._indices_info = self._indices_loader.get()
    return self._indices_info

  @property
  def indices(self):
    if self._indices is None:
      self._indices, self._indices_info = self._indices_loader.get()
    return self._indices

  @property
  def nb_files(self):
    return len(self._indices)

  # ==================== pickling ==================== #
  @property
  def data_info(self):
    return (self._indices_info, self._data,
            self._length, self._return_name)

  def _restore_data(self, info):
    (self._indices_info, self._data,
        self._length, self._return_name) = info
    # deserialize indices
    ids_type, info = self._indices_info
    if ids_type == 'mapping':
      self._indices = info
    elif ids_type == 'file':
      self._indices = np.genfromtxt(info, dtype=_indices_dtype,
                                    delimiter=' ')

  # ==================== override from Data ==================== #
  @property
  def shape(self):
    """ This is just an "UPPER" estimation, some data points might be lost
    during preprocessing each indices by recipes.
    """
    if self._length is None:
      self._length = sum((end - start)
                      for name, (start, end) in self.indices.items())
    ret_shape = [(self._length,) + dat.shape[1:]
                 for dat in self.data]
    return tuple(ret_shape) if self.is_data_list else ret_shape[0]

  def __str__(self):
    name = ctext('IndexedData', 'cyan')
    s = '<%s: Indices(type:"%s" length:%d)>\n' % \
        (name, self.indices_info[0], len(self.indices))
    for dat in self.data:
      s += '   (%s)%s: %s %s\n' % \
          (dat.__class__.__name__,
              ctext(str(dat.data_info), 'yellow'),
              dat.shape,
              str(dat.dtype))
    return s[:-1]

  # ==================== Strings ==================== #
  def keys(self):
    return self.indices.keys()

  def __getitem__(self, key):
    if is_string(key):
      key = slice(*self.indices[key])
    return super(IndexedData, self).__getitem__(key)

# ===========================================================================
# Multiprocessing Feeder
# ===========================================================================
def _dummy_batch_filter(x):
  return x


class Feeder(Data):
  """ multiprocessing Feeder to 1 comsumer
  Process1    Process2 ...    Process3
      |          |     |          |
       ------- Map Function ------
      |          |     |          |
       ----- Reduce Function -----
       \            |            /
         ---- Return batches ---

  This feeder return a non-deterministic order of data, hence,
  cannot be reproducible

  map_function: (name, x, transcription)
  reduce_function: (list of objects returned from map_function)

  Parameters
  ----------
  indices: path(csv file), list, ndarray, dict
      indices represent following information: [name, start_id, end_id]
      if indices is dictionary, it must in the form: {name: (start, end)}
  dtype: string or numpy.dtype
      all data return from this feeder are converted to given dtype
      if None, original dtype is kept
  batch_filter: call-able
      must be a function has take a list of np.ndarray as first arguments
      ([X]) or ([X, y]), you can return None to ignore given batch, return the
      data for accepting the batch
  batch_mode: 'batch' or 'file' (string type)
      'batch' mode return shuffling and return everything in small batches
      'file' mode return [(file_name, order_index_from_0, data...), ...]
  ncpu: int
      number of CPU used for multiprocessing
  buffer_size: int
      A process will perform processing on a group of `buffer_size` number of
      data points, then, a list of results are returned to the main process.
      The higher this number the more powerful batch shuffling.
  hwm: int
      "high water mark" for SEND socket, is a hard limit on the
      maximum number of outstanding messages ØMQ shall queue
      in memory for any single peer that the specified socket
      is communicating with.
  mpi_backend: {'pyzmq', 'python'}
      the A.P.I for message passing between process, sometimes
      python Queue is faster than pyZMQ, but if pyZMQ is faster
      it could up to 35%.

  Example
  -------
  >>> ds = F.Dataset(os.path.join(temppath, 'ds'), read_only=True)
  >>> feeder = F.Feeder(ds['X'], indices=ds['indices.csv'],
  >>>                   ncpu=2, buffer_size=2, hwm=12)
  >>> feeder.set_recipes([
  >>>     F.recipes.TransLoader(ds['transcription.dict'], dtype='int32'),
  >>>     F.recipes.CreateBatch()
  >>> ])
  >>> for i, j in feeder.set_batch(12, seed=1208251813, shuffle_level=2):
  >>>     for x, y in zip(i, j):
  >>>         print(transcription_test[str(x.tolist())] == y)

  Note
  ----
  set(ncpu=1) if you want a reproducible results
   - Memory transferring in Queue is always the bottleneck of multiprocessing
  3 supporting mode for shuffling:
   - shuffle_level=0: only shuffling the indices
   - shuffle_level=1: shuffle the buffered batch (e.g. 12 files in the indices)
   - shuffle_level=2: shuffle each returned batch
  * you must balance 2 number: buffer_size and hwm, so the
  amount of data cached by all processed does not excess the RAM

  """

  def __init__(self, data_desc, dtype=None,
               batch_filter=None, batch_mode='batch',
               ncpu=1, buffer_size=8, hwm=86,
               mpi_backend='python'):
    super(Feeder, self).__init__(data=as_tuple(data_desc, t=IndexedData),
                                 read_only=True)
    # find intersection of all indices in IndexedData
    self._indices_keys = async(
        lambda: np.array(
            list(set.intersection(*[set(dat.indices.keys())
                                    for dat in self._data])),
            dtype=str)
    )()
    # ====== desire dtype ====== #
    nb_data = sum(len(dat._data) for dat in self._data)
    self._output_dtype = as_tuple(dtype, N=nb_data)
    # ====== Set default recipes ====== #
    self._recipes = RecipeList()
    self._recipes.set_feeder_info(nb_desc=len(self._data))
    self.set_multiprocessing(ncpu, buffer_size, hwm, mpi_backend)
    # ====== cache shape information ====== #
    # store first dimension
    self._cache_shape = None
    # if the recipes changed the shape need to be recalculated
    self._recipes_changed = False
    # ====== Iteration information ====== #
    self._running_iter = []
    # ====== batch mode ====== #
    if batch_filter is None:
      batch_filter = _dummy_batch_filter
    elif not hasattr(batch_filter, '__call__'):
      raise ValueError('batch_filter must be a function has 1 or 2 '
                       'parameters (X) or (X, y).')
    # check if batch_filter Picklable
    try:
      cPickle.dumps(batch_filter, protocol=2)
    except Exception:
      raise ValueError("`batch_filter` must be pickle-able, which must be "
                       "top-level function.")

    self._batch_filter = batch_filter
    # check batch_mode
    batch_mode = str(batch_mode).lower()
    if batch_mode not in ("batch", 'file'):
      raise ValueError("Only support `batch_mode`: 'file'; 'batch', but "
                       "given value: '%s'" % batch_mode)
    self._batch_mode = batch_mode

  # ==================== pickling ==================== #
  @property
  def data_info(self):
    return (self._data, self.indices_keys, self._recipes,
            self._output_dtype, self._cache_shape,
            self._batch_mode, self._batch_filter,
            self.ncpu, self.buffer_size, self.hwm)

  def _restore_data(self, info):
    (self._data, self._indices_keys, self._recipes,
     self._output_dtype, self._cache_shape,
     self._batch_mode, self._batch_filter,
     self.ncpu, self.buffer_size, self.hwm) = info
    # ====== basic attributes ====== #
    self._recipes_changed = False
    self._running_iter = []

  # ==================== multiprocessing ==================== #
  def set_multiprocessing(self, ncpu, buffer_size=None, hwm=None,
                          mpi_backend=None):
    self.ncpu = ncpu
    if buffer_size is not None:
      self.buffer_size = int(buffer_size)
    if hwm is not None:
      self.hwm = int(hwm)
    if mpi_backend is not None:
      self.mpi_backend = str(mpi_backend)
    return self

  def set_batch(self, batch_size=None, batch_filter=None, batch_mode=None,
                seed=-1, start=None, end=None, shuffle_level=None):
    # ====== check batch_filter ====== #
    if batch_filter is not None:
      if not hasattr(batch_filter, '__call__'):
        raise ValueError(
            'batch_filter must be a function has 1 arguments (X)')
      self._batch_filter = batch_filter
    # ====== check batch_mode ====== #
    if batch_mode is not None:
      batch_mode = str(batch_mode).lower()
      if batch_mode not in ("batch", 'file'):
        raise ValueError("Only support `batch_mode`: 'file'; 'batch', but "
                         "given value: '%s'" % batch_mode)
      self._batch_mode = batch_mode
    return super(Feeder, self).set_batch(batch_size=batch_size, seed=seed,
                                         start=start, end=end,
                                         shuffle_level=shuffle_level)

  def set_recipes(self, *recipes):
    self._recipes_changed = True
    self._recipes.set_recipes(recipes)
    self.shape # re-calculate cached shape
    return self

  # ==================== override from Data ==================== #
  @property
  def nb_files(self):
    return len(self.indices_keys)

  @property
  def indices_keys(self):
    if not isinstance(self._indices_keys, np.ndarray):
      self._indices_keys = self._indices_keys.get()
    return self._indices_keys

  @property
  def dtype(self):
    """ This is only return the desire dtype for input
    Data, not the ones outputed by Feeder. """
    all_dtypes = []
    for dat in self._data:
      for d in dat._data:
        all_dtypes.append(d.dtype)
    return tuple([j if i is None else i
                  for i, j in zip(self._output_dtype, all_dtypes)])

  @property
  def shape(self):
    """ This is just an "UPPER" estimation, some data points might be lost
    during preprocessing each indices by recipes.
    """
    # ====== first time calculate the shape ====== #
    if self._cache_shape is None or self._recipes_changed:
      # for each Descriptor, create list of pairs: (name, length)
      shapes_indices = []
      for dat in self._data:
        indices = []
        length = 0
        for name in self.indices_keys:
          start, end = dat.indices[name]
          lng = end - start
          length += lng
          indices.append((name, lng))
        # modify shapes by estimted length from indices
        shapes = (dat.shape,) if is_number(dat.shape[0]) \
            else dat.shape
        # NOTE: the indices is copy for each shape (i.e. data),
        # hence, it will create some overhead in shape_transform
        for shp in [(length,) + shp[1:] for shp in shapes]:
          shapes_indices.append((shp, list(indices)))
      # Recipes shape_transform
      shapes = tuple([
          shp for shp, ids in self._recipes.shape_transform(shapes_indices)
      ])
      del shapes_indices
      self._cache_shape = tuple(shapes)
      self._recipes_changed = False
    # ====== get the cached shape ====== #
    if any(s[0] == 0 for s in self._cache_shape):
      raise RuntimeError("Feeder has `length=0` change the recipes to retain "
                         "minimum of `length>=1`, shape: %s" % str(self._cache_shape))
    return self._cache_shape

  def __str__(self):
    padding = '   '
    s = '<%s: #keys:%d #iter:%d #CPU:%s #Buffer:%d #HWM:%d mode:"%s">\n' % \
        (ctext('Feeder', 'cyan'), len(self.indices_keys),
            len(self._running_iter), self.ncpu, self.buffer_size,
            self.hwm, self._batch_mode)
    # ====== Shape and dtype ====== #
    shape = self.shape # this is always list of shape
    s += padding + ctext("Shape: ", 'magenta') + \
        ', '.join((str(s) for s in shape)) + '\n'
    s += padding + ctext("Dtype: ", 'magenta') + \
        ', '.join((str(dt) for dt in self.dtype)) + '\n'
    # ====== print recipes ====== #
    s += padding + ctext('Recipes:', 'magenta') + '\n'
    for recipe in self._recipes:
      s += '\n'.join(['\t' + i for i in str(recipe).split('\n')])
      s += '\n'
    # ====== print data descriptor ====== #
    s += padding + ctext('Descriptor:', 'magenta') + '\n'
    for desc in self._data:
      s += '\n'.join(['\t' + i for i in str(desc).split('\n')])
      s += '\n'
    return s[:-1]

  # ==================== Strings ==================== #
  def __iter__(self):
    # ====== get start and end for indices ====== #
    start = _apply_approx(self.nb_files, self._start)
    end = _apply_approx(self.nb_files, self._end)
    all_keys = self.indices_keys[start:end]
    # ====== shuffle the indices ====== #
    rng = None
    shuffle_level = self._shuffle_level
    if self._seed is not None:
      rng = np.random.RandomState(self._seed)
      all_keys = all_keys[rng.permutation(self.nb_files)]
      if shuffle_level < 1:
        rng = None
      # reset the seed
      self._seed = None
    batch_size = self._batch_size
    batch_filter = self._batch_filter
    process_func = self._recipes.process
    # ====== prepare data, indices and dtype ====== #
    data_indices_dtype = []
    i = 0
    for dat in self._data:
      for d in dat._data:
        data_indices_dtype.append(
            (d, dat.indices, self._output_dtype[i]))
        i += 1

    # ====== create wrapped functions ====== #
    def map_func(jobs):
      if self.buffer_size == 1:
        jobs = [jobs]
      # calculating batch results
      batch = []
      for name in jobs:
        X = []
        for dat, ids, dtype in data_indices_dtype:
          start, end = ids[name]
          # data can be list of Data, or just 1 Data
          dat = dat[start:end]
          if dat.dtype != dtype:
            dat = dat.astype(dtype)
          X.append(dat)
        X = process_func(name, X)
        # ignore None returned result
        if X is not None:
          batch.append(X)
      # choose grouping function
      if self._batch_mode == 'batch':
        X = _batch_grouping(batch, batch_size, rng, batch_filter)
      elif self._batch_mode == 'file':
        X = _file_grouping(batch, batch_size, rng, batch_filter)
      return X

    # ====== track and return ====== #
    it = MPI(jobs=all_keys, func=map_func, ncpu=self.ncpu,
             batch=self.buffer_size, hwm=self.hwm,
             backend=self.mpi_backend)
    self._running_iter.append(it)
    return iter(it)

  def save_cache(self, path, name=None, dtype=None, batch_size=1024):
    """ Save all preprocessed data to a Dataset

    Parameters
    ----------
    path: string
        path to a folder
    name: None, or list of string
        specific name for each returned `numpy.ndarray` during iteration
    dtype: None, or list of dtype, or single dtype
        specific dtype for all or each of returned `numpy.ndarray`
        during iteration
    batch_size: int
        amount of samples for each batch (higher the faster iteration)

    Note
    ----
    Only returned `numpy.ndarray` are saved
    """
    from odin.fuel.dataset import Dataset
    if not is_string(path):
      raise ValueError("`path` must be string path to a folder.")
    if os.path.exists(path) and os.path.isfile(path):
      raise ValueError("`path` is a file, required a folder for "
                       "saving all cache data.")
    # ====== start caching ====== #
    prog = Progbar(target=len(self),
                   name='Saving cache of preprocessed data',
                   print_report=True, print_summary=True)
    ds = Dataset(path, override=True)
    with self.set_batch_context(batch_size=int(batch_size), seed=None,
                                start=0, end=-1, shuffle_level=0):
      for X in self:
        if not isinstance(X, (tuple, list)):
          X = (X,)
        n = 0
        i = 0
        # saving preprocessed data
        for x in X:
          if isinstance(x, np.ndarray):
            # checking name
            if name is None:
              x_name = 'X%d' % i
            else:
              x_name = name[i]
            # checking dtype
            if isinstance(dtype, (tuple, list)):
              x = x.astype(dtype[i])
            elif dtype is not None:
              x = x.astype(dtype)
            # saving to the dataset
            if x_name in ds:
              ds[x_name].append(x)
            else:
              ds[(x_name, 'memmap')] = x
            # update samples count, and data count
            n = x.shape[0]
            i += 1
        # print progress
        prog.add(n)
    # ====== flush and close everything ====== #
    ds.flush()
    ds.close()
    with open(os.path.join(path, 'README'), 'wb') as f:
      f.write(str(self))
    # end
    # ====== check one more time ====== #
    ds = Dataset(path, read_only=True)
    print(ds)
    print(ctext("Dataset size:", 'cyan'), ds.size, '(MB)')
    ds.close()
    return self

  def stop_all(self):
    """ Call this method to stop all processes in case you
    spamming to many iteration
    """
    for i in self._running_iter:
      i.terminate()
    self._running_iter = []

  def __del__(self):
    self.stop_all()
