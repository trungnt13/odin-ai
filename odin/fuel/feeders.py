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
import shutil
from six.moves import zip, zip_longest, range
from collections import Mapping

import numpy as np

from odin.utils import (segment_list, one_hot, flatten_list, is_string,
                        Progbar, UnitTimer, get_system_status, batching,
                        get_process_status, SharedCounter, as_tuple, ctext)
from odin.utils.mpi import MPI, async

from .data import MutableData, as_data
from .dataset import Dataset
from .recipes import FeederList, FeederRecipe
from .utils import NoSQL


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
            (name, [list of data], [list of others]),
            (name, [list of data], [list of others]),
            (name, [list of data], [list of others]),
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
                   for name, X, y in batch]
        # shuffle if possible
        if rng is not None:
            [rng.shuffle(i) for i in indices]
        # ====== create batch of data ====== #
        for idx in zip_longest(*indices):
            ret = []
            for start, (name, X, y) in zip(idx, batch):
                # skip if the one data that is not enough
                if start is None: continue
                # pick data from each given input
                end = start + batch_size
                _ = [x[start:end] for x in X] + [i[start:end] for i in y]
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
    """ Return: [(name, index, data...), ...]
        NOTE: each element in batch is one file
    """
    # ====== shuffle the file ====== #
    if rng is not None:
        rng.shuffle(batch)
    # ====== return batched files with index for ordering ====== #
    for name, X, Y in batch:
        n = X[0].shape[0]
        ret = list(X) + list(Y)
        for i, (start, end) in enumerate(batching(n, batch_size)):
            r = [name, i] + [j[start:end] for j in ret]
            yield tuple(batch_filter(r))


def _weird_grouping(batch):
    pass

# ===========================================================================
# DataDescriptor
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
        indices = np.genfromtxt(indices, dtype=_indices_dtype, delimiter=' ')
        indices_info = ('file', str(indices))
    # list or tuple form: (name, (start, end))
    elif isinstance(indices, (tuple, list, np.ndarray)):
        if len(indices[0]) == 2:
            indices = [(name, start, end) for name, (start, end) in indices]
        indices = np.array(indices, dtype=_indices_dtype)
        indices_info = ('array', indices)
    # dictionary: name -> (start, end) or (name, start, end)
    elif isinstance(indices, Mapping):
        if isinstance(indices, NoSQL):
            indices_info = ('nosql', indices)
        indices = np.array([i if len(i) == 3 else (i[0], i[1][0], i[1][1])
                            for i in indices.iteritems()],
                           dtype=_indices_dtype)
        if indices_info is None:
            indices_info = ('array', indices)
    else:
        raise ValueError('Unsupport `indices` type: "%s".' % type(indices))
    return indices, indices_info


class DataDescriptor(MutableData):

    def __init__(self, data, indices):
        super(DataDescriptor, self).__init__()
        # ====== load indices ====== #
        self._indices_loader = async(_preprocessing_indices,
            callback=lambda result: self._loaded_callback())(indices)
        self._indices_info = None
        self._indices = None
        # ====== Load data ====== #
        if not isinstance(data, (tuple, list)):
            data = (data,)
        data = tuple([as_data(d) for d in data])
        # check all data have the same shape[0]
        length = len(data[0])
        if any(d.shape[0] != length for d in data):
            raise ValueError('All Data must have the same length '
                             '(i.e. shape[0]), the given data have '
                             'shape: %s' % str([d.shape for d in data]))
        self._data = data
        # ====== states variables ====== #
        self._length = None
        # if True return name during __iter__
        self._return_name = False

    # ==================== specials ==================== #
    def merge(self, *desc):
        """Merge multiple DataDescriptor into 1"""
        desc = [d for d in desc if isinstance(d, DataDescriptor)]
        # ====== get all data ====== #
        all_data = []
        for d in desc:
            all_data += d.data
        # ====== merge indces ====== #
        new_indices = []
        return DataDescriptor(data=all_data, indices=new_indices)

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
    def _loaded_callback(self):
        self._new_args = (self.indices_info, self.data)

    def restore_data(self):
        if self._new_args is None:
            raise RuntimeError("Indices have not been loaded before calling "
                               "cPickle.dump on this class.")
        self._indices_info, self._data = self._new_args
        # deserialize indices
        ids_type, info = self._indices_info
        if ids_type == 'array':
            self._indices = info
        elif ids_type == 'file':
            self._indices = np.genfromtxt(info, dtype=_indices_dtype,
                                          delimiter=' ')
        elif ids_type == 'nosql':
            self._indices = info

    # ==================== override from Data ==================== #
    @property
    def shape(self):
        """ This is just an "UPPER" estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        if self._length is None:
            self._length = sum((end - start)
                               for name, start, end in self.indices)
        ret_shape = [(self._length,) + d.shape[1:]
                     for d in self.data]
        return ret_shape[0] if len(ret_shape) == 1 else tuple(ret_shape)

    def __str__(self):
        name = ctext('DataDescriptor', 'cyan')
        s = '<%s: Indices(type:"%s" length:%d)>\n' % \
            (name, self.indices_info[0], len(self.indices))
        for dat in self.data:
            s += '   (%s)%s: %s %s\n' % \
                (dat.__class__.__name__,
                    ctext(str(dat.path), 'yellow'),
                    dat.shape, str(dat.dtype))
        return s[:-1]

    # ==================== Strings ==================== #
    def set_return_name(self, return_name):
        self._return_name = bool(return_name)
        return self

    def __iter__(self):
        def _create_iter():
            ret_name = bool(self._return_name)
            yield None # just return for initialize the iteration
            for name, start, end in self.indices:
                dat = [d[start: end] for d in self.data]
                if ret_name:
                    dat = [name] + dat
                yield dat[0] if len(dat) == 1 else dat
        it = _create_iter()
        it.next()
        return it

    def __del__(self):
        pass


# ===========================================================================
# Multiprocessing Feeder
# ===========================================================================
class Feeder(MutableData):
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
    batch_filter: callable
        must be a function has take a list of np.ndarray as first arguments
        ([X]) or ([X, y]), you can return None to ignore given batch, return the
        data for accepting the batch
    batch_mode: 'batch' or 'file' (string type)
        'batch' mode return shuffling and return everything in small batches
        'file' mode return [(file_name, order_index, data...), ...]
    ncpu: int
        number of CPU used for multiprocessing
    buffer_size: int
        A process will perform processing on a group of `buffer_size` number of
        data points, then, a list of results are returned to the main process.
        The higher this number the more powerful batch shuffling.
    maximum_queue_size: int (default: 66)
        maximum number of batch will be cached in Queue before main process
        get it and feed to the GPU (if there are too many results in Queue, a
        deadlock will happen)

    Example
    -------
    >>> ds = F.Dataset(os.path.join(temppath, 'ds'), read_only=True)
    >>> feeder = F.Feeder(ds['X'], indices=ds['indices.csv'],
    >>>                   ncpu=2, buffer_size=2, maximum_queue_size=12)
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
    * you must balance 2 number: buffer_size and maximum_queue_size, so the
    amount of data cached by all processed does not excess the RAM

    """

    def __init__(self, data, indices, dtype=None,
                 batch_filter=lambda x: x, batch_mode='batch',
                 ncpu=1, buffer_size=8, maximum_queue_size=66):
        super(Feeder, self).__init__()
        # ====== load indices ====== #
        # indices always sorted in [(name, start, end), ...]
        if isinstance(indices, str) and os.path.isfile(indices):
            self._indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
        elif isinstance(indices, (tuple, list)):
            if len(indices[0]) == 2: # form: (name, (start, end))
                indices = [(name, start, end) for name, (start, end) in indices]
            self._indices = np.asarray(indices)
        elif isinstance(indices, np.ndarray):
            self._indices = indices
        elif isinstance(indices, Mapping): # name -> (start, end)
            self._indices = np.asarray([as_tuple(i) + as_tuple(j)
                                        for i, j in indices.iteritems()])
        else:
            raise ValueError('Unsupport indices type: "%s".' % type(indices))
        # ====== Load data ====== #
        if not isinstance(data, (tuple, list)):
            data = (data,)
        data = tuple([as_data(d) for d in data])
        length = len(data[0])
        if any(len(d) != length for d in data):
            raise ValueError('All Data must have the same length '
                             '(i.e. shape[0]).')
        self._data = data
        # ====== desire dtype ====== #
        self._outtype = None if dtype is None else as_tuple(dtype, N=len(self._data))
        # ====== Set default recipes ====== #
        self._recipes = FeederList()
        self.set_multiprocessing(ncpu, buffer_size, maximum_queue_size)
        # ====== cache shape information ====== #
        # store first dimension
        self.__cache_indices_id = id(self._indices)
        self._cache_shape = None
        self.__running_iter = []
        # ====== batch mode ====== #
        if batch_filter is None:
            batch_filter = lambda args: args
        elif not callable(batch_filter):
            raise ValueError('batch_filter must be a function has 1 or 2 '
                             'parameters (X) or (X, y).')
        self._batch_filter = batch_filter
        batch_mode = str(batch_mode).lower()
        if batch_mode not in ("batch", 'file'):
            raise ValueError("Only support `batch_mode`: 'file'; 'batch', but "
                             "given value: '%s'" % batch_mode)
        self._batch_mode = batch_mode

    # ==================== pickling ==================== #
    def __getstate__(self):
        return (_dump_data_info(self._data), self._indices, self._outtype,
                self._recipes, self.ncpu, self.buffer_size,
                self.maximum_queue_size)

    def __setstate__(self, states):
        (data, self._indices, self._outtype,
         self._recipes, self.ncpu, self.buffer_size,
         self.maximum_queue_size) = states
        self._data = _load_data_info(data)
        self.__cache_indices_id = id(self._indices)
        self._cache_shape = None
        self.__running_iter = []

    # ==================== multiprocessing ==================== #
    def set_multiprocessing(self, ncpu=None, buffer_size=None, maximum_queue_size=None):
        if ncpu is not None:
            self.ncpu = ncpu
        if buffer_size is not None:
            self.buffer_size = buffer_size
        if maximum_queue_size is not None:
            self.maximum_queue_size = maximum_queue_size
        return self

    def set_batch(self, batch_size=None, batch_filter=None, batch_mode=None,
                  seed=-1, start=None, end=None, shuffle_level=None):
        # ====== check batch_filter ====== #
        if batch_filter is not None:
            if not callable(batch_filter):
                raise ValueError('batch_filter must be a function has 1 or 2 '
                                 'parameters (X) or (X, y).')
            self._batch_filter = batch_filter
        # ====== chec batch_mode ====== #
        if batch_mode is not None:
            batch_mode = str(batch_mode).lower()
            if batch_mode not in ("batch", 'file'):
                raise ValueError("Only support `batch_mode`: 'file'; 'batch', but "
                                 "given value: '%s'" % batch_mode)
            self._batch_mode = batch_mode
        return super(Feeder, self).set_batch(batch_size=batch_size, seed=seed,
                                             start=start, end=end,
                                             shuffle_level=shuffle_level)

    def set_recipes(self, recipes):
        # filter out None value
        recipes = flatten_list(as_tuple(recipes))
        recipes = [i for i in recipes
                   if i is not None and isinstance(i, FeederRecipe)]
        if len(recipes) > 0:
            self._recipes = FeederList(*recipes)
        return self

    def stop_all(self):
        """ Call this method to stop all processes in case you
        spamming to many iteration
        """
        for i in self.__running_iter:
            i.stop()
        self.__running_iter = []

    # ==================== override from Data ==================== #
    @property
    def nb_files(self):
        return len(self._indices)

    @property
    def shape(self):
        """ This is just an "UPPER" estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        # ====== first time calculate the shape ====== #
        if self._cache_shape is None or id(self._indices) != self.__cache_indices_id:
            indices = {name: int(end) - int(start)
                       for name, start, end in self._indices}
            n = sum(indices.itervalues())
            shape = [(n,) + d.shape[1:] for d in self._data]
            shape, indices = self._recipes.shape_transform(shape, indices)
            if len(shape) == 1:
                shape = shape[0]
            self.__cache_indices_id = id(self._indices)
            self._cache_shape = shape
        # ====== get the cached shape ====== #
        else:
            shape = self._cache_shape
        return tuple(shape)

    def __str__(self):
        if self._data is None:
            name = 'None'
            dtype = 'unknown'
        else:
            name = ' '.join([i.name for i in self._data])
            dtype = self.dtype
        return '<Feeders dataset: %s, shape: %s, type: %s, #iter: %d>' % \
        (name, self.shape, dtype, len(self.__running_iter))

    # ==================== Strings ==================== #
    def __iter__(self):
        # ====== check ====== #
        if self._recipes is None:
            raise ValueError('You must "set_recipes" first')
        # ====== get start and end for indices ====== #
        n = self._indices.shape[0]
        start = _apply_approx(n, self._start)
        end = _apply_approx(n, self._end)
        indices = self._indices[start:end]
        outtype = self._outtype
        # ====== shuffle the indices ====== #
        rng = None
        shuffle_level = self._shuffle_level
        if self._seed is not None:
            rng = np.random.RandomState(self._seed)
            indices = indices[rng.permutation(indices.shape[0])]
            if shuffle_level < 1:
                rng = None
            # reset the seed
            self._seed = None
        batch_size = self._batch_size
        batch_filter = self._batch_filter
        process_func = self._recipes.process

        # ====== create wrapped functions ====== #
        def map_func(jobs):
            batch = []
            for name, start, end in jobs:
                start = int(start)
                end = int(end)
                # data can be list of Data, or just 1 Data
                if outtype is not None:
                    x = [np.array(d[start:end], dtype=t) for d, t in zip(self._data, outtype)]
                else:
                    x = [np.array(d[start:end]) for d in self._data]
                x = process_func(name, x, [])
                if x is not None:
                    # not care about return kwargs (only: name, X, y)
                    batch.append(x[:3])
            # choose grouping function
            if self._batch_mode == 'batch':
                return _batch_grouping(batch, batch_size, rng, batch_filter)
            elif self._batch_mode == 'file':
                return _file_grouping(batch, batch_size, rng, batch_filter)

        def reduce_func(results):
            # perform batch level permutation
            if rng is not None and self._shuffle_level > 1:
                permutation = rng.permutation(results[0].shape[0])
                # different shape NO shuffle
                results = [r[permutation] for r in results]
            # convert batch to tuple object if possible
            if isinstance(results, (tuple, list)) and len(results) == 1:
                results = results[0]
            elif isinstance(results, list):
                results = tuple(results)
            return results
        # ====== track and return ====== #
        it = MPI(indices, map_func, reduce_func,
                 ncpu=self.ncpu,
                 buffer_size=self.buffer_size,
                 maximum_queue_size=self.maximum_queue_size,
                 chunk_scheduler=True)
        self.__running_iter.append(it)
        return it

    def save_cache(self, path, datatype='memmap'):
        """ Save all preprocessed data to a Dataset """
        if not isinstance(path, str) or os.path.isfile(path):
            raise ValueError('path must be string path to a folder.')
        if os.path.exists(path):
            print('Remove old dataset at path:', path)
            shutil.rmtree(path)

        ds = Dataset(path)
        # ====== start caching ====== #
        prog = Progbar(target=self.shape[0], name='Caching',
                       print_report=True, print_summary=True)
        for X in self:
            if not isinstance(X, (tuple, list)):
                X = (X,)
            # saving preprocessed data
            for i, x in enumerate(X):
                name = 'data%d' % i
                if name in ds: ds[name].append(x)
                else: ds[(name, datatype)] = x
            # print progress
            prog.add(X[0].shape[0])
        ds.flush()
        ds.close()
        # end
        return self

    def __del__(self):
        self.stop_all()
