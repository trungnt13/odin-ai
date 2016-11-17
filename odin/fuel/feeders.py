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
from abc import ABCMeta
from collections import Counter
from six import add_metaclass
from six.moves import zip, zip_longest, range

import numpy as np

from odin.utils import (segment_list, one_hot,
                        Progbar, UnitTimer, get_system_status,
                        get_process_status, SharedCounter, as_tuple)
from odin.utils.mpi import MPI

from .data import Data, MutableData, as_data
from .dataset import Dataset
from .recipes import FeederList, CreateBatch

# ===========================================================================
# Multiprocessing Feeders
# ===========================================================================
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)


def split_feeder(data, indices_path, transcription,
                 partitions=[0.6, 0.2, 0.2], seed=1208251813,
                 ncpu=1, buffer_size=12, maximum_queue_size=66):
    """ Fast utilities to split single indices into multiple
    Feeder parts
    """
    partitions = np.cumsum(partitions).tolist()
    if partitions[-1] > 1:
        raise Exception("The sum of all partitions must be smaller than 1.0")
    # ====== load indices ====== #
    if isinstance(indices_path, str):
        if os.path.isdir(indices_path):
            p = [p for p in os.listdir(indices_path)
                 if 'indices' in p][0]
            indices_path = os.path.join(indices_path, p)
        indices = np.genfromtxt(indices_path, dtype=str, delimiter=' ')
    else:
        indices = np.array(indices_path)
    # ====== shuffle the indices ====== #
    np.random.seed(seed)
    idx = np.random.permutation(indices.shape[0])
    indices = indices[idx]
    # ====== partitions ====== #
    partitions = (np.array([0] + partitions) * indices.shape[0]).astype(int)
    partitions = [indices[i:j] for i, j in zip(partitions, partitions[1:])]
    return [Feeder(data, p, transcription,
                   ncpu=ncpu, buffer_size=buffer_size,
                   maximum_queue_size=maximum_queue_size)
            for p in partitions]


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
    buffer_size: int
        the amount of data each process keep before return to main
        process.

    Example
    -------
    >>> ds = F.Dataset(os.path.join(temppath, 'ds'), read_only=True)
    >>> feeder = F.Feeder(indices=ds['indices.csv'],
    >>>                   ncpu=2, buffer_size=2, maximum_queue_size=12)
    >>> feeder.set_recipes([
    >>>     F.recipes.DataLoader(ds['X']),
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
                 ncpu=1, buffer_size=12):
        super(Feeder, self).__init__()
        # ====== load indices ====== #
        if isinstance(indices, str) and os.path.isfile(indices):
            self._indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
        elif isinstance(indices, (tuple, list)):
            self._indices = np.asarray(indices)
        elif isinstance(indices, np.ndarray):
            self._indices = indices
        elif isinstance(indices, dict):
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
        # store first dimension
        self._initial_shape = np.sum(self._indices[:, -1].astype('int32') -
                                     self._indices[:, -2].astype('int32'))
        # ====== desire dtype ====== #
        self._outtype = None if dtype is None else as_tuple(dtype, N=len(self._data))
        # ====== Set default recipes ====== #
        self.recipes = FeederList(CreateBatch())
        # never use all available CPU
        self.set_multiprocessing(ncpu, buffer_size)
        self.__running_iter = []

    def set_multiprocessing(self, ncpu=None, buffer_size=None):
        if ncpu is not None:
            self.ncpu = ncpu
        if buffer_size is not None:
            self.buffer_size = buffer_size
        return self

    def set_recipes(self, recipes):
        # filter out None value
        recipes = [i for i in as_tuple(recipes) if i is not None]
        if len(recipes) > 0:
            self.recipes = FeederList(*recipes)
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
    def shape(self):
        """ This is just an "UPPER" estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        shape = [d.shape for d in self._data]
        shape = [(self._initial_shape,) + s[1:] for s in shape]
        shape = self.recipes.shape_transform(shape)
        if len(shape) == 1:
            shape = shape[0]
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
        if self.recipes is None:
            raise ValueError('You must "set_recipes" first')
        # ====== get start and end for indices ====== #
        n = self._indices.shape[0]
        start = _apply_approx(n, self._start)
        end = _apply_approx(n, self._end)
        indices = self._indices[start:end]
        outtype = self._outtype
        # ====== shuffle the indices ====== #
        rng = None
        if self._seed is not None:
            rng = np.random.RandomState(self._seed)
            indices = indices[rng.permutation(indices.shape[0])]
            # reset the seed
            self._seed = None
        # ====== create iter and its identity ====== #
        process_func = self.recipes.process
        group_func = self.recipes.group
        self.recipes.prepare(
            batch_size=self._batch_size,
            seed=rng.randint(10e6) if rng is not None else None,
            shuffle_level=self._shuffle_level,
        )

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
                x = process_func(name, x)
                if x is not None:
                    batch.append(x)
            return group_func(batch)

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
                 maximum_queue_size=144)
        self.__running_iter.append(it)
        return it

    def save_cache(self, path, name, dtype='float32',
                   datatype='memmap', print_progress=True):
        """ Save all preprocessed data to a Dataset """
        if not isinstance(path, str) or os.path.isfile(path):
            raise ValueError('path must be string path to a folder.')
        if not isinstance(name, (tuple, list, np.ndarray)):
            name = (name,)
        if not isinstance(dtype, (tuple, list, np.ndarray)):
            dtype = (dtype,)

        if len(dtype) < len(name):
            dtype = (dtype[0],) * len(name)
        elif len(dtype) > len(name):
            dtype = dtype[:len(name)]

        ds = Dataset(path)
        for i in name:
            if i in ds:
                raise ValueError('Data with name:"%s" already existed in '
                                 'the dataset' % i)
        # ====== start caching ====== #
        if print_progress:
            prog = Progbar(target=self.shape[0], title='Caching:')
        for X in self:
            if not isinstance(X, (tuple, list)):
                X = (X,)
            # saving preprocessed data
            for x, nam, typ in zip(X, name, dtype):
                if nam in ds: ds[nam].append(x)
                else: ds[(nam, datatype)] = x
            # print progress
            if print_progress:
                prog.add(X[0].shape[0])
        ds.flush()
        ds.close()
        # end
        return self

    def __del__(self):
        self.stop_all()
