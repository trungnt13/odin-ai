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
import math
import time
import types
import inspect
from abc import ABCMeta
from collections import Counter
from six import add_metaclass
from six.moves import zip, zip_longest, range
from multiprocessing import cpu_count, Process, Queue

import numpy as np

from odin import SIG_TERMINATE_ITERATOR
from odin.utils import (segment_list, segment_axis, one_hot,
                        Progbar, UnitTimer, get_system_status,
                        get_process_status, SharedCounter, as_tuple)
from odin.utils.decorators import cache

from .data import Data, MutableData
from .dataset import Dataset
from .utils import MmapDict
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
    shuffle_level: int (0-3)
        0 - only shuffle the indices list
        1 - shuffle the indices list and enable shuffling in all recipes
    maximum_queue_size: int (default: 66)
        maximum number of batch will be cached in Queue before main process
        get it and feed to the GPU (if the too many results in Queue, all
        subprocess will be paused)

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

    def __init__(self, indices, ncpu=1, buffer_size=12, maximum_queue_size=144):
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
        # backup original indices
        self._original_indices = self._indices
        # ====== data ====== #
        self._data = None
        # ====== Set default recipes ====== #
        self.recipe = FeederList(CreateBatch())
        # never use all available CPU
        if ncpu is None:
            ncpu = cpu_count() - 1
        self.ncpu = max(min(ncpu, 2 * cpu_count() - 1), 1)
        self.maximum_queue_size = maximum_queue_size
        # ====== default ====== #
        self._buffer_size = buffer_size
        self._batch_size = 256
        self._seed = None
        self._start = 0.
        self._end = 1.
        # ====== manage all iteration ====== #
        self._all_iter = {}
        # store iter identity, so every iter has unique identity
        self._nb_created_iter = 0

    def set_recipes(self, recipes):
        # filter out None value
        recipes = as_tuple(recipes)
        recipes = [i for i in recipes if i is not None]
        if len(recipes) > 0:
            self.recipe = FeederList(*recipes)
            self._indices = self.recipe.preprocess_indices(self._original_indices)
        return self

    def stop_all(self):
        """ Call this method to stop all processes in case you
        spamming to many iteration
        """
        for i in self._all_iter.values():
            try:
                i.next()
                i.send(SIG_TERMINATE_ITERATOR)
                for j in i:
                    pass
            except:
                pass
        self._all_iter = {}

    def get_running_iter(self, include_identity=False):
        """ Get all currently running iteration """
        if include_identity:
            return self._all_iter.items()
        return self._all_iter.values()

    # ==================== override from Data ==================== #
    @property
    def shape(self):
        """ This is just an "UPPER" estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        shape = self._indices.shape
        # ====== recipe process shape ====== #
        if self.recipe is not None:
            shape = tuple(self.recipe.shape_transform(shape))
        else:
            shape = tuple(shape)
        # ====== post process ====== #
        if not isinstance(shape[0], tuple):
            shape = (shape,)
        return shape if len(shape) > 1 else shape[0]

    def __str__(self):
        if self._data is None:
            name = 'None'
            dtype = 'unknown'
        else:
            name = self._data.name
            dtype = self.dtype
        return '<Feeders dataset "%s": shape %s, type "<%s">' % \
        (name, self.shape, dtype)

    # ==================== Strings ==================== #
    def _prepare_iter(self, batch_size, buffer_size, ntasks, jobs, seed,
                      iter_identity):
        """
        No LOCK
        -------
        2-2: 0.68 0.66 0.66
        4-4: 0.59 0.59 0.62

        LOCK
        ----
        2-2: 0.69 0.66 0.66
        4-4: 0.6 0.6 0.58
        """
        map_func = self.recipe.map
        reduce_func = self.recipe.reduce
        maximum_queue_size = self.maximum_queue_size
        self.recipe.init(ntasks, batch_size,
                         seed if self._shuffle_level > 0 else None)
        rng = None if seed is None else np.random.RandomState(seed)

        # data, jobs, map_function, results
        def work_multi(j, map, reduce, res, buffer_size, shared_counter):
            # 1 Data share between all processes
            batch = []
            n = len(j)
            for count, info in enumerate(j):
                # map tasks, if only 1 Data, just apply map on it, else apply
                # map on list of Data
                if len(info) == 1:
                    _ = map(info)
                elif len(info) == 2:
                    _ = map(info[0], info[1])
                else:
                    _ = map(info[0], info[1:])
                # append to batch
                batch.append(_)
                # reduce tasks
                if len(batch) == buffer_size or count == n - 1:
                    # check if we need to wait for the consumer here
                    while shared_counter.value > maximum_queue_size:
                        time.sleep(0.1)
                    # CRITICAL: the nb_returned will be stored from last
                    # batch and added to the shared_counter which can cause
                    # a deadlock, so it must be reseted to 0 after each batch
                    nb_returned = 0
                    # reduce and return the batch
                    for b in reduce(batch):
                        if b is not None:
                            res.put(b)
                            nb_returned += 1
                            del b
                    # new batch
                    del batch; batch = []
                    # increase shared counter (this number must perfectly
                    # counted, only 1 mismatch and deadlock happen)
                    if nb_returned > 0:
                        shared_counter.add(nb_returned)
            # ending signal
            res.put(None)
        #######################################################
        yield None # stop here wait for main iterator start
        # Queue maxsize is max_length (maximum number of items can be in queue)
        results = Queue(maxsize=0)
        shared_counter = SharedCounter()
        processes = [Process(target=work_multi,
                             args=(j, map_func, reduce_func,
                                   results, buffer_size, shared_counter))
                     for i, j in enumerate(jobs)]
        # start the workers
        [p.start() for p in processes]
        # return the results
        forced_terminated = False
        working_processes = len(processes)
        while working_processes > 0:
            # storing batch and return when cache is full
            batch = results.get()
            if batch is None:
                working_processes -= 1
            else:
                # perform batch level permutation
                if rng is not None and self._shuffle_level > 1:
                    permutation = rng.permutation(batch[0].shape[0])
                    # different shape NO shuffle
                    batch = [b[permutation] for b in batch]
                # return batch and check for returned signal
                if (yield batch[0] if len(batch) == 1
                        else tuple(batch)) == SIG_TERMINATE_ITERATOR:
                    forced_terminated = True
                    break
                del batch
                # decrease Queue size counter
                shared_counter.add(-1)
        # ====== ending the iterator ====== #
        if not forced_terminated:
            # check Queue, queue must be empty
            if not results.empty():
                raise Exception('Queue results not empty, something wrong '
                                'with multiprocessing.')
            # end the worker
            [p.join() for p in processes]
        # Exit because of stop_all
        else:
            [p.terminate() for p in processes if p.is_alive()]
        results.close()
        # Finish 1 iteration, callback to remove this iter
        del self._all_iter[iter_identity]
        if forced_terminated:
            yield

    def __iter__(self):
        # ====== check ====== #
        if self.recipe is None:
            raise ValueError('You must set_recipes first')
        # ====== get start and end for indices ====== #
        n = self._indices.shape[0]
        start = _apply_approx(n, self._start)
        end = _apply_approx(n, self._end)
        indices = self._indices[start:end]
        # ====== shuffle the indices ====== #
        seed = None
        if self._seed is not None:
            np.random.seed(self._seed)
            indices = indices[np.random.permutation(indices.shape[0])]
            # seed for the iteration
            seed = np.random.randint(10e8)
            # reset the seed
            self._seed = None
        # ====== create iter and its identity ====== #
        self._nb_created_iter += 1
        it_identity = 'iter%d' % self._nb_created_iter
        it = self._prepare_iter(self._batch_size,
                                self._buffer_size,
                                len(indices),
                                segment_list(indices, n_seg=self.ncpu),
                                seed, it_identity)
        it.next() # just for initlaize the iterator
        self._all_iter[it_identity] = it
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
