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
import inspect
import math
from abc import ABCMeta
from collections import Counter
from six import add_metaclass
from six.moves import zip, zip_longest, range
from multiprocessing import cpu_count, Process, Queue

import numpy as np

from odin.utils import segment_list, segment_axis, one_hot

from .data import Data, MutableData

# ===========================================================================
# Multiprocessing Feeders
# ===========================================================================
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)


def _batch(b, rf, size):
    b = rf(b)
    if len(b) == 2:
        X, Y = b
        for i in range((X.shape[0] - 1) // size + 1):
            yield (X[i * size:(i + 1) * size],
                   Y[i * size:(i + 1) * size])
    else:
        X = b
        for i in range((X.shape[0] - 1) // size + 1):
            yield X[i * size:(i + 1) * size]


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
    transcription: dict
        if path to a file is specified, the file must specified
        <name> -> [frame1, frame2, ...]
        if list is given, the list must contain the same information
        if dictionary is given, the dict must repesent the same mapping
        above
    cache: int
        the amount of data each process keep before return to main
        process.

    Note
    ----
    set(ncpu=1) if you want a reproducible results
    * Memory transferring in Queue is always the bottleneck of multiprocessing

    """

    def __init__(self, data, indices, transcription=None,
                 ncpu=1, cache=12):
        super(Feeder, self).__init__()
        # ====== load indices ====== #
        if isinstance(indices, str) and os.path.isfile(indices):
            self._indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
        elif isinstance(indices, (tuple, list)):
            self._indices = np.asarray(indices)
        elif isinstance(indices, np.ndarray):
            self._indices = indices
        elif isinstance(indices, dict):
            self._indices = np.asarray([(i, j[0], j[1])
                                        for i, j in indices.iteritems()])
        else:
            raise ValueError('Unsupport indices type: "%s".' % type(indices))
        # first shape, based on indices
        self._initial_shape = sum(int(e) - int(s) for _, s, e in self._indices)
        # ====== Load data ====== #
        if not isinstance(data, Data):
            raise ValueError('data must be instance of odin.fuel.Data')
        self._data = data
        # set functions
        self.recipe = None
        # never use all available CPU
        if ncpu is None:
            ncpu = cpu_count() - 1
        self.ncpu = max(min(ncpu, cpu_count() - 1), 1)
        # ====== default ====== #
        self._cache = cache
        self._batch_size = 256
        self._seed = None
        self._start = 0.
        self._end = 1.
        # ====== manage all iteration ====== #
        self._n_iter = 0
        self._stop_all = False
        # ====== transcription ====== #
        # manager = Manager()
        share_dict = None
        if transcription is not None:
            if isinstance(transcription, (list, tuple)):
                share_dict = {i: j for i, j in transcription}
            elif isinstance(transcription, dict):
                share_dict = transcription
            else:
                raise Exception('Cannot understand given transcipriont information.')
        global _transcription
        _transcription = share_dict

    def set_recipe(self, *recipes):
        if len(inspect.getargspec(recipes[0].map).args) != 4:
            raise Exception('The first recipe of the feeders must '
                            'map(name, x, transcription).')
        self.recipe = FeederList(*recipes)
        return self

    def stop_all(self):
        """ Call this method to stop all processes in case you
        spamming to many iteration
        """
        self._stop_all = self._n_iter

    @property
    def shape(self):
        """ This is just an estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        s = super(Feeder, self).shape
        s = (self._initial_shape,) + s[1:]
        if self.recipe is not None:
            return self.recipe.shape_transform(s)
        else:
            return s

    # ==================== Strings ==================== #
    def _prepare_iter(self, batch_size, cache, ntasks, jobs, seed):
        results = Queue()
        map_func = self.recipe.map
        reduce_func = self.recipe.reduce
        self.recipe.init(ntasks, batch_size, seed)

        # data, jobs, map_function, results
        def work_multi(d, j, map, reduce, res, cache_size):
            # transcription is shared global variable
            transcription = _transcription
            batch = []
            for name, start, end in j:
                x = d[int(start):int(end)]
                trans = None
                if transcription is not None:
                    if name not in transcription:
                        continue # ignore the sample
                    trans = transcription[name]
                # map tasks
                _ = map(name, x, trans)
                if _ is not None:
                    batch.append(_)
                # reduce tasks
                if len(batch) == cache_size:
                    for b in reduce(batch):
                        res.put(b)
                    batch = []
            # return final batch
            if len(batch) > 0:
                for b in reduce(batch):
                    res.put(b)
            # ending signal
            res.put(None)
        yield None # stop here wait for main iterator start
        processes = [Process(target=work_multi,
                             args=(self._data, j, map_func, reduce_func,
                                   results, cache))
                     for i, j in enumerate(jobs)]
        # start the workers
        [p.start() for p in processes]
        # return the results
        exit_on_stop = False
        working_processes = len(processes)
        while working_processes > 0:
            # stop all iterations signal
            if self._stop_all:
                self._stop_all -= 1
                exit_on_stop = True
                break
            # storing batch and return when cache is full
            batch = results.get()
            if batch is None:
                working_processes -= 1
            else:
                yield batch
        # end the worker
        if not exit_on_stop:
            [p.join() for p in processes]
        else:
            [p.terminate() for p in processes if p.is_alive()]
        results.close()
        # Finish 1 iteration
        self._n_iter -= 1

    def __iter__(self):
        # ====== check ====== #
        if self.recipe is None:
            raise ValueError('You must set_recipe first')
        # ====== process ====== #
        n = self._indices.shape[0]
        start = _apply_approx(n, self._start)
        end = _apply_approx(n, self._end)
        indices = self._indices[start:end]
        # ====== shuffle the indices ====== #
        seed = None
        if self._seed is not None:
            np.random.seed(self._seed)
            indices = indices[np.random.permutation(indices.shape[0])]
            seed = np.random.randint(10e8) # seed for the iteration
            # reset the seed
            self._seed = None

        it = self._prepare_iter(self._batch_size,
                                self._cache,
                                len(indices),
                                segment_list(indices, n_seg=self.ncpu),
                                seed)
        it.next() # just for initlaize the iterator
        self._n_iter += 1
        return it


# ===========================================================================
# Recipes
# ===========================================================================
@add_metaclass(ABCMeta)
class FeederRecipe(object):
    """
    map(x): x->(data, name)
    reduce(x): x->list of returned object from map(x)

    Note
    ----
    This class should not store big amount of data, or the data
    will be replicated to all processes
    """

    def init(self, ntasks, batch_size, seed):
        pass

    def map(self, *args):
        return args

    def reduce(self, x):
        return x

    def shape_transform(self, shape):
        return shape


class FeederList(FeederRecipe):

    def __init__(self, *recipes):
        super(FeederList, self).__init__()
        self.recipes = recipes
        if len(recipes) == 0:
            raise Exception('FeederList must contains >= 1 recipe(s).')

    def init(self, ntasks, batch_size, seed):
        for i in self.recipes:
            i.init(ntasks, batch_size, seed)

    def map(self, *args):
        for f in self.recipes:
            args = (f.map(*args) if isinstance(args, (tuple, list)) else
                    f.map(args))
            # break the chain if one of the recipes get error,
            # and return None
            if args is None:
                return None
        return args

    def reduce(self, x):
        for f in self.recipes:
            x = f.reduce(x)
        return x

    def shape_transform(self, shape):
        """ Return the new shape that transformed by this Recipe """
        for i in self.recipes:
            shape = i.shape_transform(shape)
        return shape


class Normalization(FeederRecipe):
    """ Normalization """

    def __init__(self, mean=None, std=None, local_normalize=False):
        super(Normalization, self).__init__()
        self.mean = mean[:] if isinstance(mean, Data) else mean
        self.std = std[:] if isinstance(std, Data) else std
        self.local_normalize = local_normalize

    def map(self, name, x, transcription):
        if self.local_normalize:
            x = (x - x.mean(0)) / x.std(0)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return name, x, transcription


class Slice(FeederRecipe):
    """docstring for Slice"""

    def __init__(self, indices, axis=-1):
        super(Slice, self).__init__()
        # ====== validate axis ====== #
        if not isinstance(axis, int):
            raise ValueError('axis for Slice must be an integer.')
        if axis == 0:
            raise ValueError('Cannot slice the 0 (first) axis. ')
        self.axis = axis
        # ====== validate indices ====== #
        if not isinstance(indices, int) and \
        not isinstance(indices, slice) and \
        not isinstance(indices, (tuple, list)):
            raise ValueError('indices must be int, slice, or list of int '
                             'or slice instance.')
        self.indices = indices

    def map(self, name, x, transcription):
        ndim = x.ndim
        axis = self.axis % ndim
        if isinstance(self.indices, (slice, int)):
            indices = tuple([slice(None) if i != axis else self.indices
                             for i in range(ndim)])
            x = x[indices]
        else:
            indices = []
            for idx in self.indices:
                indices.append(tuple([slice(None) if i != axis else idx
                                      for i in range(ndim)]))
            x = np.hstack([x[i] for i in indices])
        return name, x, transcription

    def shape_transform(self, shape):
        axis = self.axis % len(shape)
        if isinstance(self.indices, int):
            n = 1
        elif isinstance(self.indices, slice):
            _ = self.indices.indices(shape[axis])
            n = _[1] - _[0]
        else:
            _ = []
            for idx in self.indices:
                if isinstance(idx, int):
                    _.append(1)
                elif isinstance(idx, slice):
                    idx = idx.indices(shape[axis])
                    _.append(idx[1] - idx[0])
            n = sum(_)
        return [j if i != axis else n for i, j in enumerate(shape)]


class LabelParse(FeederRecipe):

    def __init__(self, dtype, delimiter=' '):
        super(LabelParse, self).__init__()
        self.dtype = dtype
        self.delimiter = delimiter

    def map(self, name, x, transcription):
        dtype = self.dtype
        if isinstance(transcription, str):
            transcription = [dtype(i)
                             for i in transcription.split(self.delimiter)
                             if len(i) > 0]
        else:
            transcription = [dtype(i) for i in transcription]
        return name, x, transcription


class LabelOneHot(FeederRecipe):

    def __init__(self, n_classes):
        super(LabelOneHot, self).__init__()
        self._n_classes = int(n_classes)

    def map(self, name, x, transcription):
        if isinstance(transcription, str):
            transcription = [i for i in transcription.split(' ')
                             if len(i) > 0]
        transcription = [int(i) for i in transcription]
        transcription = one_hot(transcription, n_classes=self._n_classes)
        return name, x, transcription


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
    """

    def __init__(self, left_context=10, right_context=10, shift=None,
                 stack_transcription=True):
        super(Stacking, self).__init__()
        self.left_context = left_context
        self.right_context = right_context
        self.n = int(left_context) + 1 + int(right_context)
        self.shift = self.n if shift is None else int(shift)
        self.stack_transcription = stack_transcription

    def map(self, name, x, transcription):
        if x.shape[0] < self.n: # not enough data points for stacking
            return None

        idx = list(range(0, x.shape[0], self.shift))
        _ = [x[i:i + self.n].reshape(1, -1) for i in idx
             if (i + self.n) <= x.shape[0]]
        x = np.vstack(_) if len(_) > 1 else _[0]
        # ====== stacking the transcription ====== #
        if transcription is not None and self.stack_transcription:
            idx = list(range(0, len(transcription), self.shift))
            # only take the middle label
            transcription = np.asarray(
                [transcription[i + self.left_context + 1]
                 for i in idx if (i + self.n) <= len(transcription)])
        return name, x, transcription

    def shape_transform(self, shape):
        if len(shape) > 2:
            raise Exception('Stacking only support 2D array.')
        n_features = shape[-1] * self.n if len(shape) == 2 else self.n
        n = (shape[0] // self.n)
        return (n, n_features)


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
    a: the array to segment
    length: the length of each frame
    overlap: the number of array elements by which the frames should overlap
    axis: the axis to operate on; if None, act on the flattened array
    end: what to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value
    endvalue: the value to use for end='pad'

    Return
    ------
    a ndarray

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    """
    @staticmethod
    def most_common(x):
        return Counter(x).most_common()[0][0]

    def __init__(self, frame_length=256, hop_length=128,
                 end='cut', endvalue=0.,
                 transcription_transform=lambda x: Counter(x).most_common()[0][0]):
        super(Sequencing, self).__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.end = end
        self.endvalue = endvalue
        self.transcription_transform = transcription_transform

    def map(self, name, x, transcription):
        if x.shape[0] < self.frame_length: # not enough data points for sequencing
            return None

        x = segment_axis(x, self.frame_length, self.hop_length,
                         axis=0, end=self.end, endvalue=self.endvalue)
        # ====== transforming the transcription ====== #
        if self.transcription_transform is not None and transcription is not None:
            transcription = segment_axis(np.asarray(transcription),
                                         self.frame_length, self.hop_length,
                                         axis=0, end=self.end,
                                         endvalue=self.endvalue)
            transcription = np.asarray([self.transcription_transform(i)
                                        for i in transcription])

        return name, x, transcription

    def shape_transform(self, shape):
        n_features = shape[-1] if len(shape) >= 2 else 1
        n = (shape[0] - self.frame_length) / self.hop_length
        if self.end == 'cut':
            n = int(math.floor(n))
        else:
            n = int(math.ceil(n))
        mid_shape = shape[1:-1]
        return (n, self.frame_length,) + mid_shape + (n_features,)


class CreateBatch(FeederRecipe):
    """ Batching """

    def __init__(self):
        super(CreateBatch, self).__init__()
        self.rng = None
        self.batch_size = 256

    def init(self, ntasks, batch_size, seed):
        if seed is None:
            self.rng = None
        else:
            self.rng = np.random.RandomState(seed=seed)
        self.batch_size = batch_size

    def reduce(self, batch):
        X = []
        Y = []
        for name, x, y in batch:
            X.append(x)
            Y.append(y)
        X = np.vstack(X)
        Y = (np.concatenate(Y, axis=0) if isinstance(Y[0], np.ndarray)
             else np.asarray(Y))
        if self.rng is not None:
            idx = self.rng.permutation(X.shape[0])
            X = X[idx]
            if X.shape[0] == Y.shape[0]:
                Y = Y[idx]
        # ====== create batch ====== #
        for i in range((X.shape[0] - 1) // self.batch_size + 1):
            yield (X[i * self.batch_size:(i + 1) * self.batch_size],
                   Y[i * self.batch_size:(i + 1) * self.batch_size])
