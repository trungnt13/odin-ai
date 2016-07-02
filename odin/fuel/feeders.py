from __future__ import print_function, division, absolute_import

import os
from itertools import chain
from abc import ABCMeta, abstractmethod
from collections import Counter
from six import add_metaclass
from six.moves import zip, zip_longest, range
from multiprocessing import cpu_count, Process, Queue

import numpy as np

from odin.utils import segment_list, ordered_set, struct
from odin.utils import segment_axis

from .data import Data

# ===========================================================================
# Multiprocessing Feeders
# ===========================================================================
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)


def work_multi(d, j, f, r): # data, jobs, function, results
    for n, s1, e1 in j:
        x = d[s1:e1]
        r.put((f(x), n))


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


class Feeder(object):
    """ multiprocessing Feeder to 1 comsumer
    Process1    Process2 ...    Process3
        |           |               |
         ------- Map Function ------
         \          |     |        /
           --- Reduce Function ---
    This feeder return a non-deterministic order of data, hence,
    cannot be reproducible

    Parameters
    ----------
    transcription: path, dict, list
        if path to a file is specified, the file must specified
        <name> -> [frame1, frame2, ...]
        if list is given, the list must contain the same information
        if dictionary is given, the dict must repesent the same mapping
        above

    Note
    ----
    set(ncpu=1) if you want a reproducible results
    """

    def __init__(self, data, indices, transcription=None, ncpu=None, cache=10):
        super(Feeder, self).__init__()
        if not os.path.isfile(indices):
            raise ValueError('indices must path to indices.csv file')
        self.indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
        self.data = data
        # set functions
        self.recipe = None
        # never use all available CPU
        if ncpu is None:
            ncpu = cpu_count() - 1
        self.ncpu = min(ncpu, cpu_count() - 1)
        # ====== default ====== #
        self._cache = cache
        self._batch_size = 256
        self._seed = None
        self._start = 0.
        self._end = 1.
        # ====== transcription ====== #
        if transcription is not None:
            if isinstance(transcription, str) and os.path.isfile(transcription):
                _ = {}
                with open(transcription, 'r') as f:
                    for i in f:
                        i = i[:-1].split(' ')
                        _[i[0]] = [j for j in i[1:] if len(j) > 0]
                transcription = _
            elif isinstance(transcription, (list, tuple)):
                transcription = {i: j for i, j in transcription}
        self.transcription = transcription

    def set_batch(self, batch_size=None, seed=None, start=None, end=None):
        if isinstance(batch_size, int) and batch_size > 0:
            self._batch_size = batch_size
        self._seed = seed
        if start is not None and start > 0. - 1e-12:
            self._start = start
        if end is not None and end > 0. - 1e-12:
            self._end = end
        return self

    def set_recipe(self, *recipes):
        self.recipe = FeederList(*recipes)
        return self

    def _prepare_iter_multi(self):
        results = Queue()
        batch_size = self._batch_size
        cache = self._cache
        jobs = self.jobs
        ntasks = self.ntasks
        map_func = self.recipe.map
        reduce_func = self.recipe.reduce
        transcription = self.transcription

        processes = [Process(target=work_multi,
                             args=(self.data, j, map_func, results))
                     for i, j in enumerate(jobs)]
        # start the workers
        [p.start() for p in processes]
        # return the results
        batch = []
        for i in range(ntasks):
            x, name = results.get()
            if transcription is not None:
                name = (transcription[name]
                        if isinstance(transcription, dict)
                        else transcription(name))
            batch.append((x, name))
            if len(batch) == cache:
                for i in _batch(batch, reduce_func, batch_size):
                    yield i
                batch = []
        # end the worker
        [p.join() for p in processes]
        results.close()
        # return last batch
        if len(batch) > 0:
            for i in _batch(batch, reduce_func, batch_size):
                yield i

    def _prepare_iter_single(self):
        batch_size = self._batch_size
        cache = self._cache
        jobs = self.jobs
        map_func = self.recipe.map
        reduce_func = self.recipe.reduce
        transcription = self.transcription

        batch = []
        for name, s, e in chain(*jobs):
            x = map_func(self.data[s:e])
            if transcription is not None:
                name = (transcription[name]
                        if isinstance(transcription, dict)
                        else transcription(name))
            batch.append((x, name))
            if len(batch) == cache:
                for i in _batch(batch, reduce_func, batch_size):
                    yield i
                batch = []
        # return last batch
        if len(batch) > 0:
            for i in _batch(batch, reduce_func, batch_size):
                yield i

    def __iter__(self):
        # ====== check ====== #
        if self.recipe is None:
            raise ValueError('You must set_recipe first')
        # ====== process ====== #
        n = self.indices.shape[0]
        start = _apply_approx(n, self._start)
        end = _apply_approx(n, self._end)
        indices = self.indices[start:end]
        if self._seed is not None:
            np.random.seed(self._seed)
            indices = indices[np.random.permutation(indices.shape[0])]
            self._seed = None
        indices = [(i, int(s), int(e)) for i, s, e in indices]
        self.ntasks = len(indices)
        self.jobs = segment_list(indices, n_seg=self.ncpu)

        if self.ncpu >= 2:
            it = self._prepare_iter_multi()
        else:
            it = self._prepare_iter_single()
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

    @abstractmethod
    def map(self, x):
        pass

    @abstractmethod
    def reduce(self, x):
        pass


class FeederList(FeederRecipe):

    def __init__(self, *recipes):
        super(FeederList, self).__init__()
        self.recipes = recipes

    def map(self, x):
        for f in self.recipes:
            x = f.map(x)
        return x

    def reduce(self, x):
        for f in self.recipes:
            x = f.reduce(x)
        return x


class Normalization(FeederRecipe):
    """ Normalization """

    def __init__(self, mean=None, std=None, local_normalize=False):
        super(Normalization, self).__init__()
        self.mean = mean[:] if isinstance(mean, Data) else mean
        self.std = std[:] if isinstance(std, Data) else std
        self.local_normalize = local_normalize

    def map(self, x):
        if self.local_normalize:
            x = (x - x.mean(0)) / x.std(0)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return x

    def reduce(self, x):
        return x


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

    def __init__(self, left_context=10, right_context=10, shift=None):
        super(Stacking, self).__init__()
        self.left_context = left_context
        self.right_context = right_context
        self.n = int(left_context) + 1 + int(right_context)
        self.shift = self.n if shift is None else int(shift)

    def map(self, x):
        idx = list(range(0, x.shape[0], self.shift))
        x = np.vstack([x[i:i + self.n].reshape(1, -1)
                       for i in idx if (i + self.n) < x.shape[0]])
        return x

    def reduce(self, x):
        # label, and data
        if isinstance(x[0][1], (list, tuple)):
            _ = []
            for i, j in x: # data, name
                l = len(j)
                idx = list(range(0, l, self.shift))
                j = [j[t + self.left_context + 1]
                     for t in idx if (t + self.n) < l]
                _.append((i, j))
            x = _
        return x


class Sequencing(FeederRecipe):
    """ Sequencing
    Parameters
    ----------
    vote: None, max, set, int
        if None, a sequence of label for each data point is kept.
        if max, the label with maximum occurences will be choosed
        if int, return a number of the last labels
        if set, return lable as ordered set
    """

    def __init__(self, frame_length=256, hop_length=128,
                 end='cut', endvalue=0, vote=None):
        super(Sequencing, self).__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.end = end
        self.endvalue = endvalue
        self.vote = vote

    def map(self, x):
        return segment_axis(x, self.frame_length, self.hop_length,
                            0, self.end, self.endvalue)

    def reduce(self, x):
        # label, and data
        if isinstance(x[0][1], (list, tuple)):
            _ = []
            for x, y in x:
                y = segment_axis(np.asarray(y),
                                 self.frame_length, self.hop_length,
                                 0, self.end, self.endvalue)
                if self.vote == 'max':
                    y = np.asarray([Counter(i).most_common()[0][0] for i in y])
                elif self.vote == 'set':
                    raise NotImplementedError()
                    tmp = []
                    for i in y:
                        i = ordered_set(i)
                        if len(i) < self.frame_length:
                            i = [0]
                    tmp.append(i)
                    y = tmp
                elif isinstance(self.vote, int):
                    y = np.asarray([i[-self.vote:] for i in y])
                _.append((x, y))
            x = _
        return x


class CreateBatch(object):
    """ Batching """

    def __init__(self, seed=None):
        super(CreateBatch, self).__init__()
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = None
        self.rng = rng

    def map(self, x):
        return x

    def reduce(self, x):
        label = []
        data = []
        for i, j in x:
            data.append(i)
            label.append(j)
        label = np.concatenate(label, axis=0)
        data = np.vstack(data)
        if self.rng is not None:
            idx = self.rng.permutation(len(label))
            data = data[idx]
            label = label[idx]
        return data, label
