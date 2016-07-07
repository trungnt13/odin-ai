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
        |           |               |
         ------- Map Function ------
         \          |     |        /
           --- Reduce Function ---
    This feeder return a non-deterministic order of data, hence,
    cannot be reproducible

    map_function: (name, x, transcription)
    reduce_function: (list of objects returned from map_function)

    Parameters
    ----------
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
    """

    def __init__(self, data, indices, transcription=None,
                 ncpu=1, cache=10):
        super(Feeder, self).__init__()
        if not os.path.isfile(indices):
            raise ValueError('indices must path to indices.csv file')
        self._indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
        if not isinstance(data, Data):
            raise ValueError('data must be instance of odin.fuel.Data')
        self._data = data
        # set functions
        self.recipe = None
        # never use all available CPU
        if ncpu is None:
            ncpu = cpu_count() - 1
        self.ncpu = max(min(ncpu, cpu_count() - 1), 2)
        # ====== default ====== #
        self._cache = cache
        self._batch_size = 256
        self._seed = None
        self._start = 0.
        self._end = 1.
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
        self.recipe = FeederList(*recipes)
        return self

    # ==================== Strings ==================== #
    def _prepare_iter(self, batch_size, cache, ntasks, jobs):
        results = Queue()
        map_func = self.recipe.map
        reduce_func = self.recipe.reduce

        # data, jobs, map_function, results
        def work_multi(d, j, f, r):
            # transcription is shared global variable
            transcription = _transcription
            for name, start, end in j:
                x = d[start:end]
                trans = None
                if transcription is not None:
                    trans = transcription[name]
                r.put(f(name, x, trans))
        processes = [Process(target=work_multi,
                             args=(self._data, j, map_func, results))
                     for i, j in enumerate(jobs)]
        yield None # stop here wait for main iterator start
        # start the workers
        [p.start() for p in processes]
        # return the results
        batch = []
        for i in range(ntasks):
            batch.append(results.get())
            print('Done:', i)
            if len(batch) == cache:
                # for i in _batch(batch, reduce_func, batch_size):
                #     yield i
                batch = []
        # end the worker
        [p.join() for p in processes]
        results.close()
        # return last batch
        if len(batch) > 0:
            pass
            # for i in _batch(batch, reduce_func, batch_size):
            # yield i

    def __iter__(self):
        # ====== check ====== #
        if self.recipe is None:
            raise ValueError('You must set_recipe first')
        # ====== process ====== #
        n = self._indices.shape[0]
        start = _apply_approx(n, self._start)
        end = _apply_approx(n, self._end)
        indices = self._indices[start:end]
        if self._seed is not None:
            np.random.seed(self._seed)
            indices = indices[np.random.permutation(indices.shape[0])]
            self._seed = None
        indices = [(i, int(s), int(e)) for i, s, e in indices]

        it = self._prepare_iter(self._batch_size,
                                self._cache,
                                len(indices),
                                segment_list(indices, n_seg=self.ncpu))
        it.next() # just for initlaize the iterator
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
    def map(self, *args):
        pass

    @abstractmethod
    def reduce(self, x):
        pass


class FeederList(FeederRecipe):

    def __init__(self, *recipes):
        super(FeederList, self).__init__()
        self.recipes = recipes
        if len(recipes) == 0:
            raise Exception('FeederList must contains >= 1 recipe(s).')

    def map(self, *args):
        for f in self.recipes:
            args = (f.map(*args) if isinstance(args, (tuple, list)) else
                    f.map(args))
        return args

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

    def map(self, name, x, transcription):
        if self.local_normalize:
            x = (x - x.mean(0)) / x.std(0)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return name, x, transcription

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
