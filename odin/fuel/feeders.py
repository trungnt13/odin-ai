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
import types
from abc import ABCMeta
from collections import Counter
from six import add_metaclass
from six.moves import zip, zip_longest, range
from multiprocessing import cpu_count, Process, Queue

import numpy as np

from odin import SIG_TERMINATE_ITERATOR
from odin.utils import segment_list, segment_axis, one_hot, Progbar
from odin.utils.decorators import cache

from .data import Data, MutableData, _validate_operate_axis
from .dataset import Dataset

# ===========================================================================
# Multiprocessing Feeders
# ===========================================================================
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)


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
    buffer_size: int
        the amount of data each process keep before return to main
        process.
    shuffle_level: int (0-3)
        0 - only shuffle the indices list
        1 - shuffle the indices list and enable shuffling in all recipes

    Note
    ----
    set(ncpu=1) if you want a reproducible results
    * Memory transferring in Queue is always the bottleneck of multiprocessing

    """

    def __init__(self, data, indices, transcription=None,
                 ncpu=1, buffer_size=12):
        super(Feeder, self).__init__()
        # ====== load indices ====== #
        if isinstance(indices, str):
            if os.path.isfile(indices):
                self._indices = np.genfromtxt(indices,
                                              dtype=str, delimiter=' ')
            elif os.path.isdir(indices):
                self._indices = np.genfromtxt(os.path.join(indices, 'indices.csv'),
                                              dtype=str, delimiter=' ')
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
        if not isinstance(data, (tuple, list)):
            data = (data,)
        if any(not isinstance(d, Data) for d in data):
            raise ValueError('data must be instance of odin.fuel.Data')
        length = len(data[0])
        if any(len(d) != length for d in data):
            raise ValueError('All Data must have the same length (i.e. shape[0]).')
        self._data = data if len(data) > 1 else data[0]
        # set recipes
        self.recipe = None
        # never use all available CPU
        if ncpu is None:
            ncpu = cpu_count() - 1
        self.ncpu = max(min(ncpu, cpu_count() - 1), 1)
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
        self._transcription = share_dict

    def set_recipe(self, *recipes):
        if len(recipes) > 0:
            if len(inspect.getargspec(recipes[0].map).args) != 4:
                raise Exception('The first recipe of the feeders must '
                                'map(name, x, transcription).')
            self.recipe = FeederList(*recipes)
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
        if include_identity:
            return self._all_iter.items()
        return self._all_iter.values()

    # ==================== override from Data ==================== #
    @property
    def shape(self):
        """ This is just an "UPPER" estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        # this class has list of _data so .shape return list of shape
        shape = super(Feeder, self).shape
        if isinstance(shape[0], (tuple, list)):
            shape = [(self._initial_shape,) + s[1:] for s in shape]
        else:
            shape = (self._initial_shape,) + shape[1:]
        # ====== process each shape ====== #
        if self.recipe is not None:
            return tuple(self.recipe.shape_transform(shape))
        else:
            return tuple(shape)

    # ==================== Strings ==================== #
    def _prepare_iter(self, batch_size, buffer_size, ntasks, jobs, seed,
                      iter_identity):
        map_func = self.recipe.map
        reduce_func = self.recipe.reduce
        self.recipe.init(ntasks, batch_size,
                         seed if self._shuffle_level > 0 else None)
        rng = None if seed is None else np.random.RandomState(seed)

        # data, jobs, map_function, results
        def work_multi(j, map, reduce, res, buffer_size):
            # 1 Data share between all processes
            dat = self._data
            # transcription is shared global variable
            transcription = self._transcription
            batch = []
            n = len(j)
            for count, (name, start, end) in enumerate(j):
                # data can be list of Data, or just 1 Data
                if isinstance(dat, (tuple, list)):
                    x = [d[int(start):int(end)] for d in dat]
                else:
                    x = dat[int(start):int(end)]
                # only support 32bit datatype, it is extremely faster
                # check transcription
                trans = None
                if transcription is not None:
                    if name not in transcription:
                        continue # ignore the sample
                    trans = transcription[name]
                # map tasks, if only 1 Data, just apply map on it, else apply
                # map on list of Data
                _ = (map(name, x, trans) if len(x) > 1
                    else map(name, x[0], trans))
                if _ is not None:
                    batch.append(_)
                # reduce tasks
                if len(batch) == buffer_size or count == n - 1:
                    for b in reduce(batch):
                        res.put(b)
                    batch = []
            # ending signal
            res.put(None)
        yield None # stop here wait for main iterator start
        # Queue maxsize is max_length (maximum number of items can be in queue)
        results = Queue(maxsize=0)
        processes = [Process(target=work_multi,
                             args=(j, map_func, reduce_func,
                                   results, buffer_size))
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
                    batch = [_[rng.permutation(_.shape[0])]
                             for _ in batch]
                # return batch and check for returned signal
                if (yield batch) == SIG_TERMINATE_ITERATOR:
                    forced_terminated = True
                    break
        # Normal exit
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
                if nam not in ds:
                    ds.get_data(nam, dtype=typ, shape=(None,) + x.shape[1:],
                                datatype=datatype)
                ds.get_data(nam).append(x)
            # print progress
            if print_progress:
                prog.add(X[0].shape[0])
        ds.flush()
        ds.close()
        # end
        return self

    def __del__(self):
        self.stop_all()


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
        if mean is not None:
            if isinstance(mean, (tuple, list)):
                mean = [mean[:].astype('float32') for i in mean]
            else:
                mean = mean[:].astype('float32')
        if std is not None:
            if isinstance(std, (tuple, list)):
                std = [std[:].astype('float32') for i in std]
            else:
                std = std[:].astype('float32')
        self.mean = mean
        self.std = std
        self.local_normalize = local_normalize

    def map(self, name, X, transcription):
        if not isinstance(X, (tuple, list)):
            X = [X]
        X = [x.astype('float32') for x in X]

        if self.local_normalize:
            X = [(x - x.mean(0)) / x.std(0) for x in X]
        if self.mean is not None and self.std is not None:
            X = [(x - mean) / std
                 for x, mean, std in zip(X, self.mean, self.std)]
        return name, X if len(X) > 1 else X[0], transcription


class FeatureScaling(FeederRecipe):
    """ FeatureScaling """

    def __init__(self):
        super(FeatureScaling, self).__init__()

    def map(self, name, X, transcription):
        if isinstance(X, (tuple, list)):
            _ = []
            for x in X:
                x = x.astype('float32')
                min_ = x.min(); max_ = x.max()
                x = (x - min_) / (max_ - min_)
                _.append(x)
            X = _
        else:
            X = X.astype('float32')
            min_ = X.min(); max_ = X.max()
            X = (X - min_) / (max_ - min_)
        return name, X, transcription


class Whitening(FeederRecipe):
    """ Whitening """

    def __init__(self):
        super(Whitening, self).__init__()


class Slice(FeederRecipe):
    """ Slice
    Parameters
    ----------
    indices: int, slice, list of int(or slice)
        for example: [slice(0, 12), slice(20, 38)] will becomes
        x = np.hstack([x[0:12], x[20:38]])
    axis: int
        the axis will be applied given indices
    target_data: int
        in case Feeders is given multiple Data, target_data is
        the idx of Data that will be applied given indices
    """

    def __init__(self, indices, axis=-1, target_data=0):
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
        # ====== validate target_data ====== #
        if not isinstance(target_data, (tuple, list)):
            target_data = (target_data,)
        self._target_data = [int(i) for i in target_data]

    def map(self, name, X, transcription):
        results = []
        for _, x in enumerate(X if isinstance(X, (tuple, list)) else (X,)):
            # apply the indices if _ in target_data
            if _ in self._target_data:
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
            results.append(x)
        return (name,
                results if isinstance(X, (tuple, list)) else results[0],
                transcription)

    def shape_transform(self, shapes):
        results = []
        for target, shape in enumerate(shapes
                                  if isinstance(shapes[0], (tuple, list))
                                  else (shapes,)):
            # apply the indices if _ in target_data
            if target in self._target_data:
                axis = self.axis % len(shape) # axis in case if negative
                # int indices, just 1
                if isinstance(self.indices, int):
                    n = 1
                # slice indices
                elif isinstance(self.indices, slice):
                    _ = self.indices.indices(shape[axis])
                    n = _[1] - _[0]
                # list of slice and int
                else:
                    _ = []
                    for idx in self.indices:
                        if isinstance(idx, int):
                            _.append(1)
                        elif isinstance(idx, slice):
                            idx = idx.indices(shape[axis])
                            _.append(idx[1] - idx[0])
                    n = sum(_)
                shape = tuple([j if i != axis else n
                               for i, j in enumerate(shape)])
            results.append(shape)
        return results if isinstance(shapes[0], (tuple, list)) else results[0]


class Merge(FeederRecipe):
    """Merge
    merge a list of np.ndarray into 1 np.array
    """

    def __init__(self, merge_func=np.hstack):
        super(Merge, self).__init__()
        self.merge_func = merge_func

    def map(self, name, x, transcription):
        if not isinstance(x, np.ndarray):
            x = self.merge_func(x)
        return name, x, transcription

    def shape_transform(self, shapes):
        # just 1 shape, nothing to merge
        if not isinstance(shapes[0], (tuple, list)):
            return shapes
        # merge
        if self.merge_func == np.hstack:
            return shapes[0][:-1] + (sum(s[-1] for s in shapes),)
        elif self.merge_func == np.vstack:
            return (sum(s[0] for s in shapes),) + shapes[0][1:]
        else:
            raise Exception("We haven't support shape infer for merge_func={}"
                            ".".format(self.merge_func))


class LabelParse(FeederRecipe):

    """
    Parameters
    ----------
    dict : dictionary or function
        pass
    """

    def __init__(self, dtype, delimiter=' ', label_dict=None):
        super(LabelParse, self).__init__()
        # NO 64bit data type
        self.dtype = str(np.dtype(dtype)).replace('64', '32')
        self.delimiter = delimiter

        if label_dict is None:
            label_func = lambda x: x
        elif isinstance(label_dict, dict):
            label_func = lambda x: label_dict[x]
        elif not hasattr(label_func, '__call__'):
            raise ValueError('label_dict must be a dictionary, function or None.')
        self.label_dict = label_func

    def map(self, name, x, transcription):
        if transcription is not None:
            if isinstance(transcription, str):
                transcription = [self.label_dict(i)
                                 for i in transcription.split(self.delimiter)
                                 if len(i) > 0]
            else:
                transcription = [self.label_dict(i) for i in transcription]
            transcription = np.asarray(transcription, dtype=self.dtype)
        return name, x, transcription


class LabelOneHot(FeederRecipe):

    def __init__(self, n_classes):
        super(LabelOneHot, self).__init__()
        self._n_classes = int(n_classes)

    def map(self, name, x, transcription):
        if transcription is not None:
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

    def __init__(self, left_context=10, right_context=10, shift=None):
        super(Stacking, self).__init__()
        self.left_context = left_context
        self.right_context = right_context
        self.n = int(left_context) + 1 + int(right_context)
        self.shift = self.n if shift is None else int(shift)

    def map(self, name, X, transcription):
        if not isinstance(X, (tuple, list)):
            X = [X]
        if X[0].shape[0] < self.n: # not enough data points for stacking
            return None

        tmp = []
        for x in X:
            idx = list(range(0, x.shape[0], self.shift))
            _ = [x[i:i + self.n].ravel() for i in idx
                 if (i + self.n) <= x.shape[0]]
            x = np.asarray(_) if len(_) > 1 else _[0]
            tmp.append(x)
        X = tmp
        # ====== stacking the transcription ====== #
        if isinstance(transcription, (tuple, list, np.ndarray)):
            idx = list(range(0, len(transcription), self.shift))
            # only take the middle label
            transcription = np.asarray(
                [transcription[i + self.left_context + 1]
                 for i in idx if (i + self.n) <= len(transcription)])
        return name, X if len(X) > 1 else X[0], transcription

    def shape_transform(self, shapes):
        return_multiple = False
        if isinstance(shapes[0], (tuple, list)):
            return_multiple = True
        else:
            shapes = [shapes]
        # ====== do the shape infer ====== #
        _ = []
        for shape in shapes:
            if len(shape) > 2:
                raise Exception('Stacking only support 2D array.')
            n_features = shape[-1] * self.n if len(shape) == 2 else self.n
            n = (shape[0] // self.shift)
            _.append((n, n_features))
        return _ if return_multiple else _[0]


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
    frame_length: int
        the length of each frame
    hop_length: int
        the number of array elements by which the frames should overlap
    axis: int
        the axis to operate on; if None, act on the flattened array
    end: str
        what to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value
    endvalue: Number
        the value to use for end='pad'
    transcription_transform: callable
        a function transform a sequence of transcription value into
        desire value for 1 sample.

    Return
    ------
    a ndarray

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    """
    @staticmethod
    def most_common(x):
        return Counter(x).most_common()[0][0]

    @staticmethod
    def last_seen(x):
        return x[-1]

    def __init__(self, frame_length=256, hop_length=128,
                 end='cut', endvalue=0.,
                 transcription_transform=lambda x: Counter(x).most_common()[0][0]):
        super(Sequencing, self).__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.end = end
        self.endvalue = endvalue
        self.transcription_transform = transcription_transform

    def map(self, name, X, transcription):
        if not isinstance(X, (tuple, list)):
            X = [X]
        # not enough data points for sequencing
        if X[0].shape[0] < self.frame_length:
            return None

        X = [segment_axis(x, self.frame_length, self.hop_length,
                          axis=0, end=self.end, endvalue=self.endvalue)
             for x in X]
        # ====== transforming the transcription ====== #
        if self.transcription_transform is not None and \
        isinstance(transcription, (tuple, list, np.ndarray)):
            transcription = segment_axis(np.asarray(transcription),
                                         self.frame_length, self.hop_length,
                                         axis=0, end=self.end,
                                         endvalue=self.endvalue)
            transcription = np.asarray([self.transcription_transform(i)
                                        for i in transcription])

        return name, X if len(X) > 1 else X[0], transcription

    def shape_transform(self, shapes):
        return_multiple = False
        if isinstance(shapes[0], (tuple, list)):
            return_multiple = True
        else:
            shapes = [shapes]
        # ====== do the shape infer ====== #
        _ = []
        for shape in shapes:
            n_features = shape[-1] if len(shape) >= 2 else 1
            n = (shape[0] - self.frame_length) / self.hop_length
            if self.end == 'cut':
                n = int(math.floor(n))
            else:
                n = int(math.ceil(n))
            mid_shape = shape[1:-1]
            _.append((n, self.frame_length,) + mid_shape + (n_features,))
        return _ if return_multiple else _[0]


class Sampling(FeederRecipe):

    def __init__(self, distribution):
        raise NotImplementedError

    def reduce(self, batch):
        pass


class CreateBatch(FeederRecipe):
    """ Batching
    Parameters
    ----------
    batch_filter: callable
        must be a function has take a list of np.ndarray as first arguments
        ([X]) or ([X, y]), you can return None to ignore given batch
    """

    def __init__(self, batch_filter=None):
        super(CreateBatch, self).__init__()
        self.rng = None
        self.batch_size = 256
        if batch_filter is None:
            batch_filter = lambda *args: args
        elif not hasattr(batch_filter, '__call__'):
            raise ValueError('batch_filter must be a function has 1 or 2 '
                             'parameters (X) or (X, y).')
        self.batch_filter = batch_filter

    def init(self, ntasks, batch_size, seed):
        if seed is None:
            self.rng = None
        else:
            self.rng = np.random.RandomState(seed=seed)
        self.batch_size = batch_size

    def reduce(self, batch):
        _ = batch[0][1] # get the first x
        X = [] if isinstance(_, np.ndarray) else [[] for i in range(len(_))]
        Y = []
        for name, x, y in batch:
            # trianing data can be list of Data or just 1 Data
            if isinstance(x, (tuple, list)):
                for i, j in zip(X, x):
                    i.append(j)
            else:
                X.append(x)
            # labels can be None (no labels given)
            if y is not None:
                Y.append(y)
        # ====== stack everything into big array ====== #
        if isinstance(X[0], np.ndarray):
            X = np.vstack(X)
            shape0 = X.shape[0]
        else:
            X = [np.vstack(x) for x in X]
            shape0 = X[0].shape[0]
        Y = (np.concatenate(Y, axis=0) if len(Y) > 0 else None)
        # ====== shuffle for the whole batch ====== #
        if self.rng is not None:
            idx = self.rng.permutation(X.shape[0])
            X = (X[idx] if isinstance(X, np.ndarray)
                 else [x[idx] for x in X])
            if Y is not None and X.shape[0] == Y.shape[0]:
                Y = Y[idx]
        # ====== create batch ====== #
        batch_filter = self.batch_filter
        for i in range((shape0 - 1) // self.batch_size + 1):
            # if only one Data is given
            if isinstance(X, np.ndarray):
                x = X[i * self.batch_size:(i + 1) * self.batch_size]
                ret = (batch_filter([x]) if Y is None
                       else batch_filter([x, Y[i * self.batch_size:(i + 1) * self.batch_size]])
                       )
            # if list of Data is given
            else:
                x = [x[i * self.batch_size:(i + 1) * self.batch_size]
                     for x in X]
                ret = (x if Y is None
                       else x + [Y[i * self.batch_size:(i + 1) * self.batch_size]])
                ret = batch_filter(ret)
            # return the results
            if ret is not None:
                yield ret
