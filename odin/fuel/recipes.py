from __future__ import print_function, division, absolute_import
import os
import math
import types
import inspect
import warnings
from abc import ABCMeta
from collections import Counter
from six import add_metaclass
from six.moves import zip, range

import numpy as np

from odin.utils import (segment_list, segment_axis, one_hot,
                        Progbar, UnitTimer, get_system_status,
                        get_process_status, SharedCounter, as_tuple)
from odin.utils.decorators import functionable

from .data import Data, MutableData
from .utils import MmapDict


# ===========================================================================
# Recipes
# ===========================================================================
@add_metaclass(ABCMeta)
class FeederRecipe(object):
    """ All method of this function a called in following order
    preprocess_indices(indices): return new_indices
    init(ntasks, batch_size, seed): right before create the iter

    [multi-process]map(*x): x->(name, data)
                            return (if iterator, it will be iterated to
                                    get a list of results)
    [multi-process]reduce(x): x->(object from map(x))
                              return iterator
    [single-process]finalize(x):

    Note
    ----
    This class should not store big amount of data, or the data
    will be replicated to all processes
    """

    def preprocess_indices(self, indices):
        return indices

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

    def __len__(self):
        return len(self.recipes)

    def __str__(self):
        s = []
        for i in self.recipes:
            s.append(i.__class__.__name__)
        return '<FeederList: ' + ', '.join(s) + '>'

    def init(self, ntasks, batch_size, seed):
        for i in self.recipes:
            i.init(ntasks, batch_size, seed)

    def map(self, *args):
        for i, f in enumerate(self.recipes):
            # return iterator (iterate over all of them)
            args = f.map(*args)
            # break the chain if one of the recipes get error,
            # and return None
            if args is None:
                return None
        return args

    def reduce(self, x):
        for f in self.recipes:
            x = f.reduce(x)
        return x

    def finalize(self, x):
        for f in self.recipes:
            x = f.finalize(x)
        return x

    def preprocess_indices(self, indices):
        # sequentially preprocess the indices
        for f in self.recipes:
            indices = f.preprocess_indices(indices)
        return indices

    def shape_transform(self, shape):
        """ Return the new shape that transformed by this Recipe """
        for i in self.recipes:
            shape = i.shape_transform(shape)
        return shape


# ===========================================================================
# Loader
# ===========================================================================
class DataLoader(FeederRecipe):

    def __init__(self, data):
        super(DataLoader, self).__init__()
        # ====== Load data ====== #
        if not isinstance(data, (tuple, list)):
            data = (data,)
        if any(not isinstance(d, Data) for d in data):
            raise ValueError('data must be instance of odin.fuel.Data')
        length = len(data[0])
        if any(len(d) != length for d in data):
            raise ValueError('All Data must have the same length '
                             '(i.e. shape[0]).')
        self._data = data
        # store first dimension
        self._initial_shape = length

    def preprocess_indices(self, indices):
        self._initial_shape = sum(int(e) - int(s) for _, s, e in indices)
        return indices

    def map(self, name, info):
        start, end = int(info[0]), int(info[1])
        # data can be list of Data, or just 1 Data
        x = [d[start:end] for d in self._data]
        return name, x

    def shape_transform(self, shape):
        shape = [d.shape for d in self._data]
        shape = [(self._initial_shape,) + s[1:] for s in shape]
        return shape


class TransLoader(FeederRecipe):
    """ Load transcription from dictionary using name specifed in
    indices

    map_func:
        inputs: name, X
        outputs: name, X, loaded_transcription
    reduce_func:
        inputs: same
        outputs: same

    Parameters
    ----------
    transcription: dict
        if path to a file is specified, the file must specified
        <name> -> [frame1, frame2, ...]
        if list is given, the list must contain the same information
        if dictionary is given, the dict must repesent the same mapping
        above

    """

    def __init__(self, transcription, dtype, delimiter=' ', label_dict=None):
        super(TransLoader, self).__init__()
        # ====== transcription ====== #
        share_dict = None
        if isinstance(transcription, (list, tuple)):
            share_dict = {i: j for i, j in transcription}
        elif isinstance(transcription, dict):
            share_dict = transcription
        elif isinstance(transcription, str) and os.path.isfile(transcription):
            share_dict = MmapDict(transcription)
        else:
            raise Exception('Cannot understand given transcription information.')
        self._transcription = share_dict
        # ====== datatype and delimiter ====== #
        # NO 64bit data type
        self.dtype = str(np.dtype(dtype)).replace('64', '32')
        self.delimiter = delimiter
        # ====== label dict if available ====== #
        if label_dict is None:
            label_func = lambda x: x
        elif isinstance(label_dict, dict):
            label_func = lambda x: label_dict[x]
        elif callable(label_dict):
            label_func = label_dict
        else:
            raise ValueError('label_dict must be a dictionary, function or None.')
        self.label_dict = label_func

    def map(self, name, X):
        trans = self._transcription[name]
        # ====== parse string using delimiter ====== #
        if isinstance(trans, str):
            trans = [self.label_dict(i)
                     for i in trans.split(self.delimiter)
                     if len(i) > 0]
        else:
            trans = [self.label_dict(i) for i in trans]
        trans = np.asarray(trans, dtype=self.dtype)
        return name, X, trans


# ===========================================================================
# Basic recipes
# ===========================================================================
class Filter(FeederRecipe):

    """
    Parameters
    ----------
    filter_func: function, method
        return True if the given data is accepted for further processing
        otherwise False.

    """

    def __init__(self, filter_func):
        super(Filter, self).__init__()
        if not isinstance(filter_func, (types.FunctionType, types.MethodType)):
            raise Exception('filter_func must be FunctionType or MethodType, '
                            'but given type is: %s' % str(type(filter_func)))
        self._filter_func = functionable(filter_func)

    def map(self, name, *args):
        is_ok = self._filter_func(name, *args)
        if is_ok:
            return (name,) + args
        return None


# ===========================================================================
# Features preprocessing
# ===========================================================================
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

    def map(self, name, X, *args):
        X = [x.astype('float32') for x in X]
        if self.local_normalize:
            X = [(x - x.mean(0)) / x.std(0) for x in X]
        if self.mean is not None and self.std is not None:
            X = [(x - mean) / std
                 for x, mean, std in zip(X, self.mean, self.std)]
        return (name, X) + args


class FeatureScaling(FeederRecipe):
    """ FeatureScaling
    Scaling data into range [0, 1]
    """

    def __init__(self):
        super(FeatureScaling, self).__init__()

    def map(self, name, X, *args):
        # ====== scaling features to [0, 1] ====== #
        _ = []
        for x in X:
            x = x.astype('float32')
            min_ = x.min(); max_ = x.max()
            x = (x - min_) / (max_ - min_)
            _.append(x)
        X = _
        return (name, X) + args


class Whitening(FeederRecipe):
    """ Whitening
    TODO
    """

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

    def map(self, name, X, *args):
        results = []
        for _, x in enumerate(X):
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
        return (name, results) + args

    def shape_transform(self, shapes):
        results = []
        for target, shape in enumerate(shapes):
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
        return tuple(results)


class Merge(FeederRecipe):
    """Merge
    merge a list of np.ndarray into 1 np.array
    """

    def __init__(self, merge_func=np.hstack):
        super(Merge, self).__init__()
        self.merge_func = merge_func

    def map(self, name, X, *args):
        if len(X) > 1:
            X = self.merge_func(X)
        return (name, X) + args

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


# ===========================================================================
# Label processing
# ===========================================================================
class LabelOneHot(FeederRecipe):

    def __init__(self, n_classes):
        super(LabelOneHot, self).__init__()
        self._n_classes = int(n_classes)

    def map(self, name, X, *args):
        _ = []
        for transcription in args:
            if isinstance(transcription, str):
                transcription = [i for i in transcription.split(' ')
                                 if len(i) > 0]
            transcription = [int(i) for i in transcription]
            transcription = one_hot(transcription, n_classes=self._n_classes)
            _.append(transcription)
        return (name, X) + tuple(_)


class Name2Trans(FeederRecipe):
    """ This function convert the name (in indices) to transcription
    for given data

    Parameters
    ----------
    converter_func: callbale (2 input arguments)
        for example, lambda name, x: ['true' in name] * x[0].shape[0]
        (Note: x is always a list)

    Example
    >>> cluster_idx = ['spa-car', 'por-brz', 'spa-lac', 'spa-eur']
    >>> feeder = F.Feeder(ds['mfcc'], ds.path, ncpu=1, buffer_size=12)
    >>> feeder.set_batch(256, seed=None, shuffle_level=0)
    >>> feeder.set_recipes([
    >>>     F.recipes.NameToTranscription(
    >>>         lambda x, y: [cluster_idx.index(x)] * y.shape[0]),
    >>>     F.recipes.CreateBatch()
    >>> ])

    """

    def __init__(self, converter_func):
        super(Name2Trans, self).__init__()
        if not callable(converter_func):
            raise ValueError('"converter_func" must be callable.')
        self.converter_func = converter_func

    def map(self, name, X, *args):
        # X: is a list of ndarray
        transcription = np.array(self.converter_func(name, X))
        return name, X, transcription


# ===========================================================================
# Feature grouping
# ===========================================================================
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

    def _stacking(self, x):
        # x is ndarray
        idx = list(range(0, x.shape[0], self.shift))
        _ = [x[i:i + self.n].ravel() for i in idx
             if (i + self.n) <= x.shape[0]]
        x = np.asarray(_) if len(_) > 1 else _[0]
        return x

    def _middle_label(self, trans):
        idx = list(range(0, len(trans), self.shift))
        # only take the middle labelobject
        trans = np.asarray(
            [trans[i + self.left_context + 1]
             for i in idx if (i + self.n) <= len(trans)])
        return trans

    def map(self, name, X, *args):
        if X[0].shape[0] < self.n: # not enough data points for stacking
            warnings.warn('name="%s" has shape[0]=%d, which is not enough to stack '
                          'into %d features.' % (name, X[0].shape[0], self.n))
            return None
        X = [self._stacking(x) for x in X]
        # ====== stacking the transcription ====== #
        args = [self._middle_label(a) for a in args]
        return (name, tuple(X)) + tuple(args)

    def shape_transform(self, shapes):
        # ====== do the shape infer ====== #
        _ = []
        for shape in shapes:
            if len(shape) > 2:
                raise Exception('Stacking only support 2D array.')
            n_features = shape[-1] * self.n if len(shape) == 2 else self.n
            n = (shape[0] // self.shift)
            _.append((n, n_features))
        return tuple(_)


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

    def map(self, name, X, *args):
        # not enough data points for sequencing
        if X[0].shape[0] < self.frame_length:
            warnings.warn('name="%s" has shape[0]=%d, which is not enough to sequence '
                          'into %d features.' % (name, X[0].shape[0], self.frame_length))
            return None

        X = [segment_axis(x, self.frame_length, self.hop_length,
                          axis=0, end=self.end, endvalue=self.endvalue)
             for x in X]
        # ====== transforming the transcription ====== #
        _ = []
        if self.transcription_transform is not None:
            for a in args:
                a = segment_axis(np.asarray(a),
                                self.frame_length, self.hop_length,
                                axis=0, end=self.end,
                                endvalue=self.endvalue)
                a = np.asarray([self.transcription_transform(i)
                                for i in a])

                _.append(a)
            args = tuple(_)
        return (name, tuple(X)) + args

    def shape_transform(self, shapes):
        # ====== do the shape infer ====== #
        _ = []
        for shape in shapes:
            n_features = shape[-1] if len(shape) >= 2 else 1
            n = (shape[0] - self.frame_length) / self.hop_length
            if self.end == 'cut':
                n = int(math.floor(n))
            else:
                n = int(math.ceil(n))
            mid_shape = tuple(shape[1:-1])
            _.append((n, self.frame_length,) + mid_shape + (n_features,))
        return tuple(_)


# ===========================================================================
# Returning results
# ===========================================================================
class CreateBatch(FeederRecipe):
    """ Batching
    Parameters
    ----------
    return_name: boolean
        if True, return the name (in the indices file) with the batch
    batch_filter: callable
        must be a function has take a list of np.ndarray as first arguments
        ([X]) or ([X, y]), you can return None to ignore given batch

    Example
    -------
    >>> feeder = F.Feeder(ds['mfcc'], ds.path, ncpu=12, buffer_size=12)
    >>> feeder.set_batch(256, seed=12082518, shuffle_level=2)
    >>> feeder.set_recipes([
    >>>     F.recipes.CreateBatch(lambda x: (x[0], x[1]) if len(set(x[1])) > 1 else None)
    >>> ])

    """

    def __init__(self, batch_filter=None):
        super(CreateBatch, self).__init__()
        self.rng = None
        self.batch_size = 256
        if batch_filter is None:
            batch_filter = lambda args: args
        elif not callable(batch_filter):
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
        """ batch: contains [(name, np.ndarray-X, np.ndarray-transcription), ...] """
        length = len(batch[0]) # size of 1 batch
        nb_data = len(batch[0][1])
        X = [[] for i in range(nb_data)]
        Y = [[] for i in range(length - 2)]
        for b in batch:
            # training data can be list of Data or just 1 Data
            for i, j in zip(X, b[1]):
                i.append(j)
            # labels can be None (no labels given)
            for i, j in zip(Y, b[2:]):
                i.append(j)
        # ====== stack everything into big array ====== #
        X = [np.vstack(x) for x in X]
        shape0 = X[0].shape[0]
        Y = [np.concatenate(y, axis=0) for y in Y]
        # ====== shuffle for the whole batch ====== #
        if self.rng is not None:
            permutation = self.rng.permutation(shape0)
            X = [x[permutation] for x in X]
            Y = [y[permutation] if y.shape[0] == shape0 else y
                 for y in Y]
        # ====== create batch ====== #
        batch_filter = self.batch_filter
        for i in range((shape0 - 1) // self.batch_size + 1):
            start = i * self.batch_size
            end = start + self.batch_size
            # list of Data is given
            x = [x[start:end] for x in X]
            y = [y[start:end] for y in Y]
            ret = batch_filter(x + y)
            # always return tuple or list
            if ret is not None:
                yield ret if isinstance(ret, (tuple, list)) else (ret,)


class CreateFile(FeederRecipe):
    """ CreateFile
    Instead of divide a file into batches, return the whole file

    Parameters
    ----------
    return_name: bool
        whether return the name specifed in the indices
    """

    def __init__(self, return_name=False):
        super(CreateFile, self).__init__()
        self.return_name = return_name

    def reduce(self, batch):
        for name, X, transcription in batch:
            ret = [X] if not isinstance(X, list) else X
            # ====== transcription ====== #
            if transcription is not None:
                if not isinstance(transcription, (list, tuple)):
                    ret.append(transcription)
                else:
                    ret += transcription
            # ====== return name ====== #
            if self.return_name:
                ret = [name] + ret
            yield tuple(ret)
