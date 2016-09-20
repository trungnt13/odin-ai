from __future__ import print_function, division, absolute_import
import os
import math
import types
import inspect
from abc import ABCMeta
from collections import Counter
from six import add_metaclass
from six.moves import zip, range

import numpy as np

from odin.utils import (segment_list, segment_axis, one_hot,
                        Progbar, UnitTimer, get_system_status,
                        get_process_status, SharedCounter, as_tuple)
from odin.utils.decorators import functionable


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

    def __len__(self):
        return len(self.recipes)

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
        self._nb_args = len(inspect.getargspec(filter_func).args)
        self._filter_func = functionable(filter_func)

    def map(self, name, X, transcription):
        is_ok = False
        if self._nb_args == 1:
            is_ok = self._filter_func(name)
        elif self._nb_args == 2:
            is_ok = self._filter_func(name, X)
        else:
            is_ok = self._filter_func(name, X, transcription)
        if is_ok:
            return name, X, transcription
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
    """ FeatureScaling
    Scaling data into range [0, 1]
    """

    def __init__(self):
        super(FeatureScaling, self).__init__()

    def map(self, name, X, transcription):
        return_list = True
        if not isinstance(X, (tuple, list)):
            X = (X,)
            return_list = False
        # ====== scaling features to [0, 1] ====== #
        _ = []
        for x in X:
            x = x.astype('float32')
            min_ = x.min(); max_ = x.max()
            x = (x - min_) / (max_ - min_)
            _.append(x)
        X = _
        return name, X if return_list else X[0], transcription


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


# ===========================================================================
# Label processing
# ===========================================================================
class Name2Trans(FeederRecipe):
    """ This function convert the name (in indices) to transcription
    for given data

    Parameters
    ----------
    converter_func: callbale (2 input arguments)
        for example, lambda name, x: ['true' in name] * x.shape[0]

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

    def map(self, name, x, transcription):
        transcription = self.converter_func(name, x)
        return name, x, transcription


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
        elif callable(label_dict):
            label_func = label_dict
        else:
            raise ValueError('label_dict must be a dictionary, function or None.')
        self.label_dict = label_func

    def map(self, name, x, transcription):
        if transcription is not None:
            # ====== parse string using delimiter ====== #
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
            mid_shape = tuple(shape[1:-1])
            _.append((n, self.frame_length,) + mid_shape + (n_features,))
        return _ if return_multiple else _[0]


class Sampling(FeederRecipe):

    def __init__(self, distribution):
        raise NotImplementedError

    def reduce(self, batch):
        pass


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
            start = i * self.batch_size
            end = start + self.batch_size
            # if only one Data is given
            if isinstance(X, np.ndarray):
                x = X[start:end]
                ret = (batch_filter([x]) if Y is None
                       else batch_filter([x, Y[start:end]]))
            # if list of Data is given
            else:
                x = [x[start:end] for x in X]
                ret = (x if Y is None else x + [Y[start:end]])
                ret = batch_filter(ret)
            # return the results
            if ret is not None:
                yield ret if len(ret) > 1 else ret[0]


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
