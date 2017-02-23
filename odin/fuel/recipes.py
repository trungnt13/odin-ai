from __future__ import print_function, division, absolute_import
import os
import math
import types
import inspect
import warnings
from abc import ABCMeta
from collections import Counter
from six import add_metaclass
from six.moves import zip, zip_longest, range

import numpy as np

from odin.utils import (segment_list, segment_axis, one_hot, is_string,
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

    [multi-process]process(*x): x->(name, data)
                            return (if iterator, it will be iterated to
                                    get a list of results)
    [multi-process]group(x): x->(object from group(x))
                              return iterator

    Note
    ----
    This class should not store big amount of data, or the data
    will be replicated to all processes
    """

    def prepare(self, **kwargs):
        pass

    def shape_transform(self, shape):
        return shape

    def process(self, *args):
        return args

    def group(self, x):
        return x


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

    def prepare(self, **kwargs):
        for i in self.recipes:
            i.prepare(**kwargs)

    def process(self, *args):
        for i, f in enumerate(self.recipes):
            # return iterator (iterate over all of them)
            args = f.process(*as_tuple(args))
            # break the chain if one of the recipes get error,
            # and return None
            if args is None:
                return None
        return args

    def group(self, x):
        for f in self.recipes:
            x = f.group(x)
        return x

    def shape_transform(self, shape):
        """ Return the new shape that transformed by this Recipe """
        for i in self.recipes:
            shape = i.shape_transform(shape)
        return shape


# ===========================================================================
# Loader
# ===========================================================================
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

    def __init__(self, transcription, dtype, delimiter=' ', label_dict=None,
                 ignore_not_found=True):
        super(TransLoader, self).__init__()
        self.ignore_not_found = ignore_not_found
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

    def process(self, name, X):
        if name not in self._transcription and self.ignore_not_found:
            return None
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

    def process(self, name, *args):
        is_ok = self._filter_func(name, *args)
        if is_ok:
            return (name,) + args
        return None


# ===========================================================================
# Features preprocessing
# ===========================================================================
class Normalization(FeederRecipe):
    """ Normalization
    Note
    ----
    All computation are performed in float32, hence, the return dtype
    is always float32
    """

    def __init__(self, mean=None, std=None, local_normalize=False):
        super(Normalization, self).__init__()
        # mean
        if isinstance(mean, (tuple, list)):
            mean = [i[:].astype('float32') for i in mean]
        elif mean is not None:
            mean = mean[:].astype('float32')
        # std
        if isinstance(std, (tuple, list)):
            std = [i[:].astype('float32') for i in std]
        elif std is not None:
            std = std[:].astype('float32')
        self.mean = mean
        self.std = std
        self.local_normalize = local_normalize

    def process(self, name, X, *args):
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

    def process(self, name, X, *args):
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


# ===========================================================================
# Shape manipulation
# ===========================================================================
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

    def process(self, name, X, *args):
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

    def process(self, name, X, *args):
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

    def process(self, name, X, *args):
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
    converter_func: callbale (1 input arguments)
        for example, lambda name: 1 if 'true' in name else 0
        the return label then is duplicated for all data points in 1 file.
        (e.g. X.shape = (1208, 13), then, transcription=[ret] * 1208)

    Example
    -------
    >>> cluster_idx = ['spa-car', 'por-brz', 'spa-lac', 'spa-eur']
    >>> feeder = F.Feeder(ds['mfcc'], ds.path, ncpu=1, buffer_size=12)
    >>> feeder.set_batch(256, seed=None, shuffle_level=0)
    >>> feeder.set_recipes([
    >>>     F.recipes.NameToTranscription(lambda x: cluster_idx.index(x),
    >>>     F.recipes.CreateBatch()
    >>> ])

    """

    def __init__(self, converter_func):
        super(Name2Trans, self).__init__()
        if not callable(converter_func):
            raise ValueError('"converter_func" must be callable.')
        self.converter_func = functionable(converter_func)

    def process(self, name, X, *args):
        # X: is a list of ndarray
        label = self.converter_func(name)
        labels = [label] * X[0].shape[0]
        transcription = np.array(labels)
        return name, X, transcription


class SADindex(FeederRecipe):
    """ Speech activity indexing (i.e. only select frames
    indicated by SAD from the )

    Parameters
    ----------
    vad: dict, callable, list of (indices, data)
        anything take file name and return a list of SAD indices
    frame_length: int
        pass
    padding: int, None
        if padding is None, use previous frames for padding.
    """

    def __init__(self, vad, frame_length=256, padding=0):
        super(SADindex, self).__init__()
        if hasattr(vad, '__getitem__'):
            vad = lambda x: vad[x]
        elif isinstance(vad, (list, tuple)):
            indices, data = vad
            if is_string(indices) and os.path.exists(indices):
                indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
            indices = {name: (int(start), int(end))
                       for name, start, end in indices}
            vad = lambda x: data[indices[x][0]:indices[x][1]]
        self.vad = vad
        self.frame_length = frame_length
        self.padding = padding

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

    def process(self, name, X, *args):
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

    def process(self, name, X, *args):
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
                 transcription_transform=lambda x: x[-1]):
        super(Sequencing, self).__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.end = end
        self.endvalue = endvalue
        self.__transcription_transform = functionable(transcription_transform)

    def process(self, name, X, *args):
        # not enough data points for sequencing
        if X[0].shape[0] < self.frame_length and self.end == 'cut':
            warnings.warn('name="%s" has shape[0]=%d, which is not enough to sequence '
                          'into %d features.' % (name, X[0].shape[0], self.frame_length))
            return None

        X = [segment_axis(x, self.frame_length, self.hop_length,
                          axis=0, end=self.end, endvalue=self.endvalue)
             for x in X]
        # ====== transforming the transcription ====== #
        _ = []
        trans_transform = self.__transcription_transform
        if trans_transform is not None:
            for a in args:
                original_dtype = a.dtype
                a = segment_axis(np.asarray(a, dtype='str'),
                                self.frame_length, self.hop_length,
                                axis=0, end=self.end,
                                endvalue='__end__')
                # need to remove padded value
                a = np.asarray(
                    [trans_transform([j for j in i if j != '__end__'])
                     for i in a],
                    dtype=original_dtype
                )
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
    batch_filter: callable
        must be a function has take a list of np.ndarray as first arguments
        ([X]) or ([X, y]), you can return None to ignore given batch, return the
        data for accepting the batch
    batch_sequencing: bool
        if True, all data from different files will be processed one-by-one,
        instead of picking data from each files and return small batches from
        them

    Example
    -------
    >>> feeder = F.Feeder(ds['mfcc'], ds.path, ncpu=12, buffer_size=12)
    >>> feeder.set_batch(256, seed=12082518, shuffle_level=2)
    >>> feeder.set_recipes([
    >>>     F.recipes.CreateBatch(lambda x: (x[0], x[1]) if len(set(x[1])) > 1 else None)
    >>> ])

    """

    def __init__(self, batch_filter=None, batch_sequencing=False):
        super(CreateBatch, self).__init__()
        self.rng = None
        self.batch_size = 256
        if batch_filter is None:
            batch_filter = lambda args: args
        elif not callable(batch_filter):
            raise ValueError('batch_filter must be a function has 1 or 2 '
                             'parameters (X) or (X, y).')
        self.__batch_filter = functionable(batch_filter)

    def prepare(self, **kwargs):
        shuffle_level = kwargs.get('shuffle_level', 0)
        seed = kwargs.get('seed', None)
        self.rng = None
        if seed is not None and shuffle_level >= 1:
            self.rng = np.random.RandomState(seed=seed)
        self.batch_size = kwargs.get('batch_size', 64)

    def group(self, batch):
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
            rng = self.rng
            batch_size = self.batch_size
            batch_filter = self.__batch_filter
            indices = [list(range((b[1][0].shape[0] - 1) // batch_size + 1))
                       for b in batch]
            # shuffle if possible
            if rng is not None:
                [rng.shuffle(i) for i in indices]
            # ====== create batch of data ====== #
            for idx in zip_longest(*indices):
                ret = []
                for i, b in zip(idx, batch):
                    # skip if one of the data is not enough
                    if i is None: continue
                    # pick data from each given input
                    name = b[0]; data = b[1]; others = b[2:]
                    start = i * batch_size
                    end = start + batch_size
                    _ = [d[start:end] for d in data] + \
                    [o[start:end] for o in others]
                    ret.append(_)
                ret = [np.concatenate(x, axis=0) for x in zip(*ret)]
                # # shuffle 1 more time
                if rng is not None:
                    permutation = rng.permutation(ret[0].shape[0])
                    ret = [r[permutation] for r in ret]
                # return the batches
                for i in range((ret[0].shape[0] - 1) // batch_size + 1):
                    start = i * batch_size
                    end = start + batch_size
                    _ = batch_filter([x[start:end] for x in ret])
                    # always return tuple or list
                    if _ is not None:
                        yield _ if isinstance(_, (tuple, list)) else (ret,)


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

    def group(self, batch):
        for b in batch:
            name, X = b[0], b[1]
            Y = b[2:]
            ret = X + Y
            # ====== return name ====== #
            if self.return_name:
                ret = [name] + list(ret)
            yield tuple(ret)
