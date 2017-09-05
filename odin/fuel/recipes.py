from __future__ import print_function, division, absolute_import
import os
import math
import types
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter
from six import add_metaclass
from six.moves import cPickle
from six.moves import zip, zip_longest, range

import numpy as np

from odin.utils import (segment_list, one_hot, is_string, axis_normalize,
                        is_number, UnitTimer, get_system_status, batching,
                        get_process_status, SharedCounter, as_tuple)
from odin.preprocessing.signal import segment_axis, compute_delta
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

    [multi-process] process(*x): x->(name, data)
                                 return (if iterator, it will be iterated to
                                         get a list of results)
    [multi-process] group(x): x->(object from group(x))
                              return iterator

    Note
    ----
    This class should not store big amount of data, or the data
    will be replicated to all processes
    """

    def prepare(self, **kwargs):
        pass

    def shape_transform(self, shapes, indices):
        """
        Parameters
        ----------
        shapes: list of shape
            list of shape tuple
        indices: dict
            {name: nb_samples}
        """
        return shapes, indices

    def process(self, name, X, y, **kwargs):
        if len(kwargs) == 0:
            return name, X, y
        return name, X, y, kwargs

    def group(self, batch):
        return batch


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

    def process(self, name, X, y, **kwargs):
        for i, f in enumerate(self.recipes):
            # return iterator (iterate over all of them)
            if inspect.getargspec(f.process).keywords is None:
                args = f.process(name, X, y)
            else:
                args = f.process(name, X, y, **kwargs)
            # break the chain if one of the recipes get error,
            # and return None
            if args is None:
                return None
            # ====== otherwise keep updating arguments ====== #
            if not (3 <= len(args) <= 4):
                raise ValueError("The return value of process must contain "
                                 "name, X, y, and a dictionary represent "
                                 "additional kwargs (optional).")
            elif len(args) == 4 and not isinstance(args[-1], dict):
                raise ValueError("If process function returns 4 values "
                                 "the last value must be a dictionary.")
            name, X, y = args[:3]
            kwargs = kwargs if len(args) == 3 else args[-1]
        return name, X, y, kwargs

    def group(self, x):
        for f in self.recipes:
            x = f.group(x)
        return x

    def shape_transform(self, shapes, indices):
        """
        Parameters
        ----------
        shapes: list of shape
            list of shape tuple
        indices: dict
            {name: nb_samples}

        Return
        ------
        the new shape that transformed by this Recipe
        """
        for i in self.recipes:
            if not isinstance(indices, dict):
                raise ValueError('"indices" return in "shape_transform" of '
                                 'FeederRecipe must be a dictionary.')
            shapes, indices = i.shape_transform(shapes, indices)
        return shapes, indices


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

    def process(self, name, X, y):
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
        # append to trans list
        y.append(trans)
        return name, X, y


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

    def process(self, name, X, y):
        is_ok = self._filter_func(name)
        if is_ok:
            return name, X, y
        return None


# ===========================================================================
# Features preprocessing
# ===========================================================================
class Pooling(FeederRecipe):
    """docstring for Pooling"""

    def __init__(self, size=2, pool_func=np.mean, data_idx=0):
        super(Pooling, self).__init__()
        self.size = size
        self.pool_func = pool_func
        self.data_idx = as_tuple(data_idx, t=int)

    def process(self, name, X, y):
        X_pooled = []
        for i, x in enumerate(X):
            if i in self.data_idx:
                shape = x.shape
                x = x[:, 2:-2]
                x = x.reshape(shape[0], -1, 2)
                x = self.pool_func(x, axis=-1)
                x = x.reshape(shape[0], -1)
            X_pooled.append(x)
        return name, X_pooled, y

    def shape_transform(self, shapes, indices):
        shapes = [tuple(s[:-1] + (s[-1] // self.size - 2,))
                  if i in self.data_idx else s for i, s in enumerate(shapes)]
        return shapes, indices


class Normalization(FeederRecipe):
    """ Normalization

    Parameters
    ----------
    local_normalize: str or None
        default (True, or 'normal'), normalize the data to have mean=0, std=1
        'sigmoid', normalize by the min and max value to have all element in [0, 1]
        'tanh', normalize to have all element in [-1, 1].
    data_idx: int, or list of int
        In case multiple Data is given, only normalize in the given indices.

    Note
    ----
    Global normalization by given `mean` and `std` is performed
    before `local_normalize`.
    All computation are performed in float32, hence, the return dtype
    is always float32.
    """

    def __init__(self, mean=None, std=None, local_normalize=None,
                 data_idx=0):
        super(Normalization, self).__init__()
        # mean
        if mean is not None:
            mean = mean[:].astype('float32')
        # std
        if std is not None:
            std = std[:].astype('float32')
        self.mean = mean
        self.std = std
        self.local_normalize = str(local_normalize).lower()
        if self.local_normalize not in ('none', 'false', 'tanh', 'sigmoid',
                                        'normal', 'true'):
            raise ValueError("Not support for local_normalize=%s, you must specify "
                            "one of the following mode: none, false, tanh, sigmoid, "
                            "normal (or true)." % self.local_normalize)
        self.data_idx = None if data_idx is None else as_tuple(data_idx, t=int)

    def process(self, name, X, y):
        X_normlized = []
        data_idx = axis_normalize(self.data_idx, ndim=len(X))
        for i, x in enumerate(X):
            if i in data_idx:
                x = x.astype('float32')
                # ====== global normalization ====== #
                if self.mean is not None and self.std is not None:
                    x = (x - self.mean) / self.std
                # ====== perform local normalization ====== #
                if 'normal' in self.local_normalize or 'true' in self.local_normalize:
                    x = (x - x.mean(0)) / x.std(0)
                elif 'sigmoid' in self.local_normalize:
                    min_, max_ = np.min(x), np.max(x)
                    x = (x - min_) / (max_ - min_)
                elif 'tanh' in self.local_normalize:
                    min_, max_ = np.min(x), np.max(x)
                    x = 2 * (x - min_) / (max_ - min_) - 1
            X_normlized.append(x)
        return name, X_normlized, y


class PCAtransform(FeederRecipe):
    """ FeatureScaling
    Scaling data into range [0, 1]
    """

    def __init__(self, pca, nb_components=0.9, whiten=False,
                 data_idx=0):
        super(PCAtransform, self).__init__()
        self._pca = pca
        self.whiten = whiten
        # specified percentage of explained variance
        if nb_components < 1.:
            _ = np.cumsum(pca.explained_variance_ratio_)
            nb_components = (_ > nb_components).nonzero()[0][0] + 1
        # specified the number of components
        else:
            nb_components = int(nb_components)
        self.nb_components = nb_components
        self.data_idx = None if data_idx is None else as_tuple(data_idx, t=int)

    def process(self, name, X, y):
        # update the whiten
        data_idx = axis_normalize(self.data_idx, ndim=len(X))
        pca_whiten = self._pca.whiten
        self._pca.whiten = self.whiten
        X = [self._pca.transform(x, n_components=self.nb_components)
             if i in data_idx else x
             for i, x in enumerate(X)]
        # reset the white value
        self._pca.whiten = pca_whiten
        return name, X, y

    def shape_transform(self, shapes, indices):
        data_idx = axis_normalize(self.data_idx, ndim=len(shapes))
        shapes = [s[:-1] + (self.nb_components,)
                  if i in data_idx else s
                  for i, s in enumerate(shapes)]
        return shapes, indices


class FeatureScaling(FeederRecipe):
    """ FeatureScaling
    Scaling data into range [0, 1]
    """

    def __init__(self):
        super(FeatureScaling, self).__init__()

    def process(self, name, X, y):
        # ====== scaling features to [0, 1] ====== #
        _ = []
        for x in X:
            x = x.astype('float32')
            min_ = x.min(); max_ = x.max()
            x = (x - min_) / (max_ - min_)
            _.append(x)
        X = _
        return name, X, y


class Whitening(FeederRecipe):
    """ Whitening
    TODO
    """

    def __init__(self):
        super(Whitening, self).__init__()


class ComputeDelta(FeederRecipe):
    """Compute delta features: local estimate of the derivative
    of the input data along the selected axis.

    Parameters
    ----------
    delta: int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.
    axis: int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).
    keep_original: bool
        if False, ignore original data and only return concatenated deltas
        features
    data_idx: int, list of int, None
        if None, calculate data for all data processed by this recipes
        if int or list of int, calculte delta for the given set of data.

    Returns
    -------
    delta_data   : list(np.ndarray) [shape=(d, t) or (d, t + window)]
        delta matrix of `data`.
        return list of deltas

    """

    def __init__(self, delta=1, axis=-1, keep_original=True,
                 data_idx=None):
        super(ComputeDelta, self).__init__()
        delta = int(delta)
        if delta < 0:
            raise ValueError("delta must >= 0")
        self.delta = delta
        self.axis = axis
        self.keep_original = keep_original
        self.data_idx = None if data_idx is None else as_tuple(data_idx, t=int)

    def process(self, name, X, y):
        if self.delta > 0:
            data_idx = axis_normalize(self.data_idx, ndim=len(X))
            X = [x if i not in data_idx else
                 np.concatenate(
                     ([x] if self.keep_original else []) +
                     compute_delta(x, order=self.delta, axis=self.axis),
                     axis=self.axis)
                 for i, x in enumerate(X)]
        return name, X, y

    def shape_transform(self, shapes, indices):
        if self.delta > 0:
            n = (self.delta + 1) if self.keep_original else self.delta
            axis = self.axis
            if self.data_idx is None:
                shapes = [s[:axis] + (s[axis] * n,) + s[((axis % len(s)) + 1):]
                          for s in shapes]
            else:
                shapes = [s if i not in self.data_idx else
                          s[:axis] + (s[axis] * n,) + s[axis:]
                          for i, s in enumerate(shapes)]
        return shapes, indices


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
    target_data: int, list of int, None
        in case Feeders is given multiple Data, target_data is
        the index of Data that will be applied given indices.
        if None is given, the Slice is applied to all Data
    """

    def __init__(self, indices, axis, target_data=None):
        super(Slice, self).__init__()
        # ====== validate axis ====== #
        if not isinstance(axis, int):
            raise ValueError('axis for Slice must be an integer.')
        if axis == 0 and target_data is not None:
            raise ValueError("You can only apply Slice on axis=0 for all Data, "
                             "(i.e. 'target_data' must be None when axis=0)")
        self.axis = axis
        # ====== validate indices ====== #
        if is_number(indices):
            indices = slice(int(indices), int(indices + 1))
        elif isinstance(indices, (tuple, list)):
            indices = [i if isinstance(i, slice) else slice(int(i), int(i + 1))
                       for i in indices
                       if isinstance(i, slice) or is_number(i)]
        elif not isinstance(indices, slice):
            raise ValueError('indices must be int, slice, or list of int and slice.')
        self.indices = indices
        # ====== validate target_data ====== #
        if target_data is not None and not isinstance(target_data, (tuple, list)):
            target_data = (target_data,)
        self._target_data = target_data

    def process(self, name, X, y):
        results = []
        for _, x in enumerate(X):
            # apply the indices if _ in target_data
            if self._target_data is None or _ in self._target_data:
                ndim = x.ndim
                axis = self.axis % ndim
                # just one index given
                if isinstance(self.indices, (slice, int)):
                    indices = tuple([slice(None) if i != axis else self.indices
                                     for i in range(ndim)])
                    x = x[indices]
                # multiple indices are given
                else:
                    indices = []
                    for idx in self.indices:
                        indices.append(tuple([slice(None) if i != axis else idx
                                              for i in range(ndim)]))
                    x = np.concatenate([x[i] for i in indices], axis=self.axis)
            results.append(x)
        return name, list(results), y

    def _from_indices(self, n):
        """ This function estimates number of sample given indices """
        # slice indices
        indices = (self.indices,) if isinstance(self.indices, slice) else self.indices
        count = 0
        for idx in indices:
            idx = idx.indices(n)
            count += idx[1] - idx[0]
        return count

    def shape_transform(self, shapes, indices):
        # ====== check if first dimension is sliced ====== #
        if self.axis == 0:
            indices = {name: self._from_indices(n)
                       for name, n in indices.iteritems()}
            n = sum(indices.itervalues())
            return tuple([(n,) + s[1:] for s in shapes]), indices
        # ====== process other dimensions ====== #
        results = []
        for target, shape in enumerate(shapes):
            # apply the indices if _ in target_data
            if self._target_data is None or target in self._target_data:
                axis = self.axis % len(shape) # axis in case if negative
                # int indices, just 1
                n = self._from_indices(shape[axis])
                shape = tuple([j if i != axis else n
                               for i, j in enumerate(shape)])
            results.append(shape)
        return tuple(results), indices


class Merge(FeederRecipe):
    """Merge
    merge a list of np.ndarray into 1 np.array

    Note
    ----
    The new value will be appended to the data list
    """

    def __init__(self, merge_func=np.hstack, data_idx=None):
        super(Merge, self).__init__()
        self.merge_func = merge_func
        if merge_func not in (np.vstack, np.hstack):
            raise ValueError("Support merge function include: numpy.vstack, numpy.hstack")
        self.data_idx = None if data_idx is None \
            else as_tuple(data_idx, t=int)

    def process(self, name, X, y):
        if len(X) > 1:
            data_idx = axis_normalize(self.data_idx, ndim=len(X))
            X_old = [x for i, x in enumerate(X) if i not in data_idx]
            X_new = [x for i, x in enumerate(X) if i in data_idx]
            X = list(as_tuple(self.merge_func(X_new))) + X_old
        return name, X, y

    def shape_transform(self, shapes, indices):
        # just 1 shape, nothing to merge
        if not isinstance(shapes[0], (tuple, list)):
            return shapes, indices
        # merge
        data_idx = axis_normalize(self.data_idx, ndim=len(shapes))
        old_shapes = [s for i, s in enumerate(shapes) if i not in data_idx]
        new_shapes = [s for i, s in enumerate(shapes) if i in data_idx]
        if self.merge_func == np.hstack:
            # indices still the same
            new_shapes = new_shapes[0][:-1] + (sum(s[-1] for s in new_shapes),)
        elif self.merge_func == np.vstack:
            indices = {name: n * len(shapes) for name, n in indices.iteritems()}
            new_shapes = (sum(s[0] for s in new_shapes),) + new_shapes[0][1:]
        else:
            raise Exception("We haven't support shape infer for merge_func={}"
                            ".".format(self.merge_func))
        return (new_shapes,) + tuple(old_shapes), indices


class ExpandDims(FeederRecipe):
    """docstring for ExpandDim"""

    def __init__(self, axis, data_idx=0):
        super(ExpandDims, self).__init__()
        self.axis = int(axis)
        self.data_idx = None if data_idx is None else as_tuple(data_idx, t=int)

    def process(self, name, X, y):
        data_idx = axis_normalize(self.data_idx, ndim=len(X))
        X = [np.expand_dims(x, axis=self.axis)
             if i in data_idx else x
             for i, x in enumerate(X)]
        return name, X, y

    def shape_transform(self, shapes, indices):
        data_idx = axis_normalize(self.data_idx, ndim=len(shapes))
        new_shapes = []
        for i, s in enumerate(shapes):
            if i in data_idx:
                s = list(s)
                axis = self.axis if self.axis >= 0 else \
                    (len(s) + 1 - self.axis)
                s.insert(axis, 1)
                new_shapes.append(tuple(s))
            else:
                new_shapes.append(s)
        return tuple(new_shapes), indices


# ===========================================================================
# Label processing
# ===========================================================================
class LabelOneHot(FeederRecipe):

    def __init__(self, nb_classes, label_idx=0):
        super(LabelOneHot, self).__init__()
        self._nb_classes = int(nb_classes)
        self.label_idx = as_tuple(label_idx, t=int)

    def process(self, name, X, Y):
        _ = []
        for i, y in enumerate(Y):
            # transform into one-label y
            if i in self.label_idx:
                y = np.array([int(i) for i in y])
                y = one_hot(y, n_classes=self._nb_classes)
            _.append(y)
        return name, X, _


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

    def process(self, name, X, y):
        # X: is a list of ndarray
        label = self.converter_func(name)
        labels = [label] * X[0].shape[0]
        transcription = np.array(labels)
        y.append(transcription)
        return name, X, y


class VADindex(FeederRecipe):
    """ Voice activity indexing (i.e. only select frames
    indicated by SAD from the )

    Parameters
    ----------
    vad: dict, list of (indices, data)
        anything take file name and return a list of SAD indices
    frame_length: int
        if `frame_length`=1, simply concatenate all VAD frames.
    padding: int, None
        if padding is None, use previous frames for padding.
    filter_vad: callable
        a function take arguments: start, end
    """

    def __init__(self, vad, frame_length, padding=None, filter_vad=None):
        super(VADindex, self).__init__()
        if isinstance(vad, (list, tuple)):
            if len(vad) == 2:
                indices, data = vad
                if is_string(indices) and os.path.exists(indices):
                    indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
                vad = {name: data[int(start): int(end)]
                       for name, start, end in indices}
            else: # a list contain all information is given
                vad = {name: segments for name, segments in vad}
        elif not isinstance(vad, dict):
            raise ValueError('Unsupport "vad" type: %s' % type(vad).__name__)
        self.vad = vad
        self.padding = padding
        self.frame_length = int(frame_length)
        # ====== check filter vad ====== #
        if filter_vad is None:
            filter_vad = lambda start, end: True
        elif callable(filter_vad):
            if len(inspect.getargspec(filter_vad).args) != 2:
                raise ValueError("filter_vad must be callable that accepts 2 "
                                 "arguments: start, end")
        else:
            raise ValueError("filter_vad must be a function accept 2 arguments: "
                             "(start, end) of the VAD segment.")
        self.filter_vad = functionable(filter_vad)

    def _vad_indexing_1(self, X, indices):
        return np.concatenate([X[start:end] for start, end in indices], axis=0)

    def _vad_indexing(self, X, indices, n):
        # ====== create placeholder array ====== #
        shape = (n, self.frame_length,) + X.shape[1:]
        if self.padding is None:
            Y = np.empty(shape=shape, dtype=X.dtype)
        else:
            Y = np.full(shape=shape, fill_value=self.padding, dtype=X.dtype)
        # ====== start processing ====== #
        p = 0
        for start, end in indices:
            n = end - start
            # not enough frames
            if n <= self.frame_length:
                diff = self.frame_length - (end - start)
                if self.padding is None:
                    x = X[end - self.frame_length:end] if diff <= start else None
                else:
                    x = X[start:end]
                if x is not None:
                    Y[p, -x.shape[0]:] = x
                    p += 1
            # more frames thant the length
            elif n > self.frame_length:
                i = n // self.frame_length
                x = X[(end - i * self.frame_length):end]
                # remains (now the number of remain always smaller than
                # frame_length) do the same for `n <= self.frame_length`
                j = n - i * self.frame_length
                if j > 0:
                    diff = self.frame_length - j
                    if self.padding is None and diff <= start:
                        Y[p] = X[start - diff:start + j]
                        p += 1
                    else:
                        Y[p, -j:] = X[start:start + j]
                        p += 1
                # assign the main part
                Y[p:p + i] = np.reshape(x,
                    newshape=(i, self.frame_length,) + x.shape[1:])
                p += i
        return Y

    def _estimate_number_of_sample(self, start, end):
        if end - start < self.frame_length:
            diff = self.frame_length - (end - start)
            if self.padding is None and diff > start:
                return 0 # not enough previous segments for padding
        elif end - start > self.frame_length:
            return int(np.ceil((end - start) / self.frame_length))
        return 1

    def _slice_last_axis(self, x):
        s = [slice(None) for i in range(x.ndim - 1)] + [-1]
        return x[s]

    def process(self, name, X, y):
        # ====== return None, ignore the file ====== #
        if name not in self.vad:
            return None
        # ====== found the VAD, process it ====== #
        indices = self.vad[name]
        indices = [(start, end) for start, end in indices
                   if self.filter_vad(start, end)]
        if self.frame_length == 1:
            X = [self._vad_indexing_1(x, indices) for x in X]
            y = [self._vad_indexing_1(a, indices) for a in y]
        else:
            n = sum(self._estimate_number_of_sample(start, end)
                    for start, end in indices)
            if n > 0:
                X = [self._vad_indexing(x, indices, n) for x in X]
                y = [self._slice_last_axis(self._vad_indexing(a, indices, n))
                     for a in y]
            else:
                return None
        return name, X, y

    def shape_transform(self, shapes, indices):
        # ====== init ====== #
        if self.frame_length == 1:
            n_func = lambda start, end: end - start
            shape_func = lambda n, shape: (n,) + shape[1:]
        else:
            n_func = lambda start, end: self._estimate_number_of_sample(start, end)
            shape_func = lambda n, shape: (n, self.frame_length) + shape[1:]
        # ====== processing ====== #
        indices_new = {}
        n = 0
        for name, segments in self.vad.iteritems():
            # not found find in original indices
            if name not in indices: continue
            # found the name, and update its indices
            n_file = 0
            for start, end in segments:
                if self.filter_vad(start, end):
                    n_file += n_func(start, end)
            indices_new[name] = n_file
            n += n_file
        shapes = tuple([shape_func(n, s) for s in shapes])
        return shapes, indices_new


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
        self.shift = int(right_context) if shift is None else int(shift)

    def _stacking(self, x):
        # x is ndarray
        idx = list(range(0, x.shape[0], self.shift))
        _ = [x[i:i + self.n].reshape(1, -1) for i in idx
             if (i + self.n) <= x.shape[0]]
        x = np.concatenate(_, axis=0) if len(_) > 1 else _[0]
        return x

    def _middle_label(self, trans):
        idx = list(range(0, len(trans), self.shift))
        # only take the middle labelobject
        trans = np.asarray(
            [trans[i + self.left_context + 1]
             for i in idx if (i + self.n) <= len(trans)])
        return trans

    def process(self, name, X, y):
        if X[0].shape[0] < self.n: # not enough data points for stacking
            warnings.warn('name="%s" has shape[0]=%d, which is not enough to stack '
                          'into %d features.' % (name, X[0].shape[0], self.n))
            return None
        X = [self._stacking(x) for x in X]
        # ====== stacking the transcription ====== #
        y = [self._middle_label(a) for a in y]
        return name, X, y

    def shape_transform(self, shapes, indices):
        # ====== update the indices ====== #
        n = 0
        indices_new = {}
        for name, nb_samples in indices.iteritems():
            nb_samples = 1 + (nb_samples - self.n) // self.shift
            indices_new[name] = nb_samples
            n += nb_samples
        # ====== do the shape infer ====== #
        _ = []
        for shape in shapes:
            if len(shape) > 2:
                raise Exception('Stacking only support 2D array.')
            n_features = shape[-1] * self.n if len(shape) == 2 else self.n
            _.append((n, n_features))
        return tuple(_), indices_new


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
    endmode: 'pre', 'post'
        if "pre", padding or wrapping at the beginning of the array.
        if "post", padding or wrapping at the ending of the array.
    label_transform: callable
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

    def __init__(self, frame_length=256, hop_length=None,
                 end='cut', endvalue=0., endmode='post',
                 label_transform=lambda x: x[-1]):
        super(Sequencing, self).__init__()
        self.frame_length = int(frame_length)
        self.hop_length = frame_length // 2 if hop_length is None else int(hop_length)
        if hop_length > frame_length:
            raise ValueError("hop_length=%d must be smaller than frame_length=%d"
                             % (hop_length, frame_length))
        self.end = end
        self.endvalue = endvalue
        self.endmode = endmode
        self.__label_transform = functionable(label_transform)

    def process(self, name, X, y):
        # not enough data points for sequencing
        if X[0].shape[0] < self.frame_length and self.end == 'cut':
            warnings.warn('name="%s" has shape[0]=%d, which is not enough to sequence '
                          'into %d features.' % (name, X[0].shape[0], self.frame_length))
            return None

        X = [segment_axis(x, self.frame_length, self.hop_length, axis=0,
                    end=self.end, endvalue=self.endvalue, endmode=self.endmode)
             for x in X]
        # ====== transforming the transcription ====== #
        labs_transform = self.__label_transform
        if labs_transform is not None:
            label_list = []
            for labels in y:
                original_dtype = labels.dtype
                labels = segment_axis(np.asarray(labels, dtype='str'),
                                self.frame_length, self.hop_length,
                                axis=0, end=self.end,
                                endvalue='__end__', endmode=self.endmode)
                # need to remove padded value
                labels = np.asarray(
                    [labs_transform([j for j in i if '__end__' not in j])
                     for i in labels],
                    dtype=original_dtype
                )
                label_list.append(labels)
            y = label_list
        return name, X, y

    def shape_transform(self, shapes, indices):
        # ====== update the indices ====== #
        n = 0
        indices_new = {}
        for name, nb_samples in indices.iteritems():
            if nb_samples < self.frame_length:
                nb_samples = 0 if self.end == 'cut' else 1
            else:
                if self.end != 'cut':
                    nb_samples = np.ceil((nb_samples - self.frame_length) / self.hop_length)
                else:
                    nb_samples = np.floor((nb_samples - self.frame_length) / self.hop_length)
                nb_samples = int(nb_samples) + 1
            indices_new[name] = nb_samples
            n += nb_samples
        # ====== shape inference ====== #
        _ = []
        for shape in shapes:
            features_shape = (shape[-1],) if len(shape) >= 2 else ()
            mid_shape = tuple(shape[1:-1])
            _.append((n, self.frame_length,) + mid_shape + features_shape)
        return tuple(_), indices_new


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


class CreateFile(FeederRecipe):
    """ CreateFile
    Instead of divide a file into batches, return the whole file

    Parameters
    ----------
    return_name: bool
        whether return the name specifed in the indices

    Return
    ------
    [(name, index, data...), ...]

    """

    def __init__(self):
        super(CreateFile, self).__init__()

    def prepare(self, **kwargs):
        shuffle_level = kwargs.get('shuffle_level', 0)
        seed = kwargs.get('seed', None)
        self.rng = None
        if seed is not None and shuffle_level >= 1:
            self.rng = np.random.RandomState(seed=seed)
        self.batch_size = kwargs.get('batch_size', 1)

    def _to_numpy_array(self, x):
        if not is_string(x[0]) and len(set(i.shape[1:] for i in x)) == 1:
            return np.concatenate(x, axis=0)
        return np.array(x)

    def group(self, batch):
        # NOTE: each element in batch is one file
        # ====== shuffle ====== #
        if self.rng is not None:
            self.rng.shuffle(batch)
        for name, X, Y in batch:
            n = X[0].shape[0]
            ret = list(X) + list(Y)
            for i, (start, end) in enumerate(batching(n, self.batch_size)):
                r = [name, i] + [j[start:end] for j in ret]
                yield tuple(r)
