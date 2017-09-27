from __future__ import print_function, division, absolute_import
import os
import math
import types
import inspect
import warnings
from collections import Counter, Mapping
from six.moves import cPickle
from six.moves import zip, zip_longest, range

import numpy as np

from odin.utils import (one_hot, is_string, ctext, is_number,
                        is_primitives, as_tuple, flatten_list,
                        is_pickleable)
from odin.utils.decorators import functionable

from .utils import MmapDict
from .feeder import FeederRecipe
from .recipes_shape import *
from .recipes_norm import *


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
        raise NotImplementedError
        self.ignore_not_found = ignore_not_found
        # ====== transcription ====== #
        share_dict = None
        if isinstance(transcription, (list, tuple)):
            share_dict = {i: j for i, j in transcription}
        elif isinstance(transcription, Mapping):
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
        elif isinstance(label_dict, Mapping):
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
        # append to trans list
        X.append(trans)
        return name, X

    def shape_transform(self, shapes):
        pass


# ===========================================================================
# Basic recipes
# ===========================================================================
class Filter(FeederRecipe):

    """
    Parameters
    ----------
    filter_func: function(name, X)
        return True if the given data is accepted for further processing
        otherwise False

    """

    def __init__(self, filter_func):
        super(Filter, self).__init__()
        if not callable(filter_func):
            raise ValueError('"filter_func" must be callable.')
        if not is_pickleable(filter_func):
            raise ValueError('"filter_func" must be pickle-able.')
        self.filter_func = filter_func

    def process(self, name, X):
        if self.filter_func(name, X):
            return name, X
        return None


# ===========================================================================
# Label processing
# ===========================================================================
class LabelOneHot(FeederRecipe):

    def __init__(self, nb_classes, data_idx=()):
        super(LabelOneHot, self).__init__()
        self.nb_classes = int(nb_classes)
        self.data_idx = data_idx

    def process(self, name, X):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(X),
                                  return_tuple=True)
        X_new = []
        for idx, x in enumerate(X):
            # transform into one-label y
            if idx in data_idx:
                x = np.array(x, dtype='int32')
                x = one_hot(x, nb_classes=self.nb_classes)
            X_new.append(x)
        return name, X_new

    def shape_transform(self, shapes):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(shapes),
                                  return_tuple=True)
        return [((shp[0], self.nb_classes), ids) if idx in data_idx else
                (shp, ids)
                for idx, (shp, ids) in enumerate(shapes)]


class Name2Trans(FeederRecipe):
    """ This function convert the name (in indices) to transcription
    for given data

    Parameters
    ----------
    converter_func: callbale (1 input arguments)
        for example, lambda name: 1 if 'true' in name else 0
        the return label then is duplicated for all data points in 1 file.
        (e.g. X.shape = (1208, 13), then, transcription=[ret] * 1208)
    ref_idx: int
        the new label will be duplicated based on the length of data
        at given idx

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

    def __init__(self, converter_func, ref_idx=0):
        super(Name2Trans, self).__init__()
        if not callable(converter_func):
            raise ValueError('"converter_func" must be callable.')
        if not is_pickleable(converter_func):
            raise ValueError('"converter_func" must be pickle-able.')
        self.converter_func = converter_func
        self.ref_idx = int(ref_idx)

    def process(self, name, X):
        # X: is a list of ndarray
        ref_idx = axis_normalize(axis=self.ref_idx, ndim=len(X),
                                 return_tuple=False)
        y = self.converter_func(name)
        y = np.array([y] * X[ref_idx].shape[0])
        X.append(y)
        return name, X

    def shape_transform(self, shapes):
        ref_idx = axis_normalize(axis=self.ref_idx, ndim=len(shapes),
                                 return_tuple=False)
        ref_shp, ref_ids = shapes[ref_idx]
        ids = list(ref_ids)
        n = int(ref_shp[0])
        shapes.append(((n,), ids))
        return shapes


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
        raise NotImplementedError
        if isinstance(vad, (list, tuple)):
            if len(vad) == 2:
                indices, data = vad
                if is_string(indices) and os.path.exists(indices):
                    indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
                vad = {name: data[int(start): int(end)]
                       for name, start, end in indices}
            else: # a list contain all information is given
                vad = {name: segments for name, segments in vad}
        elif not isinstance(vad, Mapping):
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

    def shape_transform(self, shapes):
        # ====== init ====== #
        if self.frame_length == 1:
            n_func = lambda start, end: end - start
            shape_func = lambda n, shape: (n,) + shape[1:]
        else:
            n_func = lambda start, end: self._estimate_number_of_sample(start, end)
            shape_func = lambda n, shape: (n, self.frame_length) + shape[1:]
        # ====== processing ====== #
        indices_new = []
        n = 0
        indices = []
        self.vad.iteritems()
        for name, length in indices:
            # not found find in original indices
            if name not in self.vad:
                continue
            # found the name, and update its indices
            n_file = 0
            segments = self.vad[name]
            for start, end in segments:
                if self.filter_vad(start, end):
                    n_file += n_func(start, end)
            indices_new.append((name, n_file))
            n += n_file
        shapes = tuple([shape_func(n, s) for s in shapes])
        return shapes, indices_new
