from __future__ import print_function, division, absolute_import

import warnings
from collections import Counter

import numpy as np

from .feeder import FeederRecipe
from odin.utils import (axis_normalize, is_pickleable, as_tuple, is_number)
from odin.preprocessing.signal import segment_axis


# ===========================================================================
# Helper
# ===========================================================================
def most_common(x):
    return Counter(x).most_common()[0][0]


def last_seen(x):
    return x[-1]


def in_middle(x):
    if len(x) % 2 == 0: # even
        return x[len(x) // 2]
    # odd
    return x[len(x) // 2 + 1]


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
    data_idx: int, list of int, None
        in case Feeders is given multiple Data, target_data is
        the index of Data that will be applied given indices.
        if None is given, the Slice is applied to all Data
    """

    def __init__(self, indices, axis, data_idx=None):
        super(Slice, self).__init__()
        # ====== validate axis ====== #
        if not isinstance(axis, int):
            raise ValueError('axis for Slice must be an integer.')
        if axis == 0 and data_idx is not None:
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
        self.data_idx = data_idx

    def process(self, name, X):
        X_new = []
        data_idx = axis_normalize(axis=self._data_idx,
                                  ndim=len(X),
                                  return_tuple=True)
        for _, x in enumerate(X):
            # apply the indices if _ in target_data
            if _ in data_idx:
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
            X_new.append(x)
        return name, X_new

    def _from_indices(self, n):
        """ This function estimates number of sample given indices """
        # slice indices
        indices = (self.indices,) if isinstance(self.indices, slice) else self.indices
        count = 0
        for idx in indices:
            idx = idx.indices(n)
            count += idx[1] - idx[0]
        return count

    def shape_transform(self, shapes):
        data_idx = axis_normalize(axis=self._data_idx,
                                  ndim=len(shapes),
                                  return_tuple=True)
        new_shapes = []
        # ====== check if first dimension is sliced ====== #
        for idx, (shp, ids) in enumerate(shapes):
            if idx in data_idx:
                if self.axis == 0:
                    ids = [(name, self._from_indices(length))
                           for name, length in ids]
                    n = sum(i[1] for i in ids)
                    shp = (n,) + shp[1:]
                else:
                    axis = self.axis % len(shp) # axis in case if negative
                    # int indices, just 1
                    n = self._from_indices(shp[axis])
                    shp = tuple([j if i != axis else n
                                 for i, j in enumerate(shp)])
            new_shapes.append((shp, ids))
        return new_shapes


class HStack(FeederRecipe):
    """Horizontal Stacking
    merge a list of np.ndarray horizontally (i.e. the last axis)
    into 1 np.array

    Note
    ----
    The new value will be `prepended` to the data list
    """

    def __init__(self, data_idx=None):
        super(HStack, self).__init__()
        self.data_idx = data_idx

    def process(self, name, X):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(X),
                                  return_tuple=True)
        if len(X) > 1 and len(data_idx) > 1:
            X_old = [x for i, x in enumerate(X) if i not in data_idx]
            X_new = [x for i, x in enumerate(X) if i in data_idx]
            X = list(as_tuple(np.hstack(X_new))) + X_old
        return name, X

    def shape_transform(self, shapes):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(shapes),
                                  return_tuple=True)
        # just 1 shape, nothing to merge
        if len(shapes) <= 1 or len(data_idx) <= 1:
            return shapes
        # merge
        old_shapes = [(shp, ids) for idx, (shp, ids) in enumerate(shapes)
                      if idx not in data_idx]
        new_shapes = [(shp, ids) for idx, (shp, ids) in enumerate(shapes)
                      if idx in data_idx]
        # ====== horizontal stacking ====== #
        first_shape, first_ids = new_shapes[0]
        new_shapes = (
            first_shape[:-1] + (sum(shp[-1] for shp, ids in new_shapes[1:]),),
            first_ids
        )
        return new_shapes + old_shapes


class ExpandDims(FeederRecipe):
    """ ExpandDims """

    def __init__(self, axis, data_idx=0):
        super(ExpandDims, self).__init__()
        self.axis = int(axis)
        self.data_idx = data_idx

    def process(self, name, X):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(X),
                                  return_tuple=True)
        X = [np.expand_dims(x, axis=self.axis)
             if i in data_idx else x
             for i, x in enumerate(X)]
        return name, X

    def shape_transform(self, shapes):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(shapes),
                                  return_tuple=True)
        new_shapes = []
        for idx, (shp, ids) in enumerate(shapes):
            if idx in data_idx:
                shp = list(shp)
                axis = self.axis if self.axis >= 0 else \
                    (len(shp) + 1 - self.axis)
                shp.insert(axis, 1)
                shp = tuple(shp)
            new_shapes.append((shp, ids))
        return new_shapes


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
    data_idx: int, list of int, or None
    label_idx: int, list of int, None, or empty list, tuple

    Note
    ----
    Stacking recipe transforms the data and labels in different way
    """

    def __init__(self, left_context=10, right_context=10, shift=None,
                 data_idx=None, label_idx=()):
        super(Stacking, self).__init__()
        self.left_context = left_context
        self.right_context = right_context
        self.n = int(left_context) + 1 + int(right_context)
        self.shift = int(right_context) if shift is None else int(shift)
        self.data_idx = data_idx
        self.label_idx = label_idx

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
             for i in idx
             if (i + self.n) <= len(trans)])
        return trans

    def process(self, name, X):
        if X[0].shape[0] < self.n: # not enough data points for stacking
            warnings.warn('name="%s" has shape[0]=%d, which is not enough to stack '
                          'into %d features.' % (name, X[0].shape[0], self.n))
            return None
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(X),
                                  return_tuple=True)
        label_idx = axis_normalize(axis=self.label_idx,
                                   ndim=len(X),
                                   return_tuple=True)
        data_idx = [i for i in data_idx
                    if i not in label_idx]
        # ====== stacking  ====== #
        X = [self._stacking(x) if idx in data_idx else x
             for idx, x in enumerate(X)]
        X = [self._middle_label(x) if idx in label_idx else x
             for idx, x in enumerate(X)]
        return name, X

    def shape_transform(self, shapes):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(shapes),
                                  return_tuple=True)
        label_idx = axis_normalize(axis=self.label_idx,
                                   ndim=len(shapes),
                                   return_tuple=True)
        data_idx = [i for i in data_idx
                    if i not in label_idx]
        # ====== update the shape and indices ====== #
        new_shapes = []
        for idx, (shp, ids) in enumerate(shapes):
            if idx in data_idx or idx in label_idx:
                # calculate new number of samples
                n = 0; ids_new = []
                for name, nb_samples in ids:
                    nb_samples = 1 + (nb_samples - self.n) // self.shift
                    ids_new.append((name, nb_samples))
                    n += nb_samples
                # for label_idx, number of features kept as original
                if idx in label_idx:
                    shp = (n,) + shp[1:]
                # for data_idx, only apply for 2D
                elif idx in data_idx:
                    if len(shp) > 2:
                        raise Exception('Stacking only support 2D array.')
                    nb_features = shp[-1] * self.n if len(shp) == 2 else \
                        self.n
                    shp = (n, nb_features)
            new_shapes.append((shp, ids))
        # ====== do the shape infer ====== #
        return new_shapes


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
    data_idx: int, list of int, None, or empty list, tuple
        list of index of all data will be applied
    label_idx: int, list of int, None, or empty list, tuple
        list of all label will be sequenced and applied the `label_transform`

    Return
    ------
    a ndarray

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    """

    def __init__(self, frame_length=256, hop_length=None,
                 end='cut', endvalue=0., endmode='post',
                 label_transform=last_seen,
                 data_idx=None, label_idx=()):
        super(Sequencing, self).__init__()
        self.frame_length = int(frame_length)
        self.hop_length = frame_length // 2 if hop_length is None else int(hop_length)
        if hop_length > frame_length:
            raise ValueError("hop_length=%d must be smaller than frame_length=%d"
                             % (hop_length, frame_length))
        self.end = end
        self.endvalue = endvalue
        self.endmode = endmode
        # ====== transform function ====== #
        if label_transform is not None and not callable(label_transform):
            raise ValueError("`label_transform` must be callable object, but "
                             "given type: %s" % type(label_transform))
        if not is_pickleable(label_transform):
            raise ValueError("`label_transform` must be pickle-able function, "
                             "i.e. it must be top-level function.")
        self.label_transform = label_transform
        # specific index
        self.data_idx = data_idx
        self.label_idx = label_idx

    def process(self, name, X):
        # ====== not enough data points for sequencing ====== #
        if X[0].shape[0] < self.frame_length and self.end == 'cut':
            warnings.warn('name="%s" has shape[0]=%d, which is not enough to sequence '
                          'into %d features.' % (name, X[0].shape[0], self.frame_length))
            return None
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(X),
                                  return_tuple=True)
        label_idx = axis_normalize(axis=self.label_idx,
                                   ndim=len(X),
                                   return_tuple=True)
        data_idx = [i for i in data_idx
                    if i not in label_idx]
        # ====== segnments X ====== #
        X_new = []
        for idx, x in enumerate(X):
            # for data
            if idx in data_idx:
                x = segment_axis(a=x,
                                 frame_length=self.frame_length,
                                 hop_length=self.hop_length, axis=0,
                                 end=self.end, endvalue=self.endvalue,
                                 endmode=self.endmode)
            # for label
            elif idx in label_idx:
                org_dtype = x.dtype
                x = segment_axis(a=np.asarray(x, dtype='str'),
                                 frame_length=self.frame_length,
                                 hop_length=self.hop_length,
                                 axis=0, end=self.end,
                                 endvalue='__end__',
                                 endmode=self.endmode)
                # need to remove padded value
                x = np.asarray(
                    [self.label_transform([j for j in i
                                           if '__end__' not in j])
                     for i in x],
                    dtype=org_dtype
                )
            X_new.append(x)
        return name, X_new

    def shape_transform(self, shapes):
        data_idx = axis_normalize(axis=self.data_idx,
                                  ndim=len(shapes),
                                  return_tuple=True)
        label_idx = axis_normalize(axis=self.label_idx,
                                   ndim=len(shapes),
                                   return_tuple=True)
        data_idx = [i for i in data_idx
                    if i not in label_idx]
        # ====== update the indices ====== #
        new_shapes = []
        for idx, (shp, ids) in enumerate(shapes):
            if idx in data_idx or idx in label_idx:
                # transoform the indices
                n = 0; ids_new = []
                for name, nb_samples in ids:
                    if nb_samples < self.frame_length:
                        nb_samples = 0 if self.end == 'cut' else 1
                    else:
                        if self.end != 'cut':
                            nb_samples = np.ceil(
                                (nb_samples - self.frame_length) / self.hop_length)
                        else:
                            nb_samples = np.floor(
                                (nb_samples - self.frame_length) / self.hop_length)
                        nb_samples = int(nb_samples) + 1
                    ids_new.append((name, nb_samples))
                    n += nb_samples
                # transoform the shape for data
                if idx in data_idx:
                    feat_shape = (shp[-1],) if len(shp) >= 2 else ()
                    mid_shape = tuple(shp[1:-1])
                    shp = (n, self.frame_length,) + mid_shape + feat_shape
                # for labels.
                elif idx in label_idx:
                    shp = (n,)
            new_shapes.append((shp, ids))
        return new_shapes
