# ===========================================================================
# This module is created based on the code from 2 libraries: Lasagne and keras
# Original work Copyright (c) 2014-2015 keras contributors
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import division, absolute_import, print_function

import math
import numpy as np
import scipy as sp

from .cache_utils import cache_memory


# ===========================================================================
# Main
# ===========================================================================
def one_hot(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy

    Note
    ----
    if any class index in y is smaller than 0, then all of its one-hot
    values is 0.
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes), dtype='int32')
    for i, j in enumerate(y):
        if j >= 0:
            Y[i, j] = 1
    return Y


def unique_labels(y, key_func=None, return_labels=False):
    """
    Parameters
    ----------
    y: list, tuple, `numpy.ndarray`
        list of object that is label or contain label information.
    key_func: callabe
        a function transform each element of `y` into unique ID for labeling.
    return_labels: bool
        if True, return the ordered labels.

    Returns
    -------
    (callable, tuple):
        function that transform any object into unique label index
        (optional) list of ordered labels.
    """
    if not isinstance(y, (list, tuple, np.ndarray)):
        raise ValueError("`y` must be iterable (list, tuple, or numpy.ndarray).")
    # ====== Get an unique order of y ====== #
    if key_func is None or not callable(key_func):
        key_func = lambda _: str(_)
    sorted_labels = list(sorted(set(key_func(i) for i in y)))
    fast_index = {j: i for i, j in enumerate(sorted_labels)}

    def labels_index_func(x):
        x = key_func(x)
        if x in fast_index:
            return fast_index[x]
        raise ValueError("Cannot find key: '%s' in %s" %
                         (str(x), str(sorted_labels)))
    if return_labels:
        return labels_index_func, tuple(sorted_labels)
    return labels_index_func


# def replace(array, value, new_value):
#     if value is None:
#         return np.where(array == np.array(None), new_value, array)
#     return np.where(array == value, new_value, array)


# def is_ndarray(x):
#     return isinstance(x, np.ndarray)


# def masked_output(X, X_mask):
#     '''
#     Example
#     -------
#         X: [[1,2,3,0,0],
#             [4,5,0,0,0]]
#         X_mask: [[1,2,3,0,0],
#                  [4,5,0,0,0]]
#         return: [[1,2,3],[4,5]]
#     '''
#     res = []
#     for x, mask in zip(X, X_mask):
#         x = x[np.nonzero(mask)]
#         res.append(x.tolist())
#     return res


# def split_chunks(a, maxlen, overlap):
#     '''
#     Example
#     -------
#     >>> print(split_chunks(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 5, 1))
#     >>> [[1, 2, 3, 4, 5],
#          [4, 5, 6, 7, 8]]
#     '''
#     chunks = []
#     nchunks = int((max(a.shape) - maxlen) / (maxlen - overlap)) + 1
#     for i in xrange(nchunks):
#         start = i * (maxlen - overlap)
#         chunks.append(a[start: start + maxlen])

#     # ====== Some spare frames at the end ====== #
#     wasted = max(a.shape) - start - maxlen
#     if wasted >= (maxlen - overlap) / 2:
#         chunks.append(a[-maxlen:])
#     return chunks


# def ordered_set(seq):
#     seen = {}
#     result = []
#     for marker in seq:
#         if marker in seen: continue
#         seen[marker] = 1
#         result.append(marker)
#     return np.asarray(result)


# def shrink_labels(labels, maxdist=1):
#     '''
#     Example
#     -------
#     >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 1))
#     >>> [0, 1, 0, 1, 0, 4, 5, 4, 6, 0]
#     >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 2))
#     >>> [0, 1, 0, 4, 6, 0]
#     Notes
#     -----
#     Different from ordered_set, the resulted array still contain duplicate
#     if they a far away each other.
#     '''
#     maxdist = max(1, maxdist)
#     out = []
#     l = len(labels)
#     i = 0
#     while i < l:
#         out.append(labels[i])
#         last_val = labels[i]
#         dist = min(maxdist, l - i - 1)
#         j = 1
#         while (i + j < l and labels[i + j] == last_val) or (j < dist):
#             j += 1
#         i += j
#     return out


# def roll_sequences(sequences, maxlen, step, outlen, end='ignore'):
#     ''' Rolling sequences for generative RNN, for every sequence
#     of length=`maxlen` generate a small sequenc length=`outlen`, then move
#     the sequence by a number of `step` to get the next pair of (input, output)

#     Parameters
#     ----------
#     end : 'ignore', 'pad'(int)
#         ignore: just ignore the border of sequences
#         pad(int): pad given value
#     '''
#     if end == 'ignore':
#         pass
#     elif end == 'pad' or isinstance(end, (float, int, long)):
#         if end == 'pad':
#             end = 0.
#         end = np.cast[sequences.dtype](end)
#         # number of step
#         pad = (sequences.shape[0] - (maxlen + outlen)) / step
#         # ceil then multiply back => desire size for full sequence
#         pad = math.ceil(pad) * step + (maxlen + outlen)
#         pad = int(pad - sequences.shape[0])
#         if pad > 0:
#             pad = np.zeros((pad,) + sequences.shape[1:]) + pad
#             sequences = np.concatenate((sequences, pad), axis = 0)
#     # reupdate n value
#     n = int(math.ceil((sequences.shape[0] - (maxlen + outlen) + 1) / step))
#     rvalX = []
#     rvaly = []
#     for i in range(n):
#         start = i * step
#         end = start + maxlen
#         end_out = end + outlen
#         rvalX.append(sequences[start:end])
#         rvaly.append(sequences[end:end_out])
#     return np.asarray(rvalX), np.asarray(rvaly)
