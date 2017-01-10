from __future__ import print_function, division, absolute_import

from collections import defaultdict

import numpy as np

from odin.utils import as_tuple


def freqcount(x):
    """ x: list, iterable
    Return
    ------
    dict: x(obj) -> freq(int)
    """
    freq = defaultdict(int)
    for i in x:
        freq[i] += 1
    return dict(freq)


def split_train_test(X, seed, split=0.7):
    """
    Note
    ----
    This function provides the same partitions with same given seed.
    """
    if seed is not None:
        np.random.seed(seed)
        X = X[np.random.permutation(X.shape[0])]
    split = np.array(as_tuple(split, t=float))
    if any(split[1:] < split[:-1]):
        split = np.cumsum(split)
    if any(split > 1.):
        raise ValueError('split must be < 1.0, but the given split is: %s' % split)
    split = [int(i * X.shape[0]) for i in split]
    if split[0] != 0:
        split = [0] + split
    if split[-1] != X.shape[0]:
        split.append(X.shape[0])
    ret = tuple([X[start:end] for start, end in zip(split[:-1], split[1:])])
    return ret
