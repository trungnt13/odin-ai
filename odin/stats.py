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


def summary(x, axis=None):
    if isinstance(x, (tuple, list)):
        x = np.array(x)
    mean, std = np.mean(x, axis=axis), np.std(x, axis=axis)
    median = np.median(x, axis=axis)
    qu1, qu3 = np.percentile(x, [25, 75], axis=axis)
    min_, max_ = np.min(x, axis=axis), np.max(x, axis=axis)
    samples = ', '.join(["%.8f" % i
               for i in np.random.choice(x.ravel(), size=8, replace=False).tolist()])
    s = ""
    s += "***** Summary *****\n"
    s += "    Min : %.8f\n" % min_
    s += "1st Qu. : %.8f\n" % qu1
    s += " Median : %.8f\n" % median
    s += "   Mean : %.8f\n" % mean
    s += "3rd Qu. : %.8f\n" % qu3
    s += "    Max : %.8f\n" % max_
    s += "-------------------\n"
    s += "    Std : %.8f\n" % std
    s += "Samples : %s\n" % samples
    return s
