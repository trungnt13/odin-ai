from __future__ import print_function, division, absolute_import

from collections import defaultdict, Iterator

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


def summary(x, axis=None, shorten=False):
    if isinstance(x, Iterator):
        x = list(x)
    if isinstance(x, (tuple, list)):
        x = np.array(x)
    mean, std = np.mean(x, axis=axis), np.std(x, axis=axis)
    median = np.median(x, axis=axis)
    qu1, qu3 = np.percentile(x, [25, 75], axis=axis)
    min_, max_ = np.min(x, axis=axis), np.max(x, axis=axis)
    samples = ', '.join([str(i)
               for i in np.random.choice(x.ravel(), size=8, replace=False).tolist()])
    s = ""
    if not shorten:
        s += "***** Summary *****\n"
        s += "    Min : %s\n" % str(min_)
        s += "1st Qu. : %s\n" % str(qu1)
        s += " Median : %s\n" % str(median)
        s += "   Mean : %.8f\n" % mean
        s += "3rd Qu. : %s\n" % str(qu3)
        s += "    Max : %s\n" % str(max_)
        s += "-------------------\n"
        s += "    Std : %.8f\n" % std
        s += "#Samples : %d\n" % len(x)
        s += "Samples : %s\n" % samples
    else:
        s += "{#:%d|min:%s|qu1:%s|med:%s|mea:%.8f|qu3:%s|max:%s|std:%.8f}" %\
        (len(x), str(min_), str(qu1), str(median), mean, str(qu3), str(max_), std)
    return s
