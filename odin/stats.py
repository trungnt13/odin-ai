# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numbers import Number
from collections import defaultdict, Iterator, OrderedDict

import numpy as np

from odin.utils import as_tuple
from odin.config import get_rng


def stratified_sampling(x):
    pass


def train_valid_test_split(x, train=0.6, seed=None):
    # ====== check input ====== #
    if isinstance(x, dict):
        x = x.items()
    # ====== shuffle input ====== #
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(x)
    else:
        get_rng().shuffle(x)
    # ====== split ====== #
    N = len(x)
    train = int(float(train) * N)
    if train >= N:
        raise ValueError("train proportion must larger than 0 and smaller than 1.")
    valid = (N - train) // 2
    return x[:train], x[train:train + valid], x[train + valid:]


def freqcount(x, key=None, count=1, normalize=False, sort=False):
    """ x: list, iterable

    Parameters
    ----------
    key: callable
        extract the key from each item in the list
    count: callable, int
        extract the count from each item in the list
    normalize: bool
        if normalize, all the values are normalized from 0. to 1. (
        which sum up to 1. in total).
    sort: boolean
        if True, the list will be sorted in ascent order.

    Return
    ------
    dict: x(obj) -> freq(int)
    """
    freq = defaultdict(int)
    if key is None:
        key = lambda x: x
    if count is None:
        count = 1
    if isinstance(count, Number):
        _ = int(count)
        count = lambda x: _
    for i in x:
        c = count(i)
        i = key(i)
        freq[i] += c
    # always return the same order
    s = float(sum(v for v in freq.values()))
    freq = OrderedDict([(k, freq[k] / s if normalize else freq[k])
                        for k in sorted(freq.keys())])
    if sort:
        freq = OrderedDict(sorted(freq.items(), key=lambda x: x[1]))
    return freq


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


def KLdivergence(P, Q):
    """ KL(P||Q) = ∑_i • p_i • log(p_i/q_i)
    The smaller this number, the better P match Q distribution
    """
    if isinstance(P, dict) and isinstance(Q, dict):
        keys = sorted(P.keys())
        P = [P[k] for k in keys]
        Q = [Q[k] for k in keys]
    # ====== normalize to probability 0-1 ====== #
    P = np.array(P)
    P = P / np.sum(P, axis=-1)
    Q = np.array(Q)
    Q = Q / np.sum(Q, axis=-1)
    # ====== calcuate the KL-div ====== #
    D = 0
    for pi, qi in zip(P, Q):
        D += pi * np.log(pi / qi)
    return D
