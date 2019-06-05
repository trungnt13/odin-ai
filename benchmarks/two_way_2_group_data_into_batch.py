# python two_way_2_group_data_into_batch.py -m memory_profiler
# group: 165 + 10.9 MB and 24.3 (s/iter)
# group2: 164 + 43.6 MB (old method) and 15.6 (s/iter)
from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'theano,cpu,float32'
from six.moves import zip_longest, cPickle

import numpy as np
from odin import backend as K, nnet as N, fuel as F
from odin.utils import UnitTimer

from memory_profiler import profile

ds = F.Dataset('/home/trung/data/estonia_audio32')
indices = np.genfromtxt(ds['indices.csv'], dtype=str, delimiter=' ')

name, start, end = indices[0]
x0 = ds['mfcc'][int(start):int(end)]
x0 = (name, [x0, x0])

name, start, end = indices[1]
x1 = ds['mfcc'][int(start):int(end)]
x1 = (name, [x1, x1])

name, start, end = indices[2]
x2 = ds['mfcc'][int(start):int(end)]
x2 = (name, [x2, x2])


@profile
def group(batch):
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
    rng = np.random.RandomState(1234)
    batch_size = 64
    indices = [range((b[1][0].shape[0] - 1) // batch_size + 1)
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
        # # return the batches
        for i in range((ret[0].shape[0] - 1) // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            _ = [x[start:end] for x in ret]
            # always return tuple or list
            if _ is not None:
                yield _ if isinstance(_, (tuple, list)) else (ret,)


def group2(batch):
    rng = np.random.RandomState(1234)
    batch_size = 64
    length = len(batch[0]) # size of 1 batch
    nb_data = len(batch[0][1])
    X = [[] for i in range(nb_data)]
    Y = [[] for i in range(length - 2)]
    for b in batch:
        name = b[0]; data = b[1]; others = b[2:]
        # training data can be list of Data or just 1 Data
        for i, j in zip(X, data):
            i.append(j)
        # labels can be None (no labels given)
        for i, j in zip(Y, others):
            i.append(j)
    # ====== stack everything into big array ====== #
    X = [np.vstack(x) for x in X]
    shape0 = X[0].shape[0]
    Y = [np.concatenate(y, axis=0) for y in Y]
    # ====== shuffle for the whole batch ====== #
    if rng is not None:
        permutation = rng.permutation(shape0)
        X = [x[permutation] for x in X]
        Y = [y[permutation] if y.shape[0] == shape0 else y
             for y in Y]
    # ====== create batch ====== #
    for i in range((shape0 - 1) // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        # list of Data is given
        x = [x[start:end] for x in X]
        y = [y[start:end] for y in Y]
        ret = x + y
        # always return tuple or list
        if ret is not None:
            yield ret if isinstance(ret, (tuple, list)) else (ret,)


@profile
def test():
    with UnitTimer(12):
        for _ in range(12):
            for i, j in group((x0, x1, x2)):
                # print(i.shape, j.shape)
                pass
test()
