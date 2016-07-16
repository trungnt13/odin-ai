# ===========================================================================
# Just for reference, very difficult to apply
# ===========================================================================
from __future__ import print_function, division, absolute_import
import numpy as np
import numba as nb

from odin import utils

X = np.random.rand(50000, 123)


def normal(x):
    idx = list(range(0, x.shape[0], 5))
    _ = [x[i:i + 21].ravel() for i in idx
         if (i + 21) <= x.shape[0]]
    x = np.asarray(_) if len(_) > 1 else _[0]
    # np.random.shuffle(x)
    return x


with utils.UnitTimer(12):
    for i in range(12):
        x1 = normal(X)
print(x1.shape)


tmp = np.ones((20000, 2583))


@nb.jit('f8[:,:](f8[:,:], f8[:,:])', locals={}, nopython=True, nogil=True, cache=True)
def fast(x, tmp):
    idx = list(range(0, x.shape[0], 5))
    count = 0
    for _, i in enumerate(idx):
        if (i + 21) <= x.shape[0]:
            tmp[_] = x[i:i + 21].ravel()
            count += 1
    # idx = np.arange(count)
    # np.random.shuffle(idx)
    return tmp[:count]


with utils.UnitTimer(12):
    for i in range(12):
        x2 = fast(X, tmp)
print(x2.shape)
print(np.sum(x1 - x2)) # must be 0.

# Numpy time: 0.107473 (sec)
# Numba time: 0.037539 (sec) # at least 3 times faster


with utils.UnitTimer(12):
    for i in range(12):
        np.ones((100000, 2583))

with utils.UnitTimer(12):
    for i in range(12):
        np.empty((100000, 2583))
# create np.empty array is extremely faster than np.ones
# Time: 1.278843 (sec)
# Time: 0.000014 (sec)
