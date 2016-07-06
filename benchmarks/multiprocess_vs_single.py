from __future__ import print_function, division, absolute_import

import multiprocessing as mpi
from itertools import chain

import numpy as np

from odin import utils

ncpu = 2
ntasks = 8
jobs = [[10**3] * ntasks] * ncpu
# Big memory seems not affect the speed of inter-processes communication
dummy = np.ones((1000, 1000, 120), dtype='float64')
print('Size:', dummy.nbytes / 1024. / 1024., ' MB')

if True:
    with utils.UnitTimer():
        _ = []
        for i in chain(*jobs):
            count = 0
            for j in range(i):
                count += i * j ** i
            _.append(count)

if True:
    def work(jobs, results, dummy):
        count = 0
        for i in jobs:
            for j in range(i):
                # count += i * j ** i - dummy[12].sum().astype('int32')
                count += i * j ** i
            results.put(count)

    res = mpi.Queue()
    p = [mpi.Process(target=work, args=(i, res, dummy)) for i in jobs]

    with utils.UnitTimer():
        [i.start() for i in p]
        for _ in range(ntasks * ncpu):
            c = res.get()

        [i.join() for i in p]
        res.close()
