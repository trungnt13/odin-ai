from __future__ import print_function, division, absolute_import

import os
import timeit
import numpy as np

import h5py
from odin.fuel import MmapData

# ~ 116 MB of data
N = 240000
X = np.random.rand(N, 128).astype('float32')

# ====== test created dataset ====== #
start = timeit.default_timer()
hdf5 = h5py.File('tmp.hdf5', 'w')
print('Create HDF5   in:', timeit.default_timer() - start, 's')

start = timeit.default_timer()
mmap = MmapData('tmp.mmap', dtype='float32', shape=(None, 128))
print('Create Memmap in:', timeit.default_timer() - start, 's')

# ====== writing ====== #
print()
start = timeit.default_timer()
hdf5['X'] = X
print('Writing data to HDF5  :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
mmap.append(X)
print('Writing data to Memmap:', timeit.default_timer() - start, 's')

hdf5.flush(); hdf5.close()
mmap.flush(); mmap.close()

# ====== reading ====== #
print()
start = timeit.default_timer()
hdf5 = h5py.File('tmp.hdf5', 'r')
print('Load HDF5 data  :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
mmap = MmapData('tmp.mmap')
print('Load Memmap data:', timeit.default_timer() - start, 's')

print()
print('Test correctness of stored data')
print('HDF5  :', np.all(hdf5['X'][:] == X))
print('Memmap:', np.all(mmap[:] == X))

# ====== iterating over dataset ====== #
print()
start = timeit.default_timer()
for epoch in range(0, 3):
    for i in range(0, N, 256):
        x = X[i:i + 256]
print('Iterate Numpy data   :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
dat = hdf5['X']
for epoch in range(0, 3):
    for i in range(0, N, 256):
        x = dat[i:i + 256]
print('Iterate HDF5 data    :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
for epoch in range(0, 3):
    for i in range(0, N, 256):
        x = mmap[i:i + 256]
print('Iterate Memmap data  :', timeit.default_timer() - start, 's')

# ===========================================================================
# Clean-up
# ===========================================================================
if os.path.exists('tmp.hdf5'):
    os.remove('tmp.hdf5')
if os.path.exists('tmp.mmap'):
    os.remove('tmp.mmap')
