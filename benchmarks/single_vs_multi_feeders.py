# ===========================================================================
# Benchmark:
#        | Time |
# -------|------|
# single |  44  |
# ncpu=1 |  43  |
# ncpu=2 |  24  |
# ncpu=3 |  20  |
# ncpu=4 |  18  |
# ncpu=6 |  16  |
# ===========================================================================
from __future__ import print_function, division, absolute_import

import numpy as np

import os
os.environ['ODIN'] = 'float32,cpu,theano,seed=12'

from odin import backend as K
from odin import nnet as N
from odin import fuel
from odin.utils import one_hot, UnitTimer

ds = fuel.load_mspec_test()
transcription_path = os.path.join(ds.path, 'alignment.dict')
indices_path = os.path.join(ds.path, 'indices.csv')

indices = np.genfromtxt(indices_path, dtype=str, delimiter=' ')
transcription = fuel.MmapDict(transcription_path)
mean = ds['mspec_mean'][:]
std = ds['mspec_mean'][:]
cache = 5


# ===========================================================================
# Single process
# ===========================================================================
def get_data():
    """ batch_size = 128 """
    batch = []
    batch_trans = []
    for name, start, end in indices:
        start = int(start)
        end = int(end)
        data = ds['mspec'][start:end]
        data = (data - data.mean(0)) / data.std(0)
        data = (data - mean) / std
        data = np.vstack([data[i:i + 21].reshape(1, -1)
                          for i in range(0, data.shape[0], 21)
                          if i + 21 < data.shape[0]])
        trans = transcription[name]
        trans = np.array([int(i) for i in trans.split(' ') if len(i) > 0])
        trans = np.vstack([trans[i + 11].reshape(1, -1)
                          for i in range(0, trans.shape[0], 21)
                          if i + 21 < trans.shape[0]])
        batch.append(data)
        batch_trans.append(trans)
        if len(batch) == cache:
            batch = np.vstack(batch)
            trans = one_hot(np.vstack(batch_trans).ravel(), 10)

            idx = np.random.permutation(batch.shape[0])
            batch = batch[idx]
            trans = trans[idx]

            i = 0
            while i < batch.shape[0]:
                start = i
                end = i + 128
                yield batch[start:end], trans[start:end]
                i = end

            batch = []
            batch_trans = []


# ===========================================================================
# Feeder
# ===========================================================================
data = fuel.Feeder(ds['mspec'], '/Users/trungnt13/tmp/fbank/indices.csv',
                   transcription=fuel.MmapDict(transcription_path),
                   ncpu=1, cache=5)# change ncpu here
data.set_batch(batch_size=128, seed=12)
data.set_recipes(fuel.Normalization(local_normalize=True,
                                mean=ds['mspec_mean'],
                                std=ds['mspec_std']),
                fuel.Stacking(left_context=10,
                              right_context=10,
                              shift=None),
                fuel.OneHotTrans(n_classes=10),
                fuel.CreateBatch()
)
print('Number of CPU for feeders:', data.ncpu)


# ===========================================================================
# Training
# ===========================================================================
X = K.placeholder(shape=(None, 2583), name='X')
y = K.placeholder(shape=(None, 10), name='y')

f = N.Sequence([
    N.Dense(128, activation=K.linear),
    N.Dense(10, activation=K.softmax)
])
y_ = f(X)
cost_train = K.mean(K.categorical_crossentropy(y_, y))
f_train = K.function([X, y], cost_train)


# ====== single process ====== #
with UnitTimer():
    for _, (i, j) in enumerate(get_data()):
        f_train(i, j)
print(_)

# ====== multi-processes ====== #
with UnitTimer():
    for _, (i, j) in enumerate(data):
        f_train(i, j)
print(_)
