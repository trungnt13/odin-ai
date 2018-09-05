from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu'
import timeit
import random

import numpy as np

from odin.utils import UnitTimer, Progbar
from odin import backend as K, nnet as N

X1 = K.placeholder(shape=(10000, 1000), name='X1')
X2 = K.placeholder(shape=(10000, 1000), name='X2')

X3 = K.placeholder(shape=(10000, 2000), name='X3')

y1 = K.placeholder(shape=(1000, 2000), name='y1')
y2 = K.placeholder(shape=(2000, 3000), name='y2')
y3 = K.placeholder(shape=(3000, 4000), name='y3')
y4 = K.placeholder(shape=(4000, 5000), name='y4')

z = K.dot(X1, y1) + K.dot(X2, y1)
z = K.dot(z, y2)
z = K.dot(z, y3)
z = K.dot(z, y4)
print(z)
f = K.function([X1, X2, y1, y2, y3, y4], outputs=z)

X1 = X3[:, :1000]
X2 = X3[:, 1000:]
z1 = K.dot(X1, y1) + K.dot(X2, y1)
z1 = K.dot(z1, y2)
z1 = K.dot(z1, y3)
z1 = K.dot(z1, y4)
print(z1)
f1 = K.function([X3, y1, y2, y3, y4], outputs=z1)

v = [np.random.rand(*i.shape.as_list()) for i in [X1, X2, X3, y1, y2, y3, y4]]

f(v[0], v[1], v[3], v[4], v[5], v[6])
f1(v[2], v[3], v[4], v[5], v[6])

time = 0.
time1 = 0.
n = 100
prog = Progbar(target=80)
for _ in range(1, n + 1):
    prog.add(1)
    if _ % 2 == 0:
        start = timeit.timeit()
        f(v[0], v[1], v[3], v[4], v[5], v[6])
        time += timeit.timeit() - start
    else:
        start = timeit.timeit()
        f1(v[2], v[3], v[4], v[5], v[6])
        time1 += timeit.timeit() - start

print("Splitted input:", time)
print("Concatenated input:", time1)
