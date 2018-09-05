from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np

import tensorflow as tf
from tensorflow import keras

from odin import backend as K, nnet as N

np.random.seed(5218)
W = np.random.rand(12, 8, 25).astype('float32')
X = K.variable(value=W, name='X')
W = np.random.rand(25, 13)
b = np.random.rand(13)
K.initialize_all_variables()

if False:
  model = keras.Sequential()
  model.add(
      keras.layers.TimeDistributed(keras.layers.Dense(13, name='Dense',
                                                      activation='linear'),
                                   input_shape=(8, 25), name="TDNN"))
  model.get_layer(name="TDNN").set_weights([W, b])
  y = model.predict(X, batch_size=None, steps=1)
  print(y.shape, y.sum(), (y**2).sum(), y.std())

def fn(o, i):
  return i + o

y = tf.scan(fn, elems=tf.range(1, 5), initializer=tf.constant(1), reverse=False)
print(K.eval(y))
y = tf.scan(fn, elems=tf.range(1, 5), reverse=False)
print(K.eval(y))
y = tf.scan(fn, elems=tf.range(1, 5), reverse=True)
print(K.eval(y))
y = tf.scan(fn, elems=tf.range(1, 5)[::-1], reverse=False)[::-1]
print(K.eval(y))

elems = np.arange(1, 5)
y = K.scan_tensors(fn, sequences=elems, backward=False)
print(K.eval(y))
y = K.scan_tensors(fn, sequences=elems, backward=True)
print(K.eval(y))

mask = np.random.randint(0, 2, size=(12, 8))
y = K.scan_tensors(fn, sequences=np.random.rand(12, 8, 25),
                   mask=mask,
                   axis=1, reshape_outputs=True)
exit()

f1 = N.Sequence(ops=[
    N.Dense(num_units=18),
    N.Dense(num_units=13)
], debug=True, name='Core')

f2 = N.TimeDistributed(ops=f1, time_axis=1)

y = f1(X)
