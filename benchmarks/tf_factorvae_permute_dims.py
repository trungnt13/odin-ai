import os
from collections import defaultdict
from time import time

import tensorflow as tf
from tensorflow.python.autograph import to_code

from odin.utils import UnitTimer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)


def permute_dims1(z):
  z_perm = []
  for i in range(z.shape[1]):
    z_perm.append(tf.random.shuffle(z[:, i]))
  return tf.transpose(tf.stack(z_perm))


@tf.function
def permute_dims2(z):
  perm = tf.TensorArray(dtype=z.dtype,
                        size=z.shape[1],
                        dynamic_size=False,
                        clear_after_read=False,
                        element_shape=(z.shape[0],))
  for i in tf.range(z.shape[1]):
    z_i = tf.random.shuffle(z[:, i])
    perm = perm.write(i, z_i)
  return tf.transpose(perm.stack())


@tf.function
def permute_dims3(z):
  perm = tf.TensorArray(dtype=z.dtype,
                        size=z.shape[1],
                        dynamic_size=False,
                        clear_after_read=False,
                        element_shape=(z.shape[0],))
  ids = tf.range(z.shape[0], dtype=tf.int32)
  for i in tf.range(z.shape[1]):
    z_i = tf.gather(z[:, i], tf.random.shuffle(ids))
    perm = perm.write(i, z_i)
  return tf.transpose(perm.stack())


@tf.function
def permute_dims4(z):
  perm = tf.transpose(z)
  for i in tf.range(z.shape[1]):
    z_i = tf.expand_dims(tf.random.shuffle(z[:, i]), axis=0)
    perm = tf.tensor_scatter_nd_update(perm, indices=[[i]], updates=z_i)
  return tf.transpose(perm)


@tf.function
def permute_dims5(z):
  perm = tf.transpose(z)
  ids = tf.range(z.shape[0], dtype=tf.int32)
  for i in tf.range(z.shape[1]):
    z_i = tf.gather(z[:, i], tf.random.shuffle(ids))
    z_i = tf.expand_dims(z_i, axis=0)
    perm = tf.tensor_scatter_nd_update(perm, indices=[[i]], updates=z_i)
  return tf.transpose(perm)


benchmark = {}

for batch_size in (10, 100, 1024, 20000):
  for dim in (16, 64, 128, 512, 1024):
    shape = (batch_size, dim)
    z = tf.reshape(tf.range(shape[0] * shape[1], dtype=tf.float64), shape)

    print("\n Shape:", shape)
    permute_dims2(z + 1)
    permute_dims3(z + 1)
    permute_dims4(z + 1)
    permute_dims5(z + 1)

    start = time()
    z1 = permute_dims1(z)
    t1 = time() - start

    start = time()
    z2 = permute_dims2(z)
    t2 = time() - start

    start = time()
    z3 = permute_dims3(z)
    t3 = time() - start

    start = time()
    z4 = permute_dims4(z)
    t4 = time() - start

    start = time()
    z5 = permute_dims5(z)
    t5 = time() - start

    benchmark[shape] = [t1, t2, t3, t4, t5]
    tf.assert_equal(tf.reduce_mean(z1), tf.reduce_mean(z2), tf.reduce_mean(z3),
                    tf.reduce_mean(z))

for k, v in benchmark.items():
  print('(%s)' % ', '.join(['%5d' % i for i in k]),
        ', '.join(['%.3f' % i for i in v]))
