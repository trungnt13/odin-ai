import os

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
  perm = tf.transpose(z)
  for i in tf.range(z.shape[1]):
    z_i = tf.expand_dims(tf.random.shuffle(z[:, i]), axis=0)
    perm = tf.tensor_scatter_nd_update(perm, indices=[[i]], updates=z_i)
  return tf.transpose(perm)


for batch_size in (10, 100, 1024, 10000, 50000):
  for dim in (8, 16, 64, 128, 512, 1024):
    shape = (batch_size, dim)
    z = tf.reshape(tf.range(shape[0] * shape[1], dtype=tf.float64), shape)

    print("\n Shape:", shape)
    with UnitTimer():
      z1 = permute_dims1(z)
    with UnitTimer():
      z2 = permute_dims2(z)
    with UnitTimer():
      z3 = permute_dims3(z)

    tf.assert_equal(tf.reduce_mean(z1), tf.reduce_mean(z2), tf.reduce_mean(z3),
                    tf.reduce_mean(z))
