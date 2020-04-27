from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras
from tensorflow_probability.python.distributions import Binomial


class DiscreteDropout(keras.layers.Dropout):
  r""" Applies Binomial Dropout to the discrete input.

  ```none
  p ~ Bernoulli(p=dropout_rate)
  corrupted ~ Binomial(n=inputs, p=1-corrupt_rate)
  outputs = x * (1 - p) + corrupted * p
  ```

  Arguments:
    dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
    corrupt_rate: Float between 0 and 1. Fraction of the input values to drop.
    noise_shape: 1D integer tensor representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)` and
      you want the dropout mask to be the same for all timesteps,
      you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
  """

  def __init__(self,
               dropout_rate,
               corrupt_rate=0.2,
               noise_shape=None,
               seed=None,
               **kwargs):
    super().__init__(rate=dropout_rate,
                     noise_shape=noise_shape,
                     seed=seed,
                     **kwargs)
    self.corrupt_rate = corrupt_rate

  def call(self, inputs, training=None):
    if training:
      noise_shape = self._get_noise_shape(inputs)
      if noise_shape is None:
        noise_shape = tf.shape(inputs)
      inputs = tf.cast(inputs, dtype=self.dtype)
      # dropout mask
      mask = tf.random.uniform(shape=noise_shape,
                               minval=0.,
                               maxval=1.,
                               dtype=self.dtype,
                               seed=self.seed)
      keep_mask = tf.cast(mask >= self.rate, dtype=self.dtype)
      # corrupted values
      corrupted = Binomial(
          total_count=inputs,
          probs=tf.convert_to_tensor(1 - self.corrupt_rate,
                                     dtype=self.dtype)).sample(seed=self.seed)
      outputs = tf.multiply(inputs, keep_mask) + tf.multiply(
          corrupted, 1 - keep_mask)
    else:
      outputs = tf.identity(inputs)
    return outputs

  def get_config(self):
    config = super().get_config()
    config['corrupt_rate'] = self.corrupt_rate
    return config


class DropBlock(keras.layers.Layer):
  r"""DropBlock: a regularization method for convolutional neural networks.
  DropBlock is a form of structured dropout, where units in a contiguous
  region of a feature map are dropped together.

  DropBlock works better than dropout on convolutional layers due to the fact
  that activation units in convolutional layers are spatially correlated.

  Reference:
    Ghiasi, G., Lin, T.-Y., Le, Q.V., 2018. "DropBlock: A regularization
      method for convolutional networks". arXiv:1810.12890 [cs].
    https://github.com/google-research/simclr
  """

  def __init__(self,
               rate,
               blocksize,
               data_format='channels_last',
               seed=None,
               **kwargs):
    super().__init__(**kwargs)
    self.rate = tf.convert_to_tensor(rate, dtype=self.dtype, name='rate')
    self.blocksize = int(blocksize)
    self.data_format = str(data_format)
    self.seed = seed

  def call(self, inputs, training=None):
    keep_prob = 1. - self.rate
    if not training:
      return inputs
    blocksize = self.blocksize

    if self.data_format == 'channels_last':
      _, width, height, _ = net.get_shape().as_list()
    else:
      _, _, width, height = net.get_shape().as_list()
    if width != height:
      raise ValueError('Input tensor with width!=height is not supported.')

    blocksize = min(blocksize, width)
    # seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (1.0 - keep_prob) * width**2 / blocksize**2 / (
        width - blocksize + 1)**2

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
    valid_block_center = tf.logical_and(
        tf.logical_and(w_i >= int(blocksize // 2),
                       w_i < width - (blocksize - 1) // 2),
        tf.logical_and(h_i >= int(blocksize // 2),
                       h_i < width - (blocksize - 1) // 2))

    valid_block_center = tf.expand_dims(valid_block_center, 0)
    valid_block_center = tf.expand_dims(
        valid_block_center, -1 if self.data_format == 'channels_last' else 0)

    randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
    block_pattern = (
        1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
            (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if blocksize == width:
      block_pattern = tf.reduce_min(
          block_pattern,
          axis=[1, 2] if self.data_format == 'channels_last' else [2, 3],
          keepdims=True)
    else:
      if self.data_format == 'channels_last':
        ksize = [1, blocksize, blocksize, 1]
      else:
        ksize = [1, 1, blocksize, blocksize]
      block_pattern = -tf.nn.max_pool(
          -block_pattern,
          ksize=ksize,
          strides=[1, 1, 1, 1],
          padding='SAME',
          data_format='NHWC' if self.data_format == 'channels_last' else 'NCHW')

    percent_ones = tf.cast(tf.reduce_sum(
        (block_pattern)), tf.float32) / tf.cast(tf.size(block_pattern),
                                                tf.float32)
    net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
        block_pattern, net.dtype)
    return net
