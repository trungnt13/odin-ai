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
