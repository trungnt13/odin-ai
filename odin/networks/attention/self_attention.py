from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras

class SoftAttention(keras.layers.Layer):
  """

  References
  ----------
  """

  def __init__(self, units, **kwargs):
    super(SoftAttention, self).__init__(**kwargs)
    self.units = int(units)

  def build(self, input_shape):
    self.W1 = keras.layers.Dense(self.units)
    self.W2 = keras.layers.Dense(self.units)
    self.V = keras.layers.Dense(1)
    return super().build(input_shape)

  def call(self, features, hidden):
    pass

    # # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # # hidden shape == (batch_size, hidden_size)
    # # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    # hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # # score shape == (batch_size, 64, hidden_size)
    # score = tf.math.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # # attention_weights shape == (batch_size, 64, 1)
    # # you get 1 at the last axis because you are applying score to self.V
    # attention_weights = tf.math.softmax(self.V(score), axis=1)

    # # context_vector shape after sum == (batch_size, hidden_size)
    # context_vector = attention_weights * features
    # context_vector = tf.reduce_sum(context_vector, axis=1)

    # return context_vector, attention_weights


class HardAttention(keras.Model):
  pass
