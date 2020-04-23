import inspect
from numbers import Number

import tensorflow as tf
from six import string_types
from tensorflow.python import keras


def skip_connect(inputs, outputs, mode):
  ishape = inputs.shape
  oshape = outputs.shape
  if len(ishape) != len(oshape):
    n = abs(len(ishape) - len(oshape))
    # first expand
    for _ in range(n):
      if len(ishape) < len(oshape):
        inputs = tf.expand_dims(inputs, axis=1)
      else:
        outputs = tf.expand_dims(outputs, axis=1)
    # now repeat
    for i in range(1, n + 1):
      if len(ishape) < len(oshape):
        inputs = tf.repeat(inputs, outputs.shape[i], axis=i)
      else:
        outputs = tf.repeat(outputs, inputs.shape[i], axis=i)
  ### Concatenation
  if mode == 'concat':
    return tf.concat([outputs, inputs], axis=-1)
  ### Identity, a.k.a residual connection
  elif mode == 'identity':
    return inputs + outputs
  ### No support
  else:
    raise NotImplementedError("No support for skip connect mode: '%s'" % mode)
  return outputs


class SkipConnection(keras.Sequential):

  def __init__(self, layers, mode='concat', name=None):
    super().__init__(layers, name=name)
    self.mode = mode

  def call(self, inputs, training=None, mask=None):
    outputs = super().call(inputs, training=training, mask=mask)
    return skip_connect(inputs, outputs, self.mode)
