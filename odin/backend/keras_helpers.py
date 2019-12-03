import inspect

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.util import nest

__all__ = [
    'copy_keras_metadata', 'has_keras_meta', 'add_trainable_weights',
    'layer2text'
]


def has_keras_meta(tensor):
  return hasattr(tensor, '_keras_history') and hasattr(tensor, '_keras_mask')


def copy_keras_metadata(keras_tensor, *new_tensors):
  if not hasattr(keras_tensor, '_keras_history') or \
  not hasattr(keras_tensor, '_keras_mask'):
    pass
  else:
    new_tensors = nest.flatten(new_tensors)
    history = keras_tensor._keras_history
    mask = keras_tensor._keras_mask
    for t in new_tensors:
      setattr(t, '_keras_history', history)
      setattr(t, '_keras_mask', mask)
  return new_tensors[0] if len(new_tensors) == 1 else new_tensors


def add_trainable_weights(layer, *variables):
  from odin.backend import is_variable
  variables = nest.flatten(variables)
  assert all(is_variable(v) for v in variables), \
  "All objects from variables must be instance of tensorflow.Variable"

  assert isinstance(layer, Layer), \
  "layer must be instance of tensorflow.python.keras.layers.Layer"

  variables = [v for v in variables if v not in layer._trainable_weights]
  layer._trainable_weights = layer._trainable_weights + variables
  return layer


def layer2text(layer):
  assert isinstance(layer, keras.layers.Layer)
  from tensorflow.python.keras.layers.convolutional import Conv
  text = str(layer)
  name = '[%s:"%s"]' % (layer.__class__.__name__, layer.name)
  if isinstance(layer, keras.layers.Dense):
    text = '%sunits:%d bias:%s activation:%s' % \
      (name, layer.units, layer.use_bias, layer.activation.__name__)
  elif isinstance(layer, Conv):
    text = '%sfilter:%d kernel:%s stride:%s dilation:%s pad:%s bias:%s activation:%s' % \
      (name, layer.filters, layer.kernel_size, layer.strides,
       layer.dilation_rate, layer.padding, layer.use_bias,
       layer.activation.__name__)
  elif isinstance(layer, keras.layers.Activation):
    text = '%s %s' % (name, layer.activation.__name__)
  elif isinstance(layer, keras.layers.Dropout):
    text = '%s p=%.2f' % (name, layer.rate)
  elif isinstance(layer, keras.layers.BatchNormalization):
    text = '[%s:"%s"]axis=%s center:%s scale:%s trainable:%s' % \
      ('BatchRenorm' if layer.renorm else 'BatchNorm', layer.name,
       [i for i in tf.nest.flatten(layer.axis)],
       layer.center, layer.scale, layer.trainable)
  elif isinstance(layer, keras.layers.Lambda):
    spec = inspect.getfullargspec(layer.function)
    kw = dict(layer.arguments)
    if spec.defaults is not None:
      kw.update(spec.defaults)
    text = '%s <%s>(%s) default:%s' % \
      (name, layer.function.__name__,
       ', '.join(spec.args), kw)
  elif isinstance(layer, keras.layers.Reshape):
    text = '%s %s' % (name, layer.target_shape)
  elif isinstance(layer, keras.layers.Flatten):
    text = '%s %s' % (name, layer.data_format)
  return text
