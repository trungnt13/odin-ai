import inspect
from typing import Sequence, Union, Optional

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.util import nest
from tensorflow_probability.python.layers import DistributionLambda

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


def _shape(s):
  return str(s).replace('None', '_')


def layer2text(layer: Layer,
               inc_name: bool = False,
               padding: str = '',
               input_shape: Optional[Sequence[Union[None, int]]] = None) -> str:
  assert isinstance(layer, keras.layers.Layer)
  cls_name = layer.__class__.__name__
  cls_name = cls_name[:10]
  if inc_name:
    name = padding + '[%-10s:%s]' % (cls_name, layer.name)
  else:
    name = padding + '[%-10s] ' % cls_name
  ## Sequential
  if isinstance(layer, keras.Model):
    if hasattr(layer, 'input_shape'):
      input_shape = layer.input_shape
    text = padding + \
           f"{name}built:{layer.built} name:{layer.name} input:{input_shape}\n"
    for i in layer.layers:
      layer_text = layer2text(i, inc_name=False, padding=padding + ' ',
                              input_shape=input_shape)
      if input_shape is not None:
        input_shape = i.compute_output_shape(input_shape)
      text += padding + layer_text + '\n'
    return text[:-1]
  ## Dense
  text = str(layer)
  if isinstance(layer, keras.layers.Wrapper):
    text = f' {layer2text(layer.layer)} <{layer.__class__.__name__}>'
  elif isinstance(layer, keras.layers.Dense):
    text = '%sunits:%d bias:%s activ:%s %s' % \
           (name, layer.units, layer.use_bias, layer.activation.__name__,
            layer.posterior_layer.__class__.__name__
            if hasattr(layer, 'posterior_layer') else '')
  ## Conv
  elif isinstance(layer, Conv):
    text = '%sf:%d k:%s s:%s d:%s pad:%s bias:%s activ:%s' % \
           (name, layer.filters, layer.kernel_size, layer.strides,
            layer.dilation_rate, layer.padding, layer.use_bias,
            layer.activation.__name__)
  ## Activation
  elif isinstance(layer, keras.layers.Activation):
    text = '%s%s' % (name, layer.activation.__name__)
  ## Dropout
  elif isinstance(layer, keras.layers.Dropout):
    text = '%sp=%.2f' % (name, layer.rate)
  ## BatchNorm
  elif isinstance(layer, keras.layers.BatchNormalization):
    text = padding + '[%-10s] axis=%s center:%s scale:%s trainable:%s' % \
           ('BatchRenorm' if layer.renorm else 'BatchNorm',
            [i for i in tf.nest.flatten(layer.axis)],
            layer.center, layer.scale, layer.trainable)
  ## Distribution layer
  elif isinstance(layer, DistributionLambda):
    fn = layer._make_distribution_fn
    kw = dict(inspect.getclosurevars(fn).nonlocals)
    kw.pop('self', None)
    cls_name = type(layer).__name__
    text = padding + (f"[{cls_name}] {kw} "
                      f"fn:{layer._convert_to_tensor_fn.__name__} "
                      f"kw:{layer._kwargs}")
  ## Lambda
  elif isinstance(layer, keras.layers.Lambda):
    spec = inspect.getfullargspec(layer.function)
    kw = dict(layer.arguments)
    if spec.defaults is not None:
      kw.update(spec.defaults)
    text = '%s <%s>(%s) default:%s' % \
           (name, layer.function.__name__,
            ', '.join(spec.args), kw)
  ## Reshape
  elif isinstance(layer, keras.layers.Reshape):
    text = '%sshape:%s' % (name, layer.target_shape)
  ## Flatten
  elif isinstance(layer, keras.layers.Flatten):
    text = '%s %s' % (name, layer.data_format)
  ## Embedding
  elif isinstance(layer, keras.layers.Embedding):
    text = '%s in_dim:%d out_dim:%d mask0:%s seq_len:%s' % (
      name, layer.input_dim, layer.output_dim, layer.mask_zero,
      layer.input_length)
  ## All others
  else:
    text = f'{name} {layer}'
  ### input, output shape
  if hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape'):
    text += f' ->{_shape(layer.output_shape)}'
  elif input_shape is not None:
    output_shape = layer.compute_output_shape(input_shape)
    text += f' ->{_shape(output_shape)}'
  return text
