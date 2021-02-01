from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, List, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Activation, Add, AvgPool2D,
                                     BatchNormalization, Concatenate)
from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D
from tensorflow.keras.layers import (Dropout, Flatten, GlobalAvgPool2D,
                                     GlobalMaxPool2D, Lambda, Layer, MaxPool2D,
                                     Multiply, Reshape, UpSampling2D,
                                     ZeroPadding2D)
from tensorflow.python.keras.applications.imagenet_utils import correct_pad
from typing_extensions import Literal

Conv2D = partial(_Conv2D, padding='same')
DepthwiseConv2D = partial(_DepthwiseConv2D, padding='same')


def last_layer(inputs: tf.Tensor) -> Layer:
  if not hasattr(inputs, '_keras_history'):
    raise ValueError(f'inputs of type {inputs} has no _keras_history')
  return inputs._keras_history.layer


# ===========================================================================
# Helpers
# ===========================================================================
class RemoveMCMCdim(Layer):

  def call(self, x: tf.Tensor, **kwargs):
    shape = tf.shape(x)
    shape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)
    return tf.reshape(x, shape)


class RestoreMCMCdim(Layer):

  def __init__(self, n_mcmc, **kwargs):
    super().__init__(**kwargs)
    self.n_mcmc = int(n_mcmc)

  def call(self, x: tf.Tensor, **kwargs):
    shape = tf.shape(x)
    shape = tf.concat([[self.n_mcmc, shape[0] // self.n_mcmc], shape[1:]],
                      axis=0)
    return tf.reshape(x, shape)


class Resampling2D(Layer):
  """ Support upsampling and downsampling """

  def __init__(self,
               size: Tuple[int, int] = (2, 2),
               mode: Literal['max', 'avg', 'global', 'pad', 'nearest',
                             'bilinear'] = 'avg',
               **kwargs):
    super().__init__(**kwargs)
    self.downsampling = False
    if mode in ('max', 'avg', 'global'):
      self.downsampling = True
    self.size = size
    self.mode = mode

  def build(self, input_shape):
    ## downsampling
    if self.downsampling:
      if self.mode == 'max':
        self.pool = MaxPool2D(self.size, padding='same')
      elif self.mode == 'max':
        self.pool = AvgPool2D(self.size, padding='same')
      elif self.mode == 'global':
        self.pool = GlobalAvgPool2D(name=name)
      else:
        raise NotImplementedError
    ## upsampling
    else:
      if self.mode == 'pad':
        if not isinstance(self.size, (tuple, list)):
          self.size = [self.size]
        if len(size) == 1:
          self.size = list(self.size) * 2
        # this doesn't take into account odd number
        self.pool = ZeroPadding2D(padding=[
            (i - 1) * s // 2 for i, s in zip(self.size, input_shape[1:])
        ],)
      else:
        self.pool = UpSampling2D(size=self.size, interpolation=self.mode)
    self.reshape = Reshape((1, 1, input_shape[-1]))
    return super().build(input_shape)

  def call(self, inputs, **kwargs):
    x = self.pool(inputs)
    if self.mode == 'global':
      x = self.reshape(x)
    return x


class ConvGating(Layer):
  """ Split the filters in two parts then applying sigmoid gating """

  def call(self, inputs):
    activation, gate_logits = tf.split(inputs, 2, axis=-1)
    gate = tf.nn.sigmoid(gate_logits)
    return tf.multiply(gate, activation)


class SqueezeExcitation(Layer):
  """ Squeeze and Excitation """

  def __init__(self,
               se_ratio: float = 0.25,
               pool_mode: Literal['max', 'avg'] = 'avg',
               activation: Callable[..., tf.Tensor] = tf.nn.swish,
               name: str = 'squeeze_excitation'):
    super().__init__(name=name)
    self.se_ratio = se_ratio
    self.pool_mode = pool_mode
    self.activation = activation

  def build(self, input_shape):
    self.filters_in = input_shape[-1]
    self.filters = max(1, int(self.filters_in * self.se_ratio))
    if self.pool_mode == 'avg':
      self.pool = GlobalAvgPool2D(name=f'{self.name}_pool')
    elif self.pool_mode == 'max':
      self.pool = GlobalMaxPool2D(name=f'{self.name}_pool')
    else:
      raise NotImplementedError(f'Invalid pool={self.pool_mode}')
    self.reshape = Reshape((1, 1, self.filters_in), name=f'{self.name}_reshape')
    self.conv1 = Conv2D(self.filters,
                        1,
                        activation=self.activation,
                        name=f'{self.name}_conv')
    self.conv2 = Conv2D(self.filters_in,
                        1,
                        activation='sigmoid',
                        name=f'{self.name}_proj')
    return super().build(input_shape)

  def call(self, inputs, training=None, **kwargs):
    x = inputs
    if 0 < self.se_ratio <= 1:
      x = self.pool(x)
      x = self.reshape(x)
      x = self.conv1(x)
      x = self.conv2(x)
      return tf.multiply(x, inputs)


class ResidualSequential(keras.Sequential):

  def __init__(self,
               merge_mode: Literal['add', 'concat', 'none'] = 'add',
               layers: Optional[List[Layer]] = None,
               name: Optional[str] = None):
    super().__init__(layers=layers, name=name)
    if merge_mode == 'add':
      self.merger = Add()
    elif merge_mode == 'concat':
      self.merger = Concatenate(axis=-1)
    else:
      self.merger = None

  def summary(self, line_length=None, positions=None, print_fn=None):
    text = []
    super().summary(line_length=line_length,
                    positions=positions,
                    print_fn=lambda s: text.append(s))
    text.insert(1, f'Merger: {self.merger}')
    if print_fn is None:
      print_fn = print
    for line in text:
      print_fn(line)

  def call(self, inputs, training=None, mask=None):
    x = super().call(inputs, training=training, mask=mask)
    if self.merger is None:
      return x
    return self.merger([inputs, x])


# ===========================================================================
# Main layers
# ===========================================================================
def merge(
    inputs: Optional[tf.Tensor] = None,
    outputs: Optional[tf.Tensor] = None,
    mode: Literal['add', 'concat'] = 'add',
) -> Union[tf.Tensor, Layer]:
  if merge == 'add':
    layer = Add()
  elif merge == 'concat':
    layer = Concatenate(axis=-1)
  if inputs is None or outputs is None:
    return layer
  return layer([inputs, outputs])


def dense(
    inputs: Optional[tf.Tensor] = None,
    units: int = 256,
    name: str = 'dense',
    **kwargs,
) -> Union[tf.Tensor, Layer]:
  layer = Dense(units=units, name=name, **kwargs)
  if inputs is None:
    return layer
  return layer(inputs)


def normalize_image(
    inputs: Optional[tf.Tensor] = None,
    name: str = 'normalize_image',
) -> Union[tf.Tensor, Layer]:
  layer = Lambda(lambda x: 2. * x / 255. - 1., name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def flatten(
    inputs: Optional[tf.Tensor] = None,
    name: str = 'flatten',
) -> Union[tf.Tensor, Layer]:
  layer = Flatten(name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def remove_mcmc_dim(
    inputs: Optional[tf.Tensor] = None,
    name: str = 'remove_mcmc_dim',
) -> Union[tf.Tensor, Layer]:
  layer = RemoveMCMCdim(name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def restore_mcmc_dim(
    inputs: Optional[tf.Tensor] = None,
    n_mcmc: int = 1,
    name: str = 'restore_mcmc_dim',
) -> Union[tf.Tensor, Layer]:
  layer = RestoreMCMCdim(n_mcmc=n_mcmc, name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def pooling2D(
    inputs: Optional[tf.Tensor] = None,
    size: Tuple[int, int] = (2, 2),
    mode: Literal['max', 'avg', 'global'] = 'avg',
    name: str = 'pooling2D',
) -> Union[tf.Tensor, Layer]:
  """ Pooling """
  layer = Resampling2D(size, mode, name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def unpooling2D(
    inputs: Optional[tf.Tensor] = None,
    size: Tuple[int, int] = (2, 2),
    mode: Literal['pad', 'nearest', 'bilinear'] = 'nearest',
    name: str = 'unpooling2D',
) -> tf.Tensor:
  """ Upsampling or Unpooling """
  layer = Resampling2D(size, mode, name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def project_1_1(
    inputs: Optional[tf.Tensor] = None,
    filters: int = 32,
    activation: Optional[Callable[..., tf.Tensor]] = None,
    use_bias: bool = True,
    name: str = 'project_11',
) -> Union[tf.Tensor, Layer]:
  """ Projecting using (1, 1) convolution """
  layer = Conv2D(filters=int(filters),
                 kernel_size=(1, 1),
                 activation=activation,
                 use_bias=use_bias,
                 name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def strides2D(
    inputs: Optional[tf.Tensor] = None,
    kernel_size: Tuple[int, int] = (3, 3),
    strides: Tuple[int, int] = (2, 2),
    filters: int = 32,
    activation: Optional[Callable[..., tf.Tensor]] = None,
    use_bias: bool = True,
    name: str = 'strides2D',
    **kwargs,
) -> Union[tf.Tensor, Layer]:
  """ Downsampling using convolutional strides """
  layer = Conv2D(filters=int(filters),
                 kernel_size=kernel_size,
                 strides=strides,
                 activation=activation,
                 use_bias=use_bias,
                 name=name,
                 **kwargs)
  if inputs is None:
    return layer
  return layer(inputs)


def dropout2D(
    inputs: Optional[tf.Tensor] = None,
    rate: float = 0.0,
    name: str = 'dropout2D',
) -> Union[tf.Tensor, Layer]:
  x = inputs
  if rate > 0:
    layer = Dropout(rate, noise_shape=(None, 1, 1, 1), name=name)
    if inputs is None:
      return layer
    x = layer(x)
  elif inputs is None:
    return Activation('linear', name=name)
  return x


# ===========================================================================
# Main bottleneck
# ===========================================================================
def residual(
    inputs: tf.Tensor,
    ratio: float = 0.5,
    filters_out: Optional[int] = None,
    se_ratio: float = 0.25,
    gated: bool = False,
    batchnorm: bool = True,
    batchnorm_kw: Dict[str, Any] = {},
    dropout: float = 0.0,
    kernel_size: Tuple[int, int] = (3, 3),
    order: Literal['bac', 'cba'] = 'cba',
    design: Literal['bottleneck', 'inverted'] = 'inverted',
    strides: Tuple[int, int] = (1, 1),
    activation: Callable[..., tf.Tensor] = tf.nn.swish,
    merge: Literal['add', 'concat'] = 'add',
    name: str = 'residual',
) -> tf.Tensor:
  kw = locals()
  kw.pop('design')
  kw.pop('ratio')
  if design == 'bottleneck':
    kw['shrink_ratio'] = ratio
    return residual_bottleneck(kw)
  elif design == 'inverted':
    kw['expand_ratio'] = ratio
    return residual_inverted(kw)
  raise NotImplementedError(f'No support for residual design: "{design}"')


def residual_bottleneck(
    inputs: Optional[tf.Tensor] = None,
    shrink_ratio: float = 0.5,
    filters_out: Optional[int] = None,
    se_ratio: float = 0.25,
    gated: bool = False,
    batchnorm: bool = True,
    batchnorm_kw: Dict[str, Any] = {},
    dropout: float = 0.0,
    kernel_size: Tuple[int, int] = (3, 3),
    order: Literal['bac', 'cba'] = 'cba',
    strides: Tuple[int, int] = (1, 1),
    activation: Callable[..., tf.Tensor] = tf.nn.swish,
    merge_mode: Literal['add', 'concat'] = 'add',
    name: str = 'residual_bottleneck',
) -> Union[tf.Tensor, Layer]:
  assert 0.0 < shrink_ratio <= 1.0
  ## prepare
  layers = []
  batchnorm_kw = dict(axis=3, **batchnorm_kw)
  filters = max(1, int(inputs.shape[-1] * shrink_ratio))
  if filters_out is None:
    filters_out = inputs.shape[-1]
  use_bias = not batchnorm
  x = inputs
  if np.any(np.asarray(strides) >= 2):
    x = ZeroPadding2D(padding=correct_pad(x, kernel_size),
                      name=name + f'{name}_pad')(x)
    pad_mode = 'valid'
  else:
    pad_mode = 'same'
  ## squeeze
  if order == 'bac':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  layers.append(
      Conv2D(filters=int(filters),
             kernel_size=kernel_size,
             use_bias=use_bias,
             strides=strides,
             padding=pad_mode,
             name=f'{name}_conv1'))
  if order == 'cba':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  ## squeeze
  if order == 'bac':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn2'))
    layers.append(Activation(activation, name=f'{name}_act2'))
  layers.append(
      Conv2D(filters=int(filters),
             kernel_size=kernel_size,
             name=f'{name}_conv2'))
  if order == 'cba':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn2'))
    layers.append(Activation(activation, name=f'{name}_act2'))
  ## expand
  if se_ratio:
    layers.append(
        SqueezeExcitation(se_ratio=se_ratio,
                          activation=activation,
                          name=f'{name}_se'))
  layers.append(
      Conv2D(filters=filters_out * (2 if gated else 1),
             kernel_size=(1, 1),
             use_bias=use_bias,
             name=f'{name}_proj1'))
  if batchnorm:
    layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn3'))
  if gated:
    layers.append(ConvGating(name=f'{name}_gating'))
  # no residual connection if strides > 1
  if filters_out == inputs.shape[-1] and np.all(np.asarray(strides) == 1):
    if dropout > 0:
      layers.append(dropout2D(dropout, name=f'{name}_drop'))
  else:
    merge_mode = 'none'
  ## final layer
  res = ResidualSequential(merge_mode=merge_mode, layers=layers, name=name)
  if inputs is None:
    return res
  return res(inputs)


def residual_inverted(
    inputs: Optional[tf.Tensor] = None,
    expand_ratio: float = 2.,
    filters_out: Optional[int] = None,
    se_ratio: float = 0.25,
    gated: bool = False,
    batchnorm: bool = True,
    batchnorm_kw: Dict[str, Any] = {},
    dropout: float = 0.0,
    kernel_size: Tuple[int, int] = (3, 3),
    order: Literal['bac', 'cba'] = 'cba',
    strides: Tuple[int, int] = (1, 1),
    activation: Callable[..., tf.Tensor] = tf.nn.swish,
    merge_mode: Literal['add', 'concat'] = 'add',
    name: str = 'residual_inverted',
) -> Union[tf.Tensor, Layer]:
  assert expand_ratio >= 1
  ## prepare
  layers = []
  batchnorm_kw = dict(axis=3, **batchnorm_kw)
  filters = max(1, int(expand_ratio * inputs.shape[-1]))
  if filters_out is None:
    filters_out = inputs.shape[-1]
  use_bias = not batchnorm
  x = inputs
  if np.any(np.asarray(strides) >= 2):
    layers.append(
        ZeroPadding2D(padding=correct_pad(x, kernel_size), name=f'{name}_pad'))
    pad_mode = 'valid'
  else:
    pad_mode = 'same'
  ## expand
  if order == 'bac':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  layers.append(
      Conv2D(filters=filters,
             kernel_size=kernel_size,
             padding=pad_mode,
             strides=strides,
             use_bias=use_bias,
             name=f'{name}_conv1'))
  if order == 'cba':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  ## squeeze
  if order == 'bac':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn2'))
    layers.append(Activation(activation, name=f'{name}_act2'))
  layers.append(DepthwiseConv2D(kernel_size=kernel_size,
                                name=f'{name}_dwconv1'))
  if order == 'cba':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn2'))
    layers.append(Activation(activation, name=f'{name}_act2'))
  ## final
  if se_ratio:
    layers.append(
        SqueezeExcitation(se_ratio=se_ratio,
                          activation=activation,
                          name=f'{name}_se'))
  layers.append(
      Conv2D(filters=filters_out * (2 if gated else 1),
             kernel_size=(1, 1),
             use_bias=use_bias,
             name=f'{name}_proj1'))
  if batchnorm:
    layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn3'))
  if gated:
    layers.append(ConvGating(name=f'{name}_gating'))
  # no residual connection if strides > 1
  if filters_out == inputs.shape[-1] and np.all(np.asarray(strides) == 1):
    if dropout > 0:
      layers.append(dropout2D(dropout, name=f'{name}_drop'))
  else:
    merge_mode = 'none'
  ## final layer
  res = ResidualSequential(merge_mode=merge_mode, layers=layers, name=name)
  if inputs is None:
    return res
  return res(inputs)
