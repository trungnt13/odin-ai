import inspect
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
                                     Reshape, UpSampling2D,
                                     ZeroPadding2D, InputLayer)
from tensorflow.python.keras.applications.imagenet_utils import correct_pad
from tensorflow.python.keras.layers import Wrapper
from typing_extensions import Literal

Conv2D = partial(_Conv2D, padding='same')
DepthwiseConv2D = partial(_DepthwiseConv2D, padding='same')


def last_layer(inputs: tf.Tensor) -> Layer:
  """Return the last layer stored in the `_keras_history` of the output
  tensor"""
  if not hasattr(inputs, '_keras_history'):
    raise ValueError(f'inputs of type {inputs} has no _keras_history')
  return inputs._keras_history.layer


# ===========================================================================
# Helpers
# ===========================================================================
class Skip(Wrapper):
  """Skip connection"""

  def __init__(self, layer, coef=1.0, name=None, **kwargs):
    if name is None:
      name = layer.name
    super().__init__(layer, name=name, **kwargs)
    self.input_spec = self.layer.input_spec
    spec = inspect.getfullargspec(layer.call)
    args = set(spec.args + spec.kwonlyargs)
    self._call_args = args
    self.coef = float(coef)

  def get_config(self):
    cfg = super().get_config()
    cfg['coef'] = self.coef
    return cfg

  def call(self, inputs, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in self._call_args}
    outputs = self.layer.call(inputs, **kwargs)
    if self.coef != 0.:
      outputs = outputs + self.coef * inputs
    return outputs

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return f'<Skip "{self.name}" coef:{self.coef}>'


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
  """Support upsampling and downsampling"""

  def __init__(self,
               size: Tuple[int, int] = (2, 2),
               mode: Literal[
                 'max', 'avg', 'global', 'pad', 'nearest', 'bilinear'] = 'avg',
               **kwargs):
    super().__init__(**kwargs)
    self.downsampling = False
    if mode in ('max', 'avg', 'global'):
      self.downsampling = True
    self.size = size
    self.mode = mode
    self.pool = None
    self.reshape = None

  def get_config(self) -> Dict[str, Any]:
    return dict(size=self.size, mode=self.mode)

  def build(self, input_shape):
    ## downsampling
    if self.downsampling:
      if self.mode == 'max':
        self.pool = MaxPool2D(self.size, padding='same')
      elif self.mode == 'avg':
        self.pool = AvgPool2D(self.size, padding='same')
      elif self.mode == 'global':
        self.pool = GlobalAvgPool2D()
      else:
        raise NotImplementedError(f'No downsampling mode={self.mode}')
    ## upsampling
    else:
      if self.mode == 'pad':
        if not isinstance(self.size, (tuple, list)):
          self.size = [self.size]
        if len(self.size) == 1:
          self.size = list(self.size) * 2
        # this doesn't take into account odd number
        self.pool = ZeroPadding2D(padding=[
          (i - 1) * s // 2 for i, s in zip(self.size, input_shape[1:])
        ], )
      else:
        self.pool = UpSampling2D(size=self.size, interpolation=self.mode)
    self.reshape = Reshape((1, 1, input_shape[-1]))
    return super().build(input_shape)

  def call(self, inputs, **kwargs):
    x = self.pool(inputs)
    if self.mode == 'global':
      x = self.reshape(x)
    return x


class SigmoidGating(Layer):
  """Split the filters in two parts then applying sigmoid gating"""

  def call(self, inputs, **kwargs):
    activation, gate_logits = tf.split(inputs, 2, axis=-1)
    gate = tf.nn.sigmoid(gate_logits)
    return keras.layers.multiply([gate, activation])


class SqueezeExcitation(Layer):
  """Squeeze and Excitation"""

  def __init__(self,
               se_ratio: float = 0.25,
               pool_mode: Literal['max', 'avg'] = 'avg',
               activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.swish,
               conv_kw: Optional[Dict[str, Any]] = None,
               name: str = 'squeeze_excitation',
               **kwargs):
    super().__init__(name=name, **kwargs)
    if conv_kw is None:
      conv_kw = {}
    self.conv_k = conv_kw
    self.se_ratio = se_ratio
    self.pool_mode = pool_mode
    self.activation = activation

  def get_config(self):
    return dict(se_ratio=self.se_ratio, pool_mode=self.pool_mode,
                activation=self.activation)

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
                        name=f'{self.name}_conv',
                        **self.conv_kw)
    self.conv2 = Conv2D(self.filters_in,
                        1,
                        activation='sigmoid',
                        name=f'{self.name}_proj',
                        **self.conv_kw)
    return super().build(input_shape)

  def call(self, inputs, training=None, **kwargs):
    x = inputs
    if 0 < self.se_ratio <= 1:
      x = self.pool(x)
      x = self.reshape(x)
      x = self.conv1(x)
      x = self.conv2(x)
      return tf.multiply(x, inputs)


class SkipAndForget(Layer):
  """Add skip connection then gradually forget the connection during training"""

  def __init__(self, max_step: int = 10000, name: str = 'skip_and_forget'):
    super().__init__(name=name)
    self.max_step = tf.constant(max_step, dtype=self.dtype)
    self.step = tf.Variable(0., dtype=self.dtype, trainable=False)

  @property
  def skip_gate(self) -> tf.Tensor:
    return tf.maximum((self.max_step - self.step) / self.max_step, 0.)

  def call(self, inputs, training=None, **kwargs):
    if training:
      x, skip = inputs
      x = x + self.skip_gate * skip
      self.step.assign_add(1.)
      return x
    else:
      if isinstance(inputs, (tuple, list)):
        inputs = inputs[0]
      return inputs


class ResidualSequential(keras.Sequential):

  def __init__(self,
               layers: Optional[List[Layer]] = None,
               skip_mode: Literal['add', 'concat', 'none'] = 'add',
               skip_ratio: float = 1.0,
               name: Optional[str] = None):
    super().__init__(layers=layers, name=name)
    self.track_outputs = False
    self.skip_ratio = tf.convert_to_tensor(skip_ratio, dtype=self.dtype)
    self.skip_mode = skip_mode
    if skip_mode == 'add':
      self.merger = Add()
    elif skip_mode == 'concat':
      self.merger = Concatenate(axis=-1)
    else:
      self.merger = None

  def summary(self, line_length=None, positions=None, print_fn=None):
    from odin.backend.keras_helpers import layer2text
    return layer2text(self)

  def __repr__(self):
    text = f'Name: {self.name}\n'
    text += f'skip_mode: {self.skip_mode}\n'
    text += f'skip_ratio: {self.skip_ratio}\n'
    text += f'track_outputs: {self.track_outputs}\n'
    for layer in self.layers:
      layer: Layer
      text += f'{layer.__class__.__name__}:\n '
      for k, v in layer.get_config().items():
        if any(i in k for i in ('_initializer', '_regularizer', '_constraint')):
          continue
        text += f'{k}:{v} '
      text += '\n'
    return text[:-1]

  def call(self, inputs, training=None, mask=None):
    skip_inputs = inputs
    # === 1. normal Sequential network
    outputs = inputs  # handle the corner case where self.layers is empty
    last_outputs = []
    for layer in self.layers:
      # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
      # are the outputs of `layer` applied to `inputs`. At the end of each
      # iteration `inputs` is set to `outputs` to prepare for the next layer.
      kwargs = {}
      argspec = self._layer_call_argspecs[layer].args
      if 'mask' in argspec:
        kwargs['mask'] = mask
      if 'training' in argspec:
        kwargs['training'] = training

      outputs = layer(inputs, **kwargs)
      last_outputs.append((layer, outputs))

      if len(tf.nest.flatten(outputs)) != 1:
        raise ValueError('Sequential layer only support single outputs')
      # `outputs` will be the inputs to the next layer.
      inputs = outputs
      mask = getattr(outputs, '_keras_mask', None)
    # === 2. skip connection
    if self.merger is not None:
      outputs = self.merger([self.skip_ratio * skip_inputs, outputs])
    if self.track_outputs:
      outputs._last_outputs = tuple(last_outputs)
    return outputs


class MaskedConv2D(keras.layers.Conv2D):
  """Masked convolution 2D, type 'A' mask doesn't include the center entry,
  while type 'B' include the center of the kernel.

  References
  ----------
  Aaron van den Oord, et al. Conditional Image Generation with
      PixelCNN Decoders. In _Neural Information Processing Systems_, 2016.
      https://arxiv.org/abs/1606.05328
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               mask_type: Literal['A', 'B'] = 'A',
               padding='same',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    args = dict(locals())
    [args.pop(key) for key in ('self', '__class__', 'kwargs', 'mask_type')]
    super(MaskedConv2D, self).__init__(**args, **kwargs)
    h, w = self.kernel_size
    mask = np.zeros((h, w))
    mask[:h // 2, :] = 1.
    if 'a' in mask_type.lower():
      mask[h // 2, :w // 2] = 1.
    elif 'b' in mask_type.lower():
      mask[h // 2, :w // 2 + 1] = 1
    else:
      raise ValueError(f'mask_type must be "A" or "B", but given "{mask_type}"')
    mask = tf.convert_to_tensor(mask[:, :, np.newaxis, np.newaxis],
                                dtype=self.dtype,
                                name='kernel_mask')
    self.kernel_mask = mask

    # initializer
    old_initializer = self.kernel_initializer

    def init_and_apply_mask(*a, **k):
      return old_initializer(*a, **k) * mask

    self.kernel_initializer = init_and_apply_mask

    # constraint
    old_constraint = self.kernel_constraint

    def mask_constrain(w):
      if old_constraint is not None:
        w = old_constraint(w)
      return w * mask

    self.kernel_constraint = mask_constrain


# ===========================================================================
# Main layers
# ===========================================================================
def skip_and_forget(
    inputs: Optional[tf.Tensor] = None,
    max_step: int = 10000,
    name: str = 'skip_and_forget',
) -> Union[tf.Tensor, SkipAndForget]:
  """ Add skip connection then gradually forget the connection during
  training"""
  layer = SkipAndForget(max_step=max_step, name=name)
  if inputs is None:
    return layer
  return layer(inputs)


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
) -> Union[tf.Tensor, Lambda]:
  layer = Lambda(lambda x: 2. * x / 255. - 1., name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def flatten(
    inputs: Optional[tf.Tensor] = None,
    name: str = 'flatten',
) -> Union[tf.Tensor, Flatten]:
  layer = Flatten(name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def remove_mcmc_dim(
    inputs: Optional[tf.Tensor] = None,
    name: str = 'remove_mcmc_dim',
) -> Union[tf.Tensor, RemoveMCMCdim]:
  layer = RemoveMCMCdim(name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def restore_mcmc_dim(
    inputs: Optional[tf.Tensor] = None,
    n_mcmc: int = 1,
    name: str = 'restore_mcmc_dim',
) -> Union[tf.Tensor, RestoreMCMCdim]:
  layer = RestoreMCMCdim(n_mcmc=n_mcmc, name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def downsampling2D(
    inputs: Optional[tf.Tensor] = None,
    size: Tuple[int, int] = (2, 2),
    mode: Literal['max', 'avg', 'global'] = 'avg',
    name: Optional[str] = None,
) -> Union[tf.Tensor, Resampling2D]:
  """Pooling"""
  layer = Resampling2D(size, mode, name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def upsampling2D(
    inputs: Optional[tf.Tensor] = None,
    size: Tuple[int, int] = (2, 2),
    mode: Literal['pad', 'nearest', 'bilinear'] = 'nearest',
    name: Optional[str] = None,
) -> Union[tf.Tensor, Resampling2D]:
  """ Upsampling"""
  layer = Resampling2D(size, mode, name=name)
  if inputs is None:
    return layer
  return layer(inputs)


def project_1x1(
    inputs: Optional[tf.Tensor] = None,
    filters: int = 32,
    activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    use_bias: bool = True,
    name: str = 'project_1x1',
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
    activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
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
    name: Optional[str] = None,
) -> Union[tf.Tensor, Layer]:
  if rate > 0:
    layer = Dropout(rate, noise_shape=(None, 1, 1, 1), name=name)
    if inputs is None:
      return layer
    inputs = layer(inputs)
  elif inputs is None:
    return Activation('linear', name=name)
  return inputs


# ===========================================================================
# Main bottleneck
# ===========================================================================
def residual(
    inputs: Optional[tf.Tensor] = None,
    filters_in: Optional[int] = None,
    filters_out: Optional[int] = None,
    ratio: float = 2.0,
    se_ratio: float = 0.25,
    sigmoid_gating: bool = False,
    batchnorm: bool = True,
    batchnorm_kw: Optional[Dict[str, Any]] = None,
    dropout: float = 0.0,
    kernel_size: Tuple[int, int] = (3, 3),
    order: Literal['baw', 'wba'] = 'wba',
    design: Literal['bottleneck', 'inverted'] = 'inverted',
    strides: Tuple[int, int] = (1, 1),
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.swish,
    skip_mode: Literal['add', 'concat'] = 'add',
    skip_ratio: float = 1.0,
    name: Optional[str] = None,
) -> Union[tf.Tensor, ResidualSequential]:
  """A residual block, two designs are implemented:

  - 'wba', i.e. weight-batchnorm-activation (Tan et al. 2019):
    `X -> Conv -> BN -> ReLU -> DepthWise -> BN -> ReLU -> SE ->
    Conv -> BN -> Dropout -> Add(X)`
  - 'baw', i.e. batchnorm-activation-weight (He et al. 2016):
    `X -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Add(X)`

  All Convolutions are without biases if using BN

  Parameters
  ----------
  inputs : Tensor (optional)
      inputs tensor, if not provided (None), return the `ResidualSequential`
      layer
  ratio : float
      shrink ratio for bottleneck residual, and expand ratio for inverted
      residual.
  filters_in : int
      number of input filter, must be provided if inputs is None
  filters_out : int
      number of output filter for the output convolution
  se_ratio : float
      squeeze-and-excitation shrink ratio
  sigmoid_gating : boolean
      sigmoid gating the output convolution
  batchnorm : boolean
      enable batch normalization
  batchnorm_kw : Dict[str, Any] (optional)
      keyword arguments for batch normalization
  dropout : float
      dropout value on outputs before skip connection
  kernel_size : Tuple[int, int]
      filters dimensions
  order : {'baw', 'wba'}
      specific order of the residual block, 'baw' is
      batchnorm-activation-weight, and 'wba' is weight-batchnorm-activation,
      default 'wba'
  design : {'bottleneck', 'inverted'}
      residual block design, bottleneck residual or inverted residual with
      depthwise separated convolution.
  strides : Tuple[int, int]
      convolution strides
  activation : Callable[[tf.Tensor], tf.Tensor]
      activation function
  skip_mode : {'add', 'concat'}
      how to combine the outputs and the inputs in the final skip connection.
  skip_ratio : float
      scalar for scaling the inputs before adding to the skip connection
  name : str
      name for the layer

  Returns
  -------
  `tf.Tensor` or `ResidualSequential`

  References
  ----------
  He, K., et al. Identity Mappings in Deep Residual Networks. 2016
  Tan, M., et al. EfficientNet: Rethinking Model Scaling for Convolutional
      Neural Networks. 2019
  """
  if filters_in is None and inputs is None:
    raise ValueError('Unknown number of inputs filters, '
                     'either filters_in or inputs must be provided')
  if name is None:
    name = (
      'residual_bottleneck' if design == 'bottleneck' else 'residual_inverted')
  kw = locals()
  kw.pop('design')
  kw.pop('ratio')
  if design == 'bottleneck':
    kw['shrink_ratio'] = ratio
    return residual_bottleneck(**kw)
  elif design == 'inverted':
    kw['expand_ratio'] = ratio
    return residual_inverted(**kw)
  raise NotImplementedError(f'No support for residual design: "{design}"')


def residual_bottleneck(
    inputs: Optional[tf.Tensor] = None,
    filters_in: Optional[int] = None,
    filters_out: Optional[int] = None,
    shrink_ratio: float = 0.5,
    se_ratio: float = 0.25,
    sigmoid_gating: bool = False,
    batchnorm: bool = True,
    batchnorm_kw: Optional[Dict[str, Any]] = None,
    dropout: float = 0.0,
    kernel_size: Tuple[int, int] = (3, 3),
    order: Literal['baw', 'wba'] = 'wba',
    strides: Tuple[int, int] = (1, 1),
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.swish,
    skip_mode: Literal['add', 'concat'] = 'add',
    skip_ratio: float = 1.0,
    name: str = 'residual_bottleneck',
    **conv_kw,
) -> Union[tf.Tensor, ResidualSequential]:
  if batchnorm_kw is None:
    batchnorm_kw = {}
  assert 0.0 < shrink_ratio <= 1.0, (
    f'Bottleneck residual require 0 <= shrink_ratio <= 1, given {shrink_ratio}')
  ## prepare
  layers = []
  batchnorm_kw = dict(axis=3, **batchnorm_kw)
  if filters_in is None:
    filters_in = inputs.shape[-1]  # assume NHWC
  filters = max(1, int(filters_in * shrink_ratio))
  if filters_out is None:
    filters_out = filters_in
  use_bias = not batchnorm
  if np.any(np.asarray(strides) >= 2):
    inputs = ZeroPadding2D(padding=correct_pad(inputs, kernel_size),
                           name=name + f'{name}_pad')(inputs)
    pad_mode = 'valid'
  else:
    pad_mode = 'same'
  ## squeeze
  if order == 'baw':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  layers.append(
    Conv2D(filters=int(filters),
           kernel_size=kernel_size,
           use_bias=use_bias,
           strides=strides,
           padding=pad_mode,
           name=f'{name}_conv1',
           **conv_kw))
  if order == 'wba':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  ## squeeze
  if order == 'baw':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn2'))
    layers.append(Activation(activation, name=f'{name}_act2'))
  layers.append(
    Conv2D(filters=int(filters),
           kernel_size=kernel_size,
           name=f'{name}_conv2',
           **conv_kw))
  if order == 'wba':
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
    Conv2D(filters=filters_out * (2 if sigmoid_gating else 1),
           kernel_size=(1, 1),
           use_bias=use_bias,
           name=f'{name}_proj1',
           **conv_kw))
  if batchnorm:
    layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn3'))
  if sigmoid_gating:
    layers.append(SigmoidGating(name=f'{name}_gating'))
  # no residual connection if strides > 1
  if filters_out == filters_in and np.all(np.asarray(strides) == 1):
    if dropout > 0:
      layers.append(dropout2D(rate=dropout, name=f'{name}_drop'))
  else:
    skip_mode = 'none'
  ## final layer
  res = ResidualSequential(skip_mode=skip_mode, skip_ratio=skip_ratio,
                           layers=layers, name=name)
  if inputs is None:
    return res
  return res(inputs)


def residual_inverted(
    inputs: Optional[tf.Tensor] = None,
    filters_in: Optional[int] = None,
    filters_out: Optional[int] = None,
    expand_ratio: float = 2.,
    se_ratio: float = 0.25,
    sigmoid_gating: bool = False,
    batchnorm: bool = True,
    batchnorm_kw: Optional[Dict[str, Any]] = None,
    dropout: float = 0.0,
    kernel_size: Tuple[int, int] = (3, 3),
    order: Literal['baw', 'wba'] = 'wba',
    strides: Tuple[int, int] = (1, 1),
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.swish,
    skip_mode: Literal['add', 'concat'] = 'add',
    skip_ratio: float = 1.0,
    name: str = 'residual_inverted',
    **conv_kw,
) -> Union[tf.Tensor, ResidualSequential]:
  if batchnorm_kw is None:
    batchnorm_kw = {}
  assert expand_ratio >= 1, (
    f'Inverted residual only support expand_ratio >= 1, given {expand_ratio}')
  ## prepare
  layers = []
  batchnorm_kw = dict(axis=3, **batchnorm_kw)
  if filters_in is None:
    filters_in = inputs.shape[-1]  # assume NHWC
  filters = max(1, int(expand_ratio * filters_in))
  if filters_out is None:
    filters_out = filters_in
  use_bias = not batchnorm
  if np.any(np.asarray(strides) >= 2):
    layers.append(
      ZeroPadding2D(padding=correct_pad(inputs, kernel_size),
                    name=f'{name}_pad'))
    pad_mode = 'valid'
  else:
    pad_mode = 'same'
  ## expand
  if order == 'baw':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  layers.append(
    Conv2D(filters=filters,
           kernel_size=kernel_size,
           padding=pad_mode,
           strides=strides,
           use_bias=use_bias,
           name=f'{name}_conv1',
           **conv_kw))
  if order == 'wba':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn1'))
    layers.append(Activation(activation, name=f'{name}_act1'))
  ## squeeze
  if order == 'baw':
    if batchnorm:
      layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn2'))
    layers.append(Activation(activation, name=f'{name}_act2'))
  layers.append(DepthwiseConv2D(kernel_size=kernel_size,
                                name=f'{name}_dwconv1',
                                **conv_kw))
  if order == 'wba':
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
    Conv2D(filters=filters_out * (2 if sigmoid_gating else 1),
           kernel_size=(1, 1),
           use_bias=use_bias,
           name=f'{name}_proj1',
           **conv_kw))
  if batchnorm:
    layers.append(BatchNormalization(**batchnorm_kw, name=f'{name}_bn3'))
  if sigmoid_gating:
    layers.append(SigmoidGating(name=f'{name}_gating'))
  # no residual connection if strides > 1
  if filters_out == filters_in and np.all(np.asarray(strides) == 1):
    if dropout > 0:
      layers.append(dropout2D(rate=dropout, name=f'{name}_drop'))
  else:
    skip_mode = 'none'
  ## final layer
  res = ResidualSequential(skip_mode=skip_mode, skip_ratio=skip_ratio,
                           layers=layers, name=name)
  if inputs is None:
    return res
  return res(inputs)
