import inspect
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Sequence

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from odin.fuel import IterableDataset
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer, Activation, Flatten
from tensorflow_probability.python.distributions import Normal
from tensorflow_probability.python.distributions.pixel_cnn import \
  _PixelCNNNetwork
from tensorflow_probability.python.layers import DistributionLambda
from typeguard import typechecked
from typing_extensions import Literal

from odin.bay.distributions import (Blockwise, Categorical, ContinuousBernoulli,
                                    Distribution, Gamma,
                                    JointDistributionSequential,
                                    MixtureQuantizedLogistic, QuantizedLogistic,
                                    VonMises, Bernoulli, Independent)
from odin.networks.base_networks import SequentialNetwork

__all__ = [
  'CenterAt0',
  'PixelCNNDecoder',
  'mnist_networks',
  'dsprites_networks',
  'shapes3d_networks',
  'cifar_networks',
  'svhn_networks',
  'cifar10_networks',
  'cifar20_networks',
  'cifar100_networks',
  'celeba_networks',
  'get_networks',
  'get_optimizer_info',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _parse_distribution(
    input_shape: Tuple[int, int, int],
    distribution: Literal['qlogistic', 'mixqlogistic', 'bernoulli', 'gaussian'],
    n_components=10,
    n_channels=3) -> Tuple[int, DistributionLambda, Layer]:
  from odin.bay.layers import DistributionDense
  n_channels = input_shape[-1]
  last_layer = Activation('linear')
  # === 1. Quantized logistic
  if distribution == 'qlogistic':
    n_params = 2
    observation = DistributionLambda(
      lambda params: QuantizedLogistic(
        *[
          # loc
          p if i == 0 else
          # Ensure scales are positive and do not collapse to near-zero
          tf.nn.softplus(p) + tf.cast(tf.exp(-7.), tf.float32)
          for i, p in enumerate(tf.split(params, 2, -1))],
        low=0,
        high=255,
        inputs_domain='sigmoid',
        reinterpreted_batch_ndims=3),
      convert_to_tensor_fn=Distribution.sample,
      name='image'
    )
  # === 2. Mixture Quantized logistic
  elif distribution == 'mixqlogistic':
    n_params = MixtureQuantizedLogistic.params_size(
      n_components=n_components,
      n_channels=n_channels) // n_channels
    observation = DistributionLambda(
      lambda params: MixtureQuantizedLogistic(params,
                                              n_components=n_components,
                                              n_channels=n_channels,
                                              inputs_domain='sigmoid',
                                              high=255,
                                              low=0),
      convert_to_tensor_fn=Distribution.mean,
      name='image')
  # === 3. Bernoulli
  elif distribution == 'bernoulli':
    n_params = 1
    observation = DistributionDense(
      event_shape=input_shape,
      posterior=lambda p: Independent(Bernoulli(logits=p), len(input_shape)),
      projection=False,
      name="image")
  # === 4. Gaussian
  elif distribution == 'gaussian':
    n_params = 2
    observation = DistributionDense(
      event_shape=input_shape,
      posterior=lambda p: Independent(Normal(
        *tf.split(p, 2, -1)), len(input_shape)),
      projection=False,
      name="image")
  else:
    raise ValueError(f'No support for distribution {distribution}')
  return n_params, observation, last_layer


class CenterAt0(keras.layers.Layer):
  """Normalize the image pixel from [0, 1] to be centerized
  at 0 given range [-1, 1]
  """

  def __init__(self,
               enable: bool = True,
               div_255: bool = False,
               name: str = 'CenterAt0'):
    super().__init__(name=name)
    self.enable = bool(enable)
    self.div_255 = bool(div_255)

  def call(self, inputs, **kwargs):
    if self.enable:
      if self.div_255:
        inputs = inputs / 255.
      return 2. * inputs - 1.
    return inputs

  def get_config(self):
    return dict(enable=self.enable, div_255=self.div_255)

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return f'<Center [-1,1] enable:{self.enable} div255:{self.div_255}>'


class LogNorm(keras.layers.Layer):

  def __init__(self, enable: bool = True, name: str = 'LogNorm'):
    super().__init__(name=name)
    self.scale_factor = 10000
    self.eps = 1e-8
    self.enable = bool(enable)

  def call(self, x, **kwargs):
    if self.enable:
      x = x / (tf.reduce_sum(x, axis=-1, keepdims=True) + self.eps)
      x = x * self.scale_factor
      x = tf.math.log1p(x)
    return x

  def get_config(self):
    return dict(enable=self.enable)


def _prepare_cnn(activation=tf.nn.elu):
  # he_uniform is better for leaky_relu, relu
  # while he_normal is good for elu
  if activation is tf.nn.leaky_relu:
    init = tf.initializers.HeUniform()
  elif activation is tf.nn.elu:
    init = tf.initializers.HeNormal()
  else:
    init = tf.initializers.HeUniform()
  conv = partial(keras.layers.Conv2D,
                 padding='same',
                 kernel_initializer=init,
                 activation=activation)
  deconv = partial(keras.layers.Conv2DTranspose,
                   padding='same',
                   kernel_initializer=init,
                   activation=activation)
  return conv, deconv


class SkipSequential(keras.Model):

  def __init__(self, layers: Sequence[Layer] = (), name: str = 'SkipGenerator'):
    super().__init__(name=name)
    self.all_layers = list(layers)
    self.proj_layers = list()
    self.activation = list()
    linear = keras.activations.get('linear')
    for l in layers:
      if isinstance(l, keras.layers.Conv2DTranspose):
        self.activation.append(l.activation)
        self.proj_layers.append(
          keras.layers.Conv2D(l.filters, (1, 1),
                              padding='same',
                              activation='linear',
                              name=f'{l.name}_proj'))
        l.activation = linear
      else:
        self.proj_layers.append(None)
        self.activation.append(None)

  def build(self, input_shape):
    # this is a simple logic why keras don't do this by default!
    self._build_input_shape = input_shape
    return super().build(input_shape)

  @property
  def input_shape(self):
    return self._build_input_shape

  def call(self, x, **kwargs):
    z = tf.reshape(x, (-1, 1, 1, x.shape[-1]))
    for fn, proj, activation in zip(self.all_layers, self.proj_layers,
                                    self.activation):
      x = fn(x, **kwargs)
      # shortcut connection
      if proj is not None:
        z_proj = proj(z, **kwargs)
        x = activation(x + z_proj)
    return x


# ===========================================================================
# Basic Network
# ===========================================================================
@typechecked
def mnist_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = None,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
    is_semi_supervised: bool = False,
    is_hierarchical: bool = False,
    centerize_image: bool = True,
    skip_generator: bool = False,
    **kwargs,
) -> Dict[str, Layer]:
  """Network for MNIST dataset image size (28, 28, 1)"""
  from odin.bay.random_variable import RVconf
  from odin.bay.vi import BiConvLatents
  n_channels = int(kwargs.get('n_channels', 1))
  proj_dim = 196
  input_shape = (28, 28, n_channels)
  if zdim is None:
    zdim = 32
  conv, deconv = _prepare_cnn(activation=activation)
  n_params, observation, last_layer = _parse_distribution(
    input_shape, kwargs.get('distribution', 'bernoulli'))
  encoder = SequentialNetwork(
    [
      keras.layers.InputLayer(input_shape),
      CenterAt0(enable=centerize_image),
      conv(32, 5, strides=1, name='encoder0'),  # 28, 28, 32
      conv(32, 5, strides=2, name='encoder1'),  # 14, 14, 32
      conv(64, 5, strides=1, name='encoder2'),  # 14, 14, 64
      conv(64, 5, strides=2, name='encoder3'),  # 7 , 7 , 64
      keras.layers.Flatten(),
      keras.layers.Dense(proj_dim, activation='linear', name='encoder_proj')
    ],
    name='Encoder',
  )
  layers = [
    keras.layers.Dense(proj_dim, activation='linear', name='decoder_proj'),
    keras.layers.Reshape((7, 7, proj_dim // 49)),  # 7, 7, 4
    deconv(64, 5, strides=2, name='decoder2'),  # 14, 14, 64
    BiConvLatents(conv(64, 5, strides=1, name='decoder3'),  # 14, 14, 64
                  encoder=encoder.layers[3],
                  filters=16, kernel_size=14, strides=7,
                  disable=True,
                  name='latents2'),
    deconv(32, 5, strides=2, name='decoder4'),  # 28, 28, 32
    conv(32, 5, strides=1, name='decoder5'),  # 28, 28, 32
    conv(n_channels * n_params, 1, strides=1, activation='linear',
         name='decoder6'),
    last_layer
  ]
  layers = [i.layer if isinstance(i, BiConvLatents) and not is_hierarchical
            else i
            for i in layers]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='SkipDecoder')
  else:
    decoder = SequentialNetwork(layers=layers, name='Decoder')
  latents = RVconf((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVconf(
      10,
      'onehot',
      projection=True,
      name=kwargs.get('labels_name', 'digits'),
    ).create_posterior()
  return networks


fashionmnist_networks = partial(mnist_networks, labels_name='fashion')
binarizedmnist_networks = mnist_networks
omniglot_networks = partial(mnist_networks, n_channels=3)


# ===========================================================================
# CIFAR10
# ===========================================================================
class PixelCNNDecoder(keras.Model):

  def __init__(self, input_shape, zdim, n_components, dtype, name):
    super().__init__(name)
    # create the pixelcnn decoder
    self.pixelcnn = _PixelCNNNetwork(dropout_p=0.3,
                                     num_resnet=1,
                                     num_hierarchies=1,
                                     num_filters=32,
                                     num_logistic_mix=n_components,
                                     use_weight_norm=False)
    self.pixelcnn.build((None,) + input_shape)
    self.dense = keras.layers.Dense(units=int(np.prod(input_shape)),
                                    activation='tanh',
                                    name='decoder0')
    self.reshape = keras.layers.Reshape(input_shape)

  def call(self, inputs, training=None):
    h = self.dense(inputs)
    h = self.reshape(h)
    return self.pixelcnn(h, training=training)


@typechecked
def cifar_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = None,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
    is_semi_supervised: bool = False,
    is_hierarchical: bool = False,
    centerize_image: bool = True,
    skip_generator: bool = False,
    **kwargs,
) -> Dict[str, Layer]:
  """Network for CIFAR dataset image size (32, 32, 3)"""
  from odin.bay.random_variable import RVconf
  from odin.bay.vi.autoencoder.hierarchical_vae import BiConvLatents
  if zdim is None:
    zdim = 256
  n_channels = kwargs.get('n_channels', 3)
  input_shape = (32, 32, n_channels)
  conv, deconv = _prepare_cnn(activation=activation)
  n_classes = kwargs.get('n_classes', 10)
  proj_dim = 8 * 8 * 8
  ## output distribution
  n_params, observation, last_layer = _parse_distribution(
    input_shape, kwargs.get('distribution', 'qlogistic'))
  ## encoder
  encoder = SequentialNetwork(
    [
      CenterAt0(enable=centerize_image),
      conv(32, 4, strides=1, name='encoder0'),  # 32, 32, 32
      conv(32, 4, strides=2, name='encoder1'),  # 16, 16, 32
      conv(64, 4, strides=1, name='encoder2'),  # 16, 16, 64
      conv(64, 4, strides=2, name='encoder3'),  # 8, 8, 64
      keras.layers.Flatten(),
      keras.layers.Dense(proj_dim, activation='linear', name='encoder_proj')
    ],
    name='Encoder',
  )
  layers = [
    keras.layers.Dense(proj_dim, activation='linear', name='decoder_proj'),
    keras.layers.Reshape((8, 8, proj_dim // 64)),  # 8, 8, 4
    deconv(64, 4, strides=2, name='decoder1'),  # 16, 16, 64
    BiConvLatents(conv(64, 4, strides=1, name='decoder2'),  # 16, 16, 64
                  encoder=encoder.layers[3],
                  filters=32, kernel_size=8, strides=4,
                  disable=True,
                  name='latents1'),
    deconv(32, 4, strides=2, name='decoder3'),  # 32, 32, 32
    BiConvLatents(conv(32, 4, strides=1, name='decoder4'),  # 32, 32, 32
                  encoder=encoder.layers[1],
                  filters=16, kernel_size=8, strides=4,
                  disable=True,
                  name='latents2'),
    conv(n_channels * n_params,  # 32, 32, 3
         1,
         strides=1,
         activation='linear',
         name='decoder5'),
    last_layer
  ]
  layers = [i.layer if isinstance(i, BiConvLatents) and not is_hierarchical
            else i
            for i in layers]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='SkipDecoder')
  else:
    decoder = SequentialNetwork(layers=layers, name='Decoder')
  ## others
  latents = RVconf((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  # create the observation of MixtureQuantizedLogistic
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVconf(n_classes,
                                'onehot',
                                projection=True,
                                name='labels').create_posterior()
  return networks


cifar10_networks = partial(cifar_networks, n_classes=10)
cifar20_networks = partial(cifar_networks, n_classes=20)
cifar100_networks = partial(cifar_networks, n_classes=100)
svhn_networks = partial(cifar_networks, n_classes=10)


# ===========================================================================
# dSprites
# ===========================================================================
def _dsprites_distribution(x: tf.Tensor) -> Blockwise:
  # NOTE: tried Continuous Bernoulli for dSPrites, but leads to
  # more unstable training in semi-supervised learning.
  dtype = x.dtype
  py = JointDistributionSequential([
    VonMises(loc=x[..., 0],
             concentration=tf.math.softplus(x[..., 1]),
             name='orientation'),
    Gamma(concentration=tf.math.softplus(x[..., 2]),
          rate=tf.math.softplus(x[..., 3]),
          name='scale'),
    Categorical(logits=x[..., 4:7], dtype=dtype, name='shape'),
    Bernoulli(logits=x[..., 7], dtype=dtype, name='x_position'),
    Bernoulli(logits=x[..., 8], dtype=dtype, name='y_position'),
  ])
  return Blockwise(py, name='shapes2d')


@typechecked
def dsprites_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = None,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
    is_semi_supervised: bool = False,
    is_hierarchical: bool = False,
    centerize_image: bool = True,
    skip_generator: bool = False,
    **kwargs,
) -> Dict[str, Layer]:
  from odin.bay.random_variable import RVconf
  from odin.bay.vi.autoencoder import BiConvLatents
  if zdim is None:
    zdim = 10
  n_channels = int(kwargs.get('n_channels', 1))
  input_shape = (64, 64, n_channels)
  conv, deconv = _prepare_cnn(activation=activation)
  proj_dim = kwargs.get('proj_dim', None)
  if proj_dim is None:
    proj_dim = 128 if n_channels == 1 else 256
  else:
    proj_dim = int(proj_dim)
  n_params, observation, last_layer = _parse_distribution(
    input_shape, kwargs.get('distribution', 'bernoulli'))
  encoder = SequentialNetwork(
    [
      CenterAt0(enable=centerize_image),
      conv(32, 4, strides=2, name='encoder0'),
      conv(32, 4, strides=2, name='encoder1'),
      conv(64, 4, strides=2, name='encoder2'),
      conv(64, 4, strides=2, name='encoder3'),
      keras.layers.Flatten(),
      keras.layers.Dense(proj_dim, activation='linear', name='encoder_proj')
    ],
    name='Encoder',
  )
  # layers = [
  #   keras.layers.Dense(proj_dim, activation='linear', name='decoder_proj'),
  #   keras.layers.Reshape((4, 4, proj_dim // 16)),
  #   BiConvLatents(deconv(64, 4, strides=2, name='decoder1'),
  #                 encoder=encoder.layers[3],
  #                 filters=32, kernel_size=8, strides=4,
  #                 disable=True, name='latents1'),
  #   deconv(64, 4, strides=2, name='decoder2'),
  #   BiConvLatents(deconv(32, 4, strides=2, name='decoder3'),
  #                 encoder=encoder.layers[1],
  #                 filters=16, kernel_size=8, strides=4,
  #                 disable=True, name='latents2'),
  #   deconv(32, 4, strides=2, name='decoder4'),
  #   # NOTE: this last projection layer with linear activation is crucial
  #   # otherwise the distribution parameterized by this layer won't converge
  #   conv(n_channels * n_params,
  #        1,
  #        strides=1,
  #        activation='linear',
  #        name='decoder6'),
  #   last_layer
  # ]
  layers = [
    keras.layers.Dense(proj_dim, activation='linear', name='decoder_proj'),
    keras.layers.Reshape((4, 4, proj_dim // 16)),
    BiConvLatents(deconv(64, 4, strides=2, name='decoder1'),
                  encoder=encoder.layers[3],
                  filters=32, kernel_size=8, strides=4,
                  disable=True,
                  name='latents2'),
    deconv(64, 4, strides=2, name='decoder2'),
    deconv(32, 4, strides=2, name='decoder3'),
    deconv(32, 4, strides=2, name='decoder4'),
    # NOTE: this last projection layer with linear activation is crucial
    # otherwise the distribution parameterized by this layer won't converge
    conv(n_channels * n_params,
         1,
         strides=1,
         activation='linear',
         name='decoder6'),
    last_layer
  ]
  layers = [i.layer if isinstance(i, BiConvLatents) and not is_hierarchical
            else i
            for i in layers]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='SkipDecoder')
  else:
    decoder = SequentialNetwork(layers=layers, name='Decoder')
  latents = RVconf((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    from odin.bay.layers.dense_distribution import DistributionDense
    # TODO: update
    networks['labels'] = DistributionDense(event_shape=(5,),
                                           posterior=_dsprites_distribution,
                                           units=9,
                                           name='geometry2d')
  return networks


dsprites0_networks = dsprites_networks


# ===========================================================================
# Shapes 3D
# ===========================================================================
def _shapes3d_distribution(x: tf.Tensor) -> Blockwise:
  dtype = x.dtype
  py = JointDistributionSequential([
    VonMises(loc=x[..., 0],
             concentration=tf.math.softplus(x[..., 1]),
             name='orientation'),
    Gamma(concentration=tf.math.softplus(x[..., 2]),
          rate=tf.math.softplus(x[..., 3]),
          name='scale'),
    Categorical(logits=x[..., 4:8], dtype=dtype, name='shape'),
    ContinuousBernoulli(logits=x[..., 8], name='floor_hue'),
    ContinuousBernoulli(logits=x[..., 9], name='wall_hue'),
    ContinuousBernoulli(logits=x[..., 10], name='object_hue'),
  ])
  return Blockwise(py, name='shapes3d')


def shapes3d_networks(qz: str = 'mvndiag',
                      zdim: Optional[int] = None,
                      activation: Union[Callable, str] = tf.nn.elu,
                      is_semi_supervised: bool = False,
                      is_hierarchical: bool = False,
                      centerize_image: bool = True,
                      skip_generator: bool = False,
                      small: bool = False,
                      **kwargs) -> Dict[str, Layer]:
  if zdim is None:
    zdim = 6
  if small:
    networks = cifar_networks(qz=qz,
                              zdim=zdim,
                              activation=activation,
                              is_semi_supervised=False,
                              is_hierarchical=is_hierarchical,
                              centerize_image=centerize_image,
                              skip_generator=skip_generator,
                              distribution='bernoulli')
  else:
    networks = dsprites_networks(qz=qz,
                                 zdim=zdim,
                                 activation=activation,
                                 is_semi_supervised=False,
                                 is_hierarchical=is_hierarchical,
                                 centerize_image=centerize_image,
                                 skip_generator=skip_generator,
                                 distribution='bernoulli',
                                 n_channels=3)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    # TODO: update
    networks['labels'] = DistributionDense(event_shape=(6,),
                                           posterior=_shapes3d_distribution,
                                           units=11,
                                           name='geometry3d')
  return networks


shapes3dsmall_networks = partial(shapes3d_networks, small=True)
shapes3d0_networks = shapes3d_networks


# ===========================================================================
# Halfmoons
# ===========================================================================
def _halfmoons_distribution(x: tf.Tensor) -> Blockwise:
  dtype = x.dtype
  py = JointDistributionSequential([
    Gamma(concentration=tf.math.softplus(x[..., 0]),
          rate=tf.math.softplus(x[..., 1]),
          name='x'),
    Gamma(concentration=tf.math.softplus(x[..., 2]),
          rate=tf.math.softplus(x[..., 3]),
          name='y'),
    Gamma(concentration=tf.math.softplus(x[..., 4]),
          rate=tf.math.softplus(x[..., 5]),
          name='color'),
    Categorical(logits=x[..., 6:10], dtype=dtype, name='shape'),
  ])
  return Blockwise(py, name='shapes3d')


def halfmoons_networks(qz: str = 'mvndiag',
                       zdim: Optional[int] = None,
                       activation: Union[Callable, str] = tf.nn.elu,
                       is_semi_supervised: bool = False,
                       is_hierarchical: bool = False,
                       centerize_image: bool = True,
                       skip_generator: bool = False,
                       **kwargs) -> Dict[str, Layer]:
  if zdim is None:
    zdim = 5
  networks = dsprites_networks(qz=qz,
                               zdim=zdim,
                               activation=activation,
                               is_semi_supervised=False,
                               is_hierarchical=is_hierarchical,
                               centerize_image=centerize_image,
                               skip_generator=skip_generator,
                               distribution='bernoulli',
                               n_channels=3)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    networks['labels'] = DistributionDense(event_shape=(4,),
                                           posterior=_halfmoons_distribution,
                                           units=10,
                                           name='geometry3d')
  return networks


# ===========================================================================
# CelebA
# ===========================================================================
def _celeba_distribution(x: tf.Tensor) -> Blockwise:
  dtype = x.dtype
  py = ContinuousBernoulli(logits=x)
  return Independent(py, 1, name='attributes')


def celeba_networks(qz: str = 'mvndiag',
                    zdim: Optional[int] = None,
                    activation: Union[Callable, str] = tf.nn.elu,
                    is_semi_supervised: bool = False,
                    is_hierarchical: bool = False,
                    centerize_image: bool = True,
                    skip_generator: bool = False,
                    n_labels: int = 18,
                    **kwargs):
  from odin.bay.random_variable import RVconf
  if zdim is None:
    zdim = 45
  input_shape = (64, 64, 3)
  n_components = 10  # for Mixture Quantized Logistic
  n_channels = input_shape[-1]
  conv, deconv = _prepare_cnn(activation=activation)
  proj_dim = 512
  encoder = SequentialNetwork(
    [
      CenterAt0(enable=centerize_image),
      conv(32, 4, strides=2, name='encoder0'),
      conv(32, 4, strides=2, name='encoder1'),
      conv(64, 4, strides=2, name='encoder2'),
      conv(64, 4, strides=1, name='encoder3'),
      keras.layers.Flatten(),
      keras.layers.Dense(proj_dim, activation='linear', name='encoder_proj')
    ],
    name='Encoder',
  )
  layers = [
    keras.layers.Dense(proj_dim, activation='linear', name='decoder_proj'),
    keras.layers.Reshape((8, 8, proj_dim // 64)),
    deconv(64, 4, strides=1, name='decoder1'),
    deconv(64, 4, strides=2, name='decoder2'),
    deconv(32, 4, strides=2, name='decoder3'),
    deconv(32, 4, strides=2, name='decoder4'),
    conv(2 * n_channels,
         # MixtureQuantizedLogistic.params_size(n_components, n_channels),
         1,
         strides=1,
         activation='linear',
         name='decoder5'),
  ]
  from odin.bay import BiConvLatents
  layers = [i.layer if isinstance(i, BiConvLatents) and not is_hierarchical
            else i
            for i in layers]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='SkipDecoder')
  else:
    decoder = SequentialNetwork(layers=layers, name='Decoder')
  latents = RVconf((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = _parse_distribution(input_shape, 'qlogistic')
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    networks['labels'] = DistributionDense(event_shape=n_labels,
                                           posterior=_celeba_distribution,
                                           units=n_labels,
                                           name='attributes')
  return networks


# ===========================================================================
# Gene Networks
# ===========================================================================
@typechecked
def cortex_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = 10,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
    is_semi_supervised: bool = False,
    is_hierarchical: bool = False,
    log_norm: bool = True,
    cnn: bool = False,
    units: Sequence[int] = (256, 256, 256),
    **kwargs,
) -> Dict[str, Layer]:
  """Network for Cortex mRNA sequencing datasets"""
  from odin.bay.random_variable import RVconf
  input_shape = (558,)
  n_labels = 7
  if zdim is None:
    zdim = 10
  ## dense network
  if not cnn:
    encoder = SequentialNetwork(
      [LogNorm(enable=log_norm)] + [
        keras.layers.Dense(u, activation=activation, name=f'encoder{i}')
        for i, u in enumerate(units)
      ],
      name='encoder',
    )
    decoder = SequentialNetwork(
      [
        keras.layers.Dense(u, activation=activation, name=f'decoder{i}')
        for i, u in enumerate(units)
      ],
      name='decoder',
    )
  ## cnn
  else:
    Conv1D = partial(keras.layers.Conv1D,
                     strides=2,
                     padding='same',
                     activation=activation)
    Conv1DTranspose = partial(keras.layers.Conv1DTranspose,
                              strides=2,
                              padding='same',
                              activation=activation)
    encoder = SequentialNetwork(
      [
        LogNorm(enable=log_norm),
        keras.layers.Lambda(
          lambda x: tf.expand_dims(x, axis=-1)),  # (n, 2019, 1)
        Conv1D(32, 7, name='encoder0'),
        Conv1D(64, 5, name='encoder1'),
        Conv1D(128, 3, name='encoder2'),
        keras.layers.Flatten()
      ],
      name='encoder',
    )
    decoder = SequentialNetwork(
      [
        keras.layers.Dense(128, activation=activation, name='decoder0'),
        keras.layers.Lambda(
          lambda x: tf.expand_dims(x, axis=-1)),  # (n, 256, 1)
        Conv1DTranspose(128, 3, strides=1, name='decoder1'),
        Conv1DTranspose(64, 5, name='decoder3'),
        Conv1DTranspose(32, 7, name='decoder4'),
        Conv1DTranspose(1, 1, strides=1, name='decoder5'),
        keras.layers.Flatten()
      ],
      name='decoder',
    )
  latents = RVconf((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = RVconf(input_shape, "nb", projection=True,
                       name="mrna").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVconf(7, 'onehot', projection=True,
                                name='celltype').create_posterior()
  return networks


@typechecked
def pbmc_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = 32,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
    is_semi_supervised: bool = False,
    is_hierarchical: bool = False,
    log_norm: bool = True,
    cnn: bool = True,
    units: Sequence[int] = (512, 512, 512),
    **kwargs,
) -> Dict[str, Layer]:
  """Network for Cortex mRNA sequencing datasets"""
  from odin.bay.random_variable import RVconf
  input_shape = (2019,)
  n_labels = 32
  if zdim is None:
    zdim = 32
  ## dense network
  if not cnn:
    encoder = SequentialNetwork(
      [LogNorm(enable=log_norm)] + [
        keras.layers.Dense(u, activation=activation, name=f'encoder{i}')
        for i, u in enumerate(units)
      ],
      name='encoder',
    )
    decoder = SequentialNetwork(
      [
        keras.layers.Dense(u, activation=activation, name=f'decoder{i}')
        for i, u in enumerate(units)
      ],
      name='decoder',
    )
  ## conv network
  else:
    Conv1D = partial(keras.layers.Conv1D,
                     strides=2,
                     padding='same',
                     activation=activation)
    Conv1DTranspose = partial(keras.layers.Conv1DTranspose,
                              strides=2,
                              padding='same',
                              activation=activation)
    encoder = SequentialNetwork(
      [
        LogNorm(enable=log_norm),
        keras.layers.Lambda(
          lambda x: tf.expand_dims(x, axis=-1)),  # (n, 2019, 1)
        Conv1D(32, 7, name='encoder0'),
        Conv1D(64, 5, name='encoder1'),
        Conv1D(128, 3, name='encoder2'),
        Conv1D(128, 3, name='encoder3'),
        keras.layers.Flatten()
      ],
      name='encoder',
    )
    decoder = SequentialNetwork(
      [
        keras.layers.Dense(256, activation=activation, name='decoder0'),
        keras.layers.Lambda(
          lambda x: tf.expand_dims(x, axis=-1)),  # (n, 256, 1)
        Conv1DTranspose(128, 3, strides=1, name='decoder1'),
        Conv1DTranspose(128, 3, name='decoder2'),
        Conv1DTranspose(64, 5, name='decoder3'),
        Conv1DTranspose(32, 7, name='decoder4'),
        Conv1DTranspose(1, 1, strides=1, name='decoder5'),
        keras.layers.Flatten()
      ],
      name='decoder',
    )
  latents = RVconf((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = RVconf(input_shape, "zinb", projection=True,
                       name="mrna").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVconf(n_labels, 'nb', projection=True,
                                name='adt').create_posterior()
  return networks


# ===========================================================================
# Utils
# ===========================================================================
_DSNAME_MAP = dict(
  halfmnist='mnist'
)


def get_networks(dataset_name: [str, IterableDataset],
                 *,
                 is_semi_supervised: bool,
                 is_hierarchical: bool,
                 qz: str = 'mvndiag',
                 zdim: Optional[int] = None,
                 **kwargs) -> Dict[str, Layer]:
  """ Return dictionary of networks for encoder, decoder, observation, latents
  and labels (in case of semi-supervised learning) """
  if isinstance(dataset_name, IterableDataset):
    dataset_name = dataset_name.name.lower()
  if zdim is not None and zdim <= 0:
    zdim = None
  dataset_name = str(dataset_name).lower().strip()
  dataset_name = _DSNAME_MAP.get(dataset_name, dataset_name)
  for k, fn in globals().items():
    if isinstance(k, string_types) and (inspect.isfunction(fn) or
                                        isinstance(fn, partial)):
      k = k.split('_')[0]
      if k == dataset_name:
        return fn(qz=qz,
                  zdim=zdim,
                  is_semi_supervised=is_semi_supervised,
                  is_hierarchical=is_hierarchical,
                  **kwargs)
  raise ValueError('Cannot find pre-implemented network for '
                   f'dataset with name="{dataset_name}"')


def get_optimizer_info(
    dataset_name: str,
    batch_size: int = 64,
) -> Dict[str, Any]:
  """Return information for optimizing networks of given datasets,
  this is create with the assumption that batch_size=32

  Parameters
  ----------
  dataset_name : str
      name of datasets, e.g. 'mnist', 'dsprites', 'shapes3d'
  batch_size : int
      mini-batch size

  Returns
  -------
  Dict[str, Any]
      'max_iter' : int,
          number of iterations,
      'learning_rate' : `tf.optimizers.schedules.ExponentialDecay`
          learning rate

  """
  dataset_name = str(dataset_name).strip().lower()
  dataset_name = _DSNAME_MAP.get(dataset_name, dataset_name)
  decay_rate = 0.996
  decay_steps = 10000
  init_lr = 1e-3
  ### image networks
  if dataset_name == 'halfmoons':
    n_epochs = 200
    n_samples = 3200
  elif dataset_name == 'mnist':
    n_epochs = 800
    n_samples = 55000
  elif dataset_name == 'fashionmnist':
    n_epochs = 1000
    n_samples = 55000
  elif dataset_name == 'omniglot':
    n_epochs = 1000
    n_samples = 19280
  elif 'svhn' in dataset_name:
    n_epochs = 2000
    n_samples = 69594
  elif 'cifar' in dataset_name:
    n_epochs = 2500
    n_samples = 48000
    init_lr = 5e-4
  # dsrpites datasets
  elif 'dsprites' in dataset_name:
    n_epochs = 400
    n_samples = 663552
  # sahpes datasets
  elif 'shapes3d' in dataset_name:
    n_epochs = 250 if 'small' in dataset_name else 400
    n_samples = 432000
    init_lr = 2e-4
  elif 'celeba' in dataset_name:
    n_epochs = 2000 if 'small' in dataset_name else 3000
    n_samples = 162770
    init_lr = 2e-4
  ### gene networks
  elif 'cortex' in dataset_name:
    n_epochs = 500
    n_samples = 5000
    init_lr = 1e-4
  elif 'pbmc' in dataset_name:
    n_epochs = 500
    n_samples = 5000
    init_lr = 1e-4
  else:
    raise NotImplementedError(
      f'No predefined optimizer information for dataset {dataset_name}')
  max_iter = int((n_samples / batch_size) * n_epochs)
  lr = tf.optimizers.schedules.ExponentialDecay(init_lr,
                                                decay_steps=decay_steps,
                                                decay_rate=decay_rate,
                                                staircase=True)
  return dict(max_iter=max_iter, learning_rate=lr)
