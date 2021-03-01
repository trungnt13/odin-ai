# References
# ----------
# Kim, H., Mnih, A., 2018. Disentangling by factorising,
#   in: Dy, J., Krause, A. (Eds.), Proceedings of Machine
#   Learning Research. PMLR, Stockholmsmässan, Stockholm
#   Sweden, pp. 2649–2658.
import inspect
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python.distributions.pixel_cnn import \
    _PixelCNNNetwork
from tensorflow_probability.python.layers import DistributionLambda
from typeguard import typechecked

from odin.networks import residuals as rsd
from odin.bay.distributions import (Blockwise, Categorical, ContinuousBernoulli,
                                    Distribution, Gamma,
                                    JointDistributionSequential, PixelCNNpp,
                                    VonMises, Bernoulli)

__all__ = [
    'mnist_networks',
    'dsprites_networks',
    'shapes3dsmall_networks',
    'shapes3d_networks',
    'celebasmall_networks',
    'celeba_networks',
    'get_networks',
    'get_optimizer_info',
]


# ===========================================================================
# Helpers
# ===========================================================================
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


def _prepare_cnn(activation=tf.nn.leaky_relu):
  # he_uniform is better for leaky_relu
  if activation is tf.nn.leaky_relu:
    init = 'he_uniform'
  else:
    init = 'glorot_uniform'
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

  def __init__(self, layers: List[Layer] = [], name: str = 'SkipGenerator'):
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
    zdim: Optional[int] = 16,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
    is_semi_supervised: bool = False,
    centerize_image: bool = True,
    skip_generator: bool = False,
    **kwargs,
) -> Dict[str, Layer]:
  """Network for MNIST dataset image size (28, 28, 1)"""
  from odin.bay.random_variable import RVmeta
  n_channels = int(kwargs.get('n_channels', 1))
  proj_dim = int(kwargs.get('proj_dim', 128))
  input_shape = (28, 28, n_channels)
  if zdim is None:
    zdim = 16
  conv, deconv = _prepare_cnn(activation=activation)
  encoder = keras.Sequential(
      [
          CenterAt0(enable=centerize_image),
          conv(32, 5, strides=1, name='encoder0'),
          conv(32, 5, strides=2, name='encoder1'),
          conv(64, 5, strides=1, name='encoder2'),
          conv(64, 5, strides=2, name='encoder3'),
          keras.layers.Flatten(),
          keras.layers.Dense(proj_dim, activation='linear', name='encoder4')
      ],
      name='encoder',
  )
  layers = [
      keras.layers.Dense(proj_dim, activation='linear', name='decoder0'),
      keras.layers.Dense(7 * 7 * 64, activation=activation, name='decoder1'),
      keras.layers.Reshape((7, 7, 64)),
      deconv(64, 5, strides=2, name='decoder2'),
      deconv(64, 5, strides=1, name='decoder3'),
      deconv(32, 5, strides=2, name='decoder4'),
      deconv(32, 5, strides=1, name='decoder5'),
      conv(n_channels, 1, strides=1, activation='linear', name='decoder6'),
      keras.layers.Flatten()
  ]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='skipdecoder')
  else:
    decoder = keras.Sequential(layers=layers, name='decoder')
  latents = RVmeta((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = RVmeta(input_shape, "bernoulli", projection=False,
                       name="image").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVmeta(
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
class _PixelCNNDecoder(keras.Model):

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
    zdim: Optional[int] = 32,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
    is_semi_supervised: bool = False,
    centerize_image: bool = True,
    skip_generator: bool = False,
    **kwargs,
) -> Dict[str, Layer]:
  """Network for CIFAR dataset image size (32, 32, 3)"""
  from odin.bay.random_variable import RVmeta
  if zdim is None:
    zdim = 32
  n_components = 10
  decoder_input = kwargs.get('decoder_input', zdim)
  n_channels = int(kwargs.get('n_channels', 3))
  input_shape = (32, 32, n_channels)
  conv, deconv = _prepare_cnn(activation=activation)
  n_classes = kwargs.get('n_classes', 10)
  # the number of parameters per pixel.
  n_coeffs = n_channels * (n_channels - 1) // 2
  n_out = n_channels * 2 + n_coeffs + 1
  n_out_total = n_out * n_components
  ## encoder
  inputs = keras.Input(input_shape)
  x = CenterAt0(enable=centerize_image)(inputs)
  x = rsd.project_1_1(x, filters=32, activation=activation, name='encoder1')
  x = rsd.residual_inverted(x,
                            expand_ratio=2,
                            activation=activation,
                            strides=1,
                            name='encoder2')
  x = rsd.squeeze_and_excitation(x, activation=activation, name='encoder3')
  x = rsd.residual_inverted(x,
                            expand_ratio=2,
                            activation=activation,
                            strides=2,
                            name='encoder4')
  x = rsd.squeeze_and_excitation(x, activation=activation, name='encoder5')
  x = rsd.project_1_1(x, filters=64, activation=activation, name='encoder6')
  x = rsd.residual_inverted(x,
                            expand_ratio=2,
                            activation=activation,
                            strides=2,
                            name='encoder7')
  x = rsd.squeeze_and_excitation(x, activation=activation, name='encoder8')
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(256, activation=activation, name='encoder9')(x)
  encoder = keras.Model(inputs=inputs, outputs=x, name='encoder')
  ## create the decoder
  inputs = keras.Input((decoder_input,))
  x = keras.layers.Dense(zdim * 8 * 8, activation=activation,
                         name='decoder1')(inputs)
  x = keras.layers.Reshape((8, 8, zdim))(x)
  x = rsd.project_1_1(x, filters=64, activation=activation, name='decoder2')
  x = rsd.unpooling(x, 2, name='decoder3')
  x = rsd.residual_inverted(x,
                            expand_ratio=2,
                            activation=activation,
                            name='decoder4')
  x = rsd.squeeze_and_excitation(x, activation=activation, name='decoder5')
  x = rsd.project_1_1(x, 32, activation=activation, name='decoder6')
  x = rsd.unpooling(x, 2, name='decoder7')
  x = rsd.residual_inverted(x,
                            expand_ratio=2,
                            activation=activation,
                            name='decoder8')
  x = rsd.squeeze_and_excitation(x, activation=activation, name='decoder9')
  x = rsd.project_1_1(x,
                      PixelCNNpp.params_size(n_components, n_channels),
                      activation=activation)
  decoder = keras.Model(inputs=inputs, outputs=x, name='decoder')
  ## others
  latents = RVmeta((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  # create the observation of MixtureQuantizedLogistic
  observation = DistributionLambda(
      lambda params: PixelCNNpp(params,
                                n_components=n_components,
                                n_channels=n_channels,
                                inputs_domain='sigmoid',
                                high=255,
                                low=0),
      convert_to_tensor_fn=Distribution.mean,
      name='image',
  )
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVmeta(n_classes,
                                'onehot',
                                projection=True,
                                name='labels').create_posterior()
  return networks


cifar10_networks = partial(cifar_networks, n_classes=10)
cifar20_networks = partial(cifar_networks, n_classes=20)
cifar100_networks = partial(cifar_networks, n_classes=100)


# ===========================================================================
# dSprites
# ===========================================================================
def _dsprites_distribution(x: tf.Tensor) -> Blockwise:
  # NOTE: tried Continuous Bernoulli for dSPrites, but leads to
  # more unstable training in semi-supervised learning.
  dtype = x.dtype
  py = JointDistributionSequential([
      VonMises(loc=0.,
               concentration=tf.math.softplus(x[..., 0]),
               name='orientation'),
      Gamma(concentration=tf.math.softplus(x[..., 1]),
            rate=tf.math.softplus(x[..., 2]),
            name='scale'),
      Categorical(logits=x[..., 3:6], dtype=dtype, name='shape'),
      Bernoulli(logits=x[..., 6], dtype=dtype, name='x_position'),
      Bernoulli(logits=x[..., 7], dtype=dtype, name='y_position'),
  ])
  return Blockwise(py, name='shapes2d')


@typechecked
def dsprites_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = 10,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
    is_semi_supervised: bool = False,
    centerize_image: bool = True,
    skip_generator: bool = False,
    **kwargs,
) -> Dict[str, Layer]:
  from odin.bay.random_variable import RVmeta
  if zdim is None:
    zdim = 10
  distribution = str(kwargs.get('distribution', 'bernoulli'))
  assert distribution in ('bernoulli', 'gaussian'), \
    f'Only support Bernoulli or Gaussian output, given: {distribution}'
  n_channels = int(kwargs.get('n_channels', 1))
  input_shape = (64, 64, n_channels)
  conv, deconv = _prepare_cnn(activation=activation)
  proj_dim = kwargs.get('proj_dim', None)
  if proj_dim is None:
    proj_dim = 128 if n_channels == 1 else 256
  else:
    proj_dim = int(proj_dim)
  encoder = keras.Sequential(
      [
          CenterAt0(enable=centerize_image),
          conv(32, 4, strides=2, name='encoder0'),
          conv(32, 4, strides=2, name='encoder1'),
          conv(64, 4, strides=2, name='encoder2'),
          conv(64, 4, strides=2, name='encoder3'),
          keras.layers.Flatten(),
          keras.layers.Dense(proj_dim, activation='linear', name='encoder4')
      ],
      name='encoder',
  )
  layers = [
      keras.layers.Dense(proj_dim, activation=activation, name='decoder0'),
      keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 1, 1, proj_dim))),
      keras.layers.Dense(4 * 4 * 64, activation=activation, name='decoder1'),
      keras.layers.Reshape((4, 4, 64)),
      deconv(64, 4, strides=2, name='decoder2'),
      deconv(32, 4, strides=2, name='decoder3'),
      deconv(32, 4, strides=2, name='decoder4'),
      deconv(n_channels * (1 if distribution == 'bernoulli' else 2),
             4,
             strides=2,
             name='decoder5'),
      keras.layers.Flatten()
  ]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='skipdecoder')
  else:
    decoder = keras.Sequential(layers=layers, name='decoder')
  latents = RVmeta((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = RVmeta(input_shape,
                       distribution,
                       projection=False,
                       name="image").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    networks['labels'] = DistributionDense(event_shape=(5,),
                                           posterior=_dsprites_distribution,
                                           units=8,
                                           name='geometry2d')
  return networks


def dspritessmall_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = 10,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
    is_semi_supervised: bool = False,
    centerize_image: bool = True,
    skip_generator: bool = False,
    **kwargs,
) -> Dict[str, Layer]:
  if zdim is None:
    zdim = 10
  networks = mnist_networks(qz=qz,
                            zdim=zdim,
                            activation=activation,
                            is_semi_supervised=False,
                            centerize_image=centerize_image,
                            skip_generator=skip_generator,
                            n_channels=1,
                            proj_dim=128)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    networks['labels'] = DistributionDense(event_shape=(5,),
                                           posterior=_dsprites_distribution,
                                           units=8,
                                           name='geometry2d')
  return networks


# ===========================================================================
# Shapes 3D
# ===========================================================================
def _shapes3d_distribution(x: tf.Tensor) -> Blockwise:
  dtype = x.dtype
  py = JointDistributionSequential([
      VonMises(loc=0.,
               concentration=tf.math.softplus(x[..., 0]),
               name='orientation'),
      Gamma(concentration=tf.math.softplus(x[..., 1]),
            rate=tf.math.softplus(x[..., 2]),
            name='scale'),
      Categorical(logits=x[..., 3:7], dtype=dtype, name='shape'),
      ContinuousBernoulli(logits=x[..., 7], name='floor_hue'),
      ContinuousBernoulli(logits=x[..., 8], name='wall_hue'),
      ContinuousBernoulli(logits=x[..., 9], name='object_hue'),
  ])
  return Blockwise(py, name='shapes3d')


def shapes3dsmall_networks(qz: str = 'mvndiag',
                           zdim: Optional[int] = 6,
                           activation: Union[Callable, str] = tf.nn.leaky_relu,
                           is_semi_supervised: bool = False,
                           centerize_image: bool = True,
                           skip_generator: bool = False,
                           **kwargs) -> Dict[str, Layer]:
  if zdim is None:
    zdim = 6
  networks = mnist_networks(qz=qz,
                            zdim=zdim,
                            activation=activation,
                            is_semi_supervised=False,
                            centerize_image=centerize_image,
                            skip_generator=skip_generator,
                            n_channels=3,
                            proj_dim=128)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    networks['labels'] = DistributionDense(event_shape=(6,),
                                           posterior=_shapes3d_distribution,
                                           units=10,
                                           name='geometry3d')
  return networks


def shapes3d_networks(qz: str = 'mvndiag',
                      zdim: Optional[int] = 6,
                      activation: Union[Callable, str] = tf.nn.leaky_relu,
                      is_semi_supervised: bool = False,
                      centerize_image: bool = True,
                      skip_generator: bool = False,
                      **kwargs) -> Dict[str, Layer]:
  if zdim is None:
    zdim = 6
  networks = dsprites_networks(qz=qz,
                               zdim=zdim,
                               activation=activation,
                               is_semi_supervised=False,
                               centerize_image=centerize_image,
                               skip_generator=skip_generator,
                               n_channels=3)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    networks['labels'] = DistributionDense(event_shape=(6,),
                                           posterior=_shapes3d_distribution,
                                           units=10,
                                           name='geometry3d')
  return networks


# ===========================================================================
# CelebA
# ===========================================================================
def celebasmall_networks(qz: str = 'mvndiag',
                         zdim: Optional[int] = None,
                         activation: Union[Callable, str] = tf.nn.leaky_relu,
                         is_semi_supervised: bool = False,
                         centerize_image: bool = True,
                         skip_generator: bool = False,
                         n_labels: int = 18,
                         **kwargs):
  if zdim is None:
    zdim = 32
  networks = mnist_networks(qz=qz,
                            zdim=zdim,
                            activation=activation,
                            is_semi_supervised=False,
                            centerize_image=centerize_image,
                            skip_generator=skip_generator,
                            n_channels=3,
                            proj_dim=128)
  if is_semi_supervised:
    from odin.bay.layers import DistributionDense
    networks['labels'] = DistributionDense(event_shape=int(n_labels),
                                           posterior='bernoulli',
                                           name='attributes')
  return networks


def celeba_networks(qz: str = 'mvndiag',
                    zdim: Optional[int] = None,
                    activation: Union[Callable, str] = tf.nn.relu,
                    is_semi_supervised: bool = False,
                    centerize_image: bool = True,
                    skip_generator: bool = False,
                    n_labels: int = 18,
                    **kwargs):
  from odin.bay.random_variable import RVmeta
  from odin.bay.layers import DistributionDense
  if zdim is None:
    zdim = 45
  input_shape = (64, 64, 3)
  conv, deconv = _prepare_cnn(activation=activation)
  proj_dim = 256
  encoder = keras.Sequential(
      [
          CenterAt0(enable=centerize_image),
          conv(32, 4, strides=2, name='encoder0'),
          conv(32, 4, strides=2, name='encoder1'),
          conv(64, 4, strides=2, name='encoder2'),
          conv(64, 4, strides=2, name='encoder3'),
          keras.layers.Flatten(),
          keras.layers.Dense(proj_dim, activation='linear', name='encoder5')
      ],
      name='encoder',
  )
  layers = [
      keras.layers.Dense(proj_dim, activation='linear', name='decoder0'),
      keras.layers.Reshape((4, 4, 16)),
      deconv(64, 4, strides=2, name='decoder1'),
      deconv(64, 4, strides=2, name='decoder2'),
      deconv(32, 4, strides=2, name='decoder3'),
      deconv(32, 4, strides=2, name='decoder4'),
      conv(3 * 2, 1, strides=1, activation='linear', name='decoder5'),
      keras.layers.Flatten()
  ]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='skipdecoder')
  else:
    decoder = keras.Sequential(layers=layers, name='decoder')
  latents = RVmeta((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = RVmeta(input_shape, 'gaussian', projection=False,
                       name="image").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = DistributionDense(event_shape=int(n_labels),
                                           posterior='bernoulli',
                                           name='attributes')
  return networks


# ===========================================================================
# Gene Networks
# ===========================================================================
@typechecked
def cortex_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = 10,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
    is_semi_supervised: bool = False,
    log_norm: bool = True,
    cnn: bool = False,
    units: List[int] = [256, 256, 256],
    **kwargs,
) -> Dict[str, Layer]:
  """Network for Cortex mRNA sequencing datasets"""
  from odin.bay.random_variable import RVmeta
  input_shape = (558,)
  n_labels = 7
  if zdim is None:
    zdim = 10
  ## dense network
  if not cnn:
    encoder = keras.Sequential(
        [LogNorm(enable=log_norm)] + [
            keras.layers.Dense(u, activation=activation, name=f'encoder{i}')
            for i, u in enumerate(units)
        ],
        name='encoder',
    )
    decoder = keras.Sequential(
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
    encoder = keras.Sequential(
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
    decoder = keras.Sequential(
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
  latents = RVmeta((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = RVmeta(input_shape, "nb", projection=True,
                       name="mrna").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVmeta(7, 'onehot', projection=True,
                                name='celltype').create_posterior()
  return networks


@typechecked
def pbmc_networks(
    qz: str = 'mvndiag',
    zdim: Optional[int] = 32,
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
    is_semi_supervised: bool = False,
    log_norm: bool = True,
    cnn: bool = True,
    units: List[int] = [512, 512, 512],
    **kwargs,
) -> Dict[str, Layer]:
  """Network for Cortex mRNA sequencing datasets"""
  from odin.bay.random_variable import RVmeta
  input_shape = (2019,)
  n_labels = 32
  if zdim is None:
    zdim = 32
  ## dense network
  if not cnn:
    encoder = keras.Sequential(
        [LogNorm(enable=log_norm)] + [
            keras.layers.Dense(u, activation=activation, name=f'encoder{i}')
            for i, u in enumerate(units)
        ],
        name='encoder',
    )
    decoder = keras.Sequential(
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
    encoder = keras.Sequential(
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
    decoder = keras.Sequential(
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
  latents = RVmeta((zdim,), qz, projection=True,
                   name="latents").create_posterior()
  observation = RVmeta(input_shape, "zinb", projection=True,
                       name="mrna").create_posterior()
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVmeta(n_labels, 'nb', projection=True,
                                name='adt').create_posterior()
  return networks


# ===========================================================================
# Utils
# ===========================================================================
def get_networks(dataset_name: str,
                 *,
                 qz: str = 'mvndiag',
                 zdim: Optional[int] = None,
                 is_semi_supervised: bool = False,
                 **kwargs) -> Dict[str, Layer]:
  """ Return dictionary of networks for encoder, decoder, observation, latents
  and labels (in case of semi-supervised learning) """
  dataset_name = str(dataset_name).lower().strip()
  for k, fn in globals().items():
    if isinstance(k, string_types) and (inspect.isfunction(fn) or
                                        isinstance(fn, partial)):
      k = k.split('_')[0]
      if k == dataset_name:
        return fn(qz=qz,
                  zdim=zdim,
                  is_semi_supervised=is_semi_supervised,
                  **kwargs)
  raise ValueError('Cannot find pre-implemented network for '
                   f'dataset with name="{dataset_name}"')


def get_optimizer_info(dataset_name: str) -> Tuple[int, LearningRateSchedule]:
  """Return information for optimizing networks of given datasets

  Parameters
  ----------
  dataset_name : str
      name of datasets, e.g. 'mnist', 'dsprites', 'shapes3d'

  Returns
  -------
  Tuple[int, LearningRateSchedule]
      number of iterations, learning rate

  """
  dataset_name = str(dataset_name).strip().lower()
  decay_rate = 0.96
  ### image networks
  if 'omniglot' in dataset_name:
    max_iter = 80000
    init_lr = 1e-3
    decay_steps = 5000
    decay_rate = 0.996
  elif 'fashionmnist' in dataset_name:
    max_iter = 50000
    init_lr = 1e-3
    decay_steps = 5000
    decay_rate = 0.996
  elif 'mnist' in dataset_name:
    max_iter = 35000
    init_lr = 1e-3
    decay_steps = 2500
    decay_rate = 0.996
  elif 'cifar' in dataset_name:
    max_iter = 200000
    init_lr = 0.001
    decay_steps = 5000
    decay_rate = 0.996
  elif 'dsprites' in dataset_name:
    max_iter = 120000
    init_lr = 0.001
    decay_steps = 3000
  elif 'dspritessmall' in dataset_name:
    max_iter = 100000
    init_lr = 0.001
    decay_steps = 2500
  elif 'shapes3dsmall' in dataset_name:
    max_iter = 150000
    init_lr = 5e-4
    decay_steps = 8000
  elif 'shapes3d' in dataset_name:
    max_iter = 180000
    init_lr = 1e-4
    decay_steps = 10000
  elif 'celebasmall' in dataset_name:
    max_iter = 200000
    init_lr = 5e-4
    decay_steps = 8000
  elif 'celeba' in dataset_name:
    max_iter = 200000
    init_lr = 1e-4
    decay_steps = 10000
  ### gene networks
  elif 'cortex' in dataset_name:
    max_iter = 30000
    init_lr = 1e-4
    decay_steps = 5000
  elif 'pbmc' in dataset_name:
    max_iter = 50000
    init_lr = 1e-4
    decay_steps = 8000
  else:
    raise NotImplementedError(
        f'No predefined optimizer information for dataset {dataset_name}')
  lr = tf.optimizers.schedules.ExponentialDecay(init_lr,
                                                decay_steps=decay_steps,
                                                decay_rate=decay_rate,
                                                staircase=False)
  return dict(max_iter=max_iter, learning_rate=lr)
