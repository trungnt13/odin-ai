import inspect
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Dict, List, Tuple
from typing import Union, Optional, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Layer, Wrapper, Dense, Conv2D, \
  Conv1D, Conv3D, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose, Reshape, \
  AvgPool1D, AvgPool2D, AvgPool3D, MaxPool1D, MaxPool2D, MaxPool3D, Activation, \
  UpSampling1D, UpSampling2D, UpSampling3D
from tensorflow_probability.python.distributions import Distribution, Normal, \
  MultivariateNormalDiag, Independent
from tensorflow_probability.python.layers.distribution_layer import (
  DistributionLambda)
from typing_extensions import Literal

from odin.bay.helpers import kl_divergence
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import AnnealingVAE, BetaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.networks import NetConf
from odin.utils import as_tuple

__all__ = [
  'MergeNormal',
  'BiConvLatents',
  'ParallelLatents',
  'BiDenseLatents',
  'HierarchicalVAE',
  'UnetVAE'
]

# ===========================================================================
# Helpers
# ===========================================================================
_NDIMS_CONV = {
  3: (Conv1D, Conv1DTranspose),
  4: (Conv2D, Conv2DTranspose),
  5: (Conv3D, Conv3DTranspose)
}
_NDIMS_POOL = {3: dict(avg=AvgPool1D, max=MaxPool1D),
               4: dict(avg=AvgPool2D, max=MaxPool2D),
               5: dict(avg=AvgPool3D, max=MaxPool3D)}
_NDIMS_UNPOOL = {3: UpSampling1D, 4: UpSampling2D, 5: UpSampling3D}


def _create_dist(params, event_ndims, dtype):
  loc, scale = tf.split(params, 2, axis=-1)
  scale = tf.nn.softplus(scale) + tf.cast(tf.exp(-7.), dtype)
  d = Normal(loc, scale)
  d = Independent(d, reinterpreted_batch_ndims=event_ndims)
  return d


def _upsample_by_conv(
    layer: Union[Conv1D, Conv2D, Conv3D],
    layer_t: Union[Conv1DTranspose, Conv2DTranspose, Conv3DTranspose],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    padding: Literal['same', 'valid'],
    strides: Tuple[int, int]):
  in_image = input_shape[1:-1]
  out_image = output_shape[1:-1]
  out_filters = output_shape[-1]
  # shape match only use 1x1 projection
  if in_image == out_image:
    conv = layer(filters=out_filters, kernel_size=1, name='ConvOutput')
  # upsampling
  elif all(o >= i for i, o in zip(in_image, out_image)):
    kernel = [k + (1 if padding == 'valid' and k % s != 0 else 0)
              for k, s in zip(kernel_size, strides)]
    conv = layer_t(filters=out_filters,
                   kernel_size=kernel,
                   strides=strides,
                   padding=padding,
                   name='ConvOutput')
  # cannot do downsampling
  else:
    raise RuntimeError('Cannot infer output projection for input shape: '
                       f'{input_shape} and output shape: {output_shape}')
  conv.build(input_shape)
  shape = conv.compute_output_shape(input_shape)
  tf.assert_equal(
    shape[1:], output_shape[1:],
    'Failed to infer proper convolutional operator for upsampling from '
    f'{input_shape} to {output_shape}, the resulted shape is {shape}')
  return conv


def _call(self, inputs, **kwargs):
  outputs = self._old_call(inputs, **kwargs)
  self._last_outputs = outputs
  return outputs


class MergeNormal(DistributionLambda):
  """ Merge two Gaussian based on weighed variance

  https://github.com/casperkaae/LVAE/blob/066858a3fb53bb1c529a6f12ae5afb0955722845/run_models.py#L106
  """

  def __init__(self, name='MergeNormal'):
    super().__init__(make_distribution_fn=MergeNormal.new,
                     name=name)

  @staticmethod
  def new(dists):
    q_e, q_d = dists
    mu_e = q_e.mean()
    mu_d = q_d.mean()
    prec_e = 1 / q_e.variance()
    prec_d = 1 / q_d.variance()
    mu = (mu_e * prec_e + mu_d * prec_d) / (prec_e + prec_d)
    scale = tf.math.sqrt(1 / (prec_e + prec_d))
    dist = Normal(loc=mu, scale=scale)
    if isinstance(q_e, Independent):
      ndim = q_e.reinterpreted_batch_ndims
      dist = Independent(dist, reinterpreted_batch_ndims=ndim)
    return dist


# ===========================================================================
# Wrapper for hierarchical latent variable
# ===========================================================================
class HierarchicalLatents(Wrapper):

  def __init__(self,
               layer: Layer,
               beta: float = 1.,
               disable: bool = False,
               **kwargs):
    super().__init__(layer=layer,
                     trainable=not disable and kwargs.pop('trainable', True),
                     **kwargs)
    # store the last distributions
    self._posterior = None
    self._prior = None
    self._is_sampling = False
    self._disable = bool(disable)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')
    self._latents_shape = None
    # prepare the layer
    self.input_ndim = self.layer.input_spec.min_ndim
    spec = inspect.getfullargspec(layer.call)
    self._layer_args = set(spec.args + spec.kwonlyargs)

  @property
  def layers(self) -> Sequence[Layer]:
    return [v for k, v in self.__dict__.items() if isinstance(v, Layer)]

  def sampling(self):
    """Sampling mode, forward prior samples"""
    self._is_sampling = True
    return self

  def inference(self):
    """Inference mode, forward posterior samples (require encoder states)"""
    self._is_sampling = False
    return self

  @property
  def is_inference(self) -> bool:
    return not self._is_sampling

  def enable(self):
    """Enable stochastic inference and generation for this variable"""
    self._disable = False
    self.trainable = True
    return self

  @property
  def units(self) -> int:
    return int(np.prod(self._latents_shape))

  @property
  def latents_shape(self) -> Sequence[int]:
    return self._latents_shape

  @property
  def posterior(self) -> Optional[Distribution]:
    return self._posterior

  @property
  def prior(self) -> Optional[Distribution]:
    return self._prior

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    units = None
    if self.latents_shape is not None:
      units = int(np.prod(self.latents_shape))
    return (
      f"<{self.__class__.__name__} "
      f"'{self.name}' enable:{not self._disable} "
      f"sampl:{self._is_sampling} shape:{self.latents_shape}={units} "
      f"beta:{self.beta}>")

  def kl_divergence(self,
                    analytic: bool = False,
                    reverse: bool = False,
                    free_bits: Optional[float] = None,
                    raise_not_init: bool = True) -> tf.Tensor:
    if self._disable:
      return tf.zeros((), dtype=self.dtype)
    if raise_not_init:
      if self._posterior is None:
        raise ValueError('No posterior for the hierarchical latent variable.')
      if self._prior is None:
        raise ValueError("This HierarchicalLatents haven't been called.")
    elif self._posterior is None or self._prior is None:
      return tf.zeros((), dtype=self.dtype)
    qz = self.posterior
    pz = self.prior
    kld = kl_divergence(q=qz, p=pz, analytic=analytic, reverse=reverse,
                        free_bits=free_bits)
    return self.beta * kld

  def compute_output_shape(self, input_shape) -> Sequence[Union[None, int]]:
    return self.layer.compute_output_shape(input_shape)

  def call(self, inputs, training=None, mask=None, **kwargs):
    if 'training' in self._layer_args:
      kwargs['training'] = training
    if 'mask' in self._layer_args:
      kwargs['mask'] = mask
    hidden_d = self.layer.call(inputs, **kwargs)
    return hidden_d


class BiConvLatents(HierarchicalLatents):
  """Bidirectional inference using Convolutional Network for
  hierarchical latent variables

  Parameters
  ----------
  layer : `keras.layers.Layer`
      the decoder layer for top-down (generative)
  encoder : `keras.layers.Layer`, optional
      the encoder layer for bottom-up (inference)
  pre_affine : bool
      if True, applying affine to project convolutional output to latent
      units, otherwise, use the convolutional image as is.
  output_activation : {'str', Callable}
      last activation before residual connection
  deterministic_features : bool
      if True, concatenate deterministic features to the samples from posterior
      (or prior)
  residual_coef : float
      if greater than 0, add residual connection
  merge_normal : bool
      merge two normal distribution
  """

  def __init__(
      self,
      layer: Layer,
      encoder: Optional[Layer] = None,
      filters: int = 32,
      kernel_size: Union[int, Sequence[int]] = 4,
      strides: Union[int, Sequence[int]] = 2,
      padding: Literal['valid', 'same'] = 'same',
      conv_kw: Optional[Dict[str, Any]] = None,
      pre_affine: bool = False,
      output_activation: Union[None, 'str', Callable[[Any], Any]] = None,
      deterministic_features: bool = True,
      residual_coef: float = 1.0,
      merge_normal: bool = False,
      **kwargs):
    super().__init__(layer=layer, **kwargs)
    if encoder is not None:
      encoder._old_call = encoder.call
      encoder.call = MethodType(_call, encoder)
    self.encoder = encoder
    self.pre_affine = bool(pre_affine)
    self.residual_coef = residual_coef
    self.deterministic_features = deterministic_features
    if output_activation is None and hasattr(self.layer, 'activation'):
      output_activation = self.layer.activation
    self.output_activation = keras.activations.get(output_activation)
    # === 1. for creating layer
    self._network_kw = dict(
      # parameters for loc and scale
      filters=(1 if pre_affine else 2) * filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)
    if conv_kw is not None:
      self._network_kw.update(conv_kw)
    # === 2. distribution
    if merge_normal:
      self._merge_normal = MergeNormal()
    else:
      self._merge_normal = None
    # === 2. others
    self._conv_prior = None
    self._conv_posterior = None
    self._conv_deter = None
    self._conv_out = None
    self._affine_prior = None
    self._affine_posterior = None
    self._dist_prior = None
    self._dist_posterior = None
    self.concat = keras.layers.Concatenate(axis=-1)

  @property
  def is_inference(self) -> bool:
    return (not self._is_sampling and
            self.encoder is not None and
            hasattr(self.encoder, '_last_outputs') and
            self.encoder._last_outputs is not None)

  def build(self, input_shape=None):
    super().build(input_shape)
    if self._disable:
      return
    decoder_shape = self.layer.compute_output_shape(input_shape)
    layer, layer_t = _NDIMS_CONV[self.input_ndim]
    # === 1. create projection layer
    if self.encoder is not None:
      self._conv_posterior = layer(**self._network_kw, name='ConvPosterior')
      # posterior projection (assume encoder output shape is the same as
      # decoder output shape)
      self._conv_posterior.build(
        self.concat.compute_output_shape([decoder_shape, decoder_shape]))
    # prior projection
    self._conv_prior = layer(**self._network_kw, name='ConvPrior')
    self._conv_prior.build(decoder_shape)
    # deterministic projection
    if self.deterministic_features:
      kw = dict(self._network_kw)
      if not self.pre_affine:
        kw['filters'] /= 2
      self._conv_deter = layer(**kw, name='ConvDeterministic')
      self._conv_deter.build(decoder_shape)
    # === 2. create distribution
    # compute the parameter shape for the distribution
    params_shape = self._conv_prior.compute_output_shape(decoder_shape)
    if self.pre_affine:
      def create_affine():
        return [keras.layers.Flatten(),
                Activation(self.output_activation),
                Dense(int(np.prod(params_shape[1:])) * 2),
                Reshape(params_shape[1:-1] + (params_shape[-1] * 2,))]

      self._affine_prior = keras.Sequential(create_affine(),
                                            name='PriorAffine')
      self._affine_prior.build(params_shape)
      self._affine_posterior = keras.Sequential(create_affine(),
                                                name='PosteriorAffine')
      self._affine_posterior.build(params_shape)
      params_shape = self._affine_prior.compute_output_shape(params_shape)
    make_distribution = partial(_create_dist,
                                event_ndims=len(params_shape) - 1,
                                dtype=self.dtype)
    self._dist_posterior = DistributionLambda(
      make_distribution_fn=make_distribution, name=f'{self.name}_posterior')
    self._dist_posterior.build(params_shape)
    self._dist_prior = DistributionLambda(
      make_distribution_fn=make_distribution, name=f'{self.name}_prior')
    self._dist_prior.build(params_shape)
    # dynamically infer the shape
    latents_shape = tf.convert_to_tensor(self._dist_posterior(
      keras.layers.Input(params_shape[1:]))).shape
    self._latents_shape = latents_shape[1:]
    if self.deterministic_features:
      deter_shape = self._conv_deter.compute_output_shape(decoder_shape)
      latents_shape = self.concat.compute_output_shape(
        [deter_shape, latents_shape])

    # === 3. final output affine
    if self.residual_coef > 0:
      self._conv_out = _upsample_by_conv(
        layer, layer_t,
        input_shape=latents_shape,
        output_shape=decoder_shape,
        kernel_size=self._conv_prior.kernel_size,
        padding=self._conv_prior.padding,
        strides=self._conv_prior.strides)

  def call(self, inputs, training=None, mask=None, **kwargs):
    # === 1. call the layer
    hidden_d = super().call(inputs, training=training, mask=mask, **kwargs)
    if self._disable:
      return hidden_d
    # === 2. project and create the distribution
    h_prior = self._conv_prior(hidden_d)
    if self._affine_prior is not None:
      h_prior = self._affine_prior(h_prior)
    prior = self._dist_prior(h_prior)
    self._prior = prior
    # === 3. inference
    dist = prior
    if self.is_inference:
      hidden_e = self.encoder._last_outputs
      # just stop inference if there is no Encoder state
      tf.debugging.assert_equal(tf.shape(hidden_e), tf.shape(hidden_d),
                                f'Shape of inference {hidden_e.shape} and '
                                f'generative {hidden_d.shape} mismatch. '
                                f'Change to sampling mode if possible')
      # (Kingma 2016) use add, we concat here
      h_post = self.concat([hidden_e, hidden_d])
      h_post = self._conv_posterior(h_post)
      if self._affine_posterior is not None:
        h_post = self._affine_posterior(h_post)
      posterior = self._dist_posterior(h_post)
      # (Maaloe 2016) merging two Normal distribution
      if self._merge_normal is not None:
        posterior = self._merge_normal([posterior, prior])
      self._posterior = posterior
      dist = posterior
    # === 4. output
    outputs = tf.convert_to_tensor(dist)
    if self.deterministic_features:
      hidden_deter = self._conv_deter(hidden_d)
      outputs = self.concat([outputs, hidden_deter])
    if self.residual_coef > 0.:
      outputs = self._conv_out(outputs)
      outputs = self.output_activation(outputs)
      outputs = outputs + self.residual_coef * hidden_d
    return outputs


class BiDenseLatents(HierarchicalLatents):
  """Bidirectional inference for hierarchical latent variables

  Parameters
  ----------
  layer : `keras.layers.Layer`
      the decoder layer for top-down (generative)
  encoder : `keras.layers.Layer`, optional
      the encoder layer for bottom-up (inference)
  units : int
      number of latent units
  dense_kw : `Dict[str, Any]`, optional
      keyword for initialize `Dense` layer for latents
  pool_mode : {'avg', 'max'}
      perform downsampling on images before `Dense` projection
  pool_size : int
      pooling size
  output_activation : {'str', Callable}
      last activation before residual connection
  deterministic_features : bool
      if True, concatenate deterministic features to the samples from posterior
      (or prior)
  residual_coef : float
      if greater than 0, add residual connection
  merge_normal : bool
      merge two normal distribution
  """

  def __init__(
      self,
      layer: Layer,
      encoder: Optional[Layer] = None,
      units: int = 32,
      dense_kw: Optional[Dict[str, Any]] = None,
      pool_mode: Literal['avg', 'max'] = 'avg',
      pool_size: Optional[int] = None,
      output_activation: Union[None, 'str', Callable[[Any], Any]] = None,
      deterministic_features: bool = True,
      residual_coef: float = 1.0,
      merge_normal: bool = False,
      **kwargs):
    super().__init__(layer=layer, name=kwargs.pop('name', None))
    if encoder is not None:
      encoder._old_call = encoder.call
      encoder.call = MethodType(_call, encoder)
    self.encoder = encoder
    self.residual_coef = residual_coef
    self.deterministic_features = deterministic_features
    if output_activation is None and hasattr(self.layer, 'activation'):
      output_activation = self.layer.activation
    self.output_activation = keras.activations.get(output_activation)
    # === 1. for creating layer
    self._network_kw = dict(units=2 * units)
    if dense_kw is not None:
      self._network_kw.update(dense_kw)
    # === 2. distribution
    if merge_normal:
      self._merge_normal = MergeNormal()
    else:
      self._merge_normal = None
    # === 2. others
    self._latents_shape = None
    self._dense_prior = None
    self._dense_posterior = None
    self._dense_deter = None
    self._dense_out = None
    self._dist_prior = None
    self._dist_posterior = None
    # === 3. util layers
    self.concat = keras.layers.Concatenate(axis=-1)
    self.flatten = keras.layers.Flatten()
    if pool_size is not None and pool_size > 1 and self.input_ndim > 2:
      self.pooling = _NDIMS_POOL[self.input_ndim][pool_mode](
        pool_size, name='Pooling')
      self.unpooling = _NDIMS_UNPOOL[self.input_ndim](
        pool_size, name='Unpooling')
    else:
      self.pooling = Activation('linear', name='Pooling')
      self.unpooling = Activation('linear', name='Unpooling')

  @property
  def is_inference(self) -> bool:
    return (not self._is_sampling and
            self.encoder is not None and
            hasattr(self.encoder, '_last_outputs') and
            self.encoder._last_outputs is not None)

  def build(self, input_shape=None):
    super().build(input_shape)
    if self._disable:
      return
    org_decoder_shape = self.layer.compute_output_shape(input_shape)
    # === 0. pooling
    self.pooling.build(org_decoder_shape)
    pool_decoder_shape = self.pooling.compute_output_shape(org_decoder_shape)
    decoder_shape = self.flatten.compute_output_shape(pool_decoder_shape)
    # === 1. create projection layer
    if self.encoder is not None:
      self._dense_posterior = Dense(**self._network_kw, name='DensePosterior')
      # posterior projection
      shape = self.concat.compute_output_shape([decoder_shape, decoder_shape])
      self._dense_posterior.build(shape)
    # prior projection
    self._dense_prior = Dense(**self._network_kw, name='DensePrior')
    self._dense_prior.build(decoder_shape)
    # deterministic projection
    kw = dict(self._network_kw)
    kw['units'] /= 2
    if self.deterministic_features:
      self._dense_deter = Dense(**kw, name='DenseDeterministic')
      self._dense_deter.build(decoder_shape)
    # === 2. create distribution
    # compute the parameter shape for the distribution
    params_shape = self._dense_prior.compute_output_shape(decoder_shape)

    self._dist_posterior = DistributionLambda(
      make_distribution_fn=partial(_create_dist,
                                   event_ndims=len(params_shape) - 1,
                                   dtype=self.dtype),
      name=f'{self.name}_posterior')
    self._dist_posterior.build(params_shape)
    self._dist_prior = DistributionLambda(
      make_distribution_fn=partial(_create_dist,
                                   event_ndims=len(params_shape) - 1,
                                   dtype=self.dtype),
      name=f'{self.name}_prior')
    self._dist_prior.build(params_shape)
    # dynamically infer the shape
    latents_shape = tf.convert_to_tensor(self._dist_posterior(
      keras.layers.Input(params_shape[1:]))).shape
    self._latents_shape = latents_shape[1:]
    if self.deterministic_features:
      deter_shape = self._dense_deter.compute_output_shape(decoder_shape)
      latents_shape = self.concat.compute_output_shape(
        [deter_shape, latents_shape])

    # === 3. final output affine
    if self.residual_coef > 0:
      units = int(np.prod(pool_decoder_shape[1:]))
      layers = [Dense(units),
                Reshape(pool_decoder_shape[1:]),
                self.unpooling]
      if self.input_ndim > 2:
        conv, _ = _NDIMS_CONV[self.input_ndim]
        layers.append(conv(org_decoder_shape[-1], 3, 1, padding='same'))
      self._dense_out = keras.Sequential(layers, name='DenseOutput')
      self._dense_out.build(latents_shape)

  def call(self, inputs, training=None, mask=None, **kwargs):
    # === 1. call the layer
    hidden_d = super().call(inputs, training=training, mask=mask, **kwargs)
    if self._disable:
      return hidden_d
    # === 2. project and create the distribution
    flat_hd = self.flatten(self.pooling(hidden_d))
    prior = self._dist_prior(self._dense_prior(flat_hd))
    self._prior = prior
    # === 3. inference
    dist = prior
    if self.is_inference:
      hidden_e = self.encoder._last_outputs
      # just stop inference if there is no Encoder state
      tf.debugging.assert_equal(tf.shape(hidden_e), tf.shape(hidden_d),
                                f'Shape of inference {hidden_e.shape} and '
                                f'generative {hidden_d.shape} mismatch. '
                                f'Change to sampling mode if possible')
      # (Kingma 2016) use add, we concat here
      h = self.concat([hidden_e, hidden_d])
      posterior = self._dist_posterior(
        self._dense_posterior(self.flatten(self.pooling(h))))
      # (Maaloe 2016) merging two Normal distribution
      if self._merge_normal is not None:
        posterior = self._merge_normal([posterior, prior])
      self._posterior = posterior
      dist = posterior
    # === 4. output
    outputs = tf.convert_to_tensor(dist)
    if self.deterministic_features:
      hidden_deter = self._dense_deter(flat_hd)
      outputs = self.concat([outputs, hidden_deter])
    if self.residual_coef > 0.:
      outputs = self._dense_out(outputs)
      outputs = self.output_activation(outputs)
      outputs = outputs + self.residual_coef * hidden_d
    return outputs


class ParallelLatents(HierarchicalLatents):
  """Because information will take the shortest path to flow, it
  is recommended to set the number of units to be smaller or equal than
  `z0`

  References
  ----------
  Zhao, S., Song, J., Ermon, S., 2017. Learning Hierarchical Features from
      Generative Models. arXiv:1702.08396 [cs, stat].
  """

  def __init__(
      self,
      layer: Layer,
      encoder: Optional[Layer] = None,
      filters: int = 32,
      kernel_size: Union[int, Sequence[int]] = 4,
      strides: Union[int, Sequence[int]] = 2,
      padding: Literal['valid', 'same'] = 'same',
      conv_kw: Optional[Dict[str, Any]] = None,
      output_activation: Union[None, 'str', Callable[[Any], Any]] = None,
      residual_coef: float = 1.0,
      **kwargs):
    super().__init__(layer=layer, **kwargs)
    if encoder is not None:
      encoder._old_call = encoder.call
      encoder.call = MethodType(_call, encoder)
    self.encoder = encoder
    self.residual_coef = residual_coef
    if output_activation is None and hasattr(self.layer, 'activation'):
      output_activation = self.layer.activation
    self.output_activation = keras.activations.get(output_activation)
    # === 1. for creating layer
    self._network_kw = dict(
      # parameters for loc and scale
      filters=2 * filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)
    if conv_kw is not None:
      self._network_kw.update(conv_kw)
    # === 2. others
    self._conv_posterior = None
    self._conv_out = None
    self._dist_posterior = None
    self.concat = keras.layers.Concatenate(axis=-1)

  @property
  def is_inference(self) -> bool:
    return (not self._is_sampling and
            self.encoder is not None and
            hasattr(self.encoder, '_last_outputs') and
            self.encoder._last_outputs is not None)

  def build(self, input_shape=None):
    super().build(input_shape)
    if self._disable:
      return
    decoder_shape = self.layer.compute_output_shape(input_shape)
    layer, layer_t = _NDIMS_CONV[self.input_ndim]
    # === 1. create projection layer
    assert self.encoder is not None, \
      'ParallelLatents require encoder to be specified'
    # posterior projection (assume encoder shape and decoder shape the same)
    self._conv_posterior = layer(**self._network_kw, name='ConvPosterior')
    self._conv_posterior.build(decoder_shape)
    # === 2. distribution
    params_shape = self._conv_posterior.compute_output_shape(decoder_shape)
    self._dist_posterior = DistributionLambda(
      make_distribution_fn=partial(_create_dist,
                                   event_ndims=len(params_shape) - 1,
                                   dtype=self.dtype),
      name=f'{self.name}_posterior')
    self._dist_posterior.build(params_shape)
    # dynamically infer the shape
    latents_shape = tf.convert_to_tensor(self._dist_posterior(
      keras.layers.Input(params_shape[1:]))).shape
    self._latents_shape = latents_shape[1:]
    # create the prior N(0,I)
    self._prior = Independent(
      Normal(loc=tf.zeros(self.latents_shape, dtype=self.dtype),
             scale=tf.ones(self.latents_shape, dtype=self.dtype)),
      reinterpreted_batch_ndims=len(self.latents_shape),
      name=f'{self.name}_prior')
    # === 3. final output affine
    self._conv_out = _upsample_by_conv(
      layer, layer_t,
      input_shape=latents_shape,
      output_shape=decoder_shape,
      kernel_size=self._conv_posterior.kernel_size,
      padding=self._conv_posterior.padding,
      strides=self._conv_posterior.strides)

  def call(self, inputs, training=None, mask=None, **kwargs):
    hidden_d = super().call(inputs, training=training, mask=mask, **kwargs)
    if self._disable:
      return hidden_d
    # === 2. inference
    if self.is_inference:
      hidden_e = self.encoder._last_outputs
      # just stop inference if there is no Encoder state
      tf.debugging.assert_equal(tf.shape(hidden_e), tf.shape(hidden_d),
                                f'Shape of inference {hidden_e.shape} and '
                                f'generative {hidden_d.shape} mismatch. '
                                f'Change to sampling mode if possible')
      posterior = self._dist_posterior(self._conv_posterior(hidden_e))
      self._posterior = posterior
      outputs = tf.convert_to_tensor(posterior)
    else:
      outputs = self.prior.sample(tf.shape(hidden_d)[0])
    # === 3. projection and combine
    outputs = self._conv_out(outputs)
    outputs = self.output_activation(outputs)
    return outputs + self.residual_coef * hidden_d


# ===========================================================================
# Hierarchical VAE
# ===========================================================================
class HierarchicalVAE(AnnealingVAE):
  """ A hierachical VAE with multiple stochastic layers stacked on top of the previous one
  (autoregressive):

    $q(z|x) = q(z_1|x) \mul_{i=2}^L q(z_i|z_{i-1})$

  Inference: `X -> E(->z1) -> E1(->z2) -> E2 -> z`

  Generation: `z -> D2 -> z2 -> D1 -> z1 -> D -> X~`

  The return from `encode` method: (q_z, q_z2,  q_z1)

  The return from `decode` method: (X~, p_z2, p_z1)

  Hierachical takes longer to train and often more unstable, reduce the learning rate
  is often desired.

  Parameters
  ----------
  ladder_units : List[int], optional
      number of hidden units for layers in the ladder, each element corresponding
      to a ladder latents, by default [256]
  ladder_latents : List[int], optional
      number of latents units for each latent variable in the ladder,
      by default [64]
  ladder_layers : int, optional
      number of layers for each hidden layer in the ladder, by default 2
  batchnorm : bool, optional
      use batch normalization in the ladder hidden layers, by default True
  dropout : float, optional
      dropout rate for the ladder hidden layers, by default 0.0
  activation : Callable[[tf.Tensor], tf.Tensor], optional
      activation function for the ladder hidden layers, by default tf.nn.leaky_relu
  beta : Union[float, Interpolation], optional
      a fixed beta or interpolated beta based on iteration step. It is recommended
      to keep the beta value > 0 at the beginning of training, especially when using
      powerful architecture for encoder and decoder. Otherwise, the suboptimal
      latents could drive the network to very unstable loss region which result NaNs
      during early training,
      by default `linear(vmin=1e-4, vmax=1., length=2000)`
  tie_latents : bool, optional
      tie the parameters that encoding means and standard deviation for both
      $q(z_i|z_{i-1})$ and $p(z_i|z_{i-1})$, by default False
  all_standard_prior : bool, optional
      use standard normal as prior for all latent variables, by default False

  References
  ----------
  Sønderby, C.K., Raiko, T., Maaløe, L., Sønderby, S.K., Winther, O.,
    Ladder variational autoencoders,
    Advances in Neural Information Processing Systems, 2016
  Tomczak, J.M., Welling, M., 2018. VAE with a VampPrior.
    arXiv:1705.07120 [cs, stat].
  D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, and M. Welling,
    Improved variational inference with inverse autoregressive flow, in
    Advances in neural information processing systems, 2016

  """

  def __init__(self, free_bits=0.25, **kwargs):
    super().__init__(free_bits=free_bits, **kwargs)
    found_hierarchical_vars = False
    self._hierarchical_vars = []
    self._hierarchical_vars: List[HierarchicalLatents]
    for layer in self.decoder.layers:
      if isinstance(layer, HierarchicalLatents):
        found_hierarchical_vars = True
        layer.enable()
        self._hierarchical_vars.append(layer)
    if not found_hierarchical_vars:
      raise ValueError('No HierarchicalLatents wrapper found in the decoder.')

  @contextmanager
  def sampling_mode(self):
    """Temporary switch all the hierarchical latents into sampling mode"""
    [layer.sampling() for layer in self.hierarchical_latents]
    yield self
    [layer.inference() for layer in self.hierarchical_latents]

  @property
  def hierarchical_latents(self) -> Sequence[HierarchicalLatents]:
    return tuple(self._hierarchical_vars)

  def sample_prior(self, n: int = 1, seed: int = 1) -> Sequence[tf.Tensor]:
    """Sampling from prior distribution"""
    z0 = super().sample_prior(n, seed)
    with self.sampling_mode():
      self.decode(z0, training=False)
    Z = [z0]
    for layer in self.hierarchical_latents:
      Z.append(layer.prior.sample())
    return tuple(Z)

  def sample_observation(self, n: int = 1, seed: int = 1,
                         training: bool = False) -> Distribution:
    z0 = super().sample_prior(n, seed)
    with self.sampling_mode():
      obs = self.decode(z0, training=training)
    return obs

  def sample_traverse(self,
                      inputs,
                      **kwargs) -> Distribution:
    with self.sampling_mode():
      obs = super().sample_traverse(inputs, **kwargs)
    return obs

  def get_latents(self,
                  inputs=None,
                  training=None,
                  mask=None,
                  return_prior=False,
                  **kwargs) -> Sequence[Distribution]:
    z0 = super().get_latents(inputs=inputs, training=training, mask=mask,
                             return_prior=return_prior,
                             **kwargs)
    posterior, prior = list(as_tuple(z0)), []
    if return_prior:
      posterior = list(as_tuple(z0[0]))
      prior = list(as_tuple(z0[1]))
      z0 = z0[0]
    # new encode called
    if inputs is not None:
      self.decode(z0, training=training, mask=mask)
    for layer in self.hierarchical_latents:
      posterior.append(layer.posterior)
      prior.append(layer.prior)
    if return_prior:
      return tuple(posterior), tuple(prior)
    return tuple(posterior)

  @classmethod
  def is_hierarchical(cls) -> bool:
    return True

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(
      inputs, training=training, mask=mask, **kwargs)
    for idx, layer in enumerate(self.decoder.layers):
      if not isinstance(layer, HierarchicalLatents):
        continue
      kl[f'kl_{layer.name}'] = self.beta * layer.kl_divergence(
        analytic=self.analytic, reverse=self.reverse,
        free_bits=self.free_bits)
    return llk, kl


# ===========================================================================
# Unet VAE
# ===========================================================================
def _get_full_args(fn):
  spec = inspect.getfullargspec(fn)
  return spec.args + spec.kwonlyargs


def _prepare_encoder_decoder(encoder, decoder):
  if isinstance(encoder, keras.Sequential):
    encoder = [l.layer if isinstance(l, Wrapper) else l
               for l in encoder.layers]
  if isinstance(decoder, keras.Sequential):
    decoder = [l.layer if isinstance(l, Wrapper) else l
               for l in decoder.layers]
  assert isinstance(encoder, (tuple, list)), \
    f'encoder must be list of Layer, given {encoder}'
  assert isinstance(decoder, (tuple, list)), \
    f'decoder must be list of Layer, given {decoder}'
  return encoder, decoder


class UnetVAE(BetaVAE):
  """ Unet-VAE created for CIFAR10 """

  def __init__(
      self,
      encoder: List[keras.layers.Layer],
      decoder: List[keras.layers.Layer],
      layers_map: Sequence[Tuple[str, str]] = (('encoder2', 'decoder2'),
                                               ('encoder0', 'decoder4')),
      dropout: float = 0.,
      noise: float = 0.,
      beta: float = 10.,
      free_bits: float = 2.,
      **kwargs,
  ):
    encoder, decoder = _prepare_encoder_decoder(encoder, decoder)
    super().__init__(beta=beta,
                     free_bits=free_bits,
                     encoder=encoder,
                     decoder=decoder,
                     **kwargs)
    encoder_layers = set(l.name for l in self.encoder)
    # mappping from layers in decoder to encoder
    self.layers_map = dict(
      (j, i) if i in encoder_layers else (i, j) for i, j in layers_map)
    if dropout > 0.:
      self.dropout = keras.layers.Dropout(rate=dropout)
    else:
      self.dropout = None
    if noise > 0.:
      self.noise = keras.layers.GaussianNoise(stddev=noise)
    else:
      self.noise = None

  @classmethod
  def is_hierarchical(cls) -> bool:
    return False

  def encode(self,
             inputs,
             training=None,
             mask=None,
             only_encoding=False,
             **kwargs):
    h_e = inputs
    kw = dict(training=training, mask=mask, **kwargs)
    encoder_outputs = {}
    for f_e in self.encoder:
      args = _get_full_args(f_e.call)
      h_e = f_e(h_e, **{k: v for k, v in kw.items() if k in args})
      encoder_outputs[f_e.name] = h_e
    qz_x = self.latents(h_e,
                        training=training,
                        mask=mask,
                        sample_shape=self.sample_shape)
    qz_x._encoder_outputs = encoder_outputs
    return qz_x

  def decode(self,
             latents,
             training=None,
             mask=None,
             only_decoding=False,
             **kwargs):
    h_d = latents
    kw = dict(training=training, mask=mask, **kwargs)
    encoder_outputs = latents._encoder_outputs
    for f_d in self.decoder:
      args = _get_full_args(f_d.call)
      h_d = f_d(h_d, **{k: v for k, v in kw.items() if k in args})
      if f_d.name in self.layers_map:
        h_e = encoder_outputs[self.layers_map[f_d.name]]
        if self.dropout is not None:
          h_e = self.dropout(h_e, training=training)
        if self.noise is not None:
          h_e = self.noise(h_e, training=training)
        h_d = h_d + h_e
    px_z = self.observation(h_d, training=training, mask=mask)
    return px_z


class PUnetVAE(BetaVAE):
  """ Probabilistic Unet-VAE

  # TODO
  What have been tried:

  1. Soft connection: run autoencoder as normal, only add extra regularization
      D(q(z|x)||p(z|x)) => posterior collaps in all the ladder latents
      (i.e. except the main middle latents)
  2. Semi-hard connection:
  """

  def __init__(
      self,
      encoder: List[keras.layers.Layer],
      decoder: List[keras.layers.Layer],
      layers_map: List[Tuple[str, str, int]] = (
          ('encoder2', 'decoder2', 16),
          ('encoder1', 'decoder3', 16),
          ('encoder0', 'decoder4', 16),
      ),
      beta: float = 10.,
      free_bits: float = 2.,
      name: str = 'PUnetVAE',
      **kwargs,
  ):
    encoder, decoder = _prepare_encoder_decoder(encoder, decoder)
    super().__init__(encoder=encoder,
                     decoder=decoder,
                     beta=beta,
                     name=name,
                     free_bits=free_bits,
                     **kwargs)
    encoder_name = {i.name: i for i in self.encoder}
    decoder_name = {i.name: i for i in self.decoder}
    n_latents = 0
    ladder_latents = {}
    for i, j, units in layers_map:
      if i in encoder_name and j in decoder_name:
        q = RVconf(units,
                   'mvndiag',
                   projection=True,
                   name=f'ladder_q{n_latents}').create_posterior()
        p = RVconf(units,
                   'mvndiag',
                   projection=True,
                   name=f'ladder_p{n_latents}').create_posterior()
        ladder_latents[i] = q
        ladder_latents[j] = p
        n_latents += 1
    self.ladder_latents = ladder_latents
    self.flatten = keras.layers.Flatten()

  def encode(self,
             inputs,
             training=None,
             mask=None,
             only_encoding=False,
             **kwargs):
    h_e = inputs
    kw = dict(training=training, mask=mask, **kwargs)
    Q = []
    for f_e in self.encoder:
      args = _get_full_args(f_e.call)
      h_e = f_e(h_e, **{k: v for k, v in kw.items() if k in args})
      if f_e.name in self.ladder_latents:
        h = self.flatten(h_e)
        Q.append(self.ladder_latents[f_e.name](h, training=training, mask=mask))
    qz_x = self.latents(h_e,
                        training=training,
                        mask=mask,
                        sample_shape=self.sample_shape)
    return [qz_x] + Q

  def decode(self,
             latents,
             training=None,
             mask=None,
             only_decoding=False,
             **kwargs):
    h_d = latents[0]
    kw = dict(training=training, mask=mask, **kwargs)
    P = []
    for f_d in self.decoder:
      args = _get_full_args(f_d.call)
      h_d = f_d(h_d, **{k: v for k, v in kw.items() if k in args})
      if f_d.name in self.ladder_latents:
        h = self.flatten(h_d)
        P.append(self.ladder_latents[f_d.name](h, training=training, mask=mask))
    px_z = self.observation(h_d, training=training, mask=mask)
    return [px_z] + P

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    P, Q = self.last_outputs
    n_latents = len(self.ladder_latents) // 2
    for i in range(n_latents):
      pz = [p for p in P if f'ladder_p{i}' in p.name][0]
      qz = [q for q in Q if f'ladder_q{i}' in q.name][0]
      kl[f'kl_ladder{i}'] = self.beta * kl_divergence(q=qz,
                                                      p=pz,
                                                      analytic=self.analytic,
                                                      free_bits=self.free_bits,
                                                      reverse=self.reverse)
    return llk, kl


# ===========================================================================
# Very Deep VAE
# ===========================================================================
class VeryDeepVAE(AnnealingVAE):
  """ Very Deep Variational AutoEncoder

  References
  ----------
  Sønderby, C.K., et al., 2016. Ladder variational autoencoders,
      Advances in Neural Information Processing Systems.
      Curran Associates, Inc., pp. 3738–3746.
  Kingma, D.P., et al., 2016. Improved variational inference with
      inverse autoregressive flow, Advances in Neural Information
      Processing Systems. Curran Associates, Inc., pp. 4743–4751.
  Maaløe, L., et al., 2019. BIVA: A Very Deep Hierarchy of Latent
      Variables for Generative Modeling. arXiv:1902.02102 [cs, stat].
  Child, R., 2021. Very deep {VAE}s generalize autoregressive models
      and can outperform them on images, in: International Conference
      on Learning Representations.
  Havtorn, J.D., et al., 2021. Hierarchical VAEs Know What They Don’t
      Know. arXiv:2102.08248 [cs, stat].
  """
  # TODO
