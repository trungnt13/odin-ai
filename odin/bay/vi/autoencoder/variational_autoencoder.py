from __future__ import absolute_import, annotations, division, print_function

import copy
import glob
import inspect
import os
import pickle
import warnings
from dataclasses import dataclass
from functools import partial
from itertools import zip_longest
from numbers import Number
from typing import (Any, Callable, Dict, Iterator, List, Optional, Text, Tuple,
                    Type, Union)

import numpy as np
import scipy as sp
import tensorflow as tf
from numpy import ndarray
from odin import backend as bk
from odin.backend.keras_helpers import layer2text
from odin.bay.layers import DenseDistribution, VectorDeterministicLayer
from odin.bay.random_variable import RVmeta
from odin.bay.vi._base import VariationalModel
from odin.exp.trainer import Trainer
from odin.networks import Identity, NetworkConfig, TensorTypes, TrainStep
from odin.utils import as_tuple
from odin.utils.python_utils import classproperty
from scipy.sparse import spmatrix
from six import string_types
from tensorflow import Tensor, Variable
from tensorflow.python import keras
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops.summary_ops_v2 import SummaryWriter
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.distributions import Distribution
from typing_extensions import Literal

__all__ = [
    'LayerCreator',
    'VAEStep',
    'VariationalAutoencoder',
    'VAE',
    'Autoencoder',
]

# ===========================================================================
# Types
# ===========================================================================
LayerCreator = Union[str, Layer, Type[Layer], \
                     NetworkConfig, RVmeta, \
                     Callable[[Optional[List[int]]], Layer]]


# ===========================================================================
# Helpers
# ===========================================================================
def _get_args(layer):
  spec = inspect.getfullargspec(layer.call)
  return set(spec.args + spec.kwonlyargs)


def _net2str(net):
  if isinstance(net, (keras.Sequential, tfl.DistributionLambda)):
    return layer2text(net)
  return str(net)


def _parse_layers(network, is_decoding=False, name=None) -> Layer:
  ## make sure is a list
  if isinstance(network, (tuple, list)):
    if len(network) != 1:
      raise ValueError(
          f'Only support single neural network but provide {network}')
    network = network[0]
  assert network is not None, 'network cannot be None'
  ## check different options
  cfg = network
  ## identity
  if cfg is None:
    layer = Identity(name=name)
  ## string alias of activation
  elif isinstance(cfg, string_types):
    layer = keras.layers.Activation(keras.activations.get(cfg))
  ## Callable or type
  elif (inspect.isfunction(cfg) or isinstance(cfg, partial) or
        isinstance(cfg, type)):
    layer = cfg()
  ## RVmeta
  elif isinstance(cfg, RVmeta):
    layer = cfg.create_posterior(name=name if cfg.name is None else None)
  ## the NetworkConfig
  elif isinstance(cfg, NetworkConfig):
    layer = cfg.create_network(name=name)
  ## Layer
  elif isinstance(cfg, Layer):
    layer = cfg
  ## no support
  else:
    raise ValueError(
        f"No support for network configuration of type: {type(cfg)}")
  return layer


def _iter_lists(X, Y):
  r""" Try to match the length of list-Y to list-X,
  the yield a pair of (x, y) with the condition x is not None """
  Y = Y * len(X) if len(Y) == 1 else Y
  for i, (x, y) in enumerate(zip_longest(X, Y)):
    if x is not None:
      yield i, x, y


# ===========================================================================
# Training step
# ===========================================================================
@dataclass
class VAEStep(TrainStep):
  r""" A single train step (iteration) for Variational Autoencoder """

  vae: VariationalAutoencoder
  call_kw: Dict[str, Any]

  def call(self) -> Tuple[Tensor, Dict[str, Any]]:
    llk, kl = self.vae.elbo_components(self.inputs,
                                       training=self.training,
                                       mask=self.mask)
    elbo = self.vae.elbo(llk, kl)
    loss = -tf.reduce_mean(elbo)
    metrics = dict(**llk, **kl)
    return loss, metrics


# ===========================================================================
# Model
# ===========================================================================
class VariationalAutoencoder(VariationalModel):
  r"""Base class for all variational autoencoder

  Parameters
  ----------
  encoder : LayerCreator, optional
      the encoder network, by default NetworkConfig()
  decoder : LayerCreator, optional
      the decoder network, by default NetworkConfig()
  outputs : LayerCreator, optional
      a descriptor for the input/output, by default
      `RVmeta(64, 'gaus', projection=True, name="Input")`
  latents : LayerCreator, optional
      a descriptor for the latents' distribution, by default
      `RVmeta(10, 'mvndiag', projection=True, name="Latent")`
  input_shape : Optional[List[int]], optional
      specific input_shape for the network, if not given, use the given `outputs`,
      by default None

  Returns
  ---------
    p(X|Z) : a single or a list of `tensorflow_probability.Distribution`
    q(Z|X) : a single or a list of `tensorflow_probability.Distribution`

  Layers
  --------
    encoder : list of `keras.layers.Layer`.
      Encoding inputs to latents
    decoder : list of `keras.layers.Layer`.
      Decoding latents to intermediate states
    latents : list of `keras.layers.Layer`.
      A list of the Dense layer that create the latent variable (random variable)
    observation : list of `keras.layers.Layer`.
      A list of the Dense layer that create the output variable
      (random or deterministic variable)
  """

  def __init__(
      self,
      observation: LayerCreator = RVmeta((28, 28, 1),
                                         'bernoulli',
                                         projection=True,
                                         name='image'),
      encoder: LayerCreator = NetworkConfig([512, 512],
                                            flatten_inputs=True,
                                            name="encoder"),
      decoder: LayerCreator = NetworkConfig([512, 512],
                                            flatten_inputs=True,
                                            name="decoder"),
      latents: LayerCreator = RVmeta(64,
                                     'mvndiag',
                                     projection=True,
                                     name="latents"),
      **kwargs,
  ):
    ### keras want this supports_masking on to enable support masking
    super().__init__(**kwargs)
    ### create layers
    self._encoder = _parse_layers(network=encoder, name="encoder")
    self._encoder_args = _get_args(self.encoder)
    self._latents = _parse_layers(network=latents, name="latents")
    self._latents_args = _get_args(self.latents)
    self._decoder = _parse_layers(network=decoder, name="decoder")
    self._decoder_args = _get_args(self.decoder)
    self._observation = _parse_layers(network=observation, name="observation")
    self._observation_args = _get_args(self.observation)

  @property
  def encoder(self) -> Layer:
    return self._encoder

  @property
  def decoder(self) -> Layer:
    return self._decoder

  @property
  def latents(self) -> Layer:
    return self._latents

  @property
  def observation(self) -> Layer:
    return self._observation

  @property
  def n_latents(self) -> int:
    return len(self.latents)

  @property
  def n_observation(self) -> int:
    return len(self.observation)

  @property
  def input_shape(self) -> List[int]:
    return self.encoder.input_shape

  @property
  def latent_shape(self) -> List[int]:
    return self.decoder.input_shape

  def sample_prior(self,
                   sample_shape: Union[int, List[int]] = (),
                   seed: int = 1) -> Tensor:
    r""" Sampling from prior distribution """
    return bk.atleast_2d(
        self.latents.sample(sample_shape=sample_shape, seed=seed))

  def sample_data(self,
                  sample_shape: Union[int, List[int]] = (),
                  seed: int = 1) -> Tensor:
    r""" Sample from p(X) given that the prior of X is known, this could be
    wrong since `RVmeta` often has a default prior. """
    return bk.atleast_2d(
        self.observation.sample(sample_shape=sample_shape, seed=seed))

  def generate(self,
               sample_shape: List[int] = (),
               training: Optional[bool] = None,
               seed: int = 1,
               **kwargs) -> Distribution:
    r"""Randomly generate outputs by sampling from prior distribution then
    decode it.

    Parameters
    ----------
    sample_shape : List[int], optional
        the sample shape, by default ()
    seed : int, optional
        seed for the Tensorflow random state, by default 1
    training : Optional[bool], optional
        invoke call method in which training mode, by default None

    Returns
    -------
    Union[Distribution, List[Distribution]]
        the output distribution(s)
    """
    z = self.sample_prior(sample_shape, seed)
    return self.decode(z, training=training, **kwargs)

  def encode(self,
             inputs: Union[TensorTypes, List[TensorTypes]],
             training: Optional[bool] = None,
             mask: Optional[TensorTypes] = None,
             **kwargs) -> Distribution:
    r""" Encoding inputs to latent codes """
    kw = dict(kwargs)
    if 'mask' in self._encoder_args:
      kw['mask'] = mask
    if 'training' in self._encoder_args:
      kw['training'] = training
    h_e = self.encoder(inputs, **kw)
    # create the latents distribution
    kw = {}
    if 'training' in self._latents_args:
      kw['training'] = training
    if 'mask' in self._latents_args:
      kw['mask'] = mask
    if 'sample_shape' in self._latents_args:
      kw['sample_shape'] = self.sample_shape
    qz_x = self.latents(h_e, **kw)
    # need to keep the keras mask
    qz_x._keras_mask = mask
    return qz_x

  def decode(self,
             latents: Union[TensorTypes, List[TensorTypes]],
             training: Optional[bool] = None,
             mask: Optional[Tensor] = None,
             **kwargs) -> Distribution:
    r""" Decoding latent codes, this does not guarantee output the
    reconstructed distribution """
    # stop tensorflow complaining about tensor inputs for Sequential
    c = tf.constant(0., dtype=self.dtype)
    if isinstance(latents, (tuple, list)):
      latents = [qz + c for qz in latents]
    else:
      latents = latents + c
    # flatten the sample shapes
    if self.sample_ndim > 0:
      ndim = (latents.shape.ndims - self.sample_ndim - 1)
      flat_shape = tf.shape(latents)[-ndim:]
      flat_shape = tf.concat([[-1], flat_shape], axis=0)
      latents = tf.reshape(latents, flat_shape)
    # apply the decoder and get back the sample shape
    kw = dict(kwargs)
    if 'mask' in self._decoder_args:
      kw['mask'] = mask
    if 'training' in self._decoder_args:
      kw['training'] = training
    h_d = self.decoder(latents, **kw)
    # recover the sample shape
    if self.sample_ndim > 0:
      org_shape = tf.concat(
          [self.sample_shape, [-1], tf.shape(h_d)[1:]], axis=0)
      h_d = tf.reshape(h_d, org_shape)
    # create the output distribution
    kw = {}
    if 'training' in self._observation_args:
      kw['training'] = training
    if 'mask' in self._observation_args:
      kw['mask'] = mask
    px_z = self.observation(h_d, **kw)
    # remember to store the keras mask in outputs
    px_z._keras_mask = mask
    return px_z

  def call(self,
           inputs: TensorTypes,
           training: Optional[bool] = None,
           mask: Optional[Tensor] = None,
           **kwargs) -> Tuple[Distribution, Distribution]:
    """Applying the encode-decode process for VAE

    Parameters
    ----------
    inputs : TensorTypes
        inputs' Tensors
    training : Optional[bool], optional
        training or evaluation mode, by default None
    mask : Optional[Tensor], optional
        mask, by default None

    Returns
    -------
    Union[Distribution, List[Distribution]]
        `p_{theta}(x||z)` the output distribution(s)
    Union[Distribution, List[Distribution]]
        `q_{\phi}(z||x)` the latent distribution(s)
    """
    # encode
    qz_x = self.encode(
        inputs,
        training=training,
        mask=mask,
        **{k: v for k, v in kwargs.items() if k in self._encode_func_args},
    )
    # transfer the mask from encoder to decoder here
    for qz in as_tuple(qz_x):
      if hasattr(qz, '_keras_mask') and qz._keras_mask is not None:
        mask = qz._keras_mask
        break
    # decode
    px_z = self.decode(
        qz_x,
        training=training,
        mask=mask,
        **{k: v for k, v in kwargs.items() if k in self._decode_func_args},
    )
    self._last_outputs = (px_z, qz_x)
    return self.last_outputs

  @tf.function(autograph=False)
  def marginal_log_prob(self,
                        inputs: Union[TensorTypes, List[TensorTypes]],
                        training: Optional[bool] = None,
                        mask: Optional[Tensor] = None,
                        **kwargs) -> Tuple[Tensor, Tensor]:
    """Marginal log likelihood `log(p(X))`, an biased estimation.
    With sufficient amount of MCMC samples (-> inf), the value will converges
    to `log(p(X))`

    With large amount of sample, recommending reduce the batch size to very
    small number or use CPU for the calculation `with tf.device("/CPU:0"):`

    Note: this function will need further modification for more complicated
    prior and latent space, only work for:

      - vanilla-VAE or
      - with proper prior injected into qZ_X and pZ_X using
        `qZ_X.KL_divergence.prior = ...` during `encode` or `decode` methods


    Parameters
    ----------
    inputs : TensorTypes
        inputs' Tensors
    training : Optional[bool], optional
        training or evaluation mode, by default None
    mask : Optional[Tensor], optional
        mask Tensor, by default None

    Returns
    -------
    Tuple[Tensor, Tensor]
      marginal log-likelihood : a Tensor of shape `[batch_size]`
        marginal log-likelihood of p(X)
      distortion (a.k.a reconstruction): a Dictionary mapping from distribution
        name to Tensor of shape `[batch_size]`, the negative reconstruction cost.
    """
    sample_shape = [tf.cast(tf.reduce_prod(self.sample_shape), tf.int32)]
    px_z, qz_x = self(inputs, training=training, mask=mask, **kwargs)
    ## Marginal LLK
    llk = []
    distortion = {}
    # reconstruction (a.k.a distortion)
    for i, (px, x) in enumerate(zip(as_tuple(px_z), as_tuple(inputs))):
      x_llk = px.log_prob(x)
      llk.append(x_llk)
      distortion[px.name.split('_')[0]] = x_llk
    # kl-divergence (a.k.a rate)
    for qz in tf.nest.flatten(qz_x):
      if isinstance(qz, (tfd.Deterministic, tfd.VectorDeterministic)):
        continue
      z = tf.convert_to_tensor(qz)
      # the prior is injected into the distribution during the call method of
      # DenseDistribution, or modified during the encode method by setting
      # qZ_X.KL_divergence.prior = ...
      pz = qz.KL_divergence.prior
      if pz is None:
        pz = tfd.Normal(loc=tf.zeros(qz.event_shape, dtype=z.dtype),
                        scale=tf.ones(qz.event_shape, dtype=z.dtype),
                        name='pz')
      llk_pz = pz.log_prob(z)
      llk_qz_x = qz.log_prob(z)
      llk.append(llk_pz)
      llk.append(llk_qz_x)
    # sum all llk
    iw_const = tf.math.log(tf.cast(tf.reduce_prod(sample_shape), self.dtype))
    mllk = 0.
    for i in llk:
      mllk += i
    mllk = tf.reduce_logsumexp(mllk, axis=0) - iw_const
    distortion = {
        k: tf.reduce_logsumexp(v, axis=0) - iw_const
        for k, v in distortion.items()
    }
    return mllk, distortion

  def elbo_components(
      self,
      inputs: Union[TensorTypes, List[TensorTypes]],
      training: Optional[bool] = None,
      mask: Optional[Tensor] = None,
      **kwargs,
  ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Calculate the distortion (log-likelihood) and rate (KL-divergence)
    for contruction the Evident Lower Bound (ELBO)"""
    # organize all inputs to list
    px_z, qz_x = self(inputs, training=training, mask=mask, **kwargs)
    ### llk
    llk = {}
    for o, x, px in zip(as_tuple(self.observation), as_tuple(inputs),
                        as_tuple(px_z)):
      llk[f'llk_{o.name}'] = px.log_prob(x)
    ### kl
    kl = {}
    for z, qz in zip(as_tuple(self.latents), as_tuple(qz_x)):
      if hasattr(qz, "KL_divergence"):
        kl[f'kl_{z.name}'] = qz.KL_divergence(analytic=self.analytic,
                                              reverse=self.reverse,
                                              sample_shape=None,
                                              keepdims=True)
      else:
        kl[f'kl_{z.name}'] = tf.constant(0., dtype=self.dtype)
    return llk, kl

  ################## For training
  def train_steps(self,
                  inputs: TensorTypes,
                  training: Optional[bool] = None,
                  mask: Optional[Tensor] = None,
                  call_kw: Dict[str, Any] = {}) -> Iterator[VAEStep]:
    r""" Facilitate multiple steps training for each iteration
    (similar to GAN) """
    yield VAEStep(vae=self,
                  parameters=self.trainable_variables,
                  inputs=inputs,
                  training=training,
                  mask=mask,
                  call_kw=call_kw)

  def __str__(self):
    cls = [
        i for i in type.mro(type(self)) if issubclass(i, VariationalAutoencoder)
    ]
    text = (f"{'->'.join([i.__name__ for i in cls[::-1]])} "
            f"(semi:{type(self).is_semi_supervised()})")
    text += f'\n Tensorboard : {self.tensorboard_logdir}'
    text += f'\n Analytic     : {self.analytic}'
    text += f'\n Reverse      : {self.reverse}'
    text += f'\n Sample Shape : {self.sample_shape}'
    text += f'\n Fitted        : {int(self.step.numpy())}(iters)'
    text += f'\n MD5 checksum: {self.md5_checksum}'
    ## encoder
    for i, encoder in enumerate(as_tuple(self.encoder)):
      text += f"\n Encoder#{i}:\n  "
      text += "\n  ".join(_net2str(encoder).split('\n'))
    ## Decoder
    for i, decoder in enumerate(as_tuple(self.decoder)):
      text += f"\n Decoder#{i}:\n  "
      text += "\n  ".join(_net2str(decoder).split('\n'))
    ## Latent
    for i, latent in enumerate(as_tuple(self.latents)):
      text += f"\n Latent#{i}:\n  "
      text += "\n  ".join(_net2str(latent).split('\n'))
    ## Output
    for i, output in enumerate(as_tuple(self.observation)):
      text += f"\n Output#{i}:\n  "
      text += "\n  ".join(_net2str(output).split('\n'))
    ## Optimizer
    if hasattr(self, 'optimizer'):
      for i, opt in enumerate(as_tuple(self.optimizer)):
        if isinstance(opt, tf.optimizers.Optimizer):
          text += f"\n Optimizer#{i}:\n  "
          text += "\n  ".join(
              ["%s:%s" % (k, str(v)) for k, v in opt.get_config().items()])
    return text


VAE = VariationalAutoencoder


# ===========================================================================
# Simple implementation of Autoencoder
# ===========================================================================
class Autoencoder(VariationalAutoencoder):
  """The vanilla autoencoder could be interpreted as a Variational Autoencoder with
  `vector deterministic` distribution for latent codes."""

  def __init__(
      self,
      latents: LayerCreator = RVmeta(10,
                                     'vdeterministic',
                                     projection=True,
                                     name="Latents"),
      name='Autoencoder',
      **kwargs,
  ):
    for qz in as_tuple(latents):
      if isinstance(qz, RVmeta):
        qz.posterior = 'vdeterministic'
      elif isinstance(qz, DenseDistribution):
        assert qz.posterior == VectorDeterministicLayer, \
          ('Autoencoder only support VectorDeterministic posterior, '
          f'but given:{qz.posterior}')
    super().__init__(latents=latents, name=name, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      training=training,
                                      mask=mask,
                                      **kwargs)
    # this make sure no KL is leaking
    kl = {k: 0. for k, v in kl.items()}
    return llk, kl
