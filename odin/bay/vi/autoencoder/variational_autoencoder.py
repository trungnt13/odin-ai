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
from odin.networks import (Identity, ImageNet, NetworkConfig, TensorTypes,
                           TrainStep)
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


def _parse_layers(network, name=None) -> List[Layer]:
  is_decoding = 'decoder' in str(name).lower()
  ## make sure is a list
  network = list(network) if isinstance(network, (tuple, list)) else [network]
  network = [i for i in network if i is not None]
  ## check different options
  layers = []
  for cfg in network:
    ## identity
    if cfg is None:
      layers.append(Identity(name=name))
    ## Callable or type
    elif (inspect.isfunction(cfg) or isinstance(cfg, partial) or
          isinstance(cfg, type)):
      layers.append(cfg())
    ## RVmeta
    elif isinstance(cfg, RVmeta):
      layers.append(
          cfg.create_posterior(name=name if cfg.name is None else None))
    ## string type (for dataset name or network alias)
    elif isinstance(cfg, string_types):
      cfg = cfg.lower().strip()
      #
      if cfg in ('linear', 'identity'):
        layers.append(Identity(name=name))
      #
      elif cfg in ('mnist', 'fashion_mnist'):
        kw = dict(image_shape=(28, 28, 1),
                  projection_dim=128,
                  activation='relu',
                  center0=True,
                  distribution='bernoulli',
                  distribution_kw=dict(),
                  skip_connect=False,
                  convolution=True,
                  input_shape=None,
                  decoding=is_decoding)
        layers.append(ImageNet(**kw))
      #
      elif cfg in ('shapes3d', 'dsprites', 'dspritesc', 'celeba', 'stl10',
                   'legofaces', 'cifar10', 'cifar20', 'cifar100'):
        n_channels = 1 if cfg in ('dsprites', 'dspritesc') else 3
        if cfg in ('cifar10', 'cifar100', 'cifar20'):
          image_shape = (32, 32, 3)
        else:
          image_shape = (64, 64, n_channels)
        kw = dict(image_shape=image_shape,
                  projection_dim=256,
                  activation='relu',
                  center0=True,
                  distribution='bernoulli',
                  distribution_kw=dict(),
                  skip_connect=False,
                  convolution=True,
                  input_shape=None,
                  decoding=is_decoding)
        layers.append(ImageNet(**kw))
      #
      else:
        raise NotImplementedError(
            f"No predefined network for dataset with name: {cfg}")
    # the NetworkConfig
    elif isinstance(cfg, NetworkConfig):
      layers.append(cfg.create_network(name=name))
    # Layer
    elif isinstance(cfg, Layer):
      layers.append(cfg)
    # no support
    else:
      raise ValueError(
          f"No support for network configuration of type: {type(cfg)}")
  return layers


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
  pX_Z: Optional[Distribution, List[Distribution]] = None
  qZ_X: Optional[Distribution, List[Distribution]] = None

  def call(self) -> Tuple[Tensor, Dict[str, Any]]:
    llk, kl = self.vae.elbo_components(self.inputs,
                                       training=self.training,
                                       mask=self.mask)
    elbo = self.vae.elbo(llk, kl)
    metrics = dict(**llk, **kl)
    return -tf.reduce_mean(elbo), metrics


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

  Call return
  ----------
    p(X|Z) : a single or a list of `tensorflow_probability.Distribution`
    q(Z|X) : a single or a list of `tensorflow_probability.Distribution`

  Layers
  ------
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
      encoder: LayerCreator = NetworkConfig(name="Encoder"),
      decoder: LayerCreator = NetworkConfig(name="Decoder"),
      observation: LayerCreator = RVmeta(64,
                                         'gaussian',
                                         projection=True,
                                         name="Observation"),
      latents: LayerCreator = RVmeta(10,
                                     'mvndiag',
                                     projection=True,
                                     name="Latents"),
      **kwargs,
  ):
    ### keras want this supports_masking on to enable support masking
    super().__init__(**kwargs)
    ### create layers
    self._encoder = _parse_layers(network=encoder, name="Encoder")
    self._encoder_args = [_get_args(i) for i in self.encoder]
    self._latents = _parse_layers(network=latents, name="Latents")
    self._latents_args = [_get_args(i) for i in self.latents]
    self._decoder = _parse_layers(network=decoder, name="Decoder")
    self._decoder_args = [_get_args(i) for i in self.decoder]
    self._observation = _parse_layers(network=observation, name="Observation")
    self._observation_args = [_get_args(i) for i in self.observation]

  @property
  def encoder(self) -> List[Layer]:
    return self._encoder

  @property
  def decoder(self) -> List[Layer]:
    return self._decoder

  @property
  def latents(self) -> List[Layer]:
    return self._latents

  @property
  def observation(self) -> List[Layer]:
    return self._observation

  @property
  def n_latents(self) -> int:
    return len(self.latents)

  @property
  def n_observation(self) -> int:
    return len(self.observation)

  @property
  def input_shape(self) -> Union[List[int], List[List[int]]]:
    shape = [e.input_shape for e in self.encoder]
    return shape[0] if len(shape) == 1 else shape

  @property
  def latent_shape(self) -> Union[List[int], List[List[int]]]:
    shape = [d.input_shape for d in self.decoder]
    return shape[0] if len(shape) == 1 else shape

  def sample_prior(self,
                   sample_shape: Union[int, List[int]] = (),
                   seed: int = 1) -> Union[Tensor, List[Tensor]]:
    r""" Sampling from prior distribution """
    samples = []
    for latent in self.latents:
      s = bk.atleast_2d(latent.sample(sample_shape=sample_shape, seed=seed))
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def sample_data(self,
                  sample_shape: Union[int, List[int]] = (),
                  seed: int = 1) -> Union[Tensor, List[Tensor]]:
    r""" Sample from p(X) given that the prior of X is known, this could be
    wrong since `RVmeta` often has a default prior. """
    samples = []
    for output in self.observation:
      s = bk.atleast_2d(output.sample(sample_shape=sample_shape, seed=seed))
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def generate(self,
               sample_shape: List[int] = (),
               training: Optional[bool] = None,
               seed: int = 1,
               **kwargs) -> Union[Distribution, List[Distribution]]:
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
             **kwargs) -> Union[Distribution, List[Distribution]]:
    r""" Encoding inputs to latent codes """
    states = []
    for encoder, args in zip(self.encoder, self._encoder_args):
      copy_kw = dict(kwargs)
      if 'mask' in args:
        copy_kw['mask'] = mask
      if 'training' in args:
        copy_kw['training'] = training
      e = encoder(inputs, **copy_kw)
      states.append(e)
    # create the latents distribution
    qZ_X = []
    for args, (_, latent, code) in zip(self._latents_args,
                                       _iter_lists(self.latents, states)):
      kw = {}
      if 'training' in args:
        kw['training'] = training
      if 'sample_shape' in args:
        kw['sample_shape'] = self.sample_shape
      qZ_X.append(latent(code, **kw))
    qZ_X = tf.nest.flatten(qZ_X)
    # remember to store the keras mask in outputs
    for q in qZ_X:
      q._keras_mask = mask
    return qZ_X[0] if len(qZ_X) == 1 else tuple(qZ_X)

  def decode(self,
             latents: Union[TensorTypes, Distribution, List[TensorTypes],
                            List[Distribution]],
             training: Optional[bool] = None,
             mask: Optional[Tensor] = None,
             **kwargs) -> Union[Distribution, List[Distribution]]:
    r""" Decoding latent codes, this does not guarantee output the
    reconstructed distribution """
    # stop tensorflow complaining about tensor inputs for Sequential
    c = tf.constant(0., dtype=self.dtype)
    latents = [z + c for z in latents] \
      if isinstance(latents, (tuple, list)) else latents + c
    # apply the decoder and get back the sample shape
    states = []
    for decoder, args in zip(self.decoder, self._decoder_args):
      copy_kw = dict(kwargs)
      if 'mask' in args:
        copy_kw['mask'] = mask
      if 'training' in args:
        copy_kw['training'] = training
      states.append(decoder(latents, **copy_kw))
    # create the output distribution
    pX_Z = []
    for args, (_, layer, o) in zip(self._observation_args,
                                   _iter_lists(self.observation, states)):
      kw = {}
      if 'training' in args:
        kw['training'] = training
      if 'mask' in args:
        kw['mask'] = mask
      pX_Z.append(layer(o, **kw))
    pX_Z = tf.nest.flatten(pX_Z)
    # remember to store the keras mask in outputs
    for p in pX_Z:
      p._keras_mask = mask
    return pX_Z[0] if len(pX_Z) == 1 else tuple(pX_Z)

  def call(
      self,
      inputs: TensorTypes,
      training: Optional[bool] = None,
      mask: Optional[Tensor] = None,
      **kwargs
  ) -> Tuple[Union[Distribution, List[Distribution], Union[
      Distribution, List[Distribution]]]]:
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
    qZ_X = self.encode(
        inputs,
        training=training,
        mask=mask,
        **{k: v for k, v in kwargs.items() if k in self._encode_func_args},
    )
    # transfer the mask from encoder to decoder here
    for q in tf.nest.flatten(qZ_X):
      if hasattr(q, '_keras_mask') and q._keras_mask is not None:
        mask = q._keras_mask
        break
    # decode
    pX_Z = self.decode(
        qZ_X,
        training=training,
        mask=mask,
        **{k: v for k, v in kwargs.items() if k in self._decode_func_args},
    )
    outputs = (pX_Z, qZ_X)
    self._last_outputs = outputs
    return outputs

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
    pX_Z, qZ_X = self(inputs, training=training, mask=mask, **kwargs)
    ## Marginal LLK
    llk = []
    distortion = {}
    # reconstruction (a.k.a distortion)
    for i, (pX,
            x) in enumerate(zip(tf.nest.flatten(pX_Z),
                                tf.nest.flatten(inputs))):
      x_llk = pX.log_prob(x)
      llk.append(x_llk)
      distortion[pX.name.split('_')[0]] = x_llk
    # kl-divergence (a.k.a rate)
    for qZ in tf.nest.flatten(qZ_X):
      if isinstance(qZ, (tfd.Deterministic, tfd.VectorDeterministic)):
        continue
      z = tf.convert_to_tensor(qZ)
      # the prior is injected into the distribution during the call method of
      # DenseDistribution, or modified during the encode method by setting
      # qZ_X.KL_divergence.prior = ...
      pZ = qZ.KL_divergence.prior
      if pZ is None:
        pZ = tfd.Normal(loc=tf.zeros(qZ.event_shape, dtype=z.dtype),
                        scale=tf.ones(qZ.event_shape, dtype=z.dtype))
      llk_pz = pZ.log_prob(z)
      llk_qz_x = qZ.log_prob(z)
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
      inputs: Union[Tensor, List[Tensor]],
      training: Optional[bool] = None,
      mask: Optional[Tensor] = None,
      **kwargs,
  ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Calculate the distortion (log-likelihood) and rate (KL-divergence)
    for contruction the Evident Lower Bound (ELBO)"""
    # organize all inputs to list
    pX_Z, qZ_X = self(inputs, training=training, mask=mask, **kwargs)
    ### llk
    llk = {}
    for obs, x, pX in zip(self.observation, as_tuple(inputs), as_tuple(pX_Z)):
      llk[f'llk_{obs.name}'] = pX.log_prob(x)
    ### kl
    kl = {}
    for z, qZ in zip(self.latents, as_tuple(qZ_X)):
      if hasattr(qZ, "KL_divergence"):
        kl[f'kl_{z.name}'] = qZ.KL_divergence(analytic=self.analytic,
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
            f"(semi:{self.is_semi_supervised})")
    text += f'\n Tensorboard : {self.tensorboard_logdir}'
    text += f'\n Analytic     : {self.analytic}'
    text += f'\n Reverse      : {self.reverse}'
    text += f'\n Sample Shape : {self.sample_shape}'
    text += f'\n Fitted        : {int(self.step.numpy())}(iters)'
    text += f'\n MD5 checksum: {self.md5_checksum}'
    ## encoder
    for i, encoder in enumerate(tf.nest.flatten(self.encoder)):
      text += f"\n Encoder#{i}:\n  "
      text += "\n  ".join(_net2str(encoder).split('\n'))
    ## Decoder
    for i, decoder in enumerate(tf.nest.flatten(self.decoder)):
      text += f"\n Decoder#{i}:\n  "
      text += "\n  ".join(_net2str(decoder).split('\n'))
    ## Latent
    for i, latent in enumerate(self.latents):
      text += f"\n Latent#{i}:\n  "
      text += "\n  ".join(_net2str(latent).split('\n'))
    ## Output
    for i, output in enumerate(self.observation):
      text += f"\n Output#{i}:\n  "
      text += "\n  ".join(_net2str(output).split('\n'))
    ## Optimizer
    if hasattr(self, 'optimizer'):
      for i, opt in enumerate(tf.nest.flatten(self.optimizer)):
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
    for qz in tf.nest.flatten(latents):
      if isinstance(qz, RVmeta):
        qz.posterior = 'vdeterministic'
      elif isinstance(qz, DenseDistribution):
        assert qz.posterior == VectorDeterministicLayer, \
          ('Autoencoder only support VectorDeterministic posterior, '
          f'but given:{qz.posterior}')
    super().__init__(latents=latents, name=name, **kwargs)

  def elbo_components(self,
                      inputs,
                      training=None,
                      pX_Z=None,
                      qZ_X=None,
                      mask=None,
                      **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      training=training,
                                      pX_Z=pX_Z,
                                      qZ_X=qZ_X,
                                      mask=mask,
                                      **kwargs)
    # this make sure no KL is leaking
    kl = {k: 0. for k, v in kl.items()}
    return llk, kl
