from __future__ import absolute_import, division, print_function

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import zip_longest
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union, Sequence)
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow import Tensor
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.distributions import Distribution
from tqdm import tqdm

from odin import backend as bk
from odin.backend import TensorType
from odin.backend.keras_helpers import layer2text
from odin.bay.layers.dense_distribution import DistributionDense
from odin.bay.random_variable import RVconf
from odin.bay.vi._base import VariationalModel
from odin.networks import Identity, NetConf, TrainStep, SequentialNetwork
from odin.utils import as_tuple
from odin.bay.vi.utils import traverse_dims

__all__ = [
  'LayerCreator',
  'VAEStep',
  'VariationalAutoencoder',
  'SemiSupervisedVAE',
  'VAE',
]

# ===========================================================================
# Types
# ===========================================================================
LayerCreator = Union[str, Layer, Type[Layer], NetConf, RVconf,
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


def _parse_layers(network, name=None
                  ) -> Union[Layer, DistributionDense, SequentialNetwork]:
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
  elif isinstance(cfg, RVconf):
    layer = cfg.create_posterior(name=name if cfg.name is None else None)
  ## the NetConf
  elif isinstance(cfg, NetConf):
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
  """ A single train step (iteration) for Variational Autoencoder """

  vae: 'VariationalAutoencoder' = None
  call_kw: Dict[str, Any] = field(default_factory=dict)

  def call(self) -> Tuple[Tensor, Dict[str, Any]]:
    llk, kl = self.vae.elbo_components(self.inputs,
                                       training=self.training,
                                       mask=self.mask,
                                       **self.call_kw)
    # check if array is empty, return 0s
    loss = -tf.reduce_mean(self.vae.elbo(llk, kl))
    metrics = dict(**{k: tf.reduce_mean(v) for k, v in llk.items()},
                   **{k: tf.reduce_mean(v) for k, v in kl.items()})
    return loss, metrics


# ===========================================================================
# Model
# ===========================================================================
class VariationalAutoencoder(VariationalModel):
  """Base class for all variational autoencoder

  Parameters
  ----------
  encoder : LayerCreator, optional
      the encoder network, by default NetConf()
  decoder : LayerCreator, optional
      the decoder network, by default NetConf()
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
      observation: LayerCreator = RVconf((28, 28, 1), 'bernoulli',
                                         projection=True,
                                         name='image'),
      latents: Optional[LayerCreator] = RVconf(16, 'mvndiag',
                                               projection=True,
                                               name="latents"),
      encoder: Optional[LayerCreator] = None,
      decoder: Optional[LayerCreator] = None,
      **kwargs,
  ):
    if encoder is None:
      encoder = NetConf((512, 512), flatten_inputs=True, name="encoder")
    if decoder is None:
      decoder = NetConf((512, 512), flatten_inputs=True, flatten_outputs=True,
                        name="decoder")
    ### keras want this supports_masking on to enable support masking
    super().__init__(**kwargs)
    ### create layers
    # encoder
    if isinstance(encoder, (tuple, list)):
      self._encoder = [
        _parse_layers(network=e, name=f"encoder{i}")
        for i, e in enumerate(encoder)
      ]
      self._encoder_args = [_get_args(e) for e in self._encoder]
    else:
      self._encoder = _parse_layers(network=encoder, name="encoder")
      self._encoder_args = _get_args(self.encoder)
    # latents
    if isinstance(latents, (tuple, list)):
      self._latents = [
        _parse_layers(network=z, name=f"latents{i}")
        for i, z in enumerate(latents)
      ]
      self._latents_args = [_get_args(z) for z in self.latents]
    else:
      self._latents = _parse_layers(network=latents, name="latents")
      self._latents_args = _get_args(self.latents)
    # decoder
    if isinstance(decoder, (tuple, list)):
      self._decoder = [
        _parse_layers(network=d, name=f"decoder{i}")
        for i, d in enumerate(decoder)
      ]
      self._decoder_args = [_get_args(d) for d in self.decoder]
    else:
      self._decoder = _parse_layers(network=decoder, name="decoder")
      self._decoder_args = _get_args(self.decoder)
    # observation
    if isinstance(observation, (tuple, list)):
      self._observation = [
        _parse_layers(network=observation, name=f"observation{i}")
        for i, o in enumerate(observation)
      ]
      self._observation_args = [_get_args(o) for o in self.observation]
    else:
      self._observation = _parse_layers(network=observation, name="observation")
      self._observation_args = _get_args(self.observation)

  @property
  def encoder(self) -> Union[SequentialNetwork, List[SequentialNetwork]]:
    return self._encoder

  @property
  def decoder(self) -> Union[SequentialNetwork, List[SequentialNetwork]]:
    return self._decoder

  @property
  def latents(self) -> Union[DistributionDense, List[DistributionDense]]:
    return self._latents

  @property
  def observation(self) -> Union[DistributionDense, List[DistributionDense]]:
    return self._observation

  @property
  def n_latents(self) -> int:
    if isinstance(self.latents, (tuple, list)):
      return len(self.latents)
    return 1

  @property
  def n_observation(self) -> int:
    if isinstance(self.observation, (tuple, list)):
      return len(self.observation)
    return 1

  @property
  def input_shape(self) -> Sequence[Union[None, int]]:
    return self.encoder.input_shape

  @property
  def latent_shape(self) -> Sequence[Union[None, int]]:
    return self.decoder.input_shape

  def sample_prior(self, n: int = 1, seed: int = 1) -> Tensor:
    """Sampling from prior distribution"""
    return bk.atleast_2d(
      self.latents.sample(sample_shape=n, seed=seed))

  def sample_observation_prior(self, n: int = 1, seed: int = 1) -> Tensor:
    """Sample from observation prior, i.e. p(X), given that the prior of X
    is known.

    Notes
    -----
    This could be wrong since `RVmeta` often has a default prior.
    """
    return bk.atleast_2d(
      self.observation.sample(sample_shape=n, seed=seed))

  def sample_observation(self, n: int = 1, seed: int = 1,
                         training: bool = False) -> Distribution:
    """Sample observation using latents prios"""
    z = self.sample_prior(n, seed=seed)
    return self.decode(z, training=training)

  def encode(self,
             inputs: Union[TensorType, List[TensorType]],
             training: Optional[bool] = None,
             mask: Optional[TensorType] = None,
             only_encoding: bool = False,
             **kwargs) -> Distribution:
    """ Encoding inputs to latent codes """
    kw = dict(kwargs)
    if 'mask' in self._encoder_args:
      kw['mask'] = mask
    if 'training' in self._encoder_args:
      kw['training'] = training
    h_e = self.encoder(inputs, **kw)
    if only_encoding:
      return h_e
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
             latents: Union[TensorType, List[TensorType]],
             training: Optional[bool] = None,
             mask: Optional[Tensor] = None,
             only_decoding: bool = False,
             **kwargs) -> Distribution:
    """ Decoding latent codes, this does not guarantee output the
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
    # only return hidden states from the decoder
    if only_decoding:
      return h_d
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
           inputs: TensorType,
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
    qz_x = self.encode(inputs, training=training, mask=mask, **kwargs)
    # transfer the mask from encoder to decoder here
    for qz in as_tuple(qz_x):
      if hasattr(qz, '_keras_mask') and qz._keras_mask is not None:
        mask = qz._keras_mask
        break
    # decode
    px_z = self.decode(qz_x, training=training, mask=mask, **kwargs)
    return px_z, qz_x

  def marginal_log_prob(
      self,
      inputs: Union[TensorType, List[TensorType]],
      training: Optional[bool] = None,
      n_mcmc: Optional[int] = 100,
      reduce: Optional[Callable[[Tensor], Tensor]] = tf.reduce_mean,
      batch_size: int = 32,
      verbose: bool = False,
      **kwargs,
  ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
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

    Returns
    -------
    Tuple[Dict[str, Tensor], Dict[str, Tensor]]
      reconstruction : mapping from name to marginal reconstruction
      KL divergence: mapping from name to the divergence
    """
    ## prepare data
    if isinstance(inputs, (tuple, list)):
      pass
    elif not isinstance(inputs, tf.data.Dataset):
      inputs = tf.data.Dataset.from_tensor_slices(inputs).batch(int(batch_size))
    ## check the MCMC shape
    if n_mcmc is None:
      n_mcmc = int(np.prod(self.sample_shape))
    else:
      n_mcmc = int(n_mcmc)

    ## single step
    @tf.function
    def _step(X):
      ret_llk = []
      ret_kl = []
      if isinstance(X, dict):
        Q = self.encode(training=training, **X, **kwargs)
      else:
        Q = self.encode(X, training=training, **kwargs)
      Q = as_tuple(Q)
      z = [i.sample(n_mcmc) for i in Q]
      z_reshape = [tf.reshape(i, (-1, i.shape[-1])) for i in z]
      P = self.decode(z_reshape[0] if len(Q) == 1 else z_reshape,
                      training=training)
      P = as_tuple(P)
      # calculate the KL
      for qz, z in zip(Q, z):
        if hasattr(qz, 'KL_divergence'):
          pz = qz.KL_divergence.prior
          name = qz.name.split('_')[0]
          llk_q = qz.log_prob(z)
          llk_p = pz.log_prob(z)
          ret_kl.append((name, (llk_q, llk_p)))
      # calculate the LLK
      if isinstance(X, dict):
        X = X['inputs']
      X = as_tuple(X)
      for px, x in zip(P, X):
        x = tf.tile(x, [n_mcmc] + [1 for i in range(len(x.shape) - 1)])
        name = px.name.split('_')[0]
        llk_x = tf.reshape(px.log_prob(x), (n_mcmc, -1))
        ret_llk.append((name, llk_x))
      return ret_llk, ret_kl

    ## run the inputs
    llk = defaultdict(list)
    kl = defaultdict(list)
    old_sample_shape = self.sample_shape
    self._sample_shape = ()
    inputs = tqdm(inputs, desc='MarginalLLK', disable=not verbose)
    for n_batch, X in enumerate(inputs):
      ret_llk, ret_kl = _step(X)
      for name, val in ret_llk:
        llk[str(name.numpy(), 'utf-8')].append(val)
      for name, val in ret_kl:
        kl[str(name.numpy(), 'utf-8')].append(val)
    self._sample_shape = old_sample_shape
    inputs.clear()
    inputs.close()
    # concatenate
    C = tf.math.log(tf.cast(n_mcmc, self.dtype))

    def logsumexp_concat(x):
      return tf.reduce_logsumexp(tf.concat(x, axis=-1), axis=0)

    llk = {
      name: logsumexp_concat(logprobs) - C for name, logprobs in llk.items()
    }
    kl = {
      name: (
        logsumexp_concat([i for i, _ in llkqp]) - C,
        logsumexp_concat([i for _, i in llkqp]) - C,
      ) for name, llkqp in kl.items()
    }
    ## Marginal LLK
    if reduce is not None:
      llk = {i: reduce(j) for i, j in llk.items()}
      kl = {i: (reduce(q), reduce(p)) for i, (q, p) in kl.items()}
    return llk, kl

  def elbo_components(
      self,
      inputs: Union[TensorType, List[TensorType]],
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
                                              free_bits=self.free_bits,
                                              sample_shape=None,
                                              keepdims=True)
      else:
        kl[f'kl_{z.name}'] = tf.constant(0., dtype=self.dtype)
    return llk, kl

  ################## For training
  def train_steps(self,
                  inputs: Union[TensorType, List[TensorType]],
                  training: Optional[bool] = None,
                  mask: Optional[Tensor] = None,
                  name: str = '',
                  **kwargs) -> Iterator[VAEStep]:
    """Support multiple steps training for each iteration (similar to GAN)"""
    yield VAEStep(vae=self,
                  parameters=self.trainable_variables,
                  inputs=inputs,
                  training=training,
                  mask=mask,
                  name=name,
                  call_kw=kwargs)

  def __str__(self):
    cls = [
      i for i in type.mro(type(self)) if issubclass(i, VariationalAutoencoder)
    ]
    text = (f"{'->'.join([i.__name__ for i in cls[::-1]])} "
            f"(semi:{type(self).is_semi_supervised()})")
    text += f'\n Tensorboard  : {self.tensorboard_logdir}'
    text += f'\n Analytic     : {self.analytic}'
    text += f'\n Reverse      : {self.reverse}'
    text += f'\n Free-bits    : {self.free_bits}'
    text += f'\n Sample Shape : {self.sample_shape}'
    text += f'\n Fitted       : {int(self.step.numpy())}(iters)'
    text += f'\n MD5 checksum : {self.md5_checksum}'
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


# ===========================================================================
# Semi-supervised VAE
# ===========================================================================
def _is_empty_zero():
  return 0.


class SemiSupervisedVAE(VariationalAutoencoder):
  """ NOTE: this layer only 1 inputs and multi-labels"""

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  def ignore_empty(self,
                   is_empty: Tensor,
                   loss_dict: Dict[str, Tensor]
                   ) -> Dict[str, Tensor]:
    return {k: tf.cond(is_empty,
                       true_fn=_is_empty_zero,
                       false_fn=lambda: v)
            for k, v in loss_dict.items()}

  def merge_objectives(self,
                       llk_uns: Dict[str, Tensor],
                       kl_uns: Dict[str, Tensor],
                       llk_sup: Dict[str, Tensor],
                       kl_sup: Dict[str, Tensor],
                       ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    llk = dict(**{f'uns/{k}': v for k, v in llk_uns.items()},
               **{f'sup/{k}': tf.reduce_mean(v) for k, v in llk_sup.items()})
    kl = dict(**{f'uns/{k}': v for k, v in kl_uns.items()},
              **{f'sup/{k}': tf.reduce_mean(v) for k, v in kl_sup.items()})
    return llk, kl


# ===========================================================================
# Others
# ===========================================================================
VAE = VariationalAutoencoder
