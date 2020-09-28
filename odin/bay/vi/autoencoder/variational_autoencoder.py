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
from odin.bay.layers.dense_distribution import DenseDistribution
from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder.networks import ImageNet
from odin.exp.trainer import Trainer
from odin.networks import (Identity, NetworkConfig, Networks, TensorTypes,
                           TrainStep)
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
from tensorflow.python.training.tracking import base as trackable
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.distributions import Distribution
from typing_extensions import Literal

__all__ = [
    'LayerCreator',
    'VAEStep',
    'VariationalAutoencoder',
]

# ===========================================================================
# Types
# ===========================================================================
LayerCreator = Union[str, Layer, Type[Layer], \
                     NetworkConfig, RandomVariable, \
                     Callable[[Optional[List[int]]], Layer]]


# ===========================================================================
# Helpers
# ===========================================================================
def _get_args(layer):
  spec = inspect.getfullargspec(layer.call)
  return set(spec.args + spec.kwonlyargs)


def _latent_shape(z):
  if isinstance(z, tfd.Distribution):
    shape = tf.concat([z.batch_shape, z.event_shape], axis=0)
  else:
    shape = tf.convert_to_tensor(z.shape)
  return shape


def _reduce_latents(latents, mode):
  if mode is None:
    return latents
  if isinstance(mode, string_types):
    model = str(mode).strip().lower()
    if mode == 'concat':
      return tf.concat(latents, axis=-1)
    if mode == 'mean':
      return tf.reduce_mean(tf.stack(latents), axis=0)
    if mode == 'sum':
      return tf.reduce_sum(tf.stack(latents), axis=0)
    if mode == 'min':
      return tf.reduce_min(tf.stack(latents), axis=0)
    if mode == 'max':
      return tf.reduce_max(tf.stack(latents), axis=0)
    if mode == 'none':
      return latents
  return mode(latents)


def _prepare_decode_latents(fn_reduce, latents, sample_shape):
  # convert all latents to Tensor
  list_latents = True
  if isinstance(latents, tfd.Distribution) or tf.is_tensor(latents):
    list_latents = False
  latents = tf.nest.flatten(latents)
  if isinstance(sample_shape, Number):
    sample_shape = (int(sample_shape),)
  # remove sample_shape
  if sample_shape:
    # if we call tf.convert_to_tensor or tf.reshape directly here the llk
    # could go worse for some unknown reason, but using keras layers is ok!
    ndim = len(sample_shape) + 1
    reshape = keras.layers.Lambda(
        lambda x: tf.reshape(x, tf.concat([(-1,), tf.shape(x)[ndim:]], axis=0)))
    latents = [reshape(z) for z in latents]
  # decoding
  latents = _reduce_latents(latents, fn_reduce) if list_latents else latents[0]
  return latents


def _net2str(net):
  if isinstance(net, keras.Sequential):
    return layer2text(net)
  elif isinstance(net, tfl.DistributionLambda):
    return layer2text(net)
  return str(net)


def _parse_layers(network, input_shape=None, name=None) -> List[Layer]:
  is_decoding = 'decoder' in str(name).lower()
  ## identity
  if (network is None or (isinstance(network, string_types) and
                          network in ('linear', 'identity'))):
    network = Identity(name=name)
  ## Callable or type
  elif (inspect.isfunction(network) or isinstance(network, partial) or
        isinstance(network, type)):
    args = inspect.getfullargspec(network).args
    kw = dict()
    for k in ('shape', 'input_shape', 'latent_shape'):
      if k in args:
        kw[k] = input_shape
        break
    network = network(**kw)
  ## RandomVariable
  elif isinstance(network, RandomVariable):
    network = network.create_posterior(
        input_shape=input_shape, name=name if network.name is None else None)
  ## make sure is a list
  network = list(network) if isinstance(network, (tuple, list)) else [network]
  network = [i for i in network if i is not None]
  ## check different options
  layers = []
  for cfg in network:
    # string type (for dataset name or network alias)
    if isinstance(cfg, string_types):
      cfg = cfg.lower().strip()
      if cfg in ('mnist', 'fashion_mnist'):
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
      else:
        raise NotImplementedError(
            f"No predefined network for dataset with name: {cfg}")
      layers.append(ImageNet(**kw))
    # the NetworkConfig
    elif isinstance(cfg, NetworkConfig):
      layers.append(cfg.create_network(input_shape=input_shape, name=name))
    # Layer
    elif isinstance(cfg, Layer):
      try:
        cfg.input_shape
      except AttributeError:
        if not cfg.built and input_shape is not None:
          cfg(tf.keras.Input(shape=input_shape))
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

  def call(self) -> Tuple[tf.Tensor, Dict[str, Union[tf.Tensor, str]]]:
    pX_Z, qZ_X = self.vae(self.inputs,
                          training=self.training,
                          mask=self.mask,
                          **self.call_kw)
    # store so it could be reused
    self.pX_Z = pX_Z
    self.qZ_X = qZ_X
    llk, div = self.vae.elbo(self.inputs,
                             pX_Z,
                             qZ_X,
                             training=self.training,
                             mask=self.mask,
                             return_components=True)
    # sum all the components log-likelihood and divergence
    llk_sum = tf.constant(0., dtype=self.vae.dtype)
    div_sum = tf.constant(0., dtype=self.vae.dtype)
    for x in llk.values():
      llk_sum += x
    for x in div.values():
      div_sum += x
    elbo = llk_sum - div_sum
    loss = -tf.reduce_mean(elbo)
    # metrics
    metrics = llk
    metrics.update(div)
    return loss, metrics


# ===========================================================================
# Model
# ===========================================================================
class VariationalAutoencoder(Networks):
  r""" Base class for all variational autoencoder

  Parameters
  ----------
  encoder : LayerCreator, optional
      the encoder network, by default NetworkConfig()
  decoder : LayerCreator, optional
      the decoder network, by default NetworkConfig()
  outputs : LayerCreator, optional
      a descriptor for the input/output, by default
      `RandomVariable(64, 'gaus', projection=True, name="Input")`
  latents : LayerCreator, optional
      a descriptor for the latents' distribution, by default
      `RandomVariable(10, 'diag', projection=True, name="Latent")`
  reduce_latent : {'concat', 'mean', 'min', 'max', 'sum', 'none'}, optional
      how multiple latents are handled when feeding to the decoder,
      by default 'concat'
  input_shape : Optional[List[int]], optional
      specific input_shape for the network, if not given, use the given `outputs`,
      by default None
  analytic : bool, optional
      if True, use close-form solution for KL, by default False
  reverse : bool, optional
      If `True`, calculating `KL(q||p)` which optimizes `q`
      (or p_model) by greedily filling in the highest modes of data (or, in
      other word, placing low probability to where data does not occur).
      Otherwise, `KL(p||q)` a.k.a maximum likelihood, or expectation
      propagation place high probability at anywhere data occur
      (i.e. averagely fitting the data), by default True

  Raises
  ------
  ValueError
      wrong value for `reduce_latents`

  Call return
  ----------
    p(X|Z) : a single or a list of `tensorflow_probability.Distribution`
    q(Z|X) : a single or a list of `tensorflow_probability.Distribution`

  Layers
  ------
    encoder : `keras.layers.Layer`.
      Encoding inputs to latents
    decoder : `keras.layers.Layer`.
      Decoding latents to intermediate states
    latent_layers : `keras.layers.Layer`.
      A list of the Dense layer that create the latent variable (random variable)
    output_layers : `keras.layers.Layer`.
      A list of the Dense layer that create the output variable
      (random or deterministic variable)
  """

  def __init__(
      self,
      encoder: LayerCreator = NetworkConfig(),
      decoder: LayerCreator = NetworkConfig(),
      outputs: LayerCreator = RandomVariable(64,
                                             'gaus',
                                             projection=True,
                                             name="Input"),
      latents: LayerCreator = RandomVariable(10,
                                             'diag',
                                             projection=True,
                                             name="Latent"),
      reduce_latent: Literal['concat', 'mean', 'min', 'max', 'sum',
                             'none'] = 'concat',
      input_shape: Optional[List[int]] = None,
      analytic: bool = False,
      reverse: bool = True,
      **kwargs,
  ):
    ### keras want this supports_masking on to enable support masking
    self.supports_masking = True
    super().__init__(**kwargs)
    self._sample_shape = ()
    ### First, infer the right input_shape
    if not isinstance(outputs, (tuple, list)):
      outputs = [outputs]
    if input_shape is None:
      input_shape = [
          o.event_shape if hasattr(o, 'event_shape') else o.output_shape
          for o in outputs
      ]
      if len(outputs) == 1:
        input_shape = input_shape[0]
      warnings.warn(
          f"Input shape not provide, infer using output shape {input_shape}"
          f" , the final input shape is: {tf.nest.flatten(input_shape)}")
    ### prepare support multiple encoders decoders
    all_encoder = _parse_layers(network=encoder,
                                input_shape=input_shape,
                                name="Encoder")
    ### create the latents and input distribution
    if not isinstance(latents, (tuple, list)):
      latents = [latents]
    self._latent_layers = [
        _parse_layers(z,
                      input_shape=e.output_shape[1:] if e is not None else None,
                      name=f"Latents{i if i > 0 else ''}")[0]
        for i, z, e in _iter_lists(latents, all_encoder)
    ]
    self.latent_args = [_get_args(i) for i in self.latent_layers]
    # validate method for latent reduction
    latent_shape = [
        z.event_shape if hasattr(z, 'event_shape') else z.output_shape[1:]
        for z in self.latent_layers
    ]
    reduce_latent = str(reduce_latent).strip().lower()
    if reduce_latent == 'none':
      pass
    elif reduce_latent == 'concat':
      latent_shape = sum(np.array(s) for s in latent_shape).tolist()
    elif reduce_latent in ('mean', 'min', 'max', 'sum'):
      latent_shape = latent_shape[0]
    else:
      raise ValueError(f"No support for reduce_latent='{reduce_latent}'")
    self.reduce_latent = reduce_latent
    ### Create the decoder
    all_decoder = _parse_layers(decoder,
                                input_shape=latent_shape,
                                name="Decoder")
    ### Finally the output distributions
    self._output_layers = [
        _parse_layers(o,
                      input_shape=d.output_shape[1:] if d is not None else None,
                      name=f"Outputs{i}")[0]
        for i, o, d in _iter_lists(outputs, all_decoder)
    ]
    self.output_args = [_get_args(i) for i in self.output_layers]
    ### check type
    self._encoder = all_encoder[0] if len(all_encoder) == 1 else all_encoder
    self._decoder = all_decoder[0] if len(all_decoder) == 1 else all_decoder
    ### build the latent and output layers
    for layer in self.latent_layers + self.output_layers:
      if (hasattr(layer, '_batch_input_shape') and not layer.built and
          layer.projection):
        shape = layer._batch_input_shape
        # call this dummy input to build the layer
        layer(keras.Input(shape=shape[1:], batch_size=shape[0]))
    ### the training step
    self.latent_names = [i.name for i in self.latent_layers]
    # keras already use output_names, cannot override it
    self.variable_names = [i.name for i in self.output_layers]
    ### others
    if self.save_path is not None:
      self.load_weights(self.save_path, raise_notfound=False, verbose=True)
    # encode and decode arguments
    self._encode_func_args = inspect.getfullargspec(self.encode).args[1:]
    self._encoder_args = [
        inspect.getfullargspec(i.call).args[1:]
        for i in tf.nest.flatten(self.encoder)
    ]
    self._decode_func_args = inspect.getfullargspec(self.decode).args[1:]
    self._decoder_args = [
        inspect.getfullargspec(i.call).args[1:]
        for i in tf.nest.flatten(self.decoder)
    ]
    self.analytic = analytic
    self.reverse = reverse

  @property
  def sample_shape(self) -> List[int]:
    return self._sample_shape

  def set_elbo_configs(
      self,
      analytic: Optional[bool] = None,
      reverse: Optional[bool] = None,
      sample_shape: Optional[Union[int, List[int]]] = None
  ) -> VariationalAutoencoder:
    """[summary]

    Parameters
    ----------
    analytic : Optional[bool], optional
        if True use close-form solution for KL, by default None
    reverse : Optional[bool], optional
        If `True`, calculating `KL(q||p)` which optimizes `q`
        (or p_model) by greedily filling in the highest modes of data (or, in
        other word, placing low probability to where data does not occur).
        Otherwise, `KL(p||q)` a.k.a maximum likelihood, or expectation
        propagation place high probability at anywhere data occur
        (i.e. averagely fitting the data)., by default None
    sample_shape : Optional[Union[int, List[int]]], optional
        number of MCMC samples for MCMC estimation of KL-divergence,
        by default None

    Returns
    -------
    VariationalAutoencoder
        the object itself for method chaining
    """
    if analytic is not None:
      self.analytic = bool(analytic)
    if reverse is not None:
      self.reverse = bool(reverse)
    if sample_shape is not None:
      self._sample_shape = sample_shape
    return self

  @property
  def encoder(self) -> Union[Layer, List[Layer]]:
    return self._encoder

  @property
  def decoder(self) -> Union[Layer, List[Layer]]:
    return self._decoder

  @property
  def latent_layers(self) -> Union[Layer, List[Layer]]:
    return self._latent_layers

  @property
  def output_layers(self) -> Union[Layer, List[Layer]]:
    return self._output_layers

  @property
  def init_args(self) -> Dict[str, Any]:
    r""" Return a dictionary of arguments used for initialized this class """
    return self._init_args

  @classproperty
  def default_args(cls) -> Dict[str, Any]:
    r""" Return a dictionary of the default keyword arguments of all subclass start
          from VariationalAutoencoder.
    """
    kw = dict()
    args = []
    for c in type.mro(cls)[::-1]:
      if not issubclass(c, VariationalAutoencoder):
        continue
      spec = inspect.getfullargspec(c.__init__)
      args += spec.args
      if spec.defaults is not None:
        for key, val in zip(spec.args[::-1], spec.defaults[::-1]):
          kw[key] = val
    args = [i for i in set(args) if i not in kw and i != 'self']
    return kw

  @property
  def posteriors(self) -> List[DenseDistribution]:
    return self.output_layers

  @property
  def latents(self) -> List[DenseDistribution]:
    return self.latent_layers

  @property
  def n_latents(self) -> int:
    return len(self.latent_layers)

  @property
  def n_outputs(self) -> int:
    return len(self.output_layers)

  @property
  def input_shape(self) -> Union[List[int], List[List[int]]]:
    shape = [e.input_shape for e in tf.nest.flatten(self.encoder)]
    return shape[0] if len(shape) == 1 else shape

  @property
  def latent_shape(self):
    shape = [d.input_shape for d in tf.nest.flatten(self.decoder)]
    return shape[0] if len(shape) == 1 else shape

  def sample_prior(self,
                   sample_shape: Union[int, List[int]] = (),
                   seed: int = 1) -> Tensor:
    r""" Sampling from prior distribution """
    samples = []
    for latent in self.latent_layers:
      s = bk.atleast_2d(latent.sample(sample_shape=sample_shape, seed=seed))
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def sample_data(self,
                  sample_shape: Union[int, List[int]] = (),
                  seed: int = 1) -> Tensor:
    r""" Sample from p(X) given that the prior of X is known, this could be
    wrong since `RandomVariable` often has a default prior. """
    samples = []
    for output in self.output_layers:
      s = bk.atleast_2d(output.sample(sample_shape=sample_shape, seed=seed))
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def generate(self,
               sample_shape: List[int] = (),
               seed: int = 1,
               training: Optional[bool] = None,
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
             inputs: Union[TensorTypes, ListTensorTypes],
             training: Optional[bool] = None,
             mask: Optional[TensorTypes] = None,
             **kwargs) -> Union[Distribution, List[Distribution]]:
    r""" Encoding inputs to latent codes """
    outputs = []
    for encoder, args in zip(tf.nest.flatten(self.encoder), self._encoder_args):
      copy_kw = dict(kwargs)
      if 'mask' in args:
        copy_kw['mask'] = mask
      if 'training' in args:
        copy_kw['training'] = training
      e = encoder(inputs, **copy_kw)
      outputs.append(e)
    # create the latents distribution
    qZ_X = []
    for args, (_, latent, code) in zip(self.latent_args,
                                       _iter_lists(self.latent_layers,
                                                   outputs)):
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
             latents: Union[TensorTypes, Distribution],
             training: Optional[bool] = None,
             mask: Optional[Tensor] = None,
             **kwargs) -> Union[Distribution, List[Distribution]]:
    r""" Decoding latent codes, this does not guarantee output the
    reconstructed distribution """
    latents = _prepare_decode_latents(self.reduce_latent, latents,
                                      self.sample_shape)
    # apply the decoder and get back the sample shape
    outputs = []
    for decoder, args in zip(tf.nest.flatten(self.decoder), self._decoder_args):
      copy_kw = dict(kwargs)
      if 'mask' in args:
        copy_kw['mask'] = mask
      if 'training' in args:
        copy_kw['training'] = training
      out = decoder(latents, **copy_kw)
      if len(self.sample_shape) > 0:
        list_outputs = False
        if not tf.is_tensor(out):
          list_outputs = True
        out = [
            tf.reshape(
                o, tf.concat([self.sample_shape, (-1,), o.shape[1:]], axis=0))
            for o in tf.nest.flatten(out)
        ]
        if not list_outputs:
          out = out[0]
      outputs.append(out)
    # create the output distribution
    pX_Z = tf.nest.flatten([
        layer(o, training=training)
        for _, layer, o in _iter_lists(self.output_layers, outputs)
    ])
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
    pX_Z = self.decode(
        qZ_X,
        training=training,
        mask=mask,
        **{k: v for k, v in kwargs.items() if k in self._decode_func_args},
    )
    return pX_Z, qZ_X

  @tf.function(autograph=False)
  def marginal_log_prob(self,
                        inputs: TensorTypes,
                        training: Optional[bool] = None,
                        mask: Optional[Tensor] = None,
                        **kwargs) -> Tuple[Tensor, Tensor]:
    r""" Marginal log likelihood `log(p(X))`, an biased estimation.

    With sufficient amount of MCMC samples (-> inf), the value will converges
    to `log(p(X))`

    With large amount of sample, recommending reduce the batch size to very
    small number or use CPU for the calculation `with tf.device("/CPU:0"):`

    Note: this function will need further modification for more complicated
    prior and latent space, only work for:

      - vanilla-VAE or
      - with proper prior injected into qZ_X and pZ_X using
        `qZ_X.KL_divergence.prior = ...` during `encode` or `decode` methods

    Return:
      marginal log-likelihood : a Tensor of shape `[batch_size]`
        marginal log-likelihood of p(X)
      distortion : a Dictionary mapping from distribution name to Tensor
        of shape `[batch_size]`, the negative reconstruction cost.
    """
    sample_shape = [tf.cast(tf.reduce_prod(self.sample_shape), tf.int32)]
    pX_Z, qZ_X = self.call(inputs, training=training, mask=mask, **kwargs)
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

  def _elbo(
      self,
      inputs: Union[TensorTypes, List[TensorTypes]],
      pX_Z: Union[Distribution, List[Distribution]],
      qZ_X: Union[Distribution, List[Distribution]],
      mask: Optional[Tensor] = None,
      training: Optional[bool] = None
  ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    r""" The basic components of all ELBO """
    ### llk
    llk = {}
    for name, x, pX in zip(self.variable_names, inputs, pX_Z):
      llk[f'llk_{name}'] = pX.log_prob(x)
    ### kl
    kl = {}
    for name, qZ in zip(self.latent_names, qZ_X):
      if hasattr(qZ, "KL_divergence"):
        kl[f'kl_{name}'] = qZ.KL_divergence(analytic=self.analytic,
                                            reverse=self.reverse,
                                            sample_shape=self.sample_shape,
                                            keepdims=True)
      else:
        kl[f'kl_{name}'] = tf.constant(0., dtype=self.dtype)
    return llk, kl

  def elbo(self,
           inputs: Union[Tensor, List[Tensor]],
           pX_Z: Union[Distribution, List[Distribution]],
           qZ_X: Union[Distribution, List[Distribution]],
           mask: Optional[Tensor] = None,
           training: Optional[bool] = None,
           return_components: bool = False):
    r""" Calculate the distortion (log-likelihood) and rate (KL-divergence)
    for contruction the Evident Lower Bound (ELBO).

    The final ELBO is:
      `ELBO = E_{z~q(Z|X)}[log(p(X|Z))] - KL_{x~p(X)}[q(Z|X)||p(Z)]`

    Arguments:
      return_components : a Boolean. If True return the log-likelihood and the
        KL-divergence instead of final ELBO.

    Return:
      elbo : a Tensor shape `[sample_shape, batch_size]`.
      (optional) for `return_components=True`
        log-likelihood : dictionary of `Tensor` shape [sample_shape, batch_size].
          The log-likelihood or distortion
        divergence : dictionary of `Tensor` shape [sample_shape, batch_size].
          The reversed KL-divergence or rate
    """
    # organize all inputs to list
    inputs = [
        tf.convert_to_tensor(x, dtype_hint=self.dtype)
        for x in tf.nest.flatten(inputs)
    ]
    pX_Z = tf.nest.flatten(pX_Z)
    qZ_X = tf.nest.flatten(qZ_X)
    # override the default mask
    # if the processed mask from decoder is available
    # but, it still unclear should we use the original mask or the processed
    # mask here
    # if hasattr(pX_Z[0], '_keras_mask') and pX_Z[0]._keras_mask is not None:
    # mask = pX_Z[0]._keras_mask
    llk, div = self._elbo(inputs,
                          pX_Z=pX_Z,
                          qZ_X=qZ_X,
                          mask=mask,
                          training=training)
    if not (isinstance(llk, dict) and isinstance(div, dict)):
      raise RuntimeError(
          "When overriding VariationalAutoencoder _elbo method must return "
          "dictionaries for log-likelihood and KL-divergence.")
    ## only return the components, no need else here but it is clearer
    if return_components:
      return llk, div
    ## calculate the ELBO
    # sum all the components log-likelihood and divergence
    llk_sum = tf.constant(0., dtype=self.dtype)
    div_sum = tf.constant(0., dtype=self.dtype)
    for x in llk.values():  # log-likelihood
      llk_sum += x
    for x in div.values():  # kl-divergence
      tf.debugging.assert_greater(
          x,
          -1e-3,
          message=("Negative KL-divergence values, "
                   "probably because of numerical instability."))
      div_sum += x
    elbo = llk_sum - div_sum
    return elbo

  def importance_weighted(self, elbo: TensorTypes, axis: int = 0):
    r""" VAE objective can lead to overly simplified representations which
    fail to use the network’s entire modeling capacity.

    Importance weighted autoencoder (IWAE) uses a strictly tighter
    log-likelihood lower bound derived from importance weighting.

    Using more samples can only improve the tightness of the bound, and
    as our estimator is based on the log of the average importance weights,
    it does not suffer from high variance.

    Reference:
      Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
        Autoencoders. In ICLR, 2015. https://arxiv.org/abs/1509.00519
    """
    dtype = elbo.dtype
    iw_dim = tf.cast(elbo.shape[axis], dtype=dtype)
    elbo = tf.reduce_logsumexp(elbo, axis=axis) - tf.math.log(iw_dim)
    return elbo

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
    for i, latent in enumerate(self.latent_layers):
      text += f"\n Latent#{i}:\n  "
      text += "\n  ".join(_net2str(latent).split('\n'))
    ## Output
    for i, output in enumerate(self.output_layers):
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
