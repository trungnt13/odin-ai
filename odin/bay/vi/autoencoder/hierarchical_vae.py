import inspect
from contextlib import contextmanager
from types import MethodType
from typing import Any, Callable, Dict, List, Tuple
from typing import Union, Optional, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Layer, Wrapper, Dense
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow_probability.python.distributions import Distribution, Normal, \
  MultivariateNormalDiag, Independent
from tensorflow_probability.python.layers.distribution_layer import (
  DistributionLambda)
from typing_extensions import Literal

from odin.backend import TensorType
from odin.bay.helpers import kl_divergence
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import AnnealingVAE, BetaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.networks import NetConf
from odin.utils import as_tuple

__all__ = [
  'MergeNormal',
  'HierarchicalLatents',
  'HierarchicalVAE',
  'LadderVAE',
  'UnetVAE'
]


# ===========================================================================
# Helpers
# ===========================================================================
def _copy_layer(layer: Layer,
                *,
                units: Optional[int] = None,
                n_params: int = 2,
                downsample_shape: Optional[Sequence[int]] = None,
                proj_1x1: bool = False) -> Layer:
  cfg = layer.get_config()
  cls = layer.__class__
  key = 'filters' if 'filters' in cfg else 'units'
  if units is None:
    cfg[key] *= n_params
  else:
    cfg[key] = units * n_params
  cfg['name'] += '_clone'
  cfg['activation'] = 'linear'
  cfg['use_bias'] = True
  ## convolution
  if issubclass(cls, Conv):
    # switch to use convolution
    if 'Transpose' in str(cls):
      if issubclass(cls, keras.layers.Conv1DTranspose):
        cls = keras.layers.Conv1D
      elif issubclass(cls, keras.layers.Conv2DTranspose):
        cls = keras.layers.Conv2D
      elif issubclass(cls, keras.layers.Conv3DTranspose):
        cls = keras.layers.Conv3D
      cfg.pop('output_padding')
    # no strides
    cfg['strides'] = 1
    # 1x1 projection
    if proj_1x1:
      cfg['padding'] = 'SAME'
      cfg['kernel_size'] = 1
    # downsample
    elif downsample_shape is not None:
      cfg['padding'] = 'VALID'
      cfg['kernel_size'] = downsample_shape[1:-1]
  layer = cls(**cfg)
  return layer


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
  """
  """

  def __init__(self,
               layer: Union[Dense, Conv],
               encoder: Optional[Layer] = None,
               latent_units: Optional[int] = None,
               downsample: bool = False,
               merge_normal: bool = False,
               distribution: Literal['mvndiag', 'normal'] = 'normal',
               forward_mode: Literal['sample', 'mean'] = 'sample',
               beta: float = 1.,
               disable: bool = False,
               **kwargs):
    super().__init__(layer=layer, **kwargs)
    self.latent_units = latent_units
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')
    self._total_units = None
    self.event_ndim = self.layer.input_spec.min_ndim - 1
    self.downsample = bool(downsample)
    self._disable = bool(disable)
    if encoder is not None:
      encoder._old_call = encoder.call
      encoder.call = MethodType(_call, encoder)
    self.encoder = encoder
    # === 1. Layers
    spec = inspect.getfullargspec(layer.call)
    self._args = spec.args + spec.kwonlyargs
    # === 2. distribution
    assert forward_mode in ('sample', 'mean')
    self.forward_mode = forward_mode
    assert distribution in ('mvndiag', 'normal'), \
      f"Only support 'mvndiag' or 'normal' posterior, given '{distribution}'"
    # assert top_down in ('sample', 'mean', 'deterministic')
    self.distribution = distribution
    if merge_normal:
      assert distribution == 'normal', \
        'merge_normal mode only support Normal distribution'
      self._merge_normal = MergeNormal()
    else:
      self._merge_normal = None
    # === 2. others
    self._dist_prior = None
    self._dist_posterior = None
    self._proj_posterior = None
    self._proj_prior = None
    # for final output
    self._pre_proj = None
    self._proj_out = None
    # store the last distributions
    self._posterior = None
    self._prior = None
    self._is_sampling = False

  def build(self, input_shape=None):
    super().build(input_shape)
    if self._disable:
      return
    # === 0. prepare
    output_shape = self.layer.compute_output_shape(input_shape)
    if self.distribution == 'mvndiag':
      distribution = MultivariateNormalDiag
    elif self.distribution == 'normal':
      distribution = Normal
    # latent_units equal input feature dimension
    if self.latent_units is None:
      self.latent_units = output_shape[-1]

    def create_dist(params):
      loc, scale = tf.split(params, 2, axis=-1)
      scale = tf.nn.softplus(scale) + tf.cast(tf.exp(-7.), self.dtype)
      d = distribution(loc, scale, name=self.name)
      if isinstance(d, Normal):
        d = Independent(d, reinterpreted_batch_ndims=self.event_ndim)
      return d

    # === 1. create projection layer
    if self.encoder is not None:
      # posterior projection
      self._proj_posterior = _copy_layer(
        self.layer,
        units=self.latent_units,
        n_params=2,
        downsample_shape=output_shape if self.downsample else None,
        proj_1x1=False)
      shape = output_shape[:-1] + (
        self.encoder.output_shape[-1] + output_shape[-1],)
      self._proj_posterior.build(shape)
    # prior projection
    self._proj_prior = _copy_layer(
      self.layer,
      units=self.latent_units,
      n_params=2,
      downsample_shape=output_shape if self.downsample else None,
      proj_1x1=False)
    self._proj_prior.build(output_shape)
    shape = self._proj_prior.compute_output_shape(output_shape)
    self._total_units = int(np.prod(shape[1:]) // 2)
    # pre-output latents projection
    if self.downsample:
      self._pre_proj = keras.layers.Conv2DTranspose(
        filters=self.latent_units,
        kernel_size=output_shape[1:-1],
        padding='valid',
        activation='linear',
        use_bias=True)
      self._pre_proj.build(shape[:-1] + (self.latent_units,))
    else:
      self._pre_proj = lambda x: x  # identity layer
    # final output affine (1x1 projection is faster)
    self._proj_out = _copy_layer(self.layer,
                                 units=None,
                                 n_params=1,
                                 downsample_shape=None,
                                 proj_1x1=True)
    self._proj_out.build(
      output_shape[:-1] + (self.latent_units + output_shape[-1],))

    # === 2. create distribution
    self._dist_posterior = DistributionLambda(make_distribution_fn=create_dist,
                                              name=f'{self.name}_posterior')
    self._dist_prior = DistributionLambda(make_distribution_fn=create_dist,
                                          name=f'{self.name}_prior')
    self._dist_posterior.build(shape)
    self._dist_prior.build(shape)

  def sampling(self):
    """Sampling mode, forward prior samples"""
    self._is_sampling = True
    return self

  def inference(self):
    """Inference mode, forward posterior samples (require encoder states)"""
    self._is_sampling = False
    return self

  def enable(self):
    """Enable stochastic inference and generation for this variable"""
    self._disable = False
    return self

  @property
  def is_sampling(self) -> bool:
    return self._is_sampling

  @property
  def is_stochastic(self) -> bool:
    """Return True if the layer is stochastically initialized"""
    return self._dist_prior is not None

  @property
  def posterior(self) -> Optional[Distribution]:
    return self._posterior

  @property
  def prior(self) -> Optional[Distribution]:
    return self._prior

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return ("<HVars "
            f"'{self.name}' enable:{self.is_stochastic} "
            f"sampl:{self.is_sampling} "
            f"units:{self._total_units} mode:{self.forward_mode} "
            f"dist:{self.distribution} down:{self.downsample} "
            f"merge:{True if self._merge_normal else False}>")

  def kl_divergence(self,
                    analytic: bool = False,
                    reverse: bool = False,
                    free_bits: float = 0.25,
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
    return self.beta * kl_divergence(q=qz, p=pz,
                                     analytic=analytic,
                                     reverse=reverse,
                                     free_bits=free_bits)

  def compute_output_shape(self, input_shape) -> Sequence[Union[None, int]]:
    return self.layer.compute_output_shape(input_shape)

  def call(self, inputs, training=None, mask=None, **kwargs):
    # === 1. call the layer
    if 'training' in self._args:
      kwargs['training'] = training
    if 'mask' in self._args:
      kwargs['mask'] = mask
    hidden_d = self.layer.call(inputs, **kwargs)
    if self._disable:
      return hidden_d
    # === 2. project and create the distribution
    prior = self._dist_prior(self._proj_prior(hidden_d, **kwargs))
    self._prior = prior
    # === 3. inference
    dist = prior
    if (not self.is_sampling and
        self.encoder is not None and
        hasattr(self.encoder, '_last_outputs') and
        self.encoder._last_outputs is not None):
      hidden_e = self.encoder._last_outputs
      # just stop inference if there is no Encoder state
      tf.debugging.assert_equal(tf.shape(hidden_e), tf.shape(hidden_d),
                                f'Shape of inference {hidden_e.shape} and '
                                f'generative {hidden_d.shape} mismatch. '
                                f'Change to sampling mode if possible')
      # (Kingma 2016) use add, we concat here
      h = tf.concat([hidden_e, hidden_d], axis=-1)
      posterior = self._dist_posterior(self._proj_posterior(h, **kwargs))
      # (Maaloe 2016) merging two Normal distribution
      if self._merge_normal is not None:
        posterior = self._merge_normal([posterior, prior])
      self._posterior = posterior
      dist = posterior
      # erase the previous state of encoder
      self.encoder._last_outputs = None
    # === 4. sampling
    if self.forward_mode == 'mean':
      z = dist.mean()
    else:  # sample
      z = tf.convert_to_tensor(dist)
    # === 5. output projection
    z = self._pre_proj(z, **kwargs)
    outputs = tf.concat([z, hidden_d], axis=-1)
    outputs = self._proj_out(outputs)
    outputs = self.layer.activation(outputs)
    return outputs


# ===========================================================================
# Hierarchical VAE
# ===========================================================================
class HierarchicalVAE(AnnealingVAE):

  def __init__(self, free_bits=0.25, name='HierarchicalVAE', **kwargs):
    super().__init__(free_bits=free_bits, name=name, **kwargs)
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


class StackedVAE(AnnealingVAE):
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
  Sønderby, C.K., Raiko, T., Maaløe, L., Sønderby, S.K., Winther, O., 2016.
    Ladder variational autoencoders, Advances in Neural Information Processing Systems.
    Curran Associates, Inc., pp. 3738–3746.
  Tomczak, J.M., Welling, M., 2018. VAE with a VampPrior. arXiv:1705.07120 [cs, stat].
  """

  def __init__(
      self,
      ladder_units: Sequence[int] = (256,),
      ladder_latents: Sequence[int] = (64,),
      ladder_layers: int = 2,
      batchnorm: bool = True,
      batchnorm_kw: Optional[Dict[str, Any]] = None,
      dropout: float = 0.0,
      activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
      latents: Union[Layer, RVconf] = RVconf(32,
                                             'mvndiag',
                                             projection=True,
                                             name="latents"),
      tie_latents: bool = False,
      all_standard_prior: bool = False,
      stochastic_inference: bool = True,
      only_mean_up: bool = False,
      preserve_latents_order: bool = False,
      name: str = 'StackedVAE',
      **kwargs,
  ):
    super().__init__(latents=latents, name=name, **kwargs)
    self.encoder.track_outputs = True
    self.decoder.track_outputs = True
    if batchnorm_kw is None:
      batchnorm_kw = {'momentum': 0.9}
    assert len(ladder_units) == len(ladder_latents)
    self.all_standard_prior = bool(all_standard_prior)
    self.stochastic_inference = bool(stochastic_inference)
    self.only_mean_up = bool(only_mean_up)
    self.ladder_encoder = [
      NetConf([units] * ladder_layers,
              activation=activation,
              batchnorm=batchnorm,
              batchnorm_kw=batchnorm_kw,
              dropout=dropout,
              name=f'LadderEncoder{i}').create_network()
      for i, units in enumerate(ladder_units)
    ]
    self.ladder_decoder = [
      NetConf([units] * ladder_layers,
              activation=activation,
              batchnorm=batchnorm,
              batchnorm_kw=batchnorm_kw,
              dropout=dropout,
              name=f'LadderDecoder{i}').create_network()
      for i, units in enumerate(ladder_units[::-1])
    ]
    self.ladder_qz = [
      _parse_layers(RVconf(units, 'normal', projection=True, name=f'qZ{i}'))
      for i, units in enumerate(as_tuple(ladder_latents))
    ]
    if tie_latents:
      self.ladder_pz = self.ladder_qz
    else:
      self.ladder_pz = [
        _parse_layers(RVconf(units, 'normal', projection=True, name=f'pZ{i}'))
        for i, units in enumerate(as_tuple(ladder_latents))
      ]

  @classmethod
  def is_hierarchical(cls) -> bool:
    return True

  def encode(self, inputs, training=None, mask=None, only_encoding=False,
             **kwargs):
    h = self.encoder(inputs, training=training, mask=mask)
    last_outputs = h._last_outputs
    latents = []
    for e, z in zip(self.ladder_encoder, self.ladder_qz):
      # stochastic bottom-up inference
      qz = z(h, training=training, mask=mask)
      latents.append(qz)
      if self.stochastic_inference:
        h = tf.convert_to_tensor(qz)
      elif self.only_mean_up:
        # deterministic bottom-up inference
        h = qz.mean()
      h = e(h, training=training, mask=mask)
    if only_encoding:
      return h
    qz = self.latents(h,
                      training=training,
                      mask=mask,
                      sample_shape=self.sample_shape)
    latents.append(qz)
    return tuple(latents[::-1])

  def decode(self, latents, training=None, mask=None, only_decoding=False,
             **kwargs):
    h = tf.convert_to_tensor(latents[0])
    outputs = []
    for d, z in zip(self.ladder_decoder, self.ladder_pz[::-1]):
      h = d(h, training=training, mask=mask)
      pz = z(h, training=training, mask=mask)
      outputs.append(pz)
      h = tf.convert_to_tensor(h)
    h = self.decoder(h, training=training, mask=mask)
    if only_decoding:
      return h
    h = self.observation(h, training=training, mask=mask)
    outputs.append(h)
    return tuple([outputs[-1]] + outputs[:-1])

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    P, Q = self.last_outputs
    ### KL
    Qz, Pz = Q[1:], P[1:]
    for q, p, z in zip(Qz, Pz, self.ladder_qz):
      if self.all_standard_prior:
        for name, dist in [('q', i) for i in as_tuple(q)
                           ] + [('p', i) for i in as_tuple(p)]:
          kl[f'kl{name}_{z.name}'] = self.beta * dist.KL_divergence(
            analytic=self.analytic, reverse=self.reverse)
      else:
        kl[f'kl_{z.name}'] = self.beta * kl_divergence(
          q, p, analytic=self.analytic, reverse=self.reverse)
    return llk, kl

  def __str__(self):
    text = super().__str__()

    text += f"\n LadderEncoder:\n  "
    for i, layers in enumerate(self.ladder_encoder):
      text += "\n  ".join(str(layers).split('\n'))
      text += "\n  "

    text = text[:-3] + f"\n LadderDecoder:\n  "
    for i, layers in enumerate(self.ladder_decoder):
      text += "\n  ".join(str(layers).split('\n'))
      text += "\n  "

    text = text[:-3] + f"\n LadderLatents:\n  "
    for i, layers in enumerate(self.ladder_qz):
      text += "\n  ".join(str(layers).split('\n'))
      text += "\n  "
    return text[:-3]


# ===========================================================================
# Ladder VAE
# ===========================================================================
class VLadderVAE():
  """
  References
  ----------
  Zhao, S., Song, J., Ermon, S., 2017. Learning Hierarchical Features from
      Generative Models. arXiv:1702.08396 [cs, stat].
  """
  def __init__(self, **kwargs):
    super(VLadderVAE, self).__init__(**kwargs)


class LadderVAE(StackedVAE):
  """ The ladder variational autoencoder

  Similar to hierarchical VAE with 2 improvements:

  - Deterministic bottom-up inference
  - Merge q(Z|X) two Gaussian based-on weighed variance


  Parameters
  ----------
  ladder_encoder : List[Union[Layer, NetConf]], optional
      the mapping layers between latents in the encoding part
  ladder_decoder : List[Union[Layer, NetConf]], optional
      the mapping layers between latents in the decoding part
  ladder_units : List[Union[Layer, RVmeta]], optional
      number of hidden units for stochastic latents

  References
  ----------
  Sønderby, C.K., Raiko, T., Maaløe, L., Sønderby, S.K., Winther, O., 2016.
    Ladder variational autoencoders, Advances in Neural Information Processing Systems.
    Curran Associates, Inc., pp. 3738–3746.
  https://github.com/casperkaae/LVAE
  """

  def __init__(self,
               merge_gaussians: bool = True,
               name: str = 'LadderVAE',
               **kwargs):
    super().__init__(stochastic_inference=False, name=name, **kwargs)
    self.ladder_merge = MergeNormal()
    self.merge_gaussians = bool(merge_gaussians)

  def decode(self, latents, training=None, mask=None, only_decoding=False):
    h = tf.convert_to_tensor(latents[0])
    outputs = []
    for d, z, qz_e in zip(self.ladder_decoder, self.ladder_pz[::-1],
                          latents[1:]):
      h = d(h, training=training, mask=mask)
      pz = z(h, training=training, mask=mask)
      if self.merge_gaussians:
        qz = self.ladder_merge([pz, qz_e])
      else:
        qz = qz_e
      # ladder_share_params=True
      outputs.append((qz, pz))
      h = tf.convert_to_tensor(qz)
    # final decoder
    h = self.decoder(h, training=training, mask=mask)
    if only_decoding:
      return h
    h = self.observation(h, training=training, mask=mask)
    outputs.append(h)
    return tuple([outputs[-1]] + outputs[:-1])

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super(StackedVAE, self).elbo_components(inputs=inputs,
                                                      mask=mask,
                                                      training=training)
    P, Q = self.last_outputs
    for (qz, pz), lz in zip(P[1:], self.ladder_qz[::-1]):
      if self.all_standard_prior:
        kl[f'kl_{lz.name}'] = self.beta * qz.KL_divergence(
          analytic=self.analytic, reverse=self.reverse)
      else:
        # z = tf.convert_to_tensor(qz) # sampling
        # kl[f'kl_{lz.name}'] = self.beta * (qz.log_prob(z) - pz.log_prob(z))
        kl[f'kl_{lz.name}'] = self.beta * kl_divergence(
          qz, pz, analytic=self.analytic, reverse=self.reverse)
    return llk, kl


# ===========================================================================
# HVAE
# ===========================================================================
class HVAE(AnnealingVAE):
  """ Hierarchical VAE

  References
  ----------
  Tomczak, J.M., Welling, M., 2018. VAE with a VampPrior.
      arXiv:1705.07120 [cs, stat].
  """

  def __init__(
      self,
      latents: RVconf = RVconf(32, 'mvndiag', projection=True, name="latents1"),
      ladder_latents: List[int] = [16],
      connection: NetConf = NetConf(300, activation='relu'),
      name: str = 'HierarchicalVAE',
      **kwargs,
  ):
    super().__init__(latents=latents, name=name, **kwargs)
    ## create the hierarchical latents
    self.ladder_q = [
      RVconf(units, 'mvndiag', projection=True,
             name=f'ladder_q{i}').create_posterior()
      for i, units in enumerate(ladder_latents)
    ]
    self.ladder_p = [
      RVconf(units, 'mvndiag', projection=True,
             name=f'ladder_p{i}').create_posterior()
      for i, units in enumerate(ladder_latents)
    ]
    self.n_ladder = len(ladder_latents)
    ## create the connections
    self.qz_to_qz = [
      connection.create_network(name=f'qz{i}_to_qz{i + 1}')
      for i in range(self.n_ladder)
    ]
    self.qz_to_pz = [
      connection.create_network(name=f'qz{i}_to_pz{i + 1}')
      for i in range(self.n_ladder)
    ]
    self.qz_to_px = [
      connection.create_network(name=f'qz{i}_to_px')
      for i in range(self.n_ladder + 1)
    ]
    ## other layers
    self.ladder_encoders = [
      keras.models.clone_model(self.encoder) for _ in range(self.n_ladder)
    ]
    self.concat = keras.layers.Concatenate(axis=-1)
    units = sum(
      np.prod(i.event_shape)
      for i in as_tuple(self.ladder_q) + as_tuple(self.latents))
    self.pre_decoder = keras.layers.Dense(units,
                                          activation='linear',
                                          name='pre_decoder')

  def encode(self,
             inputs,
             training=None,
             mask=None,
             only_encoding=False,
             **kwargs):
    Q = super().encode(inputs,
                       training=training,
                       mask=mask,
                       only_encoding=only_encoding,
                       **kwargs)
    Q = list(as_tuple(Q))
    for i, (f_qz, f_e) in enumerate(zip(self.ladder_q, self.ladder_encoders)):
      h_e = f_e(inputs, training=training, mask=mask)
      if only_encoding:
        Q.append(h_e)
      else:
        h_z = self.qz_to_qz[i](tf.convert_to_tensor(Q[-1]),
                               training=training,
                               mask=mask)
        qz_x = f_qz(self.concat([h_e, h_z]),
                    training=training,
                    mask=mask,
                    sample_shape=self.sample_shape)
        Q.append(qz_x)
    return tuple(Q)

  def decode(self,
             latents,
             training=None,
             mask=None,
             only_decoding=False,
             **kwargs):
    h = []
    for qz, fz in zip(latents, self.qz_to_px):
      h.append(fz(tf.convert_to_tensor(qz), training=training, mask=mask))
    h = self.concat(h)
    h = self.pre_decoder(h, training=training)
    px_z = super().decode(h,
                          training=training,
                          mask=mask,
                          only_decoding=only_decoding,
                          **kwargs)
    if only_decoding:
      return px_z
    ## p(z_i|z_{i-1})
    P = []
    for i, f_pz in enumerate(self.ladder_p):
      h_z = self.qz_to_pz[i](tf.convert_to_tensor(latents[i]),
                             training=training,
                             mask=mask)
      pz_zi = f_pz(h_z, training=training, mask=mask)
      P.append(pz_zi)
    return as_tuple(px_z) + tuple(P)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    P, Q = self.last_outputs
    for i, (pz, qz) in enumerate(zip(P[-self.n_ladder:], Q[-self.n_ladder:])):
      d = self.beta * kl_divergence(q=qz,
                                    p=pz,
                                    analytic=self.analytic,
                                    free_bits=self.free_bits,
                                    reverse=self.reverse)
      kl[f'kl_ladder{i}'] = d
    return llk, kl


# ===========================================================================
# Unet VAE
# ===========================================================================
def _get_full_args(fn):
  spec = inspect.getfullargspec(fn)
  return spec.args + spec.kwonlyargs


def _prepare_encoder_decoder(encoder, decoder):
  if isinstance(encoder, keras.Sequential):
    encoder = encoder.layers
  if isinstance(decoder, keras.Sequential):
    decoder = decoder.layers
  assert isinstance(encoder, (tuple, list)), \
    f'encoder must be list of Layer, given {encoder}'
  assert isinstance(decoder, (tuple, list)), \
    f'decoder must be list of Layer, given {decoder}'
  return encoder, decoder


class UnetVAE(BetaVAE):
  """ Unet-VAE """

  def __init__(
      self,
      encoder: List[keras.layers.Layer],
      decoder: List[keras.layers.Layer],
      layers_map: List[Tuple[str, str]] = [
        ('encoder2', 'decoder2'),
        ('encoder1', 'decoder3'),
        ('encoder0', 'decoder4'),
      ],
      dropout: float = 0.,
      noise: float = 0.,
      beta: float = 10.,
      free_bits: float = 2.,
      name: str = 'UnetVAE',
      **kwargs,
  ):
    encoder, decoder = _prepare_encoder_decoder(encoder, decoder)
    super().__init__(beta=beta,
                     free_bits=free_bits,
                     encoder=encoder,
                     decoder=decoder,
                     name=name,
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
    return True

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
