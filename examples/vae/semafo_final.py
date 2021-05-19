import dataclasses
import itertools
import os
import shutil
from functools import partial
from typing import Optional, Union, Sequence, List

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import (Dense, Flatten, Concatenate, Conv2D,
                                            Conv2DTranspose, GlobalAvgPool2D,
                                            BatchNormalization, Activation,
                                            Reshape, Layer)
from tensorflow_probability.python.distributions import (Normal, Independent,
                                                         Blockwise,
                                                         JointDistributionSequential,
                                                         VonMises, Gamma,
                                                         Categorical,
                                                         Bernoulli,
                                                         RelaxedBernoulli,
                                                         Distribution,
                                                         RelaxedOneHotCategorical,
                                                         OneHotCategorical,
                                                         Uniform,
                                                         ContinuousBernoulli,
                                                         Beta)
from tensorflow_probability.python.math import clip_by_value_preserve_gradient
from tensorflow_probability.python.internal.reparameterization import \
  FULLY_REPARAMETERIZED
from tensorflow_probability.python.layers import DistributionLambda

from odin import visual as vs
from odin.backend import one_hot
from odin.bay import (DistributionDense, MVNDiagLatents, kl_divergence,
                      DisentanglementGym, VariationalAutoencoder, get_vae,
                      BiConvLatents,
                      HierarchicalVAE)
from odin.bay.vi import traverse_dims
from odin.fuel import get_dataset
from odin.networks import get_networks, SequentialNetwork, TrainStep
from odin.utils import as_tuple
from utils import get_args, train, run_multi, set_cfg, Arguments, \
  get_model_path, get_results_path, Callback, prepare_images
from tensorflow.python.training.tracking import base as trackable

# ===========================================================================
# Const and helper
# ===========================================================================
config: Optional[Arguments] = None
CURRENT_MODEL: Optional[VariationalAutoencoder] = None


def dense_networks(input_dim: Union[None, int],
                   name: str,
                   units: int = 512,
                   batchnorm: bool = False) -> Sequential:
  if batchnorm:
    layers = [
      Dense(units, use_bias=False, name=f'{name}_1'),
      BatchNormalization(),
      Activation('relu'),
      Dense(units, use_bias=False, name=f'{name}_2'),
      BatchNormalization(),
      Activation('relu'),
    ]
  else:
    layers = [
      Dense(units, activation='relu', name=f'{name}_1'),
      Dense(units, activation='relu', name=f'{name}_2'),
    ]
  if input_dim is not None:
    layers = [Input([input_dim])] + layers
  return Sequential(layers, name=name)


def to_elbo(semafo, llk, kl):
  elbo = semafo.elbo(llk, kl)
  return tf.reduce_mean(-elbo), \
         {k: tf.reduce_mean(v) for k, v in dict(**llk, **kl).items()}


# ===========================================================================
# Distributions
# ===========================================================================
def _create_normal(params):
  loc, scale = tf.split(params, 2, axis=-1)
  scale = tf.nn.softplus(scale) + tf.cast(tf.exp(-7.), params.dtype)
  d = Normal(loc, scale)
  d = Independent(d, reinterpreted_batch_ndims=loc.shape.rank - 1)
  return d


def _create_dsprites(p: tf.Tensor) -> Blockwise:
  return Blockwise(
    JointDistributionSequential([
      VonMises(loc=0.,
               concentration=tf.math.softplus(p[..., 0]),
               name='orientation'),
      Gamma(concentration=tf.math.softplus(p[..., 1]),
            rate=tf.math.softplus(p[..., 2]),
            name='scale'),
      OneHotCategorical(logits=p[..., 3:6], dtype=tf.float32, name='shape'),
      Bernoulli(logits=p[..., 6], dtype=tf.float32, name='x_position'),
      Bernoulli(logits=p[..., 7], dtype=tf.float32, name='y_position'),
    ]), name='Shapes2D')


def _create_shapes3D(p: tf.Tensor) -> Blockwise:
  return Blockwise(
    JointDistributionSequential([
      VonMises(loc=0.,
               concentration=tf.math.softplus(p[..., 0]),
               name='orientation'),
      Gamma(concentration=tf.math.softplus(p[..., 1]),
            rate=tf.math.softplus(p[..., 2]),
            name='scale'),
      Categorical(logits=p[..., 3:7], dtype=tf.float32, name='shape'),
      Bernoulli(logits=p[..., 7], dtype=tf.float32, name='floor_hue'),
      Bernoulli(logits=p[..., 8], dtype=tf.float32, name='wall_hue'),
      Bernoulli(logits=p[..., 9], dtype=tf.float32, name='object_hue'),
    ]), name='Shapes3D')


class dSpritesDistribution(Distribution):
  input_dim: int = 40 + 6 + 3 + 32 + 32
  output_dim: int = 40 + 6 + 3 + 32 + 32

  def __init__(self,
               params: tf.Tensor,
               temperature=0.5,
               validate_args=False,
               allow_nan_stats=True,
               reparams=True,
               name='dSpritesDistribution'):
    parameters = dict(locals())
    tf.assert_equal(tf.shape(params)[-1], self.input_dim)
    self._params = params
    self.reparams = bool(reparams)
    # create distributions
    if reparams:
      Dist = partial(RelaxedOneHotCategorical, temperature=temperature)
    else:
      Dist = partial(OneHotCategorical, dtype=params.dtype)
    self.orientation = Dist(logits=params[..., 0:40], name='Orientation')
    self.scale = Dist(logits=params[..., 40:46], name='Scale')
    self.shape_type = Dist(logits=params[..., 46:49], name='Shape')
    self.x_pos = Dist(logits=params[..., 49:(49 + 32)], name='X_position')
    self.y_pos = Dist(logits=params[..., (49 + 32):(49 + 64)],
                      name='Y_position')
    super(dSpritesDistribution, self).__init__(
      dtype=params.dtype,
      validate_args=validate_args,
      allow_nan_stats=allow_nan_stats,
      reparameterization_type=FULLY_REPARAMETERIZED,
      parameters=parameters,
      name=name)

  def _event_shape_tensor(self):
    return tf.constant([self.output_dim], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([self.output_dim])

  def _batch_shape_tensor(self):
    return tf.shape(self._params)[:-1]

  def _batch_shape(self):
    return self._params.shape[:-1]

  def _sample_n(self, n, seed=None, **kwargs):
    return tf.concat([
      self.orientation.sample(n, seed),
      self.scale.sample(n, seed),
      self.shape_type.sample(n, seed),
      self.x_pos.sample(n, seed),
      self.y_pos.sample(n, seed),
    ], axis=-1)

  def _log_prob(self, y, **kwargs):
    if self.reparams:
      y = clip_by_value_preserve_gradient(y, 1e-6, 1. - 1e-6)
    llk_orientation = self.orientation.log_prob(y[..., 0:40])
    llk_scale = self.scale.log_prob(y[..., 40:46])
    llk_shape = self.shape_type.log_prob(y[..., 46:49])
    llk_x = self.x_pos.log_prob(y[..., 49:(49 + 32)])
    llk_y = self.y_pos.log_prob(y[..., (49 + 32):(49 + 64)])
    llk = llk_orientation + llk_scale + llk_shape + llk_x + llk_y
    return llk

  def _mean(self, **kwargs):
    return tf.concat([
      self.orientation.probs_parameter(),
      self.scale.probs_parameter(),
      self.shape_type.probs_parameter(),
      self.x_pos.probs_parameter(),
      self.y_pos.probs_parameter(),
    ], axis=-1)


class Shapes3DDistribution(Distribution):
  input_dim: int = 15 + 8 + 4 + 10 + 10 + 10
  output_dim: int = 15 + 8 + 4 + 10 + 10 + 10

  def __init__(self,
               params: tf.Tensor,
               temperature=0.5,
               validate_args=False,
               allow_nan_stats=True,
               reparams: bool = True,
               name='Shapes3DDistribution'):
    parameters = dict(locals())
    tf.assert_equal(tf.shape(params)[-1], self.input_dim)
    self.reparams = bool(reparams)
    self._params = params
    # create distributions
    if reparams:
      Dist = partial(RelaxedOneHotCategorical, temperature=temperature)
    else:
      Dist = partial(OneHotCategorical, dtype=params.dtype)
    self.orientation = Dist(logits=params[..., 0:15], name='Orientation')
    self.scale = Dist(logits=params[..., 15:23], name='Scale')
    self.shape_type = Dist(logits=params[..., 23:27], name='Shape')
    self.floor = Dist(logits=params[..., 27:37], name='FloorHue')
    self.wall = Dist(logits=params[..., 37:47], name='WallHue')
    self.obj = Dist(logits=params[..., 47:57], name='ObjectHue')
    super(Shapes3DDistribution, self).__init__(
      dtype=params.dtype,
      validate_args=validate_args,
      allow_nan_stats=allow_nan_stats,
      reparameterization_type=FULLY_REPARAMETERIZED,
      parameters=parameters,
      name=name)

  def _event_shape_tensor(self):
    return tf.constant([self.output_dim], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([self.output_dim])

  def _batch_shape_tensor(self):
    return tf.shape(self._params)[:-1]

  def _batch_shape(self):
    return self._params.shape[:-1]

  def _sample_n(self, n, seed=None, **kwargs):
    return tf.concat([
      self.orientation.sample(n, seed),
      self.scale.sample(n, seed),
      self.shape_type.sample(n, seed),
      self.floor.sample(n, seed),
      self.wall.sample(n, seed),
      self.obj.sample(n, seed),
    ], axis=-1)

  def _log_prob(self, y, **kwargs):
    if self.reparams:
      y = clip_by_value_preserve_gradient(y, 1e-6, 1. - 1e-6)
    llk_orientation = self.orientation.log_prob(y[..., 0:15])
    llk_scale = self.scale.log_prob(y[..., 15:23])
    llk_shape = self.shape_type.log_prob(y[..., 23:27])
    llk_floor = self.x_pos.log_prob(y[..., 27:37])
    llk_wall = self.y_pos.log_prob(y[..., 37:47])
    llk_obj = self.y_pos.log_prob(y[..., 47:57])
    llk = (llk_orientation + llk_scale + llk_shape +
           llk_floor + llk_wall + llk_obj)
    return llk

  def _mean(self, **kwargs):
    return tf.concat([
      self.orientation.probs_parameter(),
      self.scale.probs_parameter(),
      self.shape_type.probs_parameter(),
      self.floor.probs_parameter(),
      self.wall.probs_parameter(),
      self.obj.probs_parameter(),
    ], axis=-1)


class DigitsDistribution(RelaxedOneHotCategorical):
  input_dim: int = 10
  output_dim: int = 10

  def __init__(self,
               logits=None,
               temperature=0.5,
               validate_args=False,
               allow_nan_stats=True,
               name='DigitsDistribution'):
    super(DigitsDistribution, self).__init__(temperature=temperature,
                                             logits=logits,
                                             probs=None,
                                             validate_args=validate_args,
                                             allow_nan_stats=allow_nan_stats,
                                             name=name)

  def _log_prob(self, y, **kwargs):
    y = clip_by_value_preserve_gradient(y, 1e-6, 1. - 1e-6)
    return super(DigitsDistribution, self)._log_prob(y, **kwargs)

  def _mean(self, **kwargs):
    return super(DigitsDistribution, self).probs_parameter()


def reparameterize(dsname: str) -> SequentialNetwork:
  if dsname in ('mnist', 'fashionmnist', 'cifar10'):
    return SequentialNetwork([
      # networks(None, 'EncoderY', batchnorm=True),
      DistributionDense(event_shape=[DigitsDistribution.output_dim],
                        projection=True,
                        posterior=DigitsDistribution,
                        units=DigitsDistribution.input_dim,
                        name='Digits_qy')],
      name='qy_z')
  elif dsname == 'dsprites':
    return SequentialNetwork([
      DistributionDense(event_shape=[dSpritesDistribution.output_dim],
                        projection=True,
                        posterior=partial(dSpritesDistribution, reparams=True),
                        units=dSpritesDistribution.input_dim,
                        name='Shapes2D_qy')],
      name='qy_z')
  elif dsname == 'shapes3d':
    return SequentialNetwork([
      DistributionDense(event_shape=[Shapes3DDistribution.output_dim],
                        projection=True,
                        posterior=partial(Shapes3DDistribution, reparams=True),
                        units=Shapes3DDistribution.input_dim,
                        name='Shapes3D_qy')],
      name='qy_z')
  raise NotImplementedError(f'No support for {dsname}.')


# ===========================================================================
# SemafoVAE
# ===========================================================================
@dataclasses.dataclass
class LatentConfig:
  encoder: int = 1
  decoder: int = 1
  filters: int = 32  # also the number of latent units
  kernel: int = 4  # recommend kernel is multiply of stride
  strides: int = 2
  post: Optional[Conv2D] = None
  prior: Optional[Conv2D] = None
  deter: Optional[Conv2D] = None
  out: Optional[Conv2DTranspose] = None
  pz: Optional[DistributionLambda] = None
  qz: Optional[DistributionLambda] = None
  beta: float = 1.0
  C: Optional[float] = None

  def initialize_latents(self, vae: VariationalAutoencoder):
    idx = self.decoder
    decoder = vae.decoder.layers[idx]
    cnn_cfg = dict(filters=2 * self.filters,  # for mean and stddev
                   strides=self.strides,
                   kernel_size=self.kernel,
                   padding='same')
    if self.post is None:
      self.post = Conv2D(**cnn_cfg, name=f'PosteriorZ{idx}')
    if self.prior is None:
      self.prior = Conv2D(**cnn_cfg, name=f'PriorZ{idx}')
    if self.deter is None:
      cfg = dict(cnn_cfg)
      cfg['filters'] = self.filters
      self.deter = Conv2D(**cfg, name=f'DeterministicZ{idx}')
    if self.out is None:
      self.out = Conv2DTranspose(
        filters=decoder.filters,
        kernel_size=self.kernel,
        strides=self.strides,
        padding='same',
        kernel_initializer=decoder.kernel_initializer,
        bias_initializer=decoder.bias_initializer,
        activation=decoder.activation,
        name=f'OutputZ{idx}')
    if self.pz is None:
      self.pz = DistributionLambda(_create_normal, name=f'pz{idx}')
    if self.qz is None:
      self.qz = DistributionLambda(_create_normal, name=f'qz{idx}')
    # remember to assign all these layer to VAE otherwise no parameters update
    layers = []
    for k in self.__annotations__.keys():
      layer = getattr(self, k)
      if isinstance(layer, Layer):
        layers.append(layer)
    setattr(vae, f'hierarchical_latents{idx}', layers)
    return self


DefaultHierarchy = dict(
  ###
  shapes3d=dict(encoder=3, decoder=2, filters=32, kernel=8, strides=4),
  ###
  dsprites=dict(encoder=3, decoder=2, filters=32, kernel=8, strides=4),
  ###
  cifar10=[
    dict(encoder=3, decoder=3, filters=32, kernel=8, strides=4),
    # dict(encoder=1, decoder=5, filters=16, kernel=8, strides=4)
  ],
  ###
  fashionmnist=dict(encoder=3, decoder=3, filters=16, kernel=14, strides=7),
  ###
  mnist=dict(encoder=3, decoder=3, filters=16, kernel=14, strides=7),
)

DefaultGamma = dict(
  shapes3d=5.,
  dsprites=5.,
  cifar10=5.,
  fashionmnist=5.,
  mnist=5.,
)

DefaultGammaPy = dict(
  shapes3d=20.,
  dsprites=20.,
  cifar10=20.,
  fashionmnist=20.,
  mnist=20.,
)

DefaultSemi = dict(
  shapes3d=(0.1, 0.1),
  dsprites=(0.1, 0.1),
  cifar10=(1000, 0.1),
  fashionmnist=(0.01, 0.1),
  mnist=(0.004, 0.1),
)

DefaultPretrain = dict(
  shapes3d=800,
  dsprites=800,
  cifar10=0,
  fashionmnist=0,
  mnist=0,
)


# ===========================================================================
# Base SemafoVAE
# ===========================================================================
class SemafoVAE(VariationalAutoencoder):

  def __init__(self,
               encoder: SequentialNetwork,
               decoder: SequentialNetwork,
               labels: DistributionDense,
               pretrain_steps: Optional[int] = None,
               coef_H_qy: float = 1.,
               gamma_py: float = None,
               gamma_uns: Optional[float] = None,
               gamma_sup: float = 1.,
               beta_uns: float = 1.,
               beta_sup: float = 1.,
               n_iw_y: int = 1,
               **kwargs):
    if not isinstance(self, HierarchicalVAE):
      decoder = SequentialNetwork(
        [layer.layer if isinstance(layer, BiConvLatents) else layer for layer in
         decoder.layers],
        name=decoder.name)
    super().__init__(encoder=encoder, decoder=decoder, **kwargs)
    self.encoder.track_outputs = True
    if not self.decoder.built:
      self.decoder(Input(self.latents.event_shape))
    self.decoder.track_outputs = True
    # === 0. other parameters
    if gamma_py is None:
      gamma_py = DefaultGammaPy[config.ds]
    if gamma_uns is None:
      gamma_uns = DefaultGamma[config.ds]
    if pretrain_steps is None:
      pretrain_steps = DefaultPretrain[config.ds]
    self.pretrain_steps = int(pretrain_steps)
    self.n_iw_y = int(n_iw_y)
    self.coef_H_qy = float(coef_H_qy)
    self.gamma_uns = float(gamma_uns)
    self.gamma_sup = float(gamma_sup)
    self.gamma_py = float(gamma_py)
    self.beta_uns = float(beta_uns)
    self.beta_sup = float(beta_sup)
    # === 1. fixed utility layers
    self.flatten = Flatten()
    self.concat = Concatenate(-1)
    self.global_avg = GlobalAvgPool2D()
    # === 2. reparameterized q(y|z)
    ydim = int(np.prod(labels.event_shape))
    zdim = int(np.prod(self.latents.event_shape))
    self.labels = reparameterize(config.ds.lower())  # q(y|z0,z1,x)
    # === 3. second VAE for y and u
    self.encoder2 = dense_networks(ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = dense_networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = labels
    self.observation2(self.decoder2.output)
    # === 4. p(z|u)
    self.latents_prior = SequentialNetwork([
      Dense(256, activation='relu'),
      MVNDiagLatents(zdim)
    ], name=f'pZ0_given_U')
    self.latents_prior.build([None] + self.latents2.event_shape)
    # === 5. others
    self.semafo_params: List[tf.Variable] = []
    self.vae_params: List[tf.Variable] = []

  def build(self, input_shape=None):
    super(SemafoVAE, self).build(input_shape)
    # vanilla VAE params
    self.semafo_params = (self.labels.trainable_variables +
                          self.encoder2.trainable_variables +
                          self.latents2.trainable_variables +
                          self.decoder2.trainable_variables +
                          self.observation2.trainable_variables +
                          self.latents_prior.trainable_variables)
    semafo_ids = set([id(p) for p in self.semafo_params])
    self.vae_params = [v for v in self.trainable_variables
                       if id(v) not in semafo_ids]

  def encode(self, inputs, training=None, **kwargs):
    if isinstance(inputs, (tuple, list)):
      x, y = inputs
    else:
      x = inputs
      y = None
    latents = super().encode(x, training=training, **kwargs)
    latents.last_outputs = [i for _, i in self.encoder.last_outputs]
    latents.labels = y
    return latents

  def decode(self, latents, training=None, **kwargs):
    z = tf.convert_to_tensor(latents)
    px_z = super().decode(z, training=training, **kwargs)
    qy_z = self.labels(z, training=training)
    return px_z, qy_z

  def encode_aux(self, y, training=None):
    h_e = self.encoder2(y, training=training)
    qu = self.latents2(h_e, training=training)
    return qu

  def decode_aux(self, qu, training=None):
    h_d = self.decoder2(qu, training=training)
    py = self.observation2(h_d, training=training)
    return py

  def call_aux(self, y, training=None):
    qu = self.encode_aux(y, training=training)
    py = self.decode_aux(qu, training=training)
    return py, qu

  def sample_prior(self,
                   n=1,
                   seed=1,
                   return_distribution=False,
                   return_labels=False,
                   **kwargs):
    dsname = config.ds.lower()
    # === 0. mnist
    if dsname in ('mnist', 'fashionmnist', 'cifar10'):
      y = np.diag([1.] * 10)
      y = np.tile(y, [int(np.ceil(n / 10)), 1])[:n]
    # === 1. shapes
    elif dsname in ('dsprites', 'shapes3d'):
      is_2d = dsname == 'dsprites'
      if is_2d:
        # orientation, scale, shape, x, y
        combinations = [40, 6, 3, 32, 32]
      else:
        # orientation, scale, shape, floor, wall, object
        combinations = [15, 8, 4, 10, 10, 10]
      per_class = int(np.ceil(n / combinations[2]))
      cls_to_labels = dict()
      rand = np.random.RandomState(seed)
      for cls in range(combinations[2]):
        y2 = [0] * combinations[2]
        y2[cls] = 1
        y = [
          np.array([y2]) if idx == 2 else
          one_hot(
            np.linspace(0, n_cls, per_class, endpoint=False, dtype=np.int32),
            num_classes=n_cls)
          for idx, n_cls in enumerate(combinations)]
        y = np.array([np.concatenate(i, -1) for i in itertools.product(*y)])
        # rand.shuffle(y) # no need shuffle yet
        cls_to_labels[cls] = y
      y = np.concatenate([x[:per_class] for x in cls_to_labels.values()], 0)
      y = y[:n].astype(np.float32)
    # === 2. no idea
    else:
      raise NotImplementedError
    # === 3. sample the prior p(z|u)
    qu_y = self.encode_aux(y, training=False)
    u = qu_y.sample(seed=seed)
    pz_u = self.latents_prior(u, training=False)
    if return_distribution:
      ret = pz_u
    else:
      ret = pz_u.mean()
    ret.labels = y
    if return_labels:
      return ret, y
    return ret

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  def get_latents(self, inputs=None, training=None, mask=None,
                  return_prior=False, **kwargs):
    if inputs is None:
      if self.last_outputs is None:
        raise RuntimeError(f'Cannot get_latents of {self.name} '
                           'both inputs and last_outputs are None.')
      (px, qy), qz = self.last_outputs
    else:
      (px, qy), qz = self(inputs, training=training, mask=mask, **kwargs)
    posterior = qz
    prior = self.latents_prior(self.encode_aux(qy, training=training),
                               training=training)
    if return_prior:
      return posterior, prior
    return posterior

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, training=training, mask=mask,
                                      **kwargs)
    (px, qy), qz = self.last_outputs
    # for hierarchical VAE
    if hasattr(px, 'kl_pairs'):
      for cfg, q, p in px.kl_pairs:
        kl[f'kl_z{cfg.decoder}'] = cfg.beta * kl_divergence(
          q, p, analytic=self.analytic, free_bits=self.free_bits)
    return llk, kl

  def elbo_sup(self, x_s, y_s, training):
    (px_z, qy_z), qz_x = self([x_s, y_s], training=training)
    py_u, qu_y = self.call_aux(y_s, training=training)
    pz_u = self.latents_prior(qu_y, training=training)

    llk = dict(
      llk_px_sup=self.gamma_sup * px_z.log_prob(x_s),
      llk_py_sup=self.gamma_sup * self.gamma_py *
                 py_u.log_prob(y_s),
      llk_qy_sup=self.gamma_sup * self.gamma_py *
                 qy_z.log_prob(tf.clip_by_value(y_s, 1e-6, 1. - 1e-6))
    )

    z = tf.convert_to_tensor(qz_x)
    kl = dict(
      kl_z_sup=self.beta_sup * (qz_x.log_prob(z) - pz_u.log_prob(z)),
      kl_u_sup=self.beta_sup * qu_y.KL_divergence(analytic=self.analytic)
    )

    # for hierarchical VAE
    if hasattr(px_z, 'kl_pairs'):
      for i, (cfg, q, p) in enumerate(px_z.kl_pairs):
        kl_qp = kl_divergence(q, p, analytic=self.analytic)
        if cfg.C is not None:
          kl_qp = tf.math.abs(kl_qp - cfg.C * np.prod(q.event_shape))
        kl[f'kl_z{i + 1}_sup'] = cfg.beta * kl_qp

    return to_elbo(self, llk, kl)

  def elbo_uns(self, x_u, training):
    (px_z, qy_z), qz_x = self(x_u, training=training)
    y_u = tf.convert_to_tensor(qy_z, dtype=self.dtype)
    py_u, qu_y = self.call_aux(y_u, training=training)
    pz_u = self.latents_prior(qu_y, training=training)

    llk = dict(
      llk_px_uns=self.gamma_uns * px_z.log_prob(x_u),
      llk_py_uns=self.gamma_uns * self.gamma_py *
                 py_u.log_prob(y_u)
    )

    z = tf.convert_to_tensor(qz_x)
    kl = dict(
      kl_z_uns=self.beta_uns * (qz_x.log_prob(z) - pz_u.log_prob(z)),
      kl_u_uns=self.beta_uns * qu_y.KL_divergence(analytic=self.analytic),
      llk_qy_uns=self.coef_H_qy *
                 qy_z.log_prob(tf.clip_by_value(y_u, 1e-6, 1. - 1e-6))
    )
    # for hierarchical VAE
    if hasattr(px_z, 'kl_pairs'):
      for i, (cfg, q, p) in enumerate(px_z.kl_pairs):
        kl_qp = kl_divergence(q, p, analytic=self.analytic)
        if cfg.C is not None:
          kl_qp = tf.math.abs(kl_qp - cfg.C * np.prod(q.event_shape))
        kl[f'kl_z{i + 1}_uns'] = cfg.beta * kl_qp

    return to_elbo(self, llk, kl)

  def train_steps(self, inputs, training=None, **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Supervised
    def elbo_sup():
      return self.elbo_sup(x_s, y_s, training)

    #   def elbo_uns():
    #     llk, kl = self.elbo_components(x_u, training=training)
    #     return to_elbo(self, llk, kl)
    #
    #   yield TrainStep(parameters=self.vae_params, func=elbo_uns)
    yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)

  def train_steps1(self, inputs, training=None, **kwargs):
    x_u, x_s, y_s = inputs

    def elbo():
      loss_s, metr_s = self.elbo_sup(x_s, y_s, training)
      loss_u, metr_u = self.elbo_uns(x_u, training)
      return loss_s + loss_u, dict(**metr_s, **metr_u)

    yield TrainStep(parameters=self.trainable_variables, func=elbo)

    # # === 1. Supervised
    # def elbo_sup():
    #   return self.elbo_sup(x_s, y_s, training)
    #
    # yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)
    #
    # # === 2. Unsupervised
    # def elbo_uns():
    #   return self.elbo_uns(x_u, training)
    #
    # yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)

  def fit(self, *args, **kwargs):
    on_batch_end = kwargs.pop('on_batch_end', [])
    if self.pretrain_steps == 0:
      self.set_training_stage(1)
    else:
      def switch_training():
        # priming q(y|z0,z1,x) so it generate some reasonable data
        if self.step.numpy() >= self.pretrain_steps and self.training_stage == 0:
          self.set_training_stage(1)

      on_batch_end.append(switch_training)
    return super(SemafoVAE, self).fit(on_batch_end=on_batch_end, *args,
                                      **kwargs)


class G10(SemafoVAE):
  def __init__(self, **kwargs):
    super(G10, self).__init__(gamma_uns=10, **kwargs)


class G20(SemafoVAE):
  def __init__(self, **kwargs):
    super(G20, self).__init__(gamma_uns=20, **kwargs)


class G40(SemafoVAE):
  def __init__(self, **kwargs):
    super(G40, self).__init__(gamma_uns=40, **kwargs)


class G80(SemafoVAE):
  def __init__(self, **kwargs):
    super(G80, self).__init__(gamma_uns=80, **kwargs)


# ===========================================================================
# Hierarchical
# ===========================================================================
class SemafoHVAE(SemafoVAE):

  def __init__(self,
               hierarchy: Optional[Sequence[LatentConfig]] = None,
               coef_deter: float = 1.0,
               coef_residual: float = 1.0,
               coef_pz_u: float = 1.0,
               **kwargs):
    super().__init__(**kwargs)
    self.coef_deter = float(coef_deter)
    self.coef_residual = float(coef_residual)
    self.coef_pz_u = float(coef_pz_u)
    self._is_sampling = False
    if hierarchy is None:
      hierarchy = [LatentConfig(**cfg) for cfg in
                   as_tuple(DefaultHierarchy[config.ds])]
    self.hierarchy = {cfg.decoder: cfg.initialize_latents(self)
                      for cfg in as_tuple(hierarchy)}
    # === 2. create connection p(z1|z0,u)
    pz_given_u = dict()
    if coef_pz_u > 0.:
      for idx, z_layers in self.hierarchy.items():
        shape = self.decoder.layers[idx].output_shape
        shape = z_layers.prior.compute_output_shape(shape)
        units = int(np.prod(shape[1:]))
        # somehow it requires string key here
        pz_given_u[str(idx)] = Sequential([
          Dense(256, activation='relu'),
          Dense(units),
          Reshape(shape[1:])
        ], name=f'pZ{idx}_given_U')
    self.pz_given_u = pz_given_u

  @classmethod
  def is_hierarchical(cls) -> bool:
    return True

  def set_sampling(self, is_sampling: bool):
    self._is_sampling = bool(is_sampling)
    return self

  def decode(self, latents, training=None, **kwargs):
    if isinstance(latents, (tuple, list)):
      priors = list(latents[1:])
      priors = priors + [None] * (len(self.hierarchy) - len(priors))
      latents = latents[0]
    else:
      priors = [None] * len(self.hierarchy)
    # === 0. prepare
    z_org = tf.convert_to_tensor(latents)
    z = z_org
    kl_pairs = []
    # === 1. q(y|z)
    qy_z = self.labels(z_org, training=training)
    labels = latents.labels if hasattr(latents, 'labels') else None
    if labels is None:
      labels = tf.convert_to_tensor(qy_z)
    # === 2. hierarchical latents
    for idx, layer in enumerate(self.decoder.layers):
      z = layer(z, training=training)
      if idx in self.hierarchy:
        z_layers: LatentConfig = self.hierarchy[idx]
        z_samples = priors.pop(0)
        if z_samples is None:
          # prior
          h_prior = z_layers.prior(z, training=training)
          if self.coef_pz_u > 0.:
            u = tf.convert_to_tensor(self.encode_aux(labels, training=training))
            u = self.pz_given_u[str(idx)](u, training=training)
            h_prior = h_prior + self.coef_pz_u * u
          pz = z_layers.pz(h_prior, training=training)
          # posterior (inference mode)
          if not self._is_sampling:
            if not hasattr(latents, 'last_outputs'):
              raise RuntimeError(
                'No encoder states found for hierarchical model')
            h_e = latents.last_outputs[z_layers.encoder]
            h_post = z_layers.post(self.concat([z, h_e]), training=training)
            qz = z_layers.qz(h_post, training=training)
          # sampling mode (just use the prior)
          else:
            qz = pz
          # track the post and prior for KL
          kl_pairs.append((z_layers, qz, pz))
          z_samples = tf.convert_to_tensor(qz)
        # deterministic connection
        if self.coef_deter > 0:
          h_deter = z_layers.deter(z, training=training)
          z_samples = self.concat([z_samples, self.coef_deter * h_deter])
        # residual connection
        residual = 0.
        if self.coef_residual > 0:
          residual = self.coef_residual * z
        # final output
        z = z_layers.out(z_samples, training=training) + residual
    # === 3. p(x|z0,z1)
    px_z = self.observation(z, training=training)
    px_z.kl_pairs = kl_pairs
    return px_z, qy_z

  def get_latents(self, inputs=None, training=None, mask=None,
                  return_prior=False, **kwargs):
    is_sampling = self._is_sampling
    self.set_sampling(False)
    if inputs is None:
      if self.last_outputs is None:
        raise RuntimeError(f'Cannot get_latents of {self.name} '
                           'both inputs and last_outputs are None.')
      (px, qy), qz = self.last_outputs
    else:
      (px, qy), qz = self(inputs, training=training, mask=mask, **kwargs)
    self.set_sampling(is_sampling)
    posterior = [qz]
    prior = [self.latents_prior(self.encode_aux(qy, training=training),
                                training=training)]
    for cfg, q, p in px.kl_pairs:
      posterior.append(q)
      prior.append(p)
    if return_prior:
      return posterior, prior
    return posterior

  def sample_traverse(self, inputs, n_traverse_points=11, n_best_latents=5,
                      min_val=-2.0, max_val=2.0, mode='linear',
                      smallest_stddev=True, training=False, mask=None):
    from odin.bay.vi import traverse_dims
    latents = self.encode(inputs, training=training, mask=mask)
    stddev = np.sum(latents.stddev(), axis=0)
    # smaller stddev is better
    if smallest_stddev:
      top_latents = np.argsort(stddev)[:int(n_best_latents)]
    else:
      top_latents = np.argsort(stddev)[::-1][:int(n_best_latents)]
    # keep the encoder states for the posteriors
    last_outputs = [i for i in latents.last_outputs]
    latents = traverse_dims(latents,
                            feature_indices=top_latents,
                            min_val=min_val, max_val=max_val,
                            n_traverse_points=n_traverse_points,
                            mode=mode)
    latents = tf.convert_to_tensor(latents)
    if not self._is_sampling:
      n_tiles = n_traverse_points * len(top_latents)
      last_outputs = [
        tf.tile(i, [n_tiles] + [1 for _ in range(i.shape.rank - 1)])
        for i in last_outputs]
      latents.last_outputs = last_outputs
    return self.decode(latents, training=training, mask=mask), top_latents


# coef_pz_u = 0
class NoSkip(SemafoHVAE):

  def __init__(self, **kwargs):
    super(NoSkip, self).__init__(coef_deter=1.0,
                                 coef_residual=1.0,
                                 coef_pz_u=0.0,
                                 **kwargs)


class NoResidual(SemafoHVAE):

  def __init__(self, **kwargs):
    super(NoResidual, self).__init__(coef_deter=1.0,
                                     coef_residual=0.0,
                                     coef_pz_u=1.0,
                                     **kwargs)


class NoAll(SemafoHVAE):

  def __init__(self, **kwargs):
    super(NoAll, self).__init__(coef_deter=0.0,
                                coef_residual=0.0,
                                coef_pz_u=0.0,
                                **kwargs)


class NoResDeter(SemafoHVAE):

  def __init__(self, **kwargs):
    super(NoResDeter, self).__init__(coef_deter=0.0,
                                     coef_residual=0.0,
                                     coef_pz_u=1.0,
                                     **kwargs)


class HG40(SemafoHVAE):
  def __init__(self, **kwargs):
    super(HG40, self).__init__(gamma_uns=40, **kwargs)


# ===========================================================================
# Training
# ===========================================================================
def _mean(dists, idx):
  m = dists[idx].mean()
  return tf.reshape(m, (m.shape[0], -1))


def main(args: Arguments):
  # === 0. prepare dataset
  ds = get_dataset(args.ds)
  train_data_map = None
  valid_data_map = None
  # === 1. create model
  model = None
  for k, v in globals().items():
    if k.lower().strip() == args.vae.lower().strip() and callable(v):
      model = v
      break
  if model is None:
    model = get_vae(args.vae)
  is_semi = model.is_semi_supervised()
  # === 2. build the model
  networks = get_networks(args.ds,
                          zdim=args.zdim,
                          is_semi_supervised=is_semi)
  # for dsprites
  if args.ds.lower() == 'dsprites':
    networks['labels'] = DistributionDense(
      event_shape=(dSpritesDistribution.output_dim,),
      posterior=partial(dSpritesDistribution, reparams=False),
      units=dSpritesDistribution.input_dim,
      name='Shapes2D_py')
  # for shapes3d
  elif args.ds.lower() == 'shapes3d':
    networks['labels'] = DistributionDense(
      event_shape=(Shapes3DDistribution.output_dim,),
      posterior=partial(Shapes3DDistribution, reparams=False),
      units=Shapes3DDistribution.input_dim,
      name='Shapes3D_py')
  # create and build the model
  model: Union[VariationalAutoencoder, SemafoVAE] = model(**networks)
  model.build((None,) + ds.shape)
  global CURRENT_MODEL
  CURRENT_MODEL = model
  # === 3. training
  if not args.eval:
    train(model, ds, args,
          label_percent=args.py if is_semi else 0.0,
          on_batch_end=[],
          on_valid_end=[Callback.save_best_llk],
          oversample_ratio=args.ratio,
          train_data_map=train_data_map,
          valid_data_map=valid_data_map)
  # === 4. evaluation
  else:
    path = get_results_path(args)
    if args.override:
      for p in [path, path + '_valid']:
        if os.path.exists(p):
          print('Override results at path:', p)
          shutil.rmtree(p)
    if not os.path.exists(path):
      os.makedirs(path)
    ### load model weights
    model.load_weights(get_model_path(args), raise_notfound=True, verbose=True)

    ### [Posterior Traverse] special case for SemafoVAE
    model.sample_prior(3)
    # if isinstance(model, SemafoVAE):
    #   for x, y in ds.create_dataset('test', label_percent=1.0,
    #                                 batch_size=36).map(data_map).take(1):
    #     pass
    #   (px, qy), qz = model([x, y], training=False)
    #   for i, j in zip(qy.mean(), y):
    #     print(i.numpy())
    #     print(j.numpy())

    ### [Prior Traverse] special case for SemafoVAE
    if isinstance(model, SemafoVAE):
      kw = dict(return_distribution=True, return_labels=True)
      if args.ds in ('mnist', 'fashionmnist', 'cifar10'):
        pz, labels = model.sample_prior(n=10, **kw)
        n_classes = 10
      elif args.ds == 'dsprites':
        pz, labels = model.sample_prior(n=3, **kw)
        n_classes = 3
      elif args.ds == 'shapes3d':
        pz, labels = model.sample_prior(n=4, **kw)
        n_classes = 4
      else:
        raise NotImplementedError(args.ds)
      mean = pz.mean()
      stddev = pz.stddev()
      n_points = 41
      n_latents = min(15, mean.shape[1])
      max_std = 4.
      if model.is_hierarchical():
        model.set_sampling(True)
      for i in range(n_classes):
        m = mean[i:i + 1]
        s = stddev[i].numpy()
        ids = np.argsort(s)[::-1][:n_latents]  # higher is better
        m = traverse_dims(m,
                          feature_indices=ids,
                          min_val=-max_std * s,
                          max_val=max_std * s,
                          n_traverse_points=n_points)
        img, _ = model.decode(m)
        # plotting
        plt.figure(figsize=(1.5 * 11, 1.5 * n_latents))
        vs.plot_images(prepare_images(img.mean(), normalize=True),
                       grids=(n_latents, n_points),
                       ax=plt.gca())
      vs.plot_save(os.path.join(path, 'prior_traverse.pdf'), verbose=True)
      # special case for hierarchical model
      if model.is_hierarchical():
        model.set_sampling(True)
        for i in range(n_classes):
          z0 = mean[i:i + 1]
          img, _ = model.decode(z0)
          # traverse the second latents prior
          pz1 = img.kl_pairs[0][2]
          shape = list(pz1.event_shape)
          s1 = np.ravel(pz1.stddev().numpy()[0])
          ids = np.argsort(s1)[::-1][:n_latents]  # higher is better
          z1 = np.reshape(pz1.mean(), (1, -1))
          z1 = traverse_dims(z1,
                             feature_indices=ids,
                             min_val=-max_std * s1,
                             max_val=max_std * s1,
                             n_traverse_points=n_points)
          z0 = tf.tile(z0, [z1.shape[0], 1])
          z1 = np.reshape(z1, [-1] + shape)
          img, _ = model.decode([z0, z1])
          # plotting
          plt.figure(figsize=(1.5 * 11, 1.5 * n_latents))
          vs.plot_images(prepare_images(img.mean(), normalize=True),
                         grids=(n_latents, n_points),
                         ax=plt.gca())
        vs.plot_save(os.path.join(path, 'prior1_traverse.pdf'), verbose=True)

    # important, otherwise, all evaluation is wrong
    if isinstance(model, SemafoHVAE):
      model.set_sampling(False)

    ### run the disentanglement gym
    gym = DisentanglementGym(model=model,
                             dataset=args.ds,
                             batch_size=32,
                             dpi=args.dpi,
                             seed=args.seed)

    with gym.run_model(n_samples=1800 if args.debug else -1,
                       partition='test'):
      gym.write_report(os.path.join(path, 'scores.txt'), verbose=True)
      # latents t-SNE
      gym.plot_latents_tsne()
      if gym.n_latent_vars > 1:
        for i in range(gym.n_latent_vars):
          gym.plot_latents_tsne(convert_fn=partial(_mean, idx=i),
                                title=f'_z{i}')
      # prior sampling
      if model.is_hierarchical():
        model.set_sampling(True)
      gym.plot_latents_sampling()
      # traverse
      if model.is_hierarchical():
        model.set_sampling(True)
        gym.plot_latents_traverse(title='_prior')
        model.set_sampling(False)
        gym.plot_latents_traverse(title='_post')
      else:
        gym.plot_latents_traverse()
      # inference
      for i in range(gym.n_latent_vars):
        gym.plot_latents_stats(latent_idx=i)
        gym.plot_latents_factors(
          convert_fn=partial(_mean, idx=i), title=f'_z{i}')
        gym.plot_correlation(
          convert_fn=partial(_mean, idx=i), title=f'_z{i}')
      gym.plot_reconstruction()
    gym.save_figures(path, verbose=True)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  set_cfg(root_path='/home/trung/exp/semafo', n_valid_batches=30)
  parsed_args = get_args(dict(py=0.004, ratio=0.1, it=200000, bs=64))
  config = parsed_args
  parsed_args.py, parsed_args.ratio = DefaultSemi[parsed_args.ds]
  run_multi(main, args=parsed_args)
