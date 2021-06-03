import dataclasses
import itertools
import os
import shutil
import warnings
from functools import partial
from typing import Optional, Union, Sequence, List

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import (Dense, Flatten, Concatenate, Conv2D,
                                            Conv2DTranspose, GlobalAvgPool2D,
                                            BatchNormalization, Activation,
                                            Reshape, Layer)
from tensorflow_probability.python.distributions import (Normal, Independent,
                                                         Blockwise,
                                                         Distribution,
                                                         RelaxedOneHotCategorical,
                                                         OneHotCategorical)
from tensorflow_probability.python.internal.reparameterization import \
  FULLY_REPARAMETERIZED, NOT_REPARAMETERIZED
from tensorflow_probability.python.layers import DistributionLambda, \
  IndependentNormal
from tensorflow_probability.python.math import clip_by_value_preserve_gradient

from odin import visual as vs
from odin.backend import one_hot
from odin.backend.keras_helpers import layer2text
from odin.bay import (DistributionDense, MVNDiagLatents, kl_divergence,
                      DisentanglementGym, VariationalAutoencoder, get_vae,
                      BiConvLatents,
                      HierarchicalVAE, BetaGammaVAE, RVconf, M2VAE)
from odin.bay.vi import traverse_dims
from odin.fuel import get_dataset, ImageDataset
from odin.ml import DimReduce
from odin.networks import get_networks, SequentialNetwork, TrainStep
from odin.utils import as_tuple
from utils import get_args, train, run_multi, set_cfg, Arguments, \
  get_model_path, get_results_path, Callback, prepare_images, get_scores_path

# ===========================================================================
# Const and helper
# ===========================================================================
config: Optional[Arguments] = None
CURRENT_MODEL: Optional[VariationalAutoencoder] = None
DS_TO_LABELS = dict(
  mnist=[f'#{i}' for i in range(10)],
  fashionmnist=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
  cifar10=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck'],
  dsprites=['square', 'ellipse', 'heart'],
  dsprites0=['square', 'ellipse', 'heart'],
  shapes3d=['cube', 'cylinder', 'sphere', 'round'],
  shapes3d0=['cube', 'cylinder', 'sphere', 'round'],
)


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


class FactorDistribution(Distribution):

  def __init__(self,
               params: tf.Tensor,
               temperature=0.5,
               dsname: str = 'mnist',
               validate_args=False,
               allow_nan_stats=True,
               reparams=True,
               name='FactorDistribution'):
    parameters = dict(locals())
    self.event_dim = FactorDistribution.event_dim(dsname)
    tf.assert_equal(tf.shape(params)[-1], self.event_dim)
    self._params = params
    self.reparams = bool(reparams)
    self.dsname = dsname.lower()
    # create distributions
    if reparams:
      Dist = partial(RelaxedOneHotCategorical, temperature=temperature)
    else:
      Dist = partial(OneHotCategorical, dtype=params.dtype)
    if self.dsname == 'dsprites':
      self._distributions = [
        Dist(logits=params[..., 0:40], name='Orientation'),
        Dist(logits=params[..., 40:46], name='Scale'),
        Dist(logits=params[..., 46:49], name='Shape'),
        Dist(logits=params[..., 49:(49 + 32)], name='X_position'),
        Dist(logits=params[..., (49 + 32):(49 + 64)], name='Y_position')]
      self._classes = [40, 6, 3, 32, 32]
    elif self.dsname == 'shapes3d':
      self._distributions = [
        Dist(logits=params[..., 0:15], name='Orientation'),
        Dist(logits=params[..., 15:23], name='Scale'),
        Dist(logits=params[..., 23:27], name='Shape'),
        Dist(logits=params[..., 27:37], name='FloorHue'),
        Dist(logits=params[..., 37:47], name='WallHue'),
        Dist(logits=params[..., 47:57], name='ObjectHue')]
      self._classes = [15, 8, 4, 10, 10, 10]
    else:
      self._distributions = [Dist(logits=params, name='Classes')]
      self._classes = [self.event_dim]
    self._classes = np.cumsum([0] + list(self._classes), dtype=np.int32)
    super(FactorDistribution, self).__init__(
      dtype=params.dtype,
      validate_args=validate_args,
      allow_nan_stats=allow_nan_stats,
      reparameterization_type=(FULLY_REPARAMETERIZED if reparams else
                               NOT_REPARAMETERIZED),
      parameters=parameters,
      name=name)

  @classmethod
  def event_dim(cls, dsname: str) -> int:
    if dsname == 'dsprites':
      return 40 + 6 + 3 + 32 + 32
    elif dsname == 'shapes3d':
      return 15 + 8 + 4 + 10 + 10 + 10
    elif dsname == 'dsprites0':
      return 3
    elif dsname == 'shapes3d0':
      return 4
    elif dsname in ('mnist', 'fashionmnist', 'cifar10'):
      return 10
    raise NotImplementedError(f'No support for dataset {dsname}')

  @property
  def distributions(self) -> Sequence[Distribution]:
    return self._distributions

  def _event_shape_tensor(self):
    return tf.constant([self.event_dim], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([self.event_dim])

  def _batch_shape_tensor(self):
    return tf.shape(self._params)[:-1]

  def _batch_shape(self):
    return self._params.shape[:-1]

  def _sample_n(self, n, seed=None, **kwargs):
    return tf.concat([
      d.sample(n, seed) for d in self._distributions
    ], axis=-1)

  def _log_prob(self, y, **kwargs):
    if self.reparams:
      y = clip_by_value_preserve_gradient(y, 1e-6, 1. - 1e-6)
    return sum(d.log_prob(y[..., s:e])
               for d, s, e in zip(self._distributions,
                                  self._classes,
                                  self._classes[1:]))

  def _mean(self, **kwargs):
    return tf.concat([d.probs_parameter() for d in self._distributions],
                     axis=-1)

  def _mode(self, **kwargs):
    return tf.concat([tf.expand_dims(tf.argmax(d.probs_parameter(), -1), -1)
                      for d in self._distributions], axis=-1)

  def probs_parameter(self, name=None):
    """Probs vec computed from non-`None` input arg (`probs` or `logits`)."""
    return tf.concat([d.probs_parameter() for d in self._distributions],
                     axis=-1)


def reparameterize(dsname: str,
                   temperature: float = 0.5,
                   reparams: bool = True,
                   nonlinear_qy_z: bool = False) -> SequentialNetwork:
  dsname = dsname.lower()
  kw = dict(reparams=reparams, temperature=temperature, dsname=dsname)
  print(f' * Reparams:{reparams}  Temperature:{temperature}')
  event_dim = FactorDistribution.event_dim(dsname)
  layer = DistributionDense(
    event_shape=[event_dim],
    projection=True,
    posterior=partial(FactorDistribution, **kw),
    units=event_dim,
    name=f'{dsname}_qy')
  if nonlinear_qy_z:
    layers = [dense_networks(None, 'EncoderY'), layer]
  else:
    layers = [layer]
  return SequentialNetwork(layers, name='qy_z')


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
  shapes3d0=dict(encoder=3, decoder=2, filters=32, kernel=8, strides=4),
  ###
  dsprites=dict(encoder=3, decoder=2, filters=32, kernel=8, strides=4),
  dsprites0=dict(encoder=3, decoder=2, filters=32, kernel=8, strides=4),
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
  shapes3d=10.,
  dsprites=10.,
  shapes3d0=10.,
  dsprites0=10.,
  cifar10=10.,
  fashionmnist=10.,
  mnist=10.,
)

DefaultGammaPy = dict(
  shapes3d=10.,
  dsprites=10.,
  shapes3d0=10.,
  dsprites0=10.,
  cifar10=10.,
  fashionmnist=10.,
  mnist=10.,
)

DefaultSemi = dict(
  shapes3d=(0.1, 0.1),
  dsprites=(0.1, 0.1),
  shapes3d0=(0.01, 0.1),
  dsprites0=(0.01, 0.1),
  cifar10=(4000, 0.1),
  fashionmnist=(0.01, 0.1),
  mnist=(0.004, 0.1),
)

DefaultPretrain = dict(
  shapes3d=800,
  dsprites=800,
  shapes3d0=800,
  dsprites0=800,
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
               reparams: bool = True,
               temperature: float = 0.5,
               nonlinear_qy_z: bool = False,
               deep_prior: bool = False,
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
    self.reparams = bool(reparams)
    self.nonlinear_qy_z = bool(nonlinear_qy_z)
    self.deep_prior = deep_prior
    # === 1. fixed utility layers
    self.flatten = Flatten()
    self.concat = Concatenate(-1)
    self.global_avg = GlobalAvgPool2D()
    # === 2. reparameterized q(y|z)
    ydim = int(np.prod(labels.event_shape))
    zdim = int(np.prod(self.latents.event_shape))
    # q(y|z0,z1,x)
    self.labels = reparameterize(config.ds.lower(),
                                 temperature=temperature,
                                 reparams=reparams,
                                 nonlinear_qy_z=nonlinear_qy_z)
    # === 3. second VAE for y and u
    self.encoder2 = dense_networks(ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = dense_networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = labels
    self.observation2(self.decoder2.output)
    # === 4. p(z|u)
    if deep_prior:
      layers = [Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                MVNDiagLatents(zdim)]
    else:
      layers = [Dense(256, activation='relu'), MVNDiagLatents(zdim)]
    self.latents_prior = SequentialNetwork(layers, name=f'pZ0_given_U')
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
                   return_mean=True,
                   **kwargs):
    dsname = config.ds.lower()
    is_2d = 'dsprites' in dsname
    # === 0. mnist
    if dsname in ('mnist', 'fashionmnist', 'cifar10'):
      y = np.diag([1.] * 10)
      y = np.tile(y, [int(np.ceil(n / 10)), 1])[:n]
    # === 1. shapes with only shapes label
    elif dsname in ('dsprites0', 'shapes3d0'):
      n_classes = 3 if is_2d else 4
      y = np.diag([1.] * n_classes)
      y = np.tile(y, [int(np.ceil(n / n_classes)), 1])[:n]
    # === 2. shapes
    elif dsname in ('dsprites', 'shapes3d'):
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
    # === 3. no idea
    else:
      raise NotImplementedError(f'No sample_prior support for dataset {dsname}')
    # === 4. sample the prior p(z|u)
    qu_y = self.encode_aux(y, training=False)
    # mean or sample
    u = qu_y.sample(seed=seed)
    pz_u = self.latents_prior(u, training=False)
    if return_distribution:
      ret = pz_u
    else:  # NOTE: sample or mean?
      ret = pz_u.mean() if return_mean else pz_u.sample()
    ret.labels = y
    if return_labels:
      return ret, y
    return ret

  def sample_fid(self, n=1, seed=1):
    # complete random samples for FID scores
    u = self.latents2.prior.sample(n, seed=seed)
    pz_u = self.latents_prior(u, training=False)
    z = pz_u.sample()
    return self.decode(z, training=False)

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
    if not self.reparams:
      y_u = tf.stop_gradient(y_u)
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

  def __str__(self):
    text = super(SemafoVAE, self).__str__()
    text += f"\n q(y|z):\n{layer2text(self.labels, padding='  ')}"
    text += f"\n p(y|u):\n{layer2text(self.observation2, padding='  ')}"
    text += f"\n Encoder-aux:\n{layer2text(self.encoder2, padding='  ')}"
    text += f"\n Decoder-aux:\n{layer2text(self.decoder2, padding='  ')}"
    text += f"\n {self.latents2}"
    return text


########### Higher Gamma values
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


########### No reparameterization for q(y|z)
class NoReparams(SemafoVAE):
  def __init__(self, **kwargs):
    super(NoReparams, self).__init__(reparams=False, **kwargs)


class NoReparamsH10(SemafoVAE):
  def __init__(self, **kwargs):
    super(NoReparamsH10, self).__init__(reparams=False,
                                        coef_H_qy=10.,
                                        **kwargs)


class NoReparamsH20(SemafoVAE):
  def __init__(self, **kwargs):
    super(NoReparamsH20, self).__init__(reparams=False,
                                        coef_H_qy=20.,
                                        **kwargs)


class NoReparamsH40(SemafoVAE):
  def __init__(self, **kwargs):
    super(NoReparamsH40, self).__init__(reparams=False,
                                        coef_H_qy=40.,
                                        **kwargs)


class T01(SemafoVAE):
  def __init__(self, **kwargs):
    super(T01, self).__init__(reparams=True, temperature=0.1, **kwargs)


class T01H10(SemafoVAE):
  def __init__(self, **kwargs):
    super(T01H10, self).__init__(reparams=True, temperature=0.1, coef_H_qy=10,
                                 **kwargs)


class T01H20(SemafoVAE):
  def __init__(self, **kwargs):
    super(T01H20, self).__init__(reparams=True, temperature=0.1, coef_H_qy=20,
                                 **kwargs)


class T01H40(SemafoVAE):
  def __init__(self, **kwargs):
    super(T01H40, self).__init__(reparams=True, temperature=0.1, coef_H_qy=40,
                                 **kwargs)


class T02(SemafoVAE):
  def __init__(self, **kwargs):
    super(T02, self).__init__(reparams=True, temperature=0.2, **kwargs)


class T02H10(SemafoVAE):
  def __init__(self, **kwargs):
    super(T02H10, self).__init__(reparams=True, temperature=0.2, coef_H_qy=10,
                                 **kwargs)


class T02H20(SemafoVAE):
  def __init__(self, **kwargs):
    super(T02H20, self).__init__(reparams=True, temperature=0.2, coef_H_qy=20,
                                 **kwargs)


class NonlinearQyz(SemafoVAE):

  def __init__(self, **kwargs):
    super(NonlinearQyz, self).__init__(reparams=False,
                                       coef_H_qy=10,
                                       nonlinear_qy_z=True,
                                       **kwargs)


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
        if self.deep_prior:
          layers = [Dense(256, activation='relu'),
                    Dense(256, activation='relu'),
                    Dense(256, activation='relu')]
        else:
          layers = [Dense(256, activation='relu')]
        pz_given_u[str(idx)] = Sequential(layers + [
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

  def sample_fid(self, n=1, seed=1):
    # complete random samples for FID scores
    u = self.latents2.prior.sample(n, seed=seed)
    pz_u = self.latents_prior(u, training=False)
    z = pz_u.sample()
    # enable sampling mode for HVAE
    sampling = self._is_sampling
    self.set_sampling(True)
    px = self.decode(z, training=False)
    self.set_sampling(sampling)
    return px

  def sample_observation(self, n: int = 1, seed: int = 1,
                         training: bool = False, **kwargs) -> Distribution:
    sampling = self._is_sampling
    self.set_sampling(True)
    x = super(SemafoHVAE, self).sample_observation(n=n, seed=seed,
                                                   training=training)
    self.set_sampling(sampling)
    return x

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
      # last_outputs = [tf.repeat(i[:0], latents.shape[0], 0)
      #                 for i in last_outputs]
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


class HNoReparamsH40(SemafoVAE):
  def __init__(self, **kwargs):
    super(HNoReparamsH40, self).__init__(reparams=False,
                                         coef_H_qy=40.,
                                         **kwargs)


class DeepPrior(SemafoVAE):
  def __init__(self, **kwargs):
    super(DeepPrior, self).__init__(deep_prior=True, **kwargs)


class DeepPriorH(SemafoHVAE):
  def __init__(self, **kwargs):
    super(DeepPriorH, self).__init__(deep_prior=True, **kwargs)


# ===========================================================================
# Baseline: M3VAE
# ===========================================================================
class CondPrior(Layer):
  def __init__(self, dim):
    super(CondPrior, self).__init__()
    self.dim = dim
    self.diag_loc_true = tf.Variable(tf.ones([dim], dtype=self.dtype) * 2)
    self.diag_loc_false = tf.Variable(tf.ones([dim], dtype=self.dtype) * -2)
    self.diag_scale_true = tf.Variable(tf.ones([dim], dtype=self.dtype))
    self.diag_scale_false = tf.Variable(tf.ones([dim], dtype=self.dtype))
    self.dist = IndependentNormal(event_shape=[dim])

  def call(self, x, **kwargs):
    loc = x * self.diag_loc_true + (1. - x) * self.diag_loc_false
    scale = x * self.diag_scale_true + (1. - x) * self.diag_scale_false
    scale = tf.clip_by_value(tf.nn.softplus(scale), 1e-3, 1e12)
    return self.dist(tf.concat([loc, scale], axis=-1))


class Classifier(Layer):
  def __init__(self, dim):
    super(Classifier, self).__init__()
    self.dim = dim
    self.weight = tf.Variable(tf.ones([self.dim]))
    self.bias = tf.Variable(tf.zeros([self.dim]))

  def call(self, x, **kwargs):
    return x * self.weight + self.bias


def multiple_onehot(params: tf.Tensor, classes: List[int]):
  dists = []
  for s, e in zip(classes, classes[1:]):
    p = params[..., s:e]
    dists.append(OneHotCategorical(logits=p, dtype=params.dtype))
  if len(dists) == 1:
    return dists[0]
  return Blockwise(distributions=dists)


class M3VAE(BetaGammaVAE):
  """Credit: https://github.com/thwjoy/ccvae"""

  def __init__(self, labels: DistributionDense, n_resamples: int = 100,
               alpha: float = 10., **kwargs):
    super().__init__(**kwargs)
    self.alpha = float(alpha)
    self.n_classes = int(np.prod(labels.event_shape))
    self.n_resamples = int(n_resamples)
    self.labels = labels
    ## prepare the classes
    if self.n_classes in (10, 3, 4):
      self.classes = [self.n_classes]
    elif self.n_classes == (15 + 8 + 4 + 30):
      self.classes = [15, 8, 4, 10, 10, 10]
    elif self.n_classes == (40 + 6 + 3 + 32 + 32):
      self.classes = [40, 6, 3, 32, 32]
    else:
      raise NotImplementedError()
    self.prior = tf.concat([
      tf.expand_dims(tf.convert_to_tensor([tf.math.log(1. / i)] * i), 0)
      for i in self.classes], axis=-1)
    self.concat = Concatenate(-1)
    self._y_prior = DistributionLambda(partial(FactorDistribution,
                                               reparams=False,
                                               dsname=config.ds))
    # partial(multiple_onehot, classes=np.cumsum([0] + self.classes)))
    ## initialize CCVAE
    self.classifier = Classifier(self.n_classes)
    self.cond_prior = CondPrior(self.n_classes)
    self.z_dim = int(np.prod(self.latents.event_shape))
    self.z_classify = self.n_classes
    self.z_style = self.z_dim - self.z_classify
    if self.z_style <= 0:
      self.z_style = self.z_dim
    self._latents = RVconf(event_shape=(self.z_style,),
                           posterior='normal',
                           projection=True,
                           name=self.latents.name).create_posterior()
    self.latents_cls = RVconf(event_shape=(self.z_classify,),
                              posterior='normal',
                              projection=True,
                              name='denotations').create_posterior()

  def sample_prior(self, n=1, seed=1, **kwargs):
    dsname = config.ds.lower()
    is_2d = 'dsprites' in dsname
    # === 0. mnist
    if dsname in ('mnist', 'fashionmnist', 'cifar10'):
      y = np.diag([1.] * 10)
      y = np.tile(y, [int(np.ceil(n / 10)), 1])[:n]
    # === 1. shapes with only shapes label
    elif dsname in ('dsprites0', 'shapes3d0'):
      n_classes = 3 if is_2d else 4
      y = np.diag([1.] * n_classes)
      y = np.tile(y, [int(np.ceil(n / n_classes)), 1])[:n]
    # === 2. shapes
    elif dsname in ('dsprites', 'shapes3d'):
      if is_2d:
        # orientation, scale, shape, x, y
        combinations = [40, 6, 3, 32, 32]
      else:
        # orientation, scale, shape, floor, wall, object
        combinations = [15, 8, 4, 10, 10, 10]
      per_class = int(np.ceil(n / combinations[2]))
      cls_to_labels = dict()
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
        cls_to_labels[cls] = y
      y = np.concatenate([x[:per_class] for x in cls_to_labels.values()], 0)
      y = y[:n].astype(np.float32)
    # === 3. no idea
    else:
      raise NotImplementedError(f'No sample_prior support for dataset {dsname}')
    z = self.latents.prior.sample(n, seed=seed)
    # y = np.concatenate([one_hot(np.mod(np.arange(n), i), i)
    #                     for i in self.classes], -1)
    qc = self.cond_prior_fn(tf.convert_to_tensor(y, dtype=self.dtype))
    c = qc.sample((), seed=seed)
    c.dist = qc
    return z, c

  def sample_traverse(self,
                      inputs,
                      n_traverse_points=11,
                      n_best_latents=5,
                      min_val=-2.0,
                      max_val=2.0,
                      mode='linear',
                      smallest_stddev=True,
                      training=False,
                      mask=None,
                      traverse_style=True):
    # style: if True, traverse style, otherwise, traverse class
    from odin.bay.vi import traverse_dims
    qz, qc = self.encode(inputs, training=training, mask=mask)
    # select style or class
    dist = qz if traverse_style else qc
    stddev = np.sum(dist.stddev(), axis=0)
    # smaller stddev is better
    if smallest_stddev:
      top_latents = np.argsort(stddev)[:int(n_best_latents)]
    else:
      top_latents = np.argsort(stddev)[::-1][:int(n_best_latents)]
    latents = traverse_dims(dist,
                            feature_indices=top_latents,
                            min_val=min_val, max_val=max_val,
                            n_traverse_points=n_traverse_points,
                            mode=mode)
    # return the decode images
    if traverse_style:
      c = qc.mean()
      c = tf.repeat(c, int(latents.shape[0] / c.shape[0]), 0)
      latents = [latents, c]
    else:
      z = qz.mean()
      z = tf.repeat(z, int(latents.shape[0] / z.shape[0]), 0)
      latents = [z, latents]
    return self.decode(latents, training=training, mask=mask), top_latents

  def cond_prior_fn(self, y):
    return self.cond_prior(y)

  def _y_prior_fn(self, alpha):
    return self._y_prior(alpha)

  def encode(self, inputs, training=None, **kwargs):
    inputs = as_tuple(inputs)
    if len(inputs):
      x = inputs[0]
      y = None
    else:
      x, y = inputs
    # encode normally
    h_e = self.encoder(x, training=training)
    qz_x = self.latents(h_e, training=training)
    qzc_x = self.latents_cls(h_e, training=training)
    return qz_x, qzc_x

  def decode(self, latents, training=None, **kwargs):
    if not isinstance(latents, (tuple, list)):
      qz_x = latents[..., :self.z_style]
      qzc_x = latents[..., self.z_style:]
    else:
      qz_x, qzc_x = latents
    z = self.concat([qz_x, qzc_x])
    px = super(M3VAE, self).decode(z, training=training, **kwargs)
    # qy
    z_class = tf.convert_to_tensor(qzc_x)
    # detach to reduce variance of gradients
    z_class = tf.stop_gradient(z_class)
    qy = self._y_prior_fn(self.classifier(z_class))
    return px, qy

  def elbo_uns(self, x_u, training=None):
    (px_u, qy_u), (qz_u, qc_u) = self(x_u, training=training)
    bs_u = x_u.shape[0]
    # basic ELBO
    llk_u = px_u.log_prob(x_u)
    kl_zu = qz_u.KL_divergence(analytic=self.analytic)
    # sample y from prior
    py_u = self._y_prior_fn(tf.tile(self.prior, [bs_u, 1]))
    y_u = tf.convert_to_tensor(py_u)
    # sample p(zc|y)
    pc_u = self.cond_prior_fn(y_u)
    # kl_c
    c = tf.convert_to_tensor(qc_u)
    kl_cu = qc_u.log_prob(c) - pc_u.log_prob(c)
    # final elbo
    return dict(llk_u=tf.reduce_mean(llk_u, 0)), \
           dict(kl_zu=tf.reduce_mean(kl_zu, 0),
                kl_cu=tf.reduce_mean(kl_cu, 0))

  def elbo_sup(self, x_s, y_s, training=None):
    (px_s, qy_s), (qz_s, qc_s) = self(x_s, training=training)
    bs_s = x_s.shape[0]
    # === 0. basic ELBO
    llk_s = px_s.log_prob(x_s)
    kl_zs = qz_s.KL_divergence(analytic=self.analytic)
    # conditional prior q(zc|y)
    pc_s = self.cond_prior_fn(y_s)
    c = tf.convert_to_tensor(qc_s)
    kl_cs = qc_s.log_prob(c) - pc_s.log_prob(c)
    # classifier loss
    k = self.n_resamples
    zs = qc_s.sample(k)
    zs = tf.reshape(zs, (-1, self.z_classify))
    logits = self.classifier(zs)
    d = self._y_prior_fn(logits)
    lqy_z = d.log_prob(tf.tile(y_s, [k, 1]))  # repeat y_s for n_mcmc
    lqy_z = tf.reshape(lqy_z, [k, bs_s])
    lqy_x = tf.reduce_logsumexp(lqy_z, 0) - tf.math.log(tf.cast(k, self.dtype))
    llk_classifier = self.alpha * lqy_x
    # log_py
    py_s = self._y_prior_fn(tf.tile(self.prior, [bs_s, 1]))
    lpy = py_s.log_prob(y_s)
    # ratio and weight
    w_s = tf.math.exp(qy_s.log_prob(y_s) - lqy_x)
    llk_s *= w_s
    kl_zs *= w_s
    kl_cs *= w_s
    # final elbo
    return dict(llk_s=tf.reduce_mean(llk_s, 0),
                llk_classifier=tf.reduce_mean(llk_classifier, 0),
                llk_py=tf.reduce_mean(lpy, 0)), \
           dict(kl_zs=tf.reduce_mean(kl_zs, 0),
                kl_cs=tf.reduce_mean(kl_cs, 0))

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    x_u, x_s, y_s = inputs
    llk_u, kl_u = self.elbo_uns(x_u, training=training)
    llk_s, kl_s = self.elbo_sup(x_s, y_s, training=training)
    # === 5. Final
    llk = dict(**llk_u, **llk_s)
    kl = dict(**kl_u, **kl_s)
    return llk, kl

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True


# ===========================================================================
# Training
# ===========================================================================
def _mean(dists, idx):
  m = dists[idx].mean()
  return tf.reshape(m, (m.shape[0], -1))


def _reshape2D(x: tf.Tensor) -> tf.Tensor:
  if x.shape.rank == 1:
    return x
  return tf.reshape(x, (x.shape[0], -1))


def concat_mean(dists: List[Distribution]) -> tf.Tensor:
  return tf.concat([_reshape2D(d.mean()) for d in dists], -1)


def semafo_latents(ds: ImageDataset,
                   model: SemafoVAE,
                   gym: DisentanglementGym,
                   path: str,
                   args: Arguments):
  if not isinstance(model, SemafoVAE):
    return
  qy = gym.px_z[1]
  py = model.call_aux(qy.mean())[0]
  y_true = gym.y_true
  y_names = ds.labels
  class_names = DS_TO_LABELS[ds.name]
  # accuracy of the prediction
  if ds.name not in ('dsprites', 'shapes3d'):
    y_true = np.expand_dims(np.argmax(y_true, -1), -1)
    y_names = ['Digits']
  with open(get_scores_path(args), 'a') as f:
    for i, (pred, pred_p, true) in enumerate(zip(qy.mode().numpy().T,
                                                 py.mode().numpy().T,
                                                 y_true.T)):
      text = f'#{i} acc_q:{accuracy_score(true, pred):.2f} ' \
             f'acc_p:{accuracy_score(true, pred_p):.2f}'
      f.write(text)
      print(text)
  # === 1. latents Traverse
  # traverse_path = os.path.join(path, 'traverses')
  # if not os.path.exists(traverse_path):
  #   os.makedirs(traverse_path)
  # if ds.name == 'dsprites':
  #   conditions = [{2: 0}, {2: 1}, {2: 2}]
  # elif ds.name == 'shapes3d':
  #   conditions = [{2: 0}, {2: 1}, {2: 2}, {2: 3}]
  # else:  # mnist, fashionmnist, cifar10
  #   conditions = [{0: i} for i in range(10)]
  # for cond in conditions:
  #   name = class_names[list(cond.values())[0]].lower()
  #   factors, ids = gym.groundtruth.sample_factors(cond, n_per_factor=3)
  #   (px_z, qy_z), qz_x = model(gym.x_true[ids])
  #   pz_u = model.latents_prior(model.encode_aux(tf.convert_to_tensor(qy_z)))
  # TODO
  # === 2. latents stats
  flatten = lambda arr: np.reshape(arr, (arr.shape[0], -1))
  for idx, (qz, pz) in enumerate(zip(gym.qz_x, gym.pz)):
    q_m = np.ravel(np.mean(qz.mean(), 0))
    p_m = np.ravel(np.mean(pz.mean(), 0))
    q_s = flatten(qz.stddev())
    p_s = flatten(pz.stddev())
    ids = np.argsort(np.mean(p_s, axis=0))
    plt.figure(figsize=(10 if len(q_m) < 80 else 18, 4))
    #
    plt.subplot(1, 2, 1)
    plt.plot(q_m[ids], color='r', label=r'$Q_{mean}$')
    plt.plot(np.mean(q_s, 0)[ids], color='b', linestyle='--',
             label=r'mean($Q_{std})$')
    plt.plot(np.max(q_s, 0)[ids], color='g', linestyle='--',
             label=r'max($Q_{std})$')
    plt.plot(np.min(q_s, 0)[ids], color='g', linestyle='--',
             label=r'min($Q_{std})$')
    plt.title('Posterior')
    plt.legend()
    #
    plt.subplot(1, 2, 2)
    plt.plot(p_m[ids], color='r', label=r'$P_{mean}$')
    plt.plot(np.mean(p_s, 0)[ids], color='b', linestyle='--',
             label=r'mean($P_{std})$')
    plt.plot(np.max(p_s, 0)[ids], color='g', linestyle='--',
             label=r'max($P_{std})$')
    plt.plot(np.min(p_s, 0)[ids], color='g', linestyle='--',
             label=r'min($P_{std})$')
    plt.title('Prior')
    plt.legend()
    #
    plt.suptitle(f'Latents #{idx}')
    plt.tight_layout(rect=[0, 0, 1, 1.1])
  vs.plot_save(os.path.join(path, 'semafo_latents_stats.pdf'), verbose=True)
  # === 4. latents t-SNE
  for idx, (qz, pz) in enumerate(zip(gym.qz_x, gym.pz)):
    n_points = 2000
    y = y_true[:n_points]
    qz = flatten(qz.mean().numpy()[:n_points])
    pz = flatten(pz.mean().numpy()[:n_points])
    py_u, qu_y = model.call_aux(gym.y_true_original[:n_points])
    pz_uy = model.latents_prior(qu_y)
    z = np.concatenate([qz, pz], -1)
    # applying t-SNE
    qz = DimReduce.TSNE(qz, framework='sklearn')
    pz = DimReduce.TSNE(pz, framework='sklearn')
    z = DimReduce.TSNE(z, framework='sklearn')
    u = DimReduce.TSNE(qu_y.mean().numpy(), framework='sklearn')
    zu = DimReduce.TSNE(flatten(pz_uy.mean().numpy()), framework='sklearn')
    for name, i in zip(y_names, y.T):
      plt.figure(figsize=(12, 3))
      vs.plot_scatter(qz, val=i, size=12, ax=(1, 5, 1), title='q(z|x)')
      vs.plot_scatter(pz, val=i, size=12, ax=(1, 5, 2), title='p(z|u)')
      vs.plot_scatter(z, val=i, size=12, ax=(1, 5, 3), title='qz*pz')
      vs.plot_scatter(u, val=i, size=12, ax=(1, 5, 4), title='q(u|y)')
      vs.plot_scatter(zu, val=i, size=12, ax=(1, 5, 5), title='p(z|u,y)')
      plt.suptitle(name, fontsize=12)
      plt.tight_layout(rect=[0, 0, 1, 1.1])
    vs.plot_save(os.path.join(path, f'qz_pz_y_{idx}.pdf'), verbose=True)


def prior_traverse(model: SemafoVAE,
                   dsname: str,
                   path: str):
  classes = DS_TO_LABELS[dsname]
  n_classes = len(classes)
  if isinstance(model, M3VAE):
    z, c = model.sample_prior(n=n_classes, seed=1)
    qc = c.dist
    mean = tf.concat([z, qc.mean()], -1)
    stddev = tf.concat([tf.ones_like(z), qc.stddev()], -1)
  elif not isinstance(model, SemafoVAE):
    pz = model.sample_prior(n=n_classes, seed=1)
    mean = pz
    stddev = tf.ones_like(mean)
  else:
    pz, labels = model.sample_prior(n=n_classes, seed=1,
                                    return_distribution=True,
                                    return_labels=True)
    mean = pz.mean()
    stddev = pz.stddev()
  n_points = 11  # TODO
  n_latents = min(20, mean.shape[1])
  max_std = 3.
  # SemafoHVAE
  if hasattr(model, 'set_sampling'):
    model.set_sampling(True)
  # first latents
  for i in range(n_classes):
    m = mean[i:i + 1]
    s = stddev[i].numpy()
    ids = np.argsort(s)[::-1][:n_latents]  # higher is better
    m = traverse_dims(m,
                      feature_indices=ids,
                      min_val=-max_std * s,
                      max_val=max_std * s,
                      n_traverse_points=n_points)
    img = as_tuple(model.decode(m))[0]
    img = prepare_images(img.mean(), True)
    plt.figure(figsize=(1.5 * n_points, 1.5 * n_latents), dpi=200)
    vs.plot_images(img, images_per_row=n_points, title=f'Class={classes[i]}')
  vs.plot_save(os.path.join(path, 'prior_traverse.pdf'), verbose=True)
  # special case for hierarchical model
  if hasattr(model, 'set_sampling'):
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
      img = prepare_images(img.mean(), True)
      plt.figure(figsize=(1.5 * n_points, 1.5 * n_latents), dpi=200)
      vs.plot_images(img, images_per_row=n_points, title=f'Class={classes[i]}')
    vs.plot_save(os.path.join(path, 'prior1_traverse.pdf'), verbose=True)
  # important, otherwise, all evaluation is wrong
  if isinstance(model, SemafoHVAE):
    model.set_sampling(False)


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
                          is_semi_supervised=model.is_semi_supervised(),
                          is_hierarchical=model.is_hierarchical())
  event_dim = FactorDistribution.event_dim(args.ds)
  networks['labels'] = DistributionDense(
    event_shape=(event_dim,),
    posterior=partial(
      FactorDistribution,
      reparams=True if issubclass(model, M2VAE) else False,
      dsname=args.ds),
    units=event_dim,
    name=f'{args.ds}_py')
  if not model.is_semi_supervised():
    del networks['labels']
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
          valid_data_map=valid_data_map,
          cache_data_memory=True if 'shapes3d' not in args.ds else False)
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
    print('Trained steps:', model.step.numpy())
    ### run the disentanglement gym
    score_path = get_scores_path(args)
    gym = DisentanglementGym(model=model,
                             dataset=args.ds,
                             batch_size=32,
                             dpi=args.dpi,
                             seed=args.seed + 5)
    with gym.run_model(n_samples=800 if args.debug else 30000,
                       device='cpu',
                       partition='test'):
      gym.write_report(score_path,
                       scores=(gym.accuracy_score, gym.log_likelihood,
                               gym.frechet_inception_distance,
                               gym.kl_divergence, gym.active_units,
                               gym.mig_score, gym.dci_score, gym.sap_score,
                               gym.clustering_score),
                       verbose=True)
      ## special case for SemafoVAE
      try:
        prior_traverse(model, args.ds, path)
      except Exception as e:
        warnings.warn(f'Failed prior_traverse {args}')
        print(e)
      semafo_latents(ds, model, gym, path, args)
      ## latents t-SNE
      gym.plot_latents_tsne()
      if gym.n_latent_vars > 1:
        for i in range(gym.n_latent_vars):
          gym.plot_latents_tsne(convert_fn=partial(_mean, idx=i),
                                title=f'_z{i}')
      ## prior sampling
      try:
        if model.is_hierarchical():
          model.set_sampling(True)
        gym.plot_latents_sampling()
      except Exception as e:
        warnings.warn(f'False plot_latents_sampling: {args}')
        print(e)
      ## traverse
      try:
        if isinstance(model, M3VAE):
          gym.plot_latents_traverse(latent_idx=0, title='_style',
                                    traverse_style=True)
          gym.plot_latents_traverse(latent_idx=1, title='_class',
                                    traverse_style=False)
        elif isinstance(model, SemafoHVAE):
          model.set_sampling(True)
          gym.plot_latents_traverse(title='_prior')
          model.set_sampling(False)
          gym.plot_latents_traverse(title='_post')
        else:
          gym.plot_latents_traverse(min_val=-2, max_val=2)
      except Exception as e:
        warnings.warn(f'False plot_latents_traverse: {args}')
        print(e)
      ## inference
      for i in range(gym.n_latent_vars):
        gym.plot_latents_stats(latent_idx=i)
        try:
          gym.plot_latents_factors(
            convert_fn=partial(_mean, idx=i), title=f'_z{i}')
          gym.plot_correlation(
            convert_fn=partial(_mean, idx=i), title=f'_z{i}')
        except Exception as e:
          warnings.warn(f'False plot_latents_factors and correlation: {args}')
          print(e)
      gym.plot_reconstruction()
    gym.save_figures(path, verbose=True)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  parsed_args = get_args(dict(py=0.0, ratio=0.1, it=200000, bs=64))
  set_cfg(root_path='/home/trung/exp/semafo',
          n_valid_batches=30,
          extra_path=dict(py=DefaultSemi[parsed_args.ds][0]))
  config = parsed_args
  # NOTE: override the default SEMI here
  if parsed_args.py == 0.0:
    parsed_args.py, parsed_args.ratio = DefaultSemi[parsed_args.ds]
  else:
    pass
  run_multi(main, args=parsed_args)
