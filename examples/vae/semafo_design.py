import os
import shutil
from functools import partial
from typing import Optional, Callable, List, Union, Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense, Flatten, Concatenate
from typing_extensions import Literal

from odin.backend import TensorType
from odin.bay import DistributionDense, MVNDiagLatents, BetaGammaVAE, \
  kl_divergence, DisentanglementGym, VariationalAutoencoder
from odin.bay.layers import RelaxedOneHotCategoricalLayer
from odin.fuel import get_dataset, ImageDataset
from odin.networks import get_networks, SequentialNetwork, TrainStep, Networks
from utils import get_args, train, run_multi, set_cfg, Arguments, \
  get_model_path, get_results_path, Callback

# ===========================================================================
# Const and helper
# ===========================================================================
dense = partial(Dense, activation='relu')
INPUT_SHAPE = ()
ds: Optional[ImageDataset] = None
valid_ds: Optional[tf.data.Dataset] = None
config: Optional[Arguments] = None


def networks(input_dim: Union[None, int], name: str) -> Sequential:
  layers = [
    dense(512, name=f'{name}_1'),
    dense(512, name=f'{name}_2'),
  ]
  if input_dim is not None:
    layers = [Input([input_dim])] + layers
  return Sequential(layers, name=name)


def to_elbo(semafo, llk, kl):
  elbo = semafo.elbo(llk, kl)
  return tf.reduce_mean(-elbo), \
         {k: tf.reduce_mean(v) for k, v in dict(**llk, **kl).items()}


# ===========================================================================
# Model
# ===========================================================================
class VAE(BetaGammaVAE):

  def __init__(self, labels=None, **kwargs):
    encoder = SequentialNetwork([
      Input(INPUT_SHAPE),
      Flatten(),
      Dense(300, activation='relu'),
      Dense(300, activation='relu'),
      Dense(300, activation='relu'),
    ], name='Encoder')
    decoder = SequentialNetwork([
      Input(32),
      Dense(300, activation='relu'),
      Dense(300, activation='relu'),
      Dense(300, activation='relu'),
    ], name='Decoder')
    latents = MVNDiagLatents(32, name='Latents')
    observation = DistributionDense(INPUT_SHAPE,
                                    posterior='bernoulli',
                                    projection=True,
                                    name='Image')
    super().__init__(encoder=encoder,
                     decoder=decoder,
                     observation=observation,
                     latents=latents)
    self.encoder.track_outputs = True
    self.decoder.track_outputs = True
    self.flatten = Flatten()
    self.concat = Concatenate(-1)


class Gamma10(VAE):

  def __init__(self, gamma=10., **kwargs):
    super().__init__(gamma=gamma, **kwargs)


class Gamma25(VAE):

  def __init__(self, gamma=25., **kwargs):
    super().__init__(gamma=gamma, **kwargs)


# ===========================================================================
# Hierarchical
# ===========================================================================
class Hierarchical(VAE):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.z1 = dict(
      post=dense(128, name='post1'),
      prior=dense(128, name='prior1'),
      deter=dense(128, name='deter1'),
      out=dense(300, name='out1'),
      pz=MVNDiagLatents(48, name='pz1'),
      qz=MVNDiagLatents(48, name='qz1'),
    )
    self._is_sampling = False

  @classmethod
  def is_hierarchical(cls) -> bool:
    return True

  def set_sampling(self, is_sampling: bool):
    self._is_sampling = bool(is_sampling)
    return self

  def get_latents(self, inputs=None, training=None, mask=None,
                  return_prior=False, **kwargs):
    is_sampling = self._is_sampling
    self.set_sampling(False)
    if inputs is None:
      if self.last_outputs is None:
        raise RuntimeError(f'Cannot get_latents of {self.name} '
                           'both inputs and last_outputs are None.')
      px, qz = self.last_outputs
    else:
      px, qz = self(inputs, training=training, mask=mask, **kwargs)
    self.set_sampling(is_sampling)
    posterior = [qz]
    prior = [qz.KL_divergence.prior]
    for name, q, p in px.kl_pairs:
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
    last_outputs = [i for i in latents._last_outputs]
    latents = traverse_dims(latents,
                            feature_indices=top_latents,
                            min_val=min_val, max_val=max_val,
                            n_traverse_points=n_traverse_points,
                            mode=mode)
    latents = tf.convert_to_tensor(latents)
    if not self._is_sampling:
      n_tiles = n_traverse_points * len(top_latents)
      last_outputs = [tf.tile(i, [n_tiles, 1]) for i in last_outputs]
      latents._last_outputs = last_outputs
    return self.decode(latents, training=training, mask=mask), top_latents

  def encode(self, inputs, training=None, **kwargs):
    latents = super().encode(inputs, training=training, **kwargs)
    latents._last_outputs = [i for _, i in self.encoder.last_outputs]
    return latents

  def decode(self, latents, training=None, **kwargs):
    z_org = tf.convert_to_tensor(latents)
    z = z_org
    kl_pairs = []
    for i, layer in enumerate(self.decoder.layers):
      z = layer(z, training=training)
      if i in [1]:
        zlayers = getattr(self, f'z{i}')
        # prior
        h_prior = zlayers['prior'](z, training=training)
        pz = zlayers['pz'](h_prior, training=training)
        if not self._is_sampling:  # posterior (inference mode)
          if not hasattr(latents, '_last_outputs'):
            raise RuntimeError('No encoder states found for hierarchical model')
          h_e = latents._last_outputs[2]
          h_post = zlayers['post'](self.concat([z, h_e]), training=training)
          qz = zlayers['qz'](h_post, training=training)
          kl_pairs.append((f'z{i}', qz, pz))
        else:  # sampling mode
          qz = pz
        # output
        h_deter = zlayers['deter'](z, training=training)
        z = zlayers['out'](self.concat([h_deter, qz]), training=training)
    px_z = self.observation(z, training=training)
    px_z.kl_pairs = kl_pairs
    return px_z

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, training=training, mask=mask,
                                      **kwargs)
    px, qz = self.last_outputs
    for name, q, p in px.kl_pairs:
      kl[f'kl_{name}'] = kl_divergence(q, p, analytic=self.analytic)
    return llk, kl


class Hierarchical1(Hierarchical):

  def build(self, input_shape=None):
    super(Hierarchical1, self).build(input_shape)

  def decode(self, latents, training=None, **kwargs):
    z_org = tf.convert_to_tensor(latents)
    z = z_org
    kl_pairs = []
    z_prev = [z_org]
    for i, layer in enumerate(self.decoder.layers):
      z = layer(z, training=training)
      if i in [1]:
        zlayers = getattr(self, f'z{i}')
        # prior
        h_prior = zlayers['prior'](z, training=training)
        pz = zlayers['pz'](h_prior, training=training)
        if not self._is_sampling:  # posterior (inference mode)
          if not hasattr(latents, '_last_outputs'):
            raise RuntimeError('No encoder states found for hierarchical model')
          h_e = latents._last_outputs[2]
          h_post = zlayers['post'](self.concat([z, h_e]), training=training)
          qz = zlayers['qz'](h_post, training=training)
          kl_pairs.append((f'z{i}', qz, pz))
        else:  # sampling mode
          qz = pz
        # output
        z_samples = tf.convert_to_tensor(qz)
        z_prev.append(z_samples)
        h_deter = zlayers['deter'](z, training=training)
        z = zlayers['out'](self.concat([h_deter, z_samples]), training=training)
    # final output
    z = self.concat([z] + z_prev)
    px_z = self.observation(z, training=training)
    px_z.kl_pairs = kl_pairs
    return px_z


class Residual(Hierarchical):

  def decode(self, latents, training=None, **kwargs):
    z_org = tf.convert_to_tensor(latents)
    z = z_org
    kl_pairs = []
    for i, layer in enumerate(self.decoder.layers):
      z = layer(z, training=training)
      if i in [1]:
        zlayers = getattr(self, f'z{i}')
        # prior
        h_prior = zlayers['prior'](z, training=training)
        pz = zlayers['pz'](h_prior, training=training)
        if not self._is_sampling:  # posterior (inference mode)
          if not hasattr(latents, '_last_outputs'):
            raise RuntimeError('No encoder states found for hierarchical model')
          h_e = latents._last_outputs[2]
          h_post = zlayers['post'](self.concat([z, h_e]), training=training)
          qz = zlayers['qz'](h_post, training=training)
          kl_pairs.append((f'z{i}', qz, pz))
        else:  # sampling mode
          qz = pz
        # output
        h_deter = zlayers['deter'](z, training=training)
        z_out = zlayers['out'](self.concat([h_deter, qz]), training=training)
        # add residual
        z = z_out + 0.3 * z  # this is hand tuned
    px_z = self.observation(z, training=training)
    px_z.kl_pairs = kl_pairs
    return px_z


class BottomUp(Hierarchical):

  def decode(self, latents, training=None, mask=None, only_decoding=False,
             **kwargs):
    x = latents
    kl_pairs = []
    for i, layer in enumerate(self.decoder.layers):
      x = layer(x, training=training)
      if i in [1]:
        z = getattr(self, f'z{i}')
        # prior
        h_prior = z['prior'](x, training=training)
        pz = z['pz'](h_prior, training=training)
        # posterior
        h_post = z['post'](self.encoder.last_outputs[2][1], training=training)
        qz = z['qz'](h_post, training=training)
        kl_pairs.append((f'z{i}', qz, pz))
        # output
        h_deter = z['deter'](x, training=training)
        x = z['out'](self.concat([h_deter, qz]), training=training)
    px = self.observation(x, training=training)
    px.kl_pairs = kl_pairs
    return px


class BiDirect(Hierarchical):

  def decode(self, latents, training=None, mask=None, only_decoding=False,
             **kwargs):
    x = latents
    kl_pairs = []
    for i, layer in enumerate(self.decoder.layers):
      x = layer(x, training=training)
      if i in [1]:
        x = self.concat([x, latents])  # shortcut from the last latents
        z = getattr(self, f'z{i}')
        # prior
        h_prior = z['prior'](x, training=training)
        pz = z['pz'](h_prior, training=training)
        # posterior
        h_post = z['post'](self.concat([x, self.encoder.last_outputs[2][1]]),
                           training=training)
        qz = z['qz'](h_post, training=training)
        kl_pairs.append((f'z{i}', qz, pz))
        # output
        h_deter = z['deter'](x, training=training)
        x = z['out'](self.concat([h_deter, qz]), training=training)
    px = self.observation(x, training=training)
    px.kl_pairs = kl_pairs
    return px


# ===========================================================================
# Semafo
# ===========================================================================
class Semafo(VAE):

  def __init__(self,
               labels,
               coef_H_qy: float = 1.,
               gamma_py: float = 10.,
               gamma_uns: Optional[float] = None,
               gamma_sup: float = 1.,
               beta_uns: float = 1.,
               beta_sup: float = 1.,
               n_iw_y: int = 1,
               **kwargs):
    super().__init__(**kwargs)
    self.n_iw_y = int(n_iw_y)
    self.coef_H_qy = float(coef_H_qy)
    if gamma_uns is None:
      gamma_uns = config.ratio / config.py  # 0.1 / 0.004
    self.gamma_uns = float(gamma_uns)
    self.gamma_sup = float(gamma_sup)
    self.gamma_py = float(gamma_py)
    self.beta_uns = float(beta_uns)
    self.beta_sup = float(beta_sup)
    self.labels_org = labels

  def build(self, input_shape=None):
    labels = self.labels_org
    self.labels = SequentialNetwork([
      networks(None, 'EncoderY'),
      DistributionDense(event_shape=[10], projection=True,
                        posterior=RelaxedOneHotCategoricalLayer,
                        posterior_kwargs=dict(temperature=0.5),
                        name='Digits')],
      name='qy_z')

    super(Semafo, self).build(input_shape)

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)

    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)

    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  @property
  def xdim(self) -> int:
    return int(np.prod(self.observation.event_shape))

  @property
  def zdim(self) -> int:
    return int(np.prod(self.latents.event_shape))

  @property
  def ydim(self) -> int:
    return int(np.prod(self.labels.layers[-1].event_shape))

  def sample_prior(self, n: int = 1, seed: int = 1, **kwargs):
    y = np.diag([1.] * 10)
    y = np.tile(y, [int(np.ceil(n / 10)), 1])[:n]
    qu_y = self.encode_aux(y, training=False)
    u = qu_y.sample(seed=seed)
    pz_u = self.latents_prior(u, training=False)
    return pz_u.mean()
    # return pz_u.sample(seed=seed)

  def decode(self, latents, training=None, **kwargs):
    z = tf.convert_to_tensor(latents)
    px_z = super().decode(z, training=training, **kwargs)
    py_z = self.labels(z, training=training)
    return px_z, py_z

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

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Supervised
    def elbo_sup():
      (px_z, qy_z), qz_x = self(x_s, training=training)
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
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)

    # === 2. Unsupervised
    def elbo_uns():
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
        llk_qy=self.coef_H_qy *
               qy_z.log_prob(tf.clip_by_value(y_u, 1e-6, 1. - 1e-6))
      )

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)


class Semafo1a(Semafo):

  def encode(self, inputs, training=None, **kwargs):
    h_e = self.encoder(inputs, training=training)
    qz_x = self.latents(h_e, training=training)
    qz_x.encoder_states = [i for _, i in h_e._last_outputs]
    return qz_x

  def decode(self, latents, training=None, **kwargs):
    z = tf.convert_to_tensor(latents)
    px_z = super(Semafo, self).decode(z, training=training, **kwargs)
    if hasattr(latents, 'encoder_states'):
      qy_z = self.labels(tf.concat(latents.encoder_states + [z], -1),
                         training=training)
    else:
      qy_z = None
    return px_z, qy_z


class Semafo1b(Semafo):

  def build(self, input_shape=None):
    labels = self.labels_org
    self.labels = SequentialNetwork(
      [DistributionDense(event_shape=[10], projection=True,
                         posterior=RelaxedOneHotCategoricalLayer,
                         posterior_kwargs=dict(temperature=0.5),
                         name='Digits')], name='qy_z')

    super(Semafo, self).build(input_shape)

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)

    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)

    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]


class Semafo1d(Semafo):

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Supervised
    def elbo_sup():
      (px_z, qy_z), qz_x = self(x_s, training=training)
      py_u, qu_y = self.call_aux(y_s, training=training)
      pz_u = self.latents_prior(qu_y, training=training)

      llk = dict(
        llk_px_sup=self.gamma_sup * px_z.log_prob(x_s),
        llk_py_sup=self.gamma_sup * self.gamma_py *
                   py_u.log_prob(y_s),
      )

      z = tf.convert_to_tensor(qz_x)
      kl = dict(
        kl_z_sup=self.beta_sup * (qz_x.log_prob(z) - pz_u.log_prob(z)),
        kl_u_sup=self.beta_sup * qu_y.KL_divergence(analytic=self.analytic)
      )
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)

    # === 2. Unsupervised
    def elbo_uns():
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
        llk_qy=self.coef_H_qy *
               qy_z.log_prob(tf.clip_by_value(y_u, 1e-6, 1. - 1e-6))
      )

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)


class Semafo1e(Semafo):

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Supervised
    def elbo_sup():
      (px_z, qy_z), qz_x = self(x_s, training=training)
      py_u, qu_y = self.call_aux(y_s, training=training)
      pz_u = self.latents_prior(qu_y, training=training)

      llk = dict(
        llk_px_sup=self.gamma_sup * px_z.log_prob(x_s),
        llk_py_sup=self.gamma_py * py_u.log_prob(y_s),
        llk_qy_sup=self.gamma_py * qy_z.log_prob(
          tf.clip_by_value(y_s, 1e-6, 1. - 1e-6))
      )

      z = tf.convert_to_tensor(qz_x)
      kl = dict(
        kl_z_sup=self.beta_sup * (qz_x.log_prob(z) - pz_u.log_prob(z)),
        kl_u_sup=self.beta_sup * qu_y.KL_divergence(analytic=self.analytic)
      )
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)

    # === 2. Unsupervised
    def elbo_uns():
      (px_z, qy_z), qz_x = self(x_u, training=training)
      y_u = tf.convert_to_tensor(qy_z, dtype=self.dtype)
      py_u, qu_y = self.call_aux(y_u, training=training)
      pz_u = self.latents_prior(qu_y, training=training)

      llk = dict(
        llk_px_uns=self.gamma_uns * px_z.log_prob(x_u),
        llk_py_uns=self.gamma_py * py_u.log_prob(y_u)
      )

      z = tf.convert_to_tensor(qz_x)
      kl = dict(
        kl_z_uns=self.beta_uns * (qz_x.log_prob(z) - pz_u.log_prob(z)),
        kl_u_uns=self.beta_uns * qu_y.KL_divergence(analytic=self.analytic),
        llk_qy=self.coef_H_qy *
               qy_z.log_prob(tf.clip_by_value(y_u, 1e-6, 1. - 1e-6))
      )

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)


# ===========================================================================
# Semafo 2
# ===========================================================================

class Semafo2(Semafo):

  def build(self, input_shape=None):
    labels = self.labels_org
    self.labels = SequentialNetwork([
      Flatten(),
      networks(None, 'EncoderY'),
      DistributionDense(event_shape=[10],
                        projection=True,
                        posterior=RelaxedOneHotCategoricalLayer,
                        posterior_kwargs=dict(temperature=0.5),
                        name='Digits')],
      name='qy_x')

    super(Semafo, self).build(input_shape)

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)

    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)

    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]

  def encode(self, inputs, training=None, **kwargs):
    qy_x = self.labels(inputs, training=training)
    h_e = self.encoder(inputs, training=training)
    h_e = self.concat([qy_x, h_e])
    qz_xy = self.latents(h_e, training=training)
    qz_xy._qy_x = qy_x
    return qz_xy

  def decode(self, latents, training=None, **kwargs):
    px_z = super(Semafo, self).decode(latents, training=training, **kwargs)
    if hasattr(latents, '_qy_x'):
      qy_x = latents._qy_x
    else:
      qy_x = None
    return px_z, qy_x


class Semafo2a(Semafo2):

  def build(self, input_shape=None):
    labels = self.labels_org
    self.labels = SequentialNetwork([
      Flatten(),
      DistributionDense(event_shape=[10],
                        projection=True,
                        posterior=RelaxedOneHotCategoricalLayer,
                        posterior_kwargs=dict(temperature=0.5),
                        name='Digits')],
      name='qy_x')

    super(Semafo, self).build(input_shape)

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)

    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)

    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]

  def encode(self, inputs, training=None, **kwargs):
    h_e = self.encoder(inputs, training=training)
    qy_x = self.labels(tf.concat([i for _, i in self.encoder.last_outputs], -1),
                       training=training)
    h_e = self.concat([qy_x, h_e])
    qz_xy = self.latents(h_e, training=training)
    qz_xy._qy_x = qy_x
    return qz_xy


class Semafo2b(Semafo2a):

  def build(self, input_shape=None):
    labels = self.labels_org
    self.labels = SequentialNetwork([
      Flatten(),
      networks(None, 'EncoderY'),
      DistributionDense(event_shape=[10],
                        projection=True,
                        posterior=RelaxedOneHotCategoricalLayer,
                        posterior_kwargs=dict(temperature=0.5),
                        name='Digits')],
      name='qy_x')

    super(Semafo, self).build(input_shape)

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)

    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)

    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]


class Semafo2Stage(Semafo):

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Update q(u|y) and p(y|u)
    def elbo_yu():
      py_u, qu_y = self.call_aux(y_s, training=training)
      llk = dict(
        llk_yu=self.gamma_sup * self.gamma_py * py_u.log_prob(y_s)
      )
      kl = dict(
        kl_uy=self.beta_sup * qu_y.KL_divergence(analytic=self.analytic)
      )
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.vae2_params, func=elbo_yu)

    # === 2. Unsupervised q(z|x) and q(x|z)
    def elbo_xz_uns():
      (px_z_u, qy_z_u), qz_x_u = self.call(x_u, training=training)

      llk = dict(
        llk_x_uns=self.gamma_uns * px_z_u.log_prob(x_u),
      )
      kl = dict(
        kl_uns=self.beta_uns * qz_x_u.KL_divergence(analytic=self.analytic),
      )
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.vae1_params, func=elbo_xz_uns)

    # === 2. Supervised q(z|x) and q(x|z)
    def elbo_xz_sup():
      (px_z_s, qy_z_s), qz_x_s = self.call(x_s, training=training)

      llk = dict(
        llk_x_sup=self.gamma_sup * px_z_s.log_prob(x_s),
        llk_y_sup=self.gamma_sup * self.gamma_py *
                  qy_z_s.log_prob(tf.clip_by_value(y_s, 1e-6, 1. - 1e-6)),
      )
      kl = dict(
        kl_sup=self.beta_sup * qz_x_s.KL_divergence(analytic=self.analytic),
      )
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.vae1_params, func=elbo_xz_sup)

  def train_steps1(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Supervised
    def elbo_sup():
      (px_z, qy_z), qz_x = self(x_s, training=training)
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
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)

    # === 2. Unsupervised
    def elbo_uns():
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
        llk_qy=self.coef_H_qy *
               qy_z.log_prob(tf.clip_by_value(y_u, 1e-6, 1. - 1e-6))
      )

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)

  def fit(self, *args, **kwargs) -> 'Networks':
    max_iter = kwargs.get('max_iter')
    on_batch_end: List[Callable] = kwargs.pop('on_batch_end', lambda: None)

    def switch_stage():
      if self.step.numpy() >= max_iter // 4:
        self.set_training_stage(1)

    on_batch_end.append(switch_stage)
    return super().fit(on_batch_end=on_batch_end, *args, **kwargs)


# ===========================================================================
# Semafo Hierarchical
# ===========================================================================
class SemafoH(Semafo):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._is_sampling = False

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
    last_outputs = [i for i in latents._last_outputs]
    latents = traverse_dims(latents,
                            feature_indices=top_latents,
                            min_val=min_val, max_val=max_val,
                            n_traverse_points=n_traverse_points,
                            mode=mode)
    latents = tf.convert_to_tensor(latents)
    if not self._is_sampling:
      n_tiles = n_traverse_points * len(top_latents)
      last_outputs = [tf.tile(i, [n_tiles, 1]) for i in last_outputs]
      latents._last_outputs = last_outputs
    return self.decode(latents, training=training, mask=mask), top_latents

  @classmethod
  def is_hierarchical(cls) -> bool:
    return True

  def set_sampling(self, is_sampling: bool):
    self._is_sampling = bool(is_sampling)
    return self

  def build(self, input_shape=None):
    self.z1 = dict(
      post=dense(128, name='post1'),
      prior=dense(128, name='prior1'),
      deter=dense(128, name='deter1'),
      out=dense(300, name='out1'),
      pz=MVNDiagLatents(48, name='pz1'),
      qz=MVNDiagLatents(48, name='qz1'),
      beta=1.,
    )

    labels = self.labels_org
    self.labels = SequentialNetwork([
      DistributionDense(event_shape=[10], projection=True,
                        posterior=RelaxedOneHotCategoricalLayer,
                        posterior_kwargs=dict(temperature=0.5),
                        name='Digits')],
      name='qy_z')

    super(Semafo, self).build(input_shape)

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)

    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)

    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]

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
    prior = [qz.KL_divergence.prior]
    for name, q, p, beta in px.kl_pairs:
      posterior.append(q)
      prior.append(p)
    if return_prior:
      return posterior, prior
    return posterior

  def encode(self, inputs, training=None, **kwargs):
    latents = super(SemafoH, self).encode(inputs, training=training, **kwargs)
    latents._last_outputs = [i for _, i in self.encoder.last_outputs]
    return latents

  def decode(self, latents, training=None, **kwargs):
    z_org = tf.convert_to_tensor(latents)
    z = z_org
    kl_pairs = []
    z_prev = [z_org]
    # === 0. hierarchical latents
    for i, layer in enumerate(self.decoder.layers):
      z = layer(z, training=training)
      if i in [1]:
        zlayers = getattr(self, f'z{i}')
        # prior
        h_prior = zlayers['prior'](z, training=training)
        pz = zlayers['pz'](h_prior, training=training)
        if not self._is_sampling:  # posterior (inference mode)
          if not hasattr(latents, '_last_outputs'):
            raise RuntimeError('No encoder states found for hierarchical model')
          h_e = latents._last_outputs[2]
          h_post = zlayers['post'](self.concat([z, h_e]), training=training)
          qz = zlayers['qz'](h_post, training=training)
          kl_pairs.append((f'z{i}', qz, pz, zlayers['beta']))
        else:  # sampling mode
          qz = pz
        # output
        z_samples = tf.convert_to_tensor(qz)
        z_prev.append(z_samples)
        h_deter = zlayers['deter'](z, training=training)
        z = zlayers['out'](self.concat([h_deter, z_samples]),
                           training=training)
    # === 1. p(x|z0,z1)
    z = self.concat(z_prev + [z])
    px_z = self.observation(z, training=training)
    px_z.kl_pairs = kl_pairs
    # === 2. q(y|z)
    py_z = self.labels(z_org, training=training)
    return px_z, py_z

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, training=training, mask=mask,
                                      **kwargs)
    px, qz = self.last_outputs
    for name, q, p, beta in px.kl_pairs:
      kl[f'kl_{name}'] = beta * kl_divergence(q, p, analytic=self.analytic)
    return llk, kl

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Supervised
    def elbo_sup():
      (px_z, qy_z), qz_x = self(x_s, training=training)
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
      for n, q, p, beta in px_z.kl_pairs:
        kl[f'kl_{n}'] = beta * kl_divergence(q, p, analytic=self.analytic)

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)

    # === 2. Unsupervised
    def elbo_uns():
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
        llk_qy=self.coef_H_qy *
               qy_z.log_prob(tf.clip_by_value(y_u, 1e-6, 1. - 1e-6))
      )
      for n, q, p, beta in px_z.kl_pairs:
        kl[f'kl_{n}'] = beta * kl_divergence(q, p, analytic=self.analytic)

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)


class SemafoHb2(SemafoH):

  def build(self, input_shape):
    self.z1 = dict(
      post=dense(128, name='post1'),
      prior=dense(128, name='prior1'),
      deter=dense(128, name='deter1'),
      out=dense(300, name='out1'),
      pz=MVNDiagLatents(48, name='pz1'),
      qz=MVNDiagLatents(48, name='qz1'),
      beta=2.0,
    )
    super(SemafoH, self).build(input_shape)


class Semafo2H(SemafoH):

  def build(self, input_shape=None):
    # === 0. q(y|x)
    labels = self.labels_org
    self.labels = SequentialNetwork([
      Flatten(),
      DistributionDense(event_shape=[10],
                        projection=True,
                        posterior=RelaxedOneHotCategoricalLayer,
                        posterior_kwargs=dict(temperature=0.5),
                        name='Digits')],
      name='qy_x')
    # === 1. Hierarchical p(z|z_i) and q(z|z_i,x)
    self.z1 = dict(
      post=dense(128, name='post1'),
      prior=dense(128, name='prior1'),
      deter=dense(128, name='deter1'),
      out=dense(300, name='out1'),
      pz=MVNDiagLatents(48, name='pz1'),
      qz=MVNDiagLatents(48, name='qz1'),
      beta=1.0,
    )
    super(Semafo, self).build(input_shape)
    # === 3. semafo
    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)
    # === 4. prior p(z|u)
    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)
    # === 5. split the params for different updates
    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]

  def encode(self, inputs, training=None, **kwargs):
    h_e = self.encoder(inputs, training=training)
    last_outputs = [i for _, i in self.encoder.last_outputs]
    qy_x = self.labels(tf.concat(last_outputs, -1), training=training)
    qz_xy = self.latents([qy_x, h_e], training=training)
    qz_xy._qy_x = qy_x
    qz_xy._last_outputs = last_outputs
    return qz_xy

  def decode(self, latents, training=None, **kwargs):
    z_org = tf.convert_to_tensor(latents)
    z = z_org
    kl_pairs = []
    for i, layer in enumerate(self.decoder.layers):
      z = layer(z, training=training)
      if i in [1]:
        zlayers: Dict[str, Any] = getattr(self, f'z{i}')
        # prior
        h_prior = zlayers['prior'](z, training=training)
        pz = zlayers['pz'](h_prior, training=training)
        if not self._is_sampling:  # posterior (inference mode)
          if not hasattr(latents, '_last_outputs'):
            raise RuntimeError('No encoder states found for hierarchical model')
          h_e = latents._last_outputs[2]
          h_post = zlayers['post'](self.concat([z, h_e]), training=training)
          qz = zlayers['qz'](h_post, training=training)
          kl_pairs.append((f'z{i}', qz, pz, zlayers['beta']))
        else:  # sampling mode
          qz = pz
        # output
        h_deter = zlayers['deter'](z, training=training)
        z = zlayers['out'](self.concat([h_deter, qz]), training=training)
    px_z = self.observation(z, training=training)
    px_z.kl_pairs = kl_pairs
    # === 2. q(y|z)
    qy_x = latents._qy_x if hasattr(latents, '_qy_x') else None
    return px_z, qy_x


# ===========================================================================
# Final
# ===========================================================================
class SemafoVAE(VariationalAutoencoder):

  def __init__(self,
               encoder,
               decoder,
               labels,
               coef_H_qy: float = 1.,
               gamma_py: float = 10.,
               gamma_uns: Optional[float] = None,
               gamma_sup: float = 1.,
               beta_uns: float = 1.,
               beta_sup: float = 1.,
               n_iw_y: int = 1,
               **kwargs):
    super().__init__(encoder=encoder, decoder=decoder, **kwargs)
    exit()
    self._is_sampling = False
    self.n_iw_y = int(n_iw_y)
    self.coef_H_qy = float(coef_H_qy)
    if gamma_uns is None:
      gamma_uns = config.ratio / config.py  # 0.1 / 0.004
    self.gamma_uns = float(gamma_uns)
    self.gamma_sup = float(gamma_sup)
    self.gamma_py = float(gamma_py)
    self.beta_uns = float(beta_uns)
    self.beta_sup = float(beta_sup)
    self.labels_org = labels
    self.z1 = dict(
      post=dense(128, name='post1'),
      prior=dense(128, name='prior1'),
      deter=dense(128, name='deter1'),
      out=dense(300, name='out1'),
      pz=MVNDiagLatents(48, name='pz1'),
      qz=MVNDiagLatents(48, name='qz1'),
      beta=1.,
    )

    labels = self.labels_org
    self.labels = SequentialNetwork([
      DistributionDense(event_shape=[10], projection=True,
                        posterior=RelaxedOneHotCategoricalLayer,
                        posterior_kwargs=dict(temperature=0.5),
                        name='Digits')],
      name='qy_z')

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**labels.get_config())
    self.observation2(self.decoder2.output)

    self.latents_prior = SequentialNetwork([
      networks(self.ydim, 'Prior'),
      MVNDiagLatents(self.zdim, name=f'{self.latents.name}_prior')
    ], name='latents_prior')
    self.latents_prior.build([None] + self.latents2.event_shape)

  def build(self, input_shape=None):
    super(SemafoVAE, self).build(input_shape)
    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]

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
    last_outputs = [i for i in latents._last_outputs]
    latents = traverse_dims(latents,
                            feature_indices=top_latents,
                            min_val=min_val, max_val=max_val,
                            n_traverse_points=n_traverse_points,
                            mode=mode)
    latents = tf.convert_to_tensor(latents)
    if not self._is_sampling:
      n_tiles = n_traverse_points * len(top_latents)
      last_outputs = [tf.tile(i, [n_tiles, 1]) for i in last_outputs]
      latents._last_outputs = last_outputs
    return self.decode(latents, training=training, mask=mask), top_latents

  @classmethod
  def is_hierarchical(cls) -> bool:
    return True

  def set_sampling(self, is_sampling: bool):
    self._is_sampling = bool(is_sampling)
    return self

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
    prior = [qz.KL_divergence.prior]
    for name, q, p, beta in px.kl_pairs:
      posterior.append(q)
      prior.append(p)
    if return_prior:
      return posterior, prior
    return posterior

  def encode(self, inputs, training=None, **kwargs):
    latents = super(SemafoH, self).encode(inputs, training=training, **kwargs)
    latents._last_outputs = [i for _, i in self.encoder.last_outputs]
    return latents

  def decode(self, latents, training=None, **kwargs):
    z_org = tf.convert_to_tensor(latents)
    z = z_org
    kl_pairs = []
    z_prev = [z_org]
    # === 0. hierarchical latents
    for i, layer in enumerate(self.decoder.layers):
      z = layer(z, training=training)
      if i in [1]:
        zlayers = getattr(self, f'z{i}')
        # prior
        h_prior = zlayers['prior'](z, training=training)
        pz = zlayers['pz'](h_prior, training=training)
        if not self._is_sampling:  # posterior (inference mode)
          if not hasattr(latents, '_last_outputs'):
            raise RuntimeError('No encoder states found for hierarchical model')
          h_e = latents._last_outputs[2]
          h_post = zlayers['post'](self.concat([z, h_e]), training=training)
          qz = zlayers['qz'](h_post, training=training)
          kl_pairs.append((f'z{i}', qz, pz, zlayers['beta']))
        else:  # sampling mode
          qz = pz
        # output
        z_samples = tf.convert_to_tensor(qz)
        z_prev.append(z_samples)
        h_deter = zlayers['deter'](z, training=training)
        z = zlayers['out'](self.concat([h_deter, z_samples]),
                           training=training)
    # === 1. p(x|z0,z1)
    z = self.concat(z_prev + [z])
    px_z = self.observation(z, training=training)
    px_z.kl_pairs = kl_pairs
    # === 2. q(y|z)
    py_z = self.labels(z_org, training=training)
    return px_z, py_z

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, training=training, mask=mask,
                                      **kwargs)
    px, qz = self.last_outputs
    for name, q, p, beta in px.kl_pairs:
      kl[f'kl_{name}'] = beta * kl_divergence(q, p, analytic=self.analytic)
    return llk, kl

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    # === 1. Supervised
    def elbo_sup():
      (px_z, qy_z), qz_x = self(x_s, training=training)
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
      for n, q, p, beta in px_z.kl_pairs:
        kl[f'kl_{n}'] = beta * kl_divergence(q, p, analytic=self.analytic)

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_sup)

    # === 2. Unsupervised
    def elbo_uns():
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
        llk_qy=self.coef_H_qy *
               qy_z.log_prob(tf.clip_by_value(y_u, 1e-6, 1. - 1e-6))
      )
      for n, q, p, beta in px_z.kl_pairs:
        kl[f'kl_{n}'] = beta * kl_divergence(q, p, analytic=self.analytic)

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)


# ===========================================================================
# Training
# ===========================================================================
def main(args: Arguments):
  model = None
  for k, v in globals().items():
    if k.lower() == args.vae and callable(v):
      model = v
      break
  # === 0. build the model
  model: BetaGammaVAE = model(
    **get_networks(args.ds,
                   is_semi_supervised=model.is_semi_supervised()))
  model.build((None,) + INPUT_SHAPE)
  is_semi = model.is_semi_supervised()
  # === 1. training
  if not args.eval:
    train(model, ds, args,
          label_percent=args.py if is_semi else 0.0,
          on_batch_end=(),
          on_valid_end=(Callback.save_best_llk,),
          oversample_ratio=args.ratio)
  # === 2. evaluation
  else:
    path = get_results_path(args)
    if args.override and os.path.exists(path):
      print('Override results at path:', path)
      shutil.rmtree(path)
      os.makedirs(path)
    # load model weights
    model.load_weights(get_model_path(args), raise_notfound=True, verbose=True)
    gym = DisentanglementGym(model=model,
                             dataset=args.ds,
                             batch_size=args.bs,
                             dpi=args.dpi,
                             seed=args.seed)
    with gym.run_model(n_samples=-1, partition='test'):
      print(gym.accuracy_score())
      # latents t-SNE
      gym.plot_latents_tsne()
      if gym.n_latent_vars > 1:
        for i in range(gym.n_latent_vars):
          gym.plot_latents_tsne(convert_fn=lambda dists: dists[i].mean(),
                                title=f'_z{i}')
      # sampling prior
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
      if model.is_hierarchical():
        model.set_sampling(False)
      for i in range(gym.n_latent_vars):
        gym.plot_latents_factors(convert_fn=lambda dists: dists[i].mean(),
                                 title=f'_z{i}')
        gym.plot_correlation(convert_fn=lambda dists: dists[i].mean(),
                             title=f'_z{i}')
        gym.plot_latents_stats(latent_idx=i)
      gym.plot_reconstruction()
    gym.save_figures(path, verbose=True)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  set_cfg(root_path='/home/trung/exp/semafo')
  args = get_args(dict(py=0.004, ratio=0.1, it=400000))
  config = args
  ds = get_dataset(args.ds)
  valid_ds = ds.create_dataset('valid', label_percent=1.0)
  INPUT_SHAPE = ds.shape
  run_multi(main, args=args)
