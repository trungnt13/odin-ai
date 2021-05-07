import os
from functools import partial
from typing import List, Type

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense, Flatten, Concatenate

from odin.backend import interpolation
from odin.bay import VariationalAutoencoder, \
  DistributionDense, MVNDiagLatents, BetaGammaVAE, DisentanglementGym, \
  kl_divergence
from odin.fuel import MNIST, get_dataset
from odin.networks import get_networks, get_optimizer_info, CenterAt0, \
  SequentialNetwork, TrainStep
from utils import get_args, train, run_multi, set_cfg, Arguments, get_model_path

# ===========================================================================
# Const and helper
# ===========================================================================
dense = partial(Dense, activation='relu')
INPUT_SHAPE = ()


def networks(input_dim, name) -> Sequential:
  return Sequential([
    Input([input_dim]),
    dense(512, name=f'{name}_1'),
    dense(512, name=f'{name}_2'),
  ], name=name)


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

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, training=training, mask=mask,
                                      **kwargs)
    px, qz = self.last_outputs
    for name, q, p in px.kl_pairs:
      kl[f'kl_{name}'] = kl_divergence(q, p, analytic=self.analytic)
    return llk, kl


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
class SemafoBase(VAE):

  def __init__(self, labels, **kwargs):
    super().__init__(**kwargs)
    self.labels = labels
    self.encoder_y = networks(self.zdim, 'EncoderY')

  def decode(self, latents, training=None, **kwargs):
    px_z = super().decode(latents, training=training, **kwargs)
    py_z = self.labels(self.encoder_y(latents, training=training),
                       training=training)
    return px_z, py_z

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
    return int(np.prod(self.labels.event_shape))


class Semafo(SemafoBase):

  def __init__(self, **kwargs):
    super().__init__(beta=interpolation.linear(1e-3, 1., 2000), **kwargs)

  def build(self, input_shape=None):
    super().build(input_shape)
    encoder: SequentialNetwork = self.encoder
    decoder: SequentialNetwork = self.decoder
    labels: DistributionDense = self.labels

    # for i, (l, t) in enumerate(encoder.last_outputs):
    #   print(f'[{i}]', l)
    #   print('  ', t)
    # for i, (l, t) in enumerate(decoder.last_outputs):
    #   print(f'[{i}]', l)
    #   print('  ', t)
    x_e = encoder.last_outputs[2][1]
    hdim = int(np.prod(x_e.shape[1:]))

    self.encoder2 = networks(hdim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.zdim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.zdim, 'Decoder2')
    self.observation2 = MVNDiagLatents(units=hdim, name='Observation2')
    self.observation2(self.decoder2.output)

    self.vae2_params = self.encoder2.trainable_variables + \
                       self.decoder2.trainable_variables + \
                       self.latents2.trainable_variables + \
                       self.observation2.trainable_variables
    vae2_params_id = set([id(i) for i in self.vae2_params])
    self.vae1_params = [v for v in self.trainable_variables
                        if id(v) not in vae2_params_id]

  def call_auxiliary(self, x, training=None):
    x = self.flatten(x)
    h_e = self.encoder2(x, training=training)
    qu = self.latents2(h_e, training=training)
    h_d = self.decoder2(qu, training=training)
    px = self.observation2(h_d, training=training)
    return px, qu

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs

    def elbo1():
      (px, py), qz = self(x_u, training=training)

      h_out = self.encoder.last_outputs[2][1]
      h_in = self.decoder.last_outputs[1][1]
      ph, qu = self.call_auxiliary(h_in, training=training)

      llk = dict(llk_x=px.log_prob(x_u))
      z = tf.convert_to_tensor(qz)
      kl = dict(kl_z=self.beta * (qz.log_prob(z) - qu.log_prob(z)))

      loss = tf.reduce_mean(-self.elbo(llk, kl))
      return loss, {k: tf.reduce_mean(v) for k, v in dict(**llk, **kl).items()}

    yield TrainStep(parameters=self.vae1_params, func=elbo1)

    def elbo2():
      (px, py), qz = self(x_u, training=training)

      h_out = self.encoder.last_outputs[2][1]
      h_in = self.decoder.last_outputs[1][1]
      ph, qu = self.call_auxiliary(h_in, training=training)

      llk = dict(llk_u=ph.log_prob(self.flatten(h_out)))
      kl = dict(kl_u=qu.KL_divergence())
      loss = tf.reduce_mean(-self.elbo(llk, kl))
      return loss, {k: tf.reduce_mean(v) for k, v in dict(**llk, **kl).items()}

    yield TrainStep(parameters=self.vae2_params, func=elbo2)


class Semafo2(SemafoBase):

  def __init__(self, n_iw: int = 10, **kwargs):
    super(Semafo2, self).__init__(**kwargs)
    self.n_iw = int(n_iw)

  def build(self, input_shape=None):
    super(Semafo2, self).build(input_shape)
    self.encoder2 = networks(self.ydim, name='Encoder2')
    self.decoder2 = networks(self.zdim, name='Decoder2')
    self.latents2 = MVNDiagLatents(self.zdim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.observation2 = DistributionDense(**self.labels.get_config())
    self.observation2(self.decoder2.output)

    self.params2 = self.encoder2.trainable_variables + \
                   self.decoder2.trainable_variables + \
                   self.latents2.trainable_variables + \
                   self.observation2.trainable_variables
    params2_ids = set([id(i) for i in self.params2])
    self.params1 = [i for i in self.trainable_variables
                    if id(i) not in params2_ids]

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

    def elbo_s():
      (px_z, qy_z), qz_x = self(x_s, training=training)
      py_u, qu_y = self.call_aux(y_s, training=training)

      z = tf.convert_to_tensor(qz_x, dtype=self.dtype)
      llk = dict(
        llk_x=px_z.log_prob(x_s),
        llk_y=10 * py_u.log_prob(y_s))
      kl = dict(
        kl_u=qu_y.KL_divergence(),
        kl_z=qz_x.log_prob(z) - qu_y.log_prob(z))
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables,
                    func=elbo_s)

    def elbo_u():
      (px_z, qy_z), qz_x = self(x_u, training=training)

      # log(q(y|z))
      y_u = tf.convert_to_tensor(qy_z, dtype=self.dtype)
      log_qy_z = qy_z.log_prob(y_u)

      # encode
      qu_y = self.encode_aux(y_u, training=training)
      pu = qu_y.KL_divergence.prior
      # decode
      u_org = qu_y.sample(self.n_iw)
      u = tf.reshape(u_org, (-1, u_org.shape[-1]))
      py_u = self.decode_aux(u, training=training)
      # log(p(y))
      log_py = py_u.log_prob(tf.tile(y_u, (self.n_iw, 1))) + \
               pu.log_prob(u) + \
               tf.reshape(qu_y.log_prob(u_org), [-1])
      log_py = tf.reduce_logsumexp(tf.reshape(log_py, (self.n_iw, -1)), 0) - \
               tf.math.log(tf.constant(self.n_iw, dtype=self.dtype))

      # D(q(z|x)||p(z|y)) ~ D(q(z|x)||q(u|y))
      z = tf.convert_to_tensor(qz_x, dtype=self.dtype)
      llk = dict(
        llk_x=px_z.log_prob(x_u))
      kl = dict(
        kl_y=log_qy_z - log_py,
        kl_z=qz_x.log_prob(z) - qu_y.log_prob(z)
      )
      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables,
                    func=elbo_u)


class Semafo3(SemafoBase):

  def build(self, input_shape=None):
    super(Semafo2, self).build(input_shape)
    self.encoder2 = networks(self.ydim, name='Encoder2')
    self.decoder2 = networks(self.zdim, name='Decoder2')
    self.latents2 = MVNDiagLatents(self.zdim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.observation2 = DistributionDense(**self.labels.get_config())
    self.observation2(self.decoder2.output)

    self.params2 = self.encoder2.trainable_variables + \
                   self.decoder2.trainable_variables + \
                   self.latents2.trainable_variables + \
                   self.observation2.trainable_variables
    params2_ids = set([id(i) for i in self.params2])
    self.params1 = [i for i in self.trainable_variables
                    if id(i) not in params2_ids]


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
  model: BetaGammaVAE = model()
  model.build((None,) + INPUT_SHAPE)
  is_semi = model.is_semi_supervised()
  # === 1. training
  if not args.eval:
    train(model, ds, args,
          label_percent=args.py if is_semi else 0.0,
          oversample_ratio=args.ratio)
  # === 2. evaluation
  else:
    model.load_weights(get_model_path(args), raise_notfound=True, verbose=True)


if __name__ == '__main__':
  set_cfg(root_path='/home/trung/exp/semafo')
  args = get_args(dict(py=0.004, ratio=0.1, it=400000))
  ds = get_dataset(args.ds)
  INPUT_SHAPE = ds.shape
  run_multi(main, args=args)
