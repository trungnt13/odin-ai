import os
import shutil
from functools import partial
from functools import partial
from typing import Optional, Callable, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense, Flatten, Concatenate

from odin.bay import DistributionDense, MVNDiagLatents, BetaGammaVAE, \
  kl_divergence, DisentanglementGym
from odin.fuel import get_dataset, ImageDataset
from odin.networks import get_networks, SequentialNetwork, TrainStep, Networks
from utils import get_args, train, run_multi, set_cfg, Arguments, \
  get_model_path, get_results_path

# ===========================================================================
# Const and helper
# ===========================================================================
dense = partial(Dense, activation='relu')
INPUT_SHAPE = ()
ds: Optional[ImageDataset] = None
valid_ds: Optional[tf.data.Dataset] = None


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

  def __init__(self,
               coef_llk_py: float = 10.,
               coef_llk_qy: float = 50.,
               n_iw_y: int = 1, **kwargs):
    super().__init__(**kwargs)
    self.n_iw_y = int(n_iw_y)
    self.coef_llk_py = float(coef_llk_py)
    self.coef_llk_qy = float(coef_llk_qy)

  def build(self, input_shape=None):
    super().build(input_shape)

    self.encoder2 = networks(self.ydim, 'Encoder2')
    self.latents2 = MVNDiagLatents(units=self.ydim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.decoder2 = networks(self.latents2.event_size, 'Decoder2')
    self.observation2 = DistributionDense(**self.labels.get_config())
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
        llk_px_sup=self.gamma * px_z.log_prob(x_s),
        llk_py_sup=self.gamma * self.coef_llk_py * py_u.log_prob(y_s),
        llk_qy_sup=self.gamma * self.coef_llk_py * qy_z.log_prob(y_s)
      )

      z = tf.convert_to_tensor(qz_x)
      kl = dict(
        kl_z_sup=self.beta * (qz_x.log_prob(z) - pz_u.log_prob(z)),
        kl_u_sup=self.beta * qu_y.KL_divergence(analytic=self.analytic)
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
        llk_px_uns=self.gamma * px_z.log_prob(x_u),
        llk_py_uns=self.gamma * self.coef_llk_py * py_u.log_prob(y_u)
      )

      z = tf.convert_to_tensor(qz_x)
      kl = dict(
        kl_z_uns=self.beta * (qz_x.log_prob(z) - pz_u.log_prob(z)),
        kl_u_uns=self.beta * qu_y.KL_divergence(analytic=self.analytic),
        llk_qy=self.coef_llk_qy * qy_z.log_prob(y_u)
      )

      return to_elbo(self, llk, kl)

    yield TrainStep(parameters=self.trainable_variables, func=elbo_uns)

  def train_steps1(self, inputs, training=None, mask=None, name='', **kwargs):
    x_u, x_s, y_s = inputs
    yield TrainStep(parameters=self.trainable_variables,
                    func=lambda: (tf.constant(0.), {}))

  def fit(self, *args, **kwargs) -> 'Networks':
    on_batch_end: List[Callable] = kwargs.pop('on_batch_end', lambda: None)

    def switch_stage():
      if self.step.numpy() > 1000:
        pass

    on_batch_end.append(switch_stage)
    return super().fit(on_batch_end=on_batch_end, *args, **kwargs)


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
    with gym.run_model(n_samples=-1, partition='valid'):
      gym.plot_latents_tsne()
      gym.plot_latents_factors()
      gym.plot_latents_stats()
      gym.plot_latents_traverse()
      gym.plot_reconstruction()
      gym.plot_latents_sampling()
    gym.save_figures(path, verbose=True)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  set_cfg(root_path='/home/trung/exp/semafo')
  args = get_args(dict(py=0.004, ratio=0.1, it=400000))
  ds = get_dataset(args.ds)
  valid_ds = ds.create_dataset('valid', label_percent=1.0)
  INPUT_SHAPE = ds.shape
  run_multi(main, args=args)
