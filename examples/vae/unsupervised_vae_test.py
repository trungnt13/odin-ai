import inspect
import os
from argparse import Namespace
from functools import partial
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dropout
from tensorflow_probability.python.distributions import Normal, Distribution, \
  Gamma, Bernoulli, Independent, ContinuousBernoulli

from odin.bay import VariationalModel, VariationalAutoencoder, \
  DistributionDense, BetaVAE, AnnealingVAE, DisentanglementGym
from odin.bay.vi import Correlation
from odin.fuel import get_dataset, ImageDataset
from odin.networks import get_networks
from utils import *


# ===========================================================================
# Classes
# ===========================================================================
def make_normal(p: tf.Tensor) -> Normal:
  loc, scale = tf.split(p, 2, axis=-1)
  scale = tf.math.softplus(scale)
  return Normal(loc=loc, scale=scale)


def make_gamma(p: tf.Tensor) -> Gamma:
  log_rate, concentration = tf.split(p, 2, -1)
  concentration = tf.nn.softplus(concentration) + tf.math.exp(-7.)
  return Gamma(log_rate=log_rate, concentration=concentration)


def make_gaussian_out(p: tf.Tensor,
                      event_shape: Sequence[int]) -> Independent:
  loc, scale = tf.split(p, 2, -1)
  loc = tf.reshape(loc, (-1,) + tuple(event_shape))
  scale = tf.reshape(scale, (-1,) + tuple(event_shape))
  scale = tf.nn.softplus(scale)
  return Independent(Normal(loc=loc, scale=scale), len(event_shape))


class MultiCapacity(BetaVAE):

  def __init__(self, args: Namespace, **kwargs):
    networks = get_networks(args.ds, zdim=args.zdim)
    zdim = args.zdim
    prior = Normal(loc=tf.zeros([zdim]), scale=tf.ones([zdim]))
    latents = [
      DistributionDense(units=zdim * 2, posterior=make_normal, prior=prior,
                        name='latents1'),
      DistributionDense(units=zdim * 2, posterior=make_normal, prior=prior,
                        name='latents2')
    ]
    networks['latents'] = latents
    super().__init__(**networks, **kwargs)

  def encode(self, inputs, training=None, **kwargs):
    h = self.encoder(inputs, training=training)
    return [qz(h, training=training, sample_shape=self.sample_shape)
            for qz in self.latents]

  def decode(self, latents, training=None, **kwargs):
    z = tf.concat(latents, -1)
    h = self.decoder(z, training=training)
    return self.observation(h, training=training)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    px, (qz1, qz2) = self(inputs, training=training)
    llk = dict(llk=px.log_prob(inputs))
    kl1 = tf.reduce_sum(qz1.KL_divergence(analytic=self.analytic), -1)
    kl2 = tf.reduce_sum(qz2.KL_divergence(analytic=self.analytic), -1)
    zdim = int(np.prod(qz1.event_shape))
    C = tf.constant(self.free_bits * zdim, dtype=self.dtype)
    kl = dict(latents1=self.beta * kl1,
              latents2=tf.abs(kl2 - C))
    return llk, kl


class Freebits(BetaVAE):

  def __init__(self, args: Namespace, free_bits=None, beta=1, **kwargs):
    networks = get_networks(args.ds, zdim=args.zdim)
    zdim = args.zdim
    prior = Normal(loc=tf.zeros([zdim]), scale=tf.ones([zdim]))
    networks['latents'] = DistributionDense(units=zdim * 2,
                                            posterior=make_normal, prior=prior,
                                            name=networks['latents'].name)
    super().__init__(free_bits=free_bits, beta=beta, **networks, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    free_bits = self.free_bits
    self.free_bits = None
    llk, kl = super(BetaVAE, self).elbo_components(
      inputs, training=training, mask=mask)
    self.free_bits = free_bits
    kl_new = {}
    for k, v in kl.items():
      if free_bits is not None:
        v = tf.maximum(self.free_bits, v)
      v = tf.reduce_sum(v, axis=-1)
      kl_new[k] = self.beta * v
    return llk, kl_new


class EquilibriumVAE(VariationalAutoencoder):

  def __init__(self, dropout: float = 0., **kwargs):
    super().__init__(**kwargs)
    self.dropout = float(dropout)

  def encode(self, inputs, training=None, **kwargs):
    if self.dropout > 0 and training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)
    return super().encode(inputs, training=training, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    free_bits = self.free_bits
    self.free_bits = None
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    self.free_bits = free_bits
    zdim = int(np.prod(self.latents.event_shape))
    C = tf.constant(self.free_bits * zdim, dtype=self.dtype)
    kl = {k: tf.math.abs(v - C) for k, v in kl.items()}
    # kl = {k: tf.math.sign(v - C) * (v - C) for k, v in kl.items()}
    return llk, kl


class GammaVAE(AnnealingVAE):

  def __init__(self, args: Namespace, **kwargs):
    networks = get_networks(args.ds, zdim=args.zdim)
    zdim = args.zdim
    prior = Gamma(rate=tf.fill([zdim], 0.3),
                  concentration=tf.fill([zdim], 0.3))
    networks['latents'] = DistributionDense(units=zdim * 2,
                                            posterior=make_gamma, prior=prior,
                                            name=networks['latents'].name)
    super().__init__(**networks, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, training=training, mask=mask)
    kl = {k: tf.reduce_sum(v, -1) for k, v in kl.items()}
    return llk, kl


# prior N(0, 2)
class Normal2VAE(Freebits):

  def __init__(self, args: Namespace, free_bits=None, beta=1, **kwargs):
    networks = get_networks(args.ds, zdim=args.zdim)
    zdim = args.zdim
    prior = Normal(loc=tf.zeros([zdim]), scale=tf.fill([zdim], 2.))
    networks['latents'] = DistributionDense(units=zdim * 2,
                                            posterior=make_normal, prior=prior,
                                            name=networks['latents'].name)
    super(Freebits, self).__init__(free_bits=free_bits, beta=beta, **networks,
                                   **kwargs)


class GaussianOut(BetaVAE):

  def __init__(self, args: Namespace, free_bits=None, beta=1., **kwargs):
    networks = get_networks(args.ds, zdim=args.zdim)
    obs: DistributionDense = networks['observation']
    event_shape = obs.event_shape
    obs_new = DistributionDense(
      event_shape=event_shape,
      units=int(np.prod(event_shape)) * 2,
      posterior=partial(make_gaussian_out, event_shape=event_shape),
      name='image')
    networks['observation'] = obs_new
    super().__init__(free_bits=free_bits, beta=beta, **networks,
                     **kwargs)


class Interpolate(BetaVAE):

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    px, qz = self.last_outputs
    z = qz.mean()
    z1 = z[0:1]
    z2 = z[1:2]
    a = tf.expand_dims(tf.linspace(0.01, 0.99, num=10), -1)
    z = z1 * a + (1 - a) * z2
    x = tf.stop_gradient(self.decode(z).mean())
    llk1, kl1 = super().elbo_components(inputs=x,
                                        mask=mask,
                                        training=training)
    for k, v in llk1.items():
      llk[k] = tf.concat([llk[k], v], 0)
    for k, v in kl1.items():
      kl[k] = tf.concat([kl[k], v], 0)
    return llk, kl


# ===========================================================================
# Extra models
# ===========================================================================

# === 1. VAE with free-bits
def model_rvae(args: Namespace):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                reverse=True)


def model_vae1(args: Namespace):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                free_bits=0.5)


def model_vae2(args: Namespace):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                free_bits=1.0)


def model_vae3(args: Namespace):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                free_bits=1.5)


# === 2. equilibrium VAE
# free-bits=0.5
def model_equilibriumvae1(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        free_bits=0.5)


# free-bits=1
def model_equilibriumvae2(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        free_bits=1.0)


# free-bits=1.5
def model_equilibriumvae3(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        free_bits=1.5)


# free-bits=0.5, dropout=0.3
def model_equilibriumvae4(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        free_bits=0.5, dropout=0.3)


# === 3. interpolate
def model_interpolate1(args: Namespace):
  return Interpolate(**get_networks(args.ds, zdim=args.zdim))


# === 4. others

def model_multicapacity(args: Namespace):
  return MultiCapacity(args, beta=5, free_bits=1.0)


# beta=1, free_bits=None
def model_freebits(args: Namespace):
  return Freebits(args, beta=1, free_bits=None)


# beta=1, free_bits=0.5
def model_freebits1(args: Namespace):
  return Freebits(args, beta=1, free_bits=0.5)


# beta=5, free_bits=0.5
def model_freebits2(args: Namespace):
  return Freebits(args, beta=5, free_bits=0.5)


def model_gamma1(args: Namespace):
  return GammaVAE(args)


def model_normal2(args: Namespace):
  return Normal2VAE(args)


def model_gaussianout(args: Namespace):
  return GaussianOut(args)


# ===========================================================================
# Main
# ===========================================================================
def evaluate(model: VariationalModel, ds: ImageDataset, args: Namespace):
  gym = DisentanglementGym(args.ds, model)
  with gym.run_vae():
    pass
  exit()
  (llk_x, llk_y,
   x_org, x_rec,
   y_true, y_pred,
   Qz, Pz) = make_prediction(args, model, 'test', take_count=10)
  (true_ids, true_labels,
   pred_ids, pred_labels) = prepare_labels(y_true, y_pred, args)
  save_figs(args, 'reconstruction',
            plot_reconstruction_images(x_org, x_rec, title='test'))
  save_figs(args, 'samples',
            plot_prior_sampling(model, args))
  for idx, qz in enumerate(Qz):
    z = qz.mean()
    corr_spear = Correlation.Spearman(z, y_true)
    save_figs(args, f'pairs_z{idx}',
              plot_latents_pairs(z, y_true, corr_spear, args))
    z_tsne = tsne_transform(z, args)
    save_figs(args, f'scatter_z{idx}',
              plot_scatter(args, z_tsne, y_true=true_labels,
                           y_pred=pred_labels))


def main(args: Namespace):
  # === 0. set configs
  ds = get_dataset(args.ds)
  # === 1. get model
  model = None
  for k, v in globals().items():
    if inspect.isfunction(v) and 'model_' == k[:6] and \
        k.split('_')[-1] == args.vae:
      model = v(args)
      model.build(ds.full_shape)
      break
  if model is None:
    model = get_model(args, return_dataset=False)
  # === 2. eval
  if args.eval:
    model.load_weights(get_model_path(args), raise_notfound=True, verbose=True)
    evaluate(model, ds, args)
  # === 3. train
  else:
    train(model, ds, args)


if __name__ == '__main__':
  set_cfg(root_path=os.path.expanduser('~/exp/unsupervised'))
  main(get_args())
