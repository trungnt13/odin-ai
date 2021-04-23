import inspect
import os
from argparse import Namespace
from functools import partial
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dropout
from tensorflow_probability.python.distributions import Normal, Distribution, \
  Gamma, Bernoulli, Independent, ContinuousBernoulli, MixtureSameFamily, \
  MultivariateNormalDiag, Categorical

from odin.bay import VariationalModel, VariationalAutoencoder, \
  DistributionDense, BetaVAE, AnnealingVAE, DisentanglementGym, RVconf, \
  BetaCapacityVAE
from odin.bay.vi import Correlation
from odin.fuel import get_dataset, ImageDataset
from odin.networks import get_networks, TrainStep
from utils import *
from matplotlib import pyplot as plt


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


class EquilibriumVAE(BetaVAE):

  def __init__(self, R: float = 0., C: float = 0.,
               dropout: float = 0., beta=1.0, **kwargs):
    kwargs.pop('free_bits', None)
    super().__init__(beta=beta, free_bits=None, **kwargs)
    self.R = float(R)
    self.C = float(C)
    self.dropout = float(dropout)

  def encode(self, inputs, training=None, **kwargs):
    if self.dropout > 0 and training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)
    return super().encode(inputs, training=training, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    llk = {k: v if self.R == 0 else -tf.math.abs(v - self.R)
           for k, v in llk.items()}
    zdim = int(np.prod(self.latents.event_shape))
    C = tf.constant(self.C * zdim, dtype=self.dtype)
    kl = {k: self.beta * tf.math.abs(v - C) for k, v in kl.items()}
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


def model_fullcov(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  zdims = int(np.prod(nets['latents'].event_shape))
  nets['latents'] = RVconf(
    event_shape=zdims,
    projection=True,
    posterior='mvntril',
    prior=Independent(Normal(tf.zeros([zdims]), tf.ones([zdims])), 1),
    name='latents').create_posterior()
  return VariationalAutoencoder(**nets, name='FullCov')


def model_gmmprior(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  latent_size = np.prod(nets['latents'].event_shape)
  n_components = 100
  loc = tf.compat.v1.get_variable(name="loc", shape=[n_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
    name="raw_scale_diag", shape=[n_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(
    name="mixture_logits", shape=[n_components])
  nets['latents'].prior = MixtureSameFamily(
    components_distribution=MultivariateNormalDiag(
      loc=loc,
      scale_diag=tf.nn.softplus(raw_scale_diag) + tf.math.exp(-7.)),
    mixture_distribution=Categorical(logits=mixture_logits),
    name="prior")
  return VariationalAutoencoder(**nets, name='GMMPrior')


def model_fullcovgmm(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  latent_size = int(np.prod(nets['latents'].event_shape))
  n_components = 100
  loc = tf.compat.v1.get_variable(name="loc", shape=[n_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
    name="raw_scale_diag", shape=[n_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(
    name="mixture_logits", shape=[n_components])
  nets['latents'] = RVconf(
    event_shape=latent_size,
    projection=True,
    posterior='mvntril',
    prior=MixtureSameFamily(
      components_distribution=MultivariateNormalDiag(
        loc=loc,
        scale_diag=tf.nn.softplus(raw_scale_diag) + tf.math.exp(-7.)),
      mixture_distribution=Categorical(logits=mixture_logits),
      name="prior"),
    name='latents').create_posterior()
  return VariationalAutoencoder(**nets, name='FullCov')


# === 1.1. Beta-Capacity VAE (Bugress 2018)

# beta=10 C=[0.01, 25]
def model_bcvae1(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaCapacityVAE(**nets, c_min=0.01, c_max=25, gamma=10, n_steps=60000)


# beta=5 C=[0.01, 25]
def model_bcvae2(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaCapacityVAE(**nets, c_min=0.01, c_max=25, gamma=5, n_steps=60000)


# beta = 0.05
def model_bvae1(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=0.05)


# beta = 0.1
def model_bvae2(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=0.1)


# beta = 0.5
def model_bvae3(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=0.5)


# beta = 2
def model_bvae4(args: Namespace):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=2)


# === 2. equilibrium VAE
# beta=1 C=0.5
def model_equilibriumvae1(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=0.5)


# beta=1 C=1
def model_equilibriumvae2(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=1.0)


# beta=1 C=1.5
def model_equilibriumvae3(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=1.5)


# beta=1 C=0.5, dropout=0.3
def model_equilibriumvae4(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=0.5, dropout=0.3)


# beta=5 C=0.5, dropout=0.
def model_equilibriumvae5(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=0.5, dropout=0., beta=5)


# beta=1 R=-50 C=0.5, dropout=0.
def model_equilibriumvae6(args: Namespace):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        R=-50, C=0.5, dropout=0., beta=5)


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
  with gym.run_model(n_samples=-1, partition='test'):
    for i in range(3):
      gym.plot_latents_traverse(n_top_latents=20, title=f'_x{i}', seed=i)
    gym.plot_latents_stats()
    gym.plot_reconstruction_images()
    gym.plot_latents_sampling()
    gym.plot_latents_factors()
    gym.plot_latents_tsne()
    gym.plot_correlation(method='spearman')
    gym.plot_correlation(method='pearson')
  gym.save_figures(get_results_path(args), verbose=True)


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
  run_multi(main)
