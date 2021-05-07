import inspect
import os
import shutil
from functools import partial
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Normal, Gamma, \
  Bernoulli, Independent, MixtureSameFamily, \
  MultivariateNormalDiag, Categorical

from odin.bay import VariationalModel, VariationalAutoencoder, \
  DistributionDense, BetaVAE, AnnealingVAE, DisentanglementGym, RVconf, \
  BetaCapacityVAE, BetaGammaVAE
from odin.bay.distributions import QuantizedLogistic
from odin.bay.layers import MixtureNormalLatents
from odin.fuel import get_dataset, ImageDataset
from odin.networks import get_networks
from odin.utils import as_tuple
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

  def __init__(self, args: Arguments, **kwargs):
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

  def __init__(self, args: Arguments, free_bits=None, beta=1, **kwargs):
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
               random_capacity: bool = False,
               dropout: float = 0., beta=1.0, **kwargs):
    kwargs.pop('free_bits', None)
    super().__init__(beta=beta, free_bits=None, **kwargs)
    self.R = float(R)
    self.C = float(C)
    self.dropout = float(dropout)
    self.random_capacity = bool(random_capacity)

  def encode(self, inputs, training=None, **kwargs):
    if self.dropout > 0 and training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)
    return super().encode(inputs, training=training, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    px, qz = self(inputs, training=training, mask=mask)
    # === 1. reconstructed information
    llk = {}
    for p, x in zip(as_tuple(px), as_tuple(inputs)):
      name = p.name.split('_')[1]
      if hasattr(p, 'distribution'):
        p = p.distribution
      if isinstance(p, Bernoulli):
        p = Bernoulli(logits=p.logits)
      elif isinstance(px, Normal):
        p = Normal(loc=p.loc, scale=p.scale)
      elif isinstance(p, QuantizedLogistic):
        p = QuantizedLogistic(loc=p.loc, scale=p.scale,
                              low=p.low, high=p.high,
                              inputs_domain=p.inputs_domain,
                              reinterpreted_batch_ndims=None)
      lk = p.log_prob(x)
      if self.R != 0.:
        lk = tf.minimum(lk, self.R)
      lk = tf.reduce_sum(lk, tf.range(1, x.shape.rank))
      llk[f'llk_{name}'] = lk
    # === 2. latent capacity
    kl = {}
    for q in as_tuple(qz):
      name = q.name.split('_')[1]
      kl_q = q.KL_divergence(analytic=self.analytic)
      if self.C > 0:
        zdim = int(np.prod(q.event_shape))
        C = tf.constant(self.C * zdim, dtype=self.dtype)
        if self.random_capacity:
          C = C * tf.random.uniform(shape=[], minval=0., maxval=1.,
                                    dtype=self.dtype)
        kl_q = tf.math.abs(kl_q - C)
      kl_q = self.beta * kl_q
      kl[f'kl_{name}'] = kl_q
    return llk, kl


class GammaVAE(AnnealingVAE):

  def __init__(self, args: Arguments, **kwargs):
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

  def __init__(self, args: Arguments, free_bits=None, beta=1, **kwargs):
    networks = get_networks(args.ds, zdim=args.zdim)
    zdim = args.zdim
    prior = Normal(loc=tf.zeros([zdim]), scale=tf.fill([zdim], 2.))
    networks['latents'] = DistributionDense(units=zdim * 2,
                                            posterior=make_normal, prior=prior,
                                            name=networks['latents'].name)
    super(Freebits, self).__init__(free_bits=free_bits, beta=beta, **networks,
                                   **kwargs)


class GaussianOut(BetaVAE):

  def __init__(self, args: Arguments, free_bits=None, beta=1., **kwargs):
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


# Gaussian mixture posterior
class GMMVAE(BetaVAE):

  def __init__(self, n_components=10, prior=None, **kwargs):
    latents = kwargs.pop('latents')
    latents = MixtureNormalLatents(units=np.prod(latents.event_shape),
                                   n_components=n_components,
                                   prior=prior,
                                   name=latents.name)
    # p: MixtureSameFamily = latents.prior
    # mix: Categorical = p.mixture_distribution
    # print(mix.probs_parameter())
    # print(p.components_distribution.distribution.loc)
    # print(p.components_distribution.distribution.scale)
    super(GMMVAE, self).__init__(latents=latents, **kwargs)


class IWGammaVAE(VariationalAutoencoder):

  def __init__(self, gamma: float = 2.0, n_iw: int = 10, **kwargs):
    super().__init__(**kwargs)
    self.gamma = gamma
    self.n_iw = n_iw

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, training, mask, **kwargs)
    llk = {k: self.gamma * v for k, v in llk.items()}
    # D(q(z)||p(z))
    ids = tf.range(inputs.shape[0], dtype=tf.int32)
    tf.random.shuffle(ids)
    ids = ids[:self.n_iw]
    x = tf.gather(inputs, ids, axis=0)
    qz_x = self.encode(x, training=training, mask=mask)
    pz = qz_x.KL_divergence.prior
    qx = tf.constant(1. / inputs.shape[0], dtype=self.dtype)
    z = tf.convert_to_tensor(qz_x)
    kl[f'kl_iw'] = tf.reduce_mean(
      qz_x.log_prob(z) - pz.log_prob(z) + tf.math.log(qx), 0)
    return llk, kl


# ===========================================================================
# Extra models
# ===========================================================================
# === 1. VAE with free-bits
def model_rvae(args: Arguments):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                reverse=True)


def model_vae1(args: Arguments):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                free_bits=0.5)


def model_vae2(args: Arguments):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                free_bits=1.0)


def model_vae3(args: Arguments):
  return VariationalAutoencoder(**get_networks(args.ds, zdim=args.zdim),
                                free_bits=1.5)


def model_fullcov(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  zdims = int(np.prod(nets['latents'].event_shape))
  nets['latents'] = RVconf(
    event_shape=zdims,
    projection=True,
    posterior='mvntril',
    prior=Independent(Normal(tf.zeros([zdims]), tf.ones([zdims])), 1),
    name='latents').create_posterior()
  return VariationalAutoencoder(**nets, name='FullCov')


def model_gmmprior(args: Arguments):
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


def model_fullcovgmm(args: Arguments):
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
def model_bcvae1(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaCapacityVAE(**nets, c_min=0.01, c_max=25, gamma=10, n_steps=60000)


# beta=5 C=[0.01, 25]
def model_bcvae2(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaCapacityVAE(**nets, c_min=0.01, c_max=25, gamma=5, n_steps=60000)


# beta = 0.05
def model_bvae1(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=0.05)


# beta = 0.1
def model_bvae2(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=0.1)


# beta = 0.5
def model_bvae3(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=0.5)


# beta = 2
def model_bvae4(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaVAE(**nets, beta=2)


# gamma=2; beta = 1.0
def model_gvae1(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaGammaVAE(**nets, beta=1.0, gamma=2.0)


# gamma=5; beta = 1.0
def model_gvae2(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaGammaVAE(**nets, beta=1.0, gamma=5.0)


# gamma=2.0; beta=2.0
def model_gvae3(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaGammaVAE(**nets, beta=2.0, gamma=2.0)


# gamma=5.0; beta=2.0
def model_gvae4(args: Arguments):
  nets = get_networks(args.ds, zdim=args.zdim)
  return BetaGammaVAE(**nets, beta=2.0, gamma=5.0)


# === 2. equilibrium VAE
# beta=1 C=0.5
def model_equilibriumvae1(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=0.5)


# beta=1 C=1
def model_equilibriumvae2(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=1.0)


# beta=1 C=1.5
def model_equilibriumvae3(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=1.5)


# beta=1 C=0.5, dropout=0.3
def model_equilibriumvae4(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=0.5, dropout=0.3)


# beta=5 C=0.5, dropout=0.
def model_equilibriumvae5(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        C=0.5, dropout=0., beta=5)


# beta=1 R=-0.1 C=0., dropout=0.
def model_equilibriumvae6(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        R=-0.1, C=0., dropout=0., beta=1.)


# beta=1 R=-0.1 C=0.5, dropout=0.
def model_equilibriumvae7(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        R=-0.1, C=0.5, dropout=0., beta=1.)


# beta=1 C=1.0, random_capacity=True
def model_equilibriumvae8(args: Arguments):
  return EquilibriumVAE(**get_networks(args.ds, zdim=args.zdim),
                        R=0.0, C=1.0, random_capacity=True, dropout=0.0,
                        beta=1.)


# === 4. IWGamma
# gamma=1.0, iw=10
def model_iwgamma1(args: Arguments):
  return IWGammaVAE(gamma=1.0, n_iw=10, **get_networks(args.ds))


# gamma=2.0, iw=10
def model_iwgamma2(args: Arguments):
  return IWGammaVAE(gamma=2.0, n_iw=10, **get_networks(args.ds))


# === 5. others
def model_multicapacity(args: Arguments):
  return MultiCapacity(args, beta=5, free_bits=1.0)


# beta=1, free_bits=None
def model_freebits(args: Arguments):
  return Freebits(args, beta=1, free_bits=None)


# beta=1, free_bits=0.5
def model_freebits1(args: Arguments):
  return Freebits(args, beta=1, free_bits=0.5)


# beta=5, free_bits=0.5
def model_freebits2(args: Arguments):
  return Freebits(args, beta=5, free_bits=0.5)


def model_gamma1(args: Arguments):
  return GammaVAE(args)


def model_normal2(args: Arguments):
  return Normal2VAE(args)


def model_gaussianout(args: Arguments):
  return GaussianOut(args)


# n_components = 10
def model_gmmvae1(args: Arguments):
  return GMMVAE(n_components=10, **get_networks(args.ds, zdim=args.zdim))


# n_components = 50
def model_gmmvae2(args: Arguments):
  return GMMVAE(n_components=50, **get_networks(args.ds, zdim=args.zdim))


# n_components = 10, prior=N(0, 1)
def model_gmmvae3(args: Arguments):
  zdim = args.zdim
  prior = Independent(Normal(loc=tf.zeros([zdim]), scale=tf.ones([zdim])), 1)
  return GMMVAE(n_components=10, prior=prior, analytic=False,
                **get_networks(args.ds, zdim=args.zdim))


# ===========================================================================
# Main
# ===========================================================================
def evaluate(model: VariationalModel, ds: ImageDataset, args: Arguments):
  # === 1. prepare path
  path = get_results_path(args)
  if args.override and os.path.exists(path):
    print('Override results at path:', path)
    shutil.rmtree(path)
    os.makedirs(path)
  # === 2. run the Gym
  gym = DisentanglementGym(dataset=args.ds, model=model)
  with gym.run_model(n_samples=-1, partition='test'):
    # should be max here
    stddev = np.max(gym.qz_x[0].stddev(), 0)
    gym.plot_distortion()
    for i in range(3):
      gym.plot_latents_traverse(n_top_latents=20, title=f'_x{i}',
                                max_val=3 * stddev,
                                min_val=-3 * stddev,
                                mode='linear', seed=i)
    gym.plot_latents_stats()
    gym.plot_reconstruction()
    gym.plot_latents_sampling()
    gym.plot_latents_factors()
    gym.plot_latents_tsne()
    gym.plot_correlation(method='spearman')
    gym.plot_correlation(method='pearson')
  gym.save_figures(path, verbose=True)


def main(args: Arguments):
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
  run_multi(main, args=get_args())
