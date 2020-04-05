from __future__ import absolute_import, division, print_function

import os
from functools import partial

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow.python import keras
from tensorflow_probability.python import distributions as tfd

from odin import visual as vs
from odin.bay import RandomVariable
from odin.bay.vi import autoencoder
from odin.fuel import BinarizedMNIST, YDisentanglement

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

# ===========================================================================
# Load dataset
# ===========================================================================
ds = BinarizedMNIST()
train = ds.create_dataset()
test = ds.create_dataset()
image_shape = ds.shape

######## other configuration
base_depth = 32
latent_size = 16
activation = 'relu'
n_samples = 5
mixture_components = 1


# ===========================================================================
# Helper
# ===========================================================================
def make_encoder_net():
  conv = partial(keras.layers.Conv2D, padding="SAME", activation=activation)

  encoder_net = keras.Sequential([
      conv(base_depth, 5, 1, input_shape=image_shape),
      conv(base_depth, 5, 2),
      conv(2 * base_depth, 5, 1),
      conv(2 * base_depth, 5, 2),
      conv(4 * latent_size, 7, padding="VALID"),
      keras.layers.Flatten(),
      keras.layers.Dense(2 * latent_size, activation=None),
  ],
                                 name="EncoderNet")
  return encoder_net


def make_decoder_net():
  deconv = partial(keras.layers.Conv2DTranspose,
                   padding="SAME",
                   activation=activation)
  conv = partial(keras.layers.Conv2D, padding="SAME", activation=activation)
  # Collapse the sample and batch dimension and convert to rank-4 tensor for
  # use with a convolutional decoder network.
  decoder_net = keras.Sequential([
      keras.layers.Lambda(lambda codes: tf.reshape(codes,
                                                   (-1, 1, 1, latent_size)),
                          batch_input_shape=(n_samples, None, latent_size)),
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(image_shape[-1], 5, activation=None),
      keras.layers.Lambda(lambda logits: tf.reshape(
          logits, tf.concat([(n_samples, -1), image_shape], axis=0))),
  ],
                                 name="DecoderNet")
  return decoder_net


def make_output_dist(decoder, codes):
  logits = decoder(codes)
  return tfd.Independent(tfd.Bernoulli(logits=logits),
                         reinterpreted_batch_ndims=len(image_shape),
                         name="image")


def make_latent_dist(encoder, images):
  images = 2 * tf.cast(images, dtype=tf.float32) - 1
  net = encoder(images)
  return tfd.MultivariateNormalDiag(
      loc=net[..., :latent_size],
      scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                tfp.math.softplus_inverse(1.0)),
      name="code")


def make_mixture_prior():
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(loc=tf.zeros([latent_size]),
                                      scale_identity_multiplier=1.0)

  loc = tf.compat.v1.get_variable(name="loc",
                                  shape=[mixture_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(name="mixture_logits",
                                             shape=[mixture_components])

  return tfd.MixtureSameFamily(
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc, scale_diag=tf.nn.softplus(raw_scale_diag)),
      mixture_distribution=tfd.Categorical(logits=mixture_logits),
      name="prior")


# ===========================================================================
# TFP vae
# ===========================================================================
class TFPVAE(keras.Model):

  def __init__(self):
    super().__init__()
    self.encoder = make_encoder_net()
    self.decoder = make_decoder_net()
    self.latent_prior = make_mixture_prior()

  def call(self, features):
    encoder, decoder, latent_prior = self.encoder, self.decoder, self.latent_prior

    approx_posterior = make_latent_dist(encoder, features)
    approx_posterior_sample = approx_posterior.sample(n_samples)
    decoder_likelihood = make_output_dist(decoder, approx_posterior_sample)

    # `distortion` is just the negative log likelihood.
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(input_tensor=distortion)
    # non-analytic KL
    rate = (approx_posterior.log_prob(approx_posterior_sample) -
            latent_prior.log_prob(approx_posterior_sample))
    avg_rate = tf.reduce_mean(input_tensor=rate)
    # elbo
    elbo_local = -(rate + distortion)
    elbo = tf.reduce_mean(input_tensor=elbo_local)
    loss = -elbo
    # iw
    importance_weighted_elbo = tf.reduce_mean(
        input_tensor=tf.reduce_logsumexp(input_tensor=elbo_local, axis=0) -
        tf.math.log(tf.cast(n_samples, dtype=tf.float32)))
    return (loss, importance_weighted_elbo, avg_distortion, avg_rate,
            approx_posterior, decoder_likelihood)

  def generate(self):
    # Decode samples from the prior for visualization.
    random_image = self.decoder(self.latent_prior.sample(16))
    return random_image


tfp_vae = TFPVAE()
# ===========================================================================
# ODIN vae
# ===========================================================================
odin_vae = autoencoder.BetaVAE(beta=1,
                               encoder=make_encoder_net(),
                               decoder=make_decoder_net(),
                               outputs=RandomVariable(event_shape=image_shape,
                                                      posterior='bernoulli',
                                                      projection=False,
                                                      name='Image'),
                               latents=RandomVariable(event_shape=latent_size,
                                                      posterior='diag',
                                                      projection=False,
                                                      name="Latent"))
x = np.random.rand(20, *image_shape).astype(np.float32)
print(odin_vae(x, n_mcmc=n_samples))
