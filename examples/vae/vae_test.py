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
from odin.exp import Trainer
from odin.fuel import BinarizedMNIST, YDisentanglement

sns.set()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

# ===========================================================================
# Load dataset
# ===========================================================================
ds = BinarizedMNIST()
train = ds.create_dataset(batch_size=64)
test = ds.create_dataset(batch_size=64)
image_shape = ds.shape

######## other configuration
save_path = "/tmp/vae_test"
base_depth = 32
latent_size = 16
activation = 'relu'
n_samples = 16
mixture_components = 1
epochs = 20
z_prior = tfd.MultivariateNormalDiag(loc=[0.] * latent_size,
                                     scale_identity_multiplier=1.).sample(16)

# delete existed files
if not os.path.exists(save_path):
  os.makedirs(save_path)
for i in os.listdir(save_path):
  os.remove(os.path.join(save_path, i))


######## helper function
def save_figures(name, model, trainer):
  fig = plt.figure(figsize=(8, 8), dpi=60)
  for i, img in enumerate(model.decode(z_prior).mean().numpy()):
    img = np.squeeze(img, axis=-1)
    ax = plt.subplot(4, 4, i + 1)
    ax.imshow(img, cmap="gray")
    ax.axis('off')
  fig.tight_layout()
  fig.savefig(os.path.join(save_path,
                           "%s_%d.png" % (name, trainer.n_iter.numpy())),
              dpi=60)
  plt.close(fig)
  trainer.plot_learning_curves(path=os.path.join(save_path, name + ".pdf"),
                               summary_steps=[100, 10])


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
                          batch_input_shape=((n_samples, None, latent_size))),
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(image_shape[-1], 5, activation=None),
  ],
                                 name="DecoderNet")
  return decoder_net


def make_output_dist(decoder, codes):
  logits = decoder(codes)
  logits = tf.reshape(logits, tf.concat([(n_samples, -1), image_shape], axis=0))
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

  def decode(self, z):
    logits = self.decoder(z)
    return tfd.Independent(tfd.Bernoulli(logits=logits),
                           reinterpreted_batch_ndims=len(image_shape),
                           name="image")


tfp_vae = TFPVAE()
# ===========================================================================
# ODIN vae
# ===========================================================================
encoder, decoder = autoencoder.create_mnist_autoencoder(latent_dim=latent_size,
                                                        base_depth=base_depth,
                                                        activation=activation)
outputs = RandomVariable(event_shape=image_shape,
                         posterior='bernoulli',
                         projection=False,
                         name='Image')
latents = RandomVariable(event_shape=latent_size,
                         posterior='diag',
                         projection=False,
                         name="Latent")
odin_vae = autoencoder.BetaVAE(beta=1,
                               encoder=encoder,
                               decoder=decoder,
                               outputs=outputs,
                               latents=latents)

# ===========================================================================
# Test fit method
# ===========================================================================
odin_fit = autoencoder.BetaVAE(beta=1,
                               encoder=encoder,
                               decoder=decoder,
                               outputs=outputs,
                               latents=latents)


def fit_callback():
  save_figures("autofit", odin_fit, odin_fit.trainer)


odin_fit.fit(train,
             test,
             epochs=epochs,
             n_mcmc=n_samples,
             valid_interval=60,
             callback=fit_callback)


# ===========================================================================
# Training
# ===========================================================================
def train_model(model):
  trainer = Trainer()
  name = type(model).__name__.lower()
  optimizer = tf.optimizers.Adam(learning_rate=0.001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-07,
                                 amsgrad=False)

  def callback():
    save_figures(name, model, trainer)

  def get_loss(inputs):
    if isinstance(model, TFPVAE):
      loss, _, llk, div = model(inputs)[:4]
      llk = -llk
    else:
      pX_Z, qZ_X = model(inputs, n_mcmc=n_samples)
      components = model.elbo(inputs, pX_Z, qZ_X, return_components=True)
      llk = tf.constant(0., dtype=model.dtype)
      div = tf.constant(0., dtype=model.dtype)
      for x in components[0].values():
        llk += x
      for x in components[1].values():
        div += x
      loss = -tf.reduce_mean(llk - div)
      llk = tf.reduce_mean(llk)
      div = tf.reduce_mean(div)
    return loss, dict(llk=llk, kl=div)

  def optimize(inputs, tape, training, n_iter):
    loss, metrics = get_loss(inputs)
    Trainer.apply_gradients(tape, optimizer, loss, model)
    return loss, metrics

  trainer.fit(train.repeat(epochs),
              valid_ds=test,
              valid_interval=60,
              optimize=optimize,
              compile_graph=True,
              callback=callback)


train_model(odin_vae)
train_model(tfp_vae)
