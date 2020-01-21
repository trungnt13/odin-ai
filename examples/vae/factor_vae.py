from __future__ import absolute_import, division, print_function

import os
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from scipy.stats import describe
from tensorflow.python import keras
from tqdm import tqdm

from odin import backend as bk
from odin import bay, networks
from odin import visual as vs
from odin.backend import Trainer, interpolation
from odin.utils import ArgController, as_tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

args = ArgController(
).add('-zdim', 'Number of latent dim', 12\
).add('-hdim', 'Number of hidden dim', 1024\
).add('-nlayer', 'Number of hidden layers', 3\
).add('-gamma', 'Gamma >0 turn on FactorVAE', 0\
).add('-beta', 'Beta > 0 BetaVAE', 1\
).parse()
# ===========================================================================
# Configs
# - Color: white
# - Shape: square, ellipse, heart
# - Scale: 6 values linearly spaced in [0.5, 1]
# - Orientation: 40 values in [0, 2pi]
# - Position X: 32 values in [0, 1]
# - Position Y: 32 values in [0, 1]
# ===========================================================================
ZDIM = int(args.zdim)
HDIM = int(args.hdim)
NLAYER = int(args.nlayer)
GAMMA = int(args.gamma)
BETA = int(args.beta)
FACTOR = GAMMA > 0
BATCH_SIZE = 512

SAVE_PATH = '/tmp/%s_vae%d_%d_%d_%d' % \
  ('factor%d' % GAMMA if FACTOR else '', BETA, ZDIM, HDIM, NLAYER)
if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)

BETA = interpolation.linear(vmin=0.,
                            vmax=float(BETA),
                            cyclical=True,
                            norm=100,
                            delayOut=10)


def process(x):
  return tf.cast(x['image'], tf.float32)


prepare = partial(Trainer.prepare,
                  postprocess=process,
                  batch_size=BATCH_SIZE,
                  parallel_postprocess=tf.data.experimental.AUTOTUNE,
                  drop_remainder=True)

# ===========================================================================
# Loading data
# ===========================================================================
train, valid, test = tfds.load(
    'dsprites:2.0.0', split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'])
input_spec = tf.data.experimental.get_structure(train)['image']
input_shape = input_spec.shape
input_dtype = input_spec.dtype

x_valid = tf.concat([
    tf.expand_dims(tf.cast(x['image'], tf.float32), axis=0)
    for x in valid.take(16)
],
                    axis=0)

vs.plot_images(np.squeeze(x_valid.numpy(), axis=-1),
               fig=plt.figure(figsize=(12, 12)))
vs.plot_save(os.path.join(SAVE_PATH, 'x.pdf'))

# ===========================================================================
# Create Model
# ===========================================================================
discriminator = networks.DenseNetwork(units=[HDIM] * NLAYER,
                                      activation='relu',
                                      batchnorm=True,
                                      flatten=True,
                                      end_layers=[keras.layers.Dense(2)],
                                      input_shape=(ZDIM,))


class FactorVAE(keras.Model):

  def __init__(self, input_shape, zdim, hdim, nlayer, posterior='bernoulli'):
    super().__init__()
    self.encoder = networks.DenseNetwork(units=[hdim] * nlayer,
                                         activation='relu',
                                         batchnorm=True,
                                         flatten=True,
                                         input_shape=input_shape)
    self.qZ = bay.layers.DenseDistribution(
        event_shape=zdim,
        posterior='normaldiag',
        prior=tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(shape=(zdim,)), scale_identity_multiplier=1.))
    self.decoder = self.encoder.transpose(input_shape=self.qZ.event_shape,
                                          tied_weights=False)
    self.pX = bay.layers.DenseDistribution(event_shape=input_shape,
                                           posterior=posterior)

  def sample(self, sample_shape=(), seed=8):
    return self.qZ.sample(sample_shape, seed=seed)

  def generate(self, Z):
    D = self.decoder(Z, training=False)
    return self.pX(D, training=False)

  def call(self, inputs, training=None):
    E = self.encoder(inputs, training=training)
    qZ = self.qZ(E, training=training)
    Z = tf.squeeze(qZ, axis=0)  # remove mcmc dimension
    D = self.decoder(Z, training=training)
    pX = self.pX(D, training=training)
    return pX, qZ


# ===========================================================================
# Create the network
# ===========================================================================
vae = FactorVAE(input_shape, zdim=ZDIM, hdim=HDIM, nlayer=NLAYER)
opt = tf.optimizers.Adam(learning_rate=0.001,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=1e-07,
                         amsgrad=False)
z_samples = vae.sample(16)
z_animation = vs.Animation(figsize=(12, 12))
x_animation = vs.Animation(figsize=(12, 12))

# trainer = Trainer.restore_checkpoint('/tmp/factor_vae')
# ===========================================================================
# Training
# ===========================================================================
labels = tf.concat([
    tf.zeros((BATCH_SIZE,), dtype=tf.int32),
    tf.ones((BATCH_SIZE,), dtype=tf.int32)
],
                   axis=0)


def permute_dims(z):
  perm = tf.transpose(z)
  for i in tf.range(z.shape[1]):
    z_i = tf.expand_dims(tf.random.shuffle(z[:, i]), axis=0)
    perm = tf.tensor_scatter_nd_update(perm, indices=[[i]], updates=z_i)
  return tf.transpose(perm)


def optimize(X, tape, n_iter, training):
  pX, qZ = vae(X, training=training)
  z = tf.squeeze(qZ, axis=0)
  beta = 1 if not training else BETA(n_iter)

  llk = pX.log_prob(X)
  kl = qZ.KL_divergence(analytic=True)
  if FACTOR:
    dZ = discriminator(z, training=training)
    total_correlation = tf.expand_dims(dZ[:, 1] - dZ[:, 0], -1)
  else:
    total_correlation = 0
  # ELBO = tf.reduce_mean(tf.reduce_logsumexp(llk - beta * kl, axis=0))
  ELBO = tf.reduce_mean(llk - beta * kl + GAMMA * total_correlation)
  loss = -ELBO

  Trainer.apply_gradients(tape, opt, loss, vae)
  # optimize the discriminator
  if tape is not None:
    tape.reset()
  if FACTOR:
    z_perm = permute_dims(z)
    dZ_perm = discriminator(z_perm, training=training)
    d = tf.concat([dZ, dZ_perm], axis=0)
    dtc_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, d))
    Trainer.apply_gradients(tape, opt, dtc_loss, discriminator)
  else:
    dtc_loss = 0.

  return loss, dict(llk=tf.reduce_mean(llk),
                    kl=tf.reduce_mean(kl),
                    tc=tf.reduce_mean(total_correlation),
                    dtc=dtc_loss,
                    beta=beta)


def callback():
  signal = Trainer.early_stop(trainer.valid_loss, threshold=0.25, verbose=1)
  if signal == Trainer.SIGNAL_BEST:
    Trainer.save_checkpoint(os.path.join(SAVE_PATH, 'model'),
                            optimizer=opt,
                            models=vae,
                            trainer=trainer)

  z_animation.plot_images(vae.generate(z_samples).mean())
  z_animation.save(os.path.join(SAVE_PATH, 'z.gif'))

  x_animation.plot_images(vae(x_valid)[0].mean())
  x_animation.save(os.path.join(SAVE_PATH, 'x.gif'))

  trainer.plot_learning_curves(os.path.join(SAVE_PATH, 'learning_curves.pdf'),
                               summary_steps=[100, 10])


trainer = Trainer()
trainer.fit(prepare(train, epochs=20, shuffle=True),
            optimize,
            valid_ds=prepare(valid),
            valid_freq=1000,
            autograph=True,
            logging_interval=2.5,
            callback=callback)
trainer.plot_learning_curves(os.path.join(SAVE_PATH, 'learning_curves.pdf'),
                             summary_steps=[100, 10])
