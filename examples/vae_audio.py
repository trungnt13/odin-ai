from __future__ import absolute_import, division, print_function

import os
from functools import partial

import numpy as np
import tensorflow as tf
from librosa import magphase, stft
from librosa.core import amplitude_to_db
from librosa.display import specshow
from matplotlib import pyplot as plt
from tensorflow.python import keras
from tqdm import tqdm

from odin import bay
from odin import networks as net
from odin import visual as vs
from odin.backend import Interpolation, Trainer
from odin.fuel import AudioFeatureLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

# ===========================================================================
# Configs
# ===========================================================================
MAX_LENGTH = 48
BUFFER_SIZE = 100
PARALLEL = tf.data.experimental.AUTOTUNE
# GaussianLayer, GammaLayer, NegativeBinomialLayer
POSTERIOR = bay.layers.NegativeBinomialLayer
BETA = partial(Interpolation.linear,
               vmin=0,
               vmax=100,
               norm=500,
               delay=50,
               cyclical=True)

# ===========================================================================
# Load the data
# ===========================================================================
audio = AudioFeatureLoader()
train, test = audio.load_fsdd()
train = audio.create_dataset(train,
                             return_path=False,
                             max_length=MAX_LENGTH,
                             cache='/tmp/fsdd_train.cache',
                             shuffle=BUFFER_SIZE,
                             parallel=PARALLEL,
                             prefetch=-1)
test = audio.create_dataset(test,
                            return_path=False,
                            max_length=MAX_LENGTH,
                            cache='/tmp/fsdd_test.cache',
                            shuffle=BUFFER_SIZE,
                            parallel=PARALLEL,
                            prefetch=-1)

for _ in test.repeat(1):
  pass
x_test = _[:16]


# ===========================================================================
# Create the model
# ===========================================================================
class VAE(keras.Model):

  def __init__(self, input_shape, zdim=16, posterior=bay.layers.NormalLayer):
    super().__init__()
    self.zdim = zdim
    self.encoder = keras.Sequential([
        keras.layers.Conv1D(32,
                            3,
                            strides=1,
                            padding='same',
                            input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv1D(64, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv1D(80, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Flatten()
    ])
    self.latent = bay.layers.DiagonalGaussianLatent(zdim)
    self.decoder = keras.Sequential([
        keras.layers.Dense(self.encoder.output_shape[1], input_shape=(zdim,)),
        keras.layers.Reshape((12, 80)),
        net.Deconv1D(80, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        net.Deconv1D(64, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        net.Deconv1D(32, 3, strides=1, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        net.Deconv1D(input_shape[-1] * posterior.params_size(1),
                     1,
                     strides=1,
                     padding='same'),
        keras.layers.Flatten()
    ])
    self.pX = posterior(event_shape=input_shape)

  def sample(self, n=1, seed=8):
    return self.latent.sample(sample_shape=n, seed=seed)

  # @tf.function
  def generate(self, z):
    assert z.shape[-1] == self.zdim
    D = self.decoder(z, training=False)
    X = self.pX(D)
    return X

  def call(self, x, training=None):
    e = self.encoder(x, training=training)
    qZ_X = self.latent(e, training=training)
    Z = tf.squeeze(qZ_X, axis=0)
    D = self.decoder(Z)
    pX = self.pX(D, training=training)
    return pX, qZ_X


vae = VAE(tf.data.experimental.get_structure(train).shape[1:],
          posterior=POSTERIOR)
z = vae.sample(n=16)
optimizer = tf.optimizers.Adam(learning_rate=0.001,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False)


# ===========================================================================
# Training
# ===========================================================================
def optimize(X, tape=None, training=None, n_iter=None):
  beta = BETA(trainer.n_iter)
  pX, qZ = vae(X, training=training)
  KL = qZ.KL_divergence(analytic=True)
  LLK = pX.log_prob(X)
  # ELBO = tf.reduce_mean(tf.reduce_logsumexp(LLK - KL, axis=0))
  ELBO = tf.reduce_mean(LLK - beta * KL, axis=0)
  loss = -ELBO

  Trainer.apply_gradients(tape, optimizer, loss, vae)
  return loss, dict(llk=tf.reduce_mean(LLK), kl=tf.reduce_mean(KL), beta=beta)


z_animation = vs.Animation(figsize=(12, 12))
xmean_animation = vs.Animation(figsize=(12, 12))
xstd_animation = vs.Animation(figsize=(12, 12))


def callback():
  z_animation.plot_spectrogram(vae.generate(z).mean())
  pX = vae(x_test, training=False)[0]
  xmean_animation.plot_spectrogram(pX.mean())
  xstd_animation.plot_spectrogram(pX.stddev())


trainer = Trainer()
trainer.fit(ds=train.repeat(800),
            valid_ds=test,
            valid_freq=500,
            optimize=optimize,
            callback=callback)
trainer.plot_learning_curves('/tmp/tmp.pdf', summary_steps=[100, 5])
z_animation.save('/tmp/tmp_z_mean.gif')
xmean_animation.save('/tmp/tmp_x_mean.gif')
xstd_animation.save('/tmp/tmp_x_std.gif')
