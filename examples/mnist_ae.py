from __future__ import absolute_import, division, print_function

import os
from functools import partial

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow import keras

from odin import backend as bk
from odin import visual as vs
from odin.backend import Trainer
from odin.bay.distribution_layers import BernoulliLayer
from odin.bay.layers import DiagonalGaussianLatent, IndependentGaussianLatent
from odin.networks import ConvNetwork, DenseNetwork

sns.set()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

BATCH_SIZE = 256

# ===========================================================================
# Load dataset and helpers
# ===========================================================================
dataset = tfds.load('binarized_mnist:1.0.0')
train = dataset['train']
valid = dataset['validation']
test = dataset['test']
input_shape = tf.data.experimental.get_structure(train)['image'].shape


def process(x):
  x = tf.cast(x['image'], tf.float32)
  return x


prepare = partial(Trainer.prepare, postprocess=process, batch_size=BATCH_SIZE)


# ===========================================================================
# Create networks
# ===========================================================================
class CVAE(keras.Model):

  def __init__(self, input_shape, zdim):
    super().__init__()
    self.encoder = ConvNetwork(filters=[32, 64],
                               input_shape=input_shape,
                               kernel_size=3,
                               strides=[2, 2],
                               extra_layers=[keras.layers.Flatten()])
    self.qZ_X = DiagonalGaussianLatent(units=zdim)
    self.decoder = self.encoder.transpose(input_shape=self.qZ_X.event_shape)
    self.pX_Z = BernoulliLayer(event_shape=input_shape)

  def sample(self, sample_shape=(), seed=8):
    return self.qZ_X.sample(sample_shape=sample_shape, seed=seed)

  @tf.function
  def generate(self, Z):
    D = self.decoder(Z, training=False)
    return self.pX_Z(D)

  def call(self, inputs, training=None, n_mcmc=1):
    E = self.encoder(inputs, training=training)
    qZ = self.qZ_X(E, training=training, n_mcmc=n_mcmc)
    Z = tf.reshape(qZ, (-1, qZ.shape[-1]))
    D = self.decoder(Z, training=training)
    output_shape = tf.concat([(n_mcmc, -1), tf.shape(D)[1:]], 0)
    D = tf.reshape(D, output_shape)
    pX = self.pX_Z(D)
    return pX, qZ


cvae = CVAE(input_shape, 32)
z_seed = cvae.sample(25)
animation = vs.Animation(figsize=(8, 8))
optimizer = tf.optimizers.Adam(learning_rate=0.001,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False)


def optimize(X, tape=None, training=None, n_iter=None):
  pX, qZ = cvae(X)
  KL = qZ.KL_divergence(analytic=True)
  LLK = pX.log_prob(X)
  ELBO = tf.reduce_mean(tf.reduce_logsumexp(LLK - KL, axis=0))
  loss = -ELBO

  Trainer.apply_gradients(tape, optimizer, loss, cvae)
  return loss, [tf.reduce_mean(LLK), tf.reduce_mean(KL)]


def callback():
  img = cvae.generate(z_seed)
  animation.plot_images(img)
  return Trainer.early_stop(trainer.valid_loss, verbose=True)


trainer = Trainer()
trainer.fit(ds=prepare(train, shuffle=True, epochs=48),
            valid_ds=prepare(valid),
            valid_freq=1200,
            optimize=optimize,
            autograph=True,
            callback=callback)
animation.save()
