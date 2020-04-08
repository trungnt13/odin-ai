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
from odin.bay.layers import (BernoulliLayer, DiagonalGaussianLatent,
                             GaussianLayer, IndependentGaussianLatent)
from odin.networks import ConvNetwork, DenseNetwork
from odin.utils import ArgController

# TODO: improve performance of VAE

sns.set()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)


args = ArgController(
).add('-ds', "1-apple/orange, 2-fashionMNIST, 3-MNIST", 1\
).add('-zdim', "number of latent units", 32\
).add('--conv', "Use convolutional neural network", False\
).parse()

BATCH_SIZE = 256
DATASET = int(args['ds'])
CONV = bool(args['conv'])
EPOCHS = 128
FREQ = 800
ZDIM = int(args['zdim'])
SUMMARY_STEPS = [500, 100]
output_dir = os.path.join(
    '/tmp', 'vae_z%d_d%s_%s' % (ZDIM, DATASET, 'conv' if CONV else 'dense'))
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
print("Output directory:", output_dir)

# ===========================================================================
# Load dataset and helpers
# ===========================================================================
# range from [-1, 1]
process = lambda data: (tf.cast(data['image'], tf.float32) / 255. - 0.5) * 2.
deprocess = lambda x: tf.cast((x / 2. + 0.5) * 255., tf.uint8)

# ====== Cycle-GAN/apple_orange ====== #
if DATASET in (1, 2):
  if DATASET == 1:
    name = 'apple2orange'
  elif DATASET == 2:
    name = 'vangogh2photo'
  trainA, trainB, validA, validB, testA, testB = \
   tfds.load('cycle_gan/%s:2.0.0' % name,
             split=['trainA[:90%]', 'trainB[:90%]',
                    'trainA[90%:]', 'trainB[90%:]',
                    'testA', 'testB'])
  train = trainA.concatenate(trainB)
  valid = validA.concatenate(validB)
  test = testA.concatenate(testB)
  input_shape = (48, 48, 3)
  Posterior = GaussianLayer

  process = lambda data: (tf.cast(
      tf.image.resize(data['image'], input_shape[:2]), tf.float32) / 255. - 0.5
                         ) * 2.
  EPOCHS = 512
  FREQ = 120
  SUMMARY_STEPS = [10, 2]
# ====== Fashion-MNIST ====== #
elif DATASET == 3:
  train, valid, test = tfds.load('fashion_mnist:3.0.0',
                                 split=['train[:80%]', 'train[80%:]', 'test'])
  input_shape = tf.data.experimental.get_structure(train)['image'].shape
  Posterior = GaussianLayer
# ====== Binarized-MNIST ====== #
else:
  dataset = tfds.load('binarized_mnist:1.0.0')
  train = dataset['train']
  valid = dataset['validation']
  test = dataset['test']
  input_shape = tf.data.experimental.get_structure(train)['image'].shape
  Posterior = BernoulliLayer
  process = lambda x: tf.cast(x['image'], tf.float32)
  deprocess = lambda x: x

prepare = partial(Trainer.prepare,
                  postprocess=process,
                  batch_size=BATCH_SIZE,
                  parallel_postprocess=tf.data.experimental.AUTOTUNE)


# ===========================================================================
# Create networks
# ===========================================================================
class VAE(keras.Model):

  def __init__(self, input_shape, zdim, use_conv=True):
    super().__init__()
    if use_conv:
      self.encoder = ConvNetwork(filters=[32, 64, 80],
                                 input_shape=input_shape,
                                 kernel_size=[3, 5, 5],
                                 strides=[1, 2, 2],
                                 extra_layers=[keras.layers.Flatten()])
      self.decoder = keras.Sequential([
          self.encoder.transpose(input_shape=(zdim,)),
          keras.layers.Conv2D(filters=input_shape[-1] *
                              int(Posterior.params_size(1)),
                              kernel_size=1,
                              use_bias=False,
                              activation='linear'),
          keras.layers.Flatten()
      ])
    else:
      self.encoder = DenseNetwork(units=[1024, 512, 256, 128],
                                  input_shape=input_shape,
                                  flatten=True)
      self.decoder = keras.Sequential([
          self.encoder.transpose(input_shape=(zdim,)),
          keras.layers.Dense(units=Posterior.params_size(input_shape),
                             use_bias=False,
                             activation='linear')
      ])
    # probabilistic layers
    self.qZ_X = DiagonalGaussianLatent(units=zdim)
    self.pX_Z = Posterior(event_shape=input_shape)

  def sample(self, sample_shape=(), seed=8):
    return self.qZ_X.sample(sample_shape=sample_shape, seed=seed)

  @tf.function
  def generate(self, Z):
    D = self.decoder(Z, training=False)
    pZ = self.pX_Z(D)
    img = deprocess(pZ.mean())
    return img

  def call(self, inputs, training=None, sample_shape=1):
    E = self.encoder(inputs, training=training)
    qZ = self.qZ_X(E, training=training, sample_shape=sample_shape)
    Z = tf.reshape(qZ, (-1, qZ.shape[-1]))
    D = self.decoder(Z, training=training)
    output_shape = tf.concat([(sample_shape, -1), tf.shape(D)[1:]], 0)
    D = tf.reshape(D, output_shape)
    pX = self.pX_Z(D)
    return pX, qZ


# ====== create the vae and its optimizer ====== #
vae = VAE(input_shape, ZDIM, use_conv=CONV)
optimizer = tf.optimizers.Adam(learning_rate=0.001,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False)
# ====== image for generation ====== #
z_seed = vae.sample(25)
animation = vs.Animation(figsize=(8, 8))
animation.plot_images(vae.generate(z_seed))


# ====== training procedures ====== #
def optimize(X, tape=None, training=None, n_iter=None):
  pX, qZ = vae(X, training=training)
  KL = qZ.KL_divergence(analytic=True)
  LLK = pX.log_prob(X)
  ELBO = tf.reduce_mean(tf.reduce_logsumexp(LLK - KL, axis=0))
  loss = -ELBO

  Trainer.apply_gradients(tape, optimizer, loss, vae)
  return loss, dict(llk=tf.reduce_mean(LLK), kl=tf.reduce_mean(KL))


def callback():
  img = vae.generate(z_seed)
  animation.plot_images(img)
  animation.save(save_freq=5)
  Trainer.early_stop(trainer.valid_loss, threshold=0.25, verbose=True)


trainer = Trainer()
trainer.fit(ds=prepare(train, shuffle=True, epochs=EPOCHS),
            valid_ds=prepare(valid),
            valid_freq=FREQ,
            optimize=optimize,
            autograph=True,
            callback=callback)

# ====== save the analysis ====== #
trainer.plot_learning_curves(path=os.path.join(output_dir,
                                               'learning_curves.pdf'),
                             summary_steps=SUMMARY_STEPS)
animation.save(path=os.path.join(output_dir, 'images.gif'))
