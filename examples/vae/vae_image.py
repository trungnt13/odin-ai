from __future__ import absolute_import, division, print_function

import os
from functools import partial

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import layers

from odin import backend as bk
from odin import visual as vs
from odin.bay import RandomVariable as RV
from odin.bay.layers import (BernoulliLayer, DiagonalGaussianLatent,
                             GaussianLayer, IndependentGaussianLatent)
from odin.bay.vi.autoencoder import (VariationalAutoencoder,
                                     create_image_autoencoder)
from odin.exp.trainer import Trainer
from odin.networks import ConvNetwork, DenseNetwork
from odin.utils import ArgController

# TODO: improve performance of VAE

sns.set()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)


args = ArgController(
).add('-ds', "1-apple/orange, 2-fashionMNIST, 3-MNIST", 3\
).add('-zdim', "number of latent units", 32\
).parse()

BATCH_SIZE = 128
DATASET = int(args['ds'])
EPOCHS = 128
FREQ = 800
ZDIM = int(args['zdim'])
SUMMARY_STEPS = [500, 100]
output_dir = os.path.join('/tmp', 'vae_z%d_d%s' % (ZDIM, DATASET))
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
print("Input shape", input_shape)

# ===========================================================================
# Create networks
# ===========================================================================
encoder, decoder = create_image_autoencoder(image_shape=input_shape,
                                            latent_shape=ZDIM,
                                            projection_dim=128)
vae = VariationalAutoencoder(
    encoder=encoder,
    decoder=decoder,
    latents=RV(ZDIM, 'diag', projection=True, name="Latent"),
    outputs=RV(input_shape, 'bernoulli', projection=False, name="Image"),
)
print(vae)
optimizer = tf.optimizers.Adam(learning_rate=1e-3,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False)
# ====== image for generation ====== #
z_seed = vae.sample_prior(25)
animation = vs.Animation(figsize=(8, 8))
animation.plot_images(vae.decode(z_seed, training=False))


# ====== training procedures ====== #
def optimize(X, training=None, n_iter=None):
  with tf.GradientTape() as tape:
    pX, qZ = vae(X, training=training)
    KL = qZ.KL_divergence(analytic=True)
    LLK = pX.log_prob(X)
    ELBO = tf.reduce_mean(LLK - KL)
    loss = -ELBO
    if training:
      Trainer.apply_gradients(tape, optimizer, loss, vae)
  return loss, dict(llk=tf.reduce_mean(LLK), kl=tf.reduce_mean(KL))


def callback():
  img = vae.decode(z_seed, training=False)
  animation.plot_images(img)
  animation.save(save_freq=5)
  Trainer.early_stop(trainer.valid_loss, threshold=0.25, verbose=True)


trainer = Trainer()
trainer.fit(prepare(train, shuffle=True, epochs=EPOCHS),
            valid_ds=prepare(valid),
            valid_freq=FREQ,
            optimize=optimize,
            compile_graph=True,
            autograph=True,
            callback=callback)

# ====== save the analysis ====== #
trainer.plot_learning_curves(path=os.path.join(output_dir,
                                               'learning_curves.pdf'),
                             summary_steps=SUMMARY_STEPS)
animation.save(path=os.path.join(output_dir, 'images.gif'))
