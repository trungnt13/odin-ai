from __future__ import absolute_import, division, print_function

import os
from functools import partial

import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.python import keras
from tqdm import tqdm

from odin import backend as bk
from odin import bay, networks
from odin.backend import Interpolation, Trainer
from odin.utils import as_tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

# ===========================================================================
# Loading data
# ===========================================================================
train, valid, test = tfds.load(
    'dsprites:2.0.0', split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'])
input_spec = tf.data.experimental.get_structure(train)['image']
input_shape = input_spec.shape
input_dtype = input_spec.dtype


def deprocess(x):
  x = (x / 2. + 0.5) * 255
  return x


def process(x):
  if isinstance(x, dict):
    x = x['image']
  x = tf.cast(x, 'float32')
  x = (x / 255 - 0.5) * 2.
  return x


N_MCMC = 1
prepare = partial(Trainer.prepare, postprocess=process, batch_size=256)
# ===========================================================================
# Create Model
# ===========================================================================
encoder = networks.DenseNetwork(units=(256, 128, 64),
                                input_shape=input_shape,
                                flatten=True)
latent = bay.layers.DenseDistribution(
    event_shape=10,
    posterior='normaldiag',
    prior=tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(shape=(10,)),
                                                   scale_identity_multiplier=1))
decoder = encoder.transpose(input_shape=latent.event_shape, tied_weights=False)
output = bay.layers.DenseDistribution(event_shape=np.prod(input_shape),
                                      posterior='normal')

# trainer = Trainer.restore_checkpoint('/tmp/vae',
#                                      [encoder, latent, decoder, output])

# ===========================================================================
# Training
# ===========================================================================
opt = tf.optimizers.Adam(learning_rate=0.001,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=1e-07,
                         amsgrad=False)


def optimize(X, tape, n_iter, training):
  E = encoder(X, training=training)
  qZ_X = latent(E, training=training, n_mcmc=N_MCMC)
  Z_X_samples = tf.reshape(qZ_X, shape=(-1, latent.event_size))
  D = decoder(Z_X_samples, training=training)
  D = tf.reshape(D, (N_MCMC, -1, D.shape[-1]))
  pX_Z = output(D, training=training)

  llk = pX_Z.log_prob(bk.flatten(X, outdim=2))
  kl = qZ_X.KL_divergence(analytic=True)
  ELBO = tf.reduce_logsumexp(llk - kl)
  loss = -ELBO

  Trainer.apply_gradients(tape, opt, loss, [encoder, latent, decoder, output])
  return loss, [tf.reduce_mean(llk, name="LLK"), tf.reduce_mean(kl, name="KL")]


def callback():
  signal = Trainer.early_stop(trainer.valid_loss, threshold=0.25, verbose=1)
  if signal == Trainer.SIGNAL_BEST:
    Trainer.save_checkpoint('/tmp/vae',
                            optimizer=opt,
                            models=[encoder, latent, decoder, output],
                            trainer=trainer)
  return signal


trainer = Trainer()
trainer.fit(prepare(train, epochs=32, parallel_postprocess=-1, shuffle=True),
            optimize,
            valid_ds=prepare(valid),
            valid_freq=5000,
            autograph=True,
            logging_interval=2.5,
            callback=callback)
