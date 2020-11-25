from __future__ import absolute_import, division, print_function

import os
import shutil
from functools import partial

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python import keras
from tqdm import tqdm

from odin import bay
from odin import networks as net
from odin import visual as vs
from odin.backend import interpolation
from odin.bay.vi.autoencoder import RVmeta, VariationalAutoencoder
from odin.training import Trainer
from odin.fuel import AudioFeatureLoader
from odin.utils import ArgController, clean_folder, partialclass

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)

tf.random.set_seed(8)
np.random.seed(8)

args = ArgController(\
).add("--override", "Override trained model", False
      ).parse()

SAVE_PATH = "/tmp/vae_audio"
if os.path.exists(SAVE_PATH) and args.override:
  clean_folder(SAVE_PATH, verbose=True)
if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
MODEL_PATH = os.path.join(SAVE_PATH, 'model')

# ===========================================================================
# Configs
# ===========================================================================
ZDIM = 32
MAX_LENGTH = 48
BUFFER_SIZE = 100
PARALLEL = tf.data.experimental.AUTOTUNE
# GaussianLayer, GammaLayer, NegativeBinomialLayer
# POSTERIOR = partialclass(bay.layers.GammaLayer,
#                          concentration_activation='softplus1',
#                          rate_activation='softplus1')
# POSTERIOR = partialclass(bay.layers.NegativeBinomialLayer,
#                          count_activation='softplus1')
POSTERIOR = bay.layers.GaussianLayer
BETA = interpolation.linear(vmin=0,
                            vmax=100,
                            norm=500,
                            delayOut=50,
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

x_test = next(iter(test))[:16]
input_shape = tf.data.experimental.get_structure(train).shape[1:]

# ===========================================================================
# Create the model
# ===========================================================================
outputs = RVmeta(event_shape=input_shape,
                         posterior='gaus',
                         projection=False,
                         name="Spectrogram")
latents = bay.layers.MultivariateNormalDiagLatent(ZDIM, name="Latents")
encoder = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv1D(
            32, 3, strides=1, padding='same', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv1D(64, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv1D(128, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Flatten()
    ],
    name="Encoder",
)
decoder = keras.Sequential(
    [
        keras.Input(shape=(ZDIM,)),
        keras.layers.Dense(encoder.output_shape[1]),
        keras.layers.Reshape((12, 128)),
        net.Deconv1D(128, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        net.Deconv1D(64, 5, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        net.Deconv1D(32, 3, strides=1, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        net.Deconv1D(input_shape[-1] * outputs.n_parameterization,
                     1,
                     strides=1,
                     padding='same'),
        keras.layers.Flatten()
    ],
    name="Decoder",
)
vae = VariationalAutoencoder(encoder=encoder,
                             decoder=decoder,
                             latents=latents,
                             outputs=outputs,
                             path=MODEL_PATH)
print(vae)
z = vae.sample_prior(sample_shape=16)

# ===========================================================================
# Training
# ===========================================================================
z_animation = vs.Animation(figsize=(12, 12))
xmean_animation = vs.Animation(figsize=(12, 12))
xstd_animation = vs.Animation(figsize=(12, 12))


def callback():
  z_animation.plot_spectrogram(vae.decode(z, training=False).mean())
  pX, qZ = vae(x_test, training=False)
  xmean_animation.plot_spectrogram(pX.mean())
  xstd_animation.plot_spectrogram(pX.stddev())


vae.fit(train=train,
        valid=test,
        max_iter=10000,
        valid_freq=500,
        checkpoint=MODEL_PATH,
        callback=callback,
        skip_fitted=True)
vae.plot_learning_curves(os.path.join(SAVE_PATH, 'learning_curves.pdf'),
                         summary_steps=[100, 10])
z_animation.save(os.path.join(SAVE_PATH, 'z_mean.gif'))
xmean_animation.save(os.path.join(SAVE_PATH, 'x_mean.gif'))
xstd_animation.save(os.path.join(SAVE_PATH, 'x_std.gif'))
