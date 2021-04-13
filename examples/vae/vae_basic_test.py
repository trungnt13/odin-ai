from __future__ import absolute_import, division, print_function

import os
import shutil
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from odin import visual as vs
from odin.bay.vi import RVconf, get_vae
from odin.fuel import MNIST, BinarizedMNIST, FashionMNIST, dSprites
from odin.ml import fast_umap
from odin.training import (Trainer, get_current_trainer, get_output_dir,
                           run_hydra)
from odin.utils import ArgController
from tensorflow.python import keras
from tensorflow.python.keras import layers

tfpl = tfp.layers
tfd = tfp.distributions

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)
tf.random.set_seed(1)
np.random.seed(1)

# ===========================================================================
# configs
# ===========================================================================
learning_rate = 1e-3
batch_size = 32
encoded_size = 16
base_depth = 32
max_iter = 50000
SAVE_PATH = "/tmp/vae_basic"

CONFIG = \
"""
ds: mnist
model:
beta: 1
"""

# ===========================================================================
# load data
# ===========================================================================
# he_uniform is better for leaky_relu
conv2D = partial(layers.Conv2D,
                 padding='same',
                 kernel_initializer='he_uniform',
                 activation=tf.nn.leaky_relu)
deconv2D = partial(layers.Conv2DTranspose,
                   padding='same',
                   kernel_initializer='he_uniform',
                   activation=tf.nn.leaky_relu)


def create_encoder(input_shape):
  return [
      layers.InputLayer(input_shape=input_shape),
      conv2D(base_depth, 5, strides=1, name='Encoder0'),
      conv2D(base_depth, 5, strides=2, name='Encoder1'),
      conv2D(2 * base_depth, 5, strides=1, name='Encoder2'),
      conv2D(2 * base_depth, 5, strides=2, name='Encoder3'),
      conv2D(4 * encoded_size, 7, strides=1, padding='valid', name='Encoder4'),
      layers.Flatten(),
      layers.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
                   activation=None,
                   name='Encoder5')
  ]


def create_decoder():
  return [
      layers.InputLayer(input_shape=[encoded_size]),
      layers.Reshape([1, 1, encoded_size]),
      deconv2D(2 * base_depth, 7, strides=1, padding='valid', name='Decoder0'),
      deconv2D(2 * base_depth, 5, strides=1, name='Decoder1'),
      deconv2D(2 * base_depth, 5, strides=2, name='Decoder2'),
      deconv2D(base_depth, 5, strides=1, name='Decoder3'),
      deconv2D(base_depth, 5, strides=2, name='Decoder4'),
      deconv2D(base_depth, 5, strides=1, name='Decoder5'),
      conv2D(1, 5, strides=1, activation=None, name='Decoder6'),
      layers.Flatten()
  ]


# ===========================================================================
# Main
# ===========================================================================
@run_hydra(output_dir=SAVE_PATH)
def main(cfg: dict):
  assert cfg.vae is not None and cfg.beta is not None, \
    f'Invalid arguments: {cfg}'
  if cfg.ds == 'bmnist':
    ds = BinarizedMNIST()
  elif cfg.ds == 'mnist':
    ds = MNIST()
  elif cfg.ds == 'fmnist':
    ds = FashionMNIST()
  else:
    raise NotImplementedError(f'No support for dataset with name={cfg.ds}')
  input_shape = ds.shape
  train = ds.create_dataset(partition='train', batch_size=batch_size)
  valid = ds.create_dataset(partition='valid', batch_size=batch_size)
  x_test, y_test = ds.numpy(partition='test',
                            batch_size=batch_size,
                            shuffle=1000,
                            label_percent=1.0)
  y_test = ds.labels[np.argmax(y_test, axis=-1)]
  ## create the prior and the network
  pz = tfd.Sample(tfd.Normal(loc=0, scale=1), sample_shape=encoded_size)
  z_samples = pz.sample(16)
  encoder = create_encoder(input_shape)
  decoder = create_decoder()
  ## create the model
  # tfp model API
  if cfg.vae == 'tfp':
    encoder.append(tfpl.MultivariateNormalTriL(encoded_size))
    encoder = keras.Sequential(encoder, name='encoder')
    decoder.append(tfpl.IndependentBernoulli(input_shape))
    decoder = keras.Sequential(decoder, name="decoder")
    vae = keras.Model(inputs=encoder.inputs,
                      outputs=[decoder(encoder.outputs[0]), encoder.outputs[0]],
                      name='tfp_vae')
  # odin model API
  else:
    encoder = keras.Sequential(encoder, name='encoder')
    decoder = keras.Sequential(decoder, name="decoder")
    vae = get_vae(cfg.vae)(
        encoder=encoder,
        decoder=decoder,
        # latents=tfpl.MultivariateNormalTriL(encoded_size),
        latents=RVconf(event_shape=(encoded_size,),
                       posterior='mvntril',
                       projection=False,
                       name="latents"),
        observation=RVconf(event_shape=input_shape,
                           posterior="bernoulli",
                           projection=False,
                           name="image"),
        name=f'odin_{cfg.vae}')
  ### training the model
  vae.build(input_shape=(None,) + input_shape)
  params = vae.trainable_variables
  opt = tf.optimizers.Adam(learning_rate=learning_rate)

  def optimize(x, training=None):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      if training:
        tape.watch(params)
      px, qz = vae(x, training=training)
      z = qz._value()
      kl = tf.reduce_mean(qz.log_prob(z) - pz.log_prob(z), axis=-1)
      nll = -tf.reduce_mean(px.log_prob(x), axis=-1)
      loss = nll + cfg.beta * kl
      if training:
        grads = tape.gradient(loss, params)
        grads_params = [(g, p) for g, p in zip(grads, params) if g is not None]
        opt.apply_gradients(grads_params)
        grads = {f'_grad/{p.name}': tf.linalg.norm(g) for p, g in grads_params}
      else:
        grads = dict()
    return loss, dict(nll=nll, kl=kl, **grads)

  def callback():
    trainer = get_current_trainer()
    x, y = x_test[:1000], y_test[:1000]
    px, qz = vae(x, training=False)
    # latents
    qz_mean = tf.reduce_mean(qz.mean(), axis=0)
    qz_std = tf.reduce_mean(qz.stddev(), axis=0)
    w = tf.reduce_sum(decoder.trainable_variables[0], axis=(0, 1, 2))
    # plot the latents and its weights
    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = plt.gca()
    l1 = ax.plot(qz_mean,
                 label='mean',
                 linewidth=1.0,
                 linestyle='--',
                 marker='o',
                 markersize=4,
                 color='r',
                 alpha=0.5)
    l2 = ax.plot(qz_std,
                 label='std',
                 linewidth=1.0,
                 linestyle='--',
                 marker='o',
                 markersize=4,
                 color='g',
                 alpha=0.5)
    ax1 = ax.twinx()
    l3 = ax1.plot(w,
                  label='weight',
                  linewidth=1.0,
                  linestyle='--',
                  marker='o',
                  markersize=4,
                  color='b',
                  alpha=0.5)
    lines = l1 + l2 + l3
    labs = [l.get_label() for l in lines]
    ax.grid(True)
    ax.legend(lines, labs)
    img_qz = vs.plot_to_image(fig)
    # reconstruction
    fig = plt.figure(figsize=(5, 5), dpi=120)
    vs.plot_images(np.squeeze(px.mean().numpy()[:25], axis=-1), grids=(5, 5))
    img_res = vs.plot_to_image(fig)
    # latents
    fig = plt.figure(figsize=(5, 5), dpi=200)
    z = fast_umap(qz.mean().numpy())
    vs.plot_scatter(z, color=y, size=12.0, alpha=0.4)
    img_umap = vs.plot_to_image(fig)
    # gradients
    grads = [(k, v) for k, v in trainer.last_train_metrics.items() if '_grad/' in k]
    encoder_grad = sum(v for k, v in grads if 'Encoder' in k)
    decoder_grad = sum(v for k, v in grads if 'Decoder' in k)
    return dict(reconstruct=img_res,
                umap=img_umap,
                latents=img_qz,
                qz_mean=qz_mean,
                qz_std=qz_std,
                w_decoder=w,
                llk_test=tf.reduce_mean(px.log_prob(x)),
                encoder_grad=encoder_grad,
                decoder_grad=decoder_grad)

  ### Create trainer and fit
  trainer = Trainer(logdir=get_output_dir())
  trainer.fit(train_ds=train.repeat(-1),
              optimize=optimize,
              valid_ds=valid,
              max_iter=max_iter,
              compile_graph=True,
              log_tag=f'{cfg.vae}_{cfg.beta}',
              on_valid_end=callback,
              valid_freq=1000)


if __name__ == "__main__":
  main(CONFIG)
