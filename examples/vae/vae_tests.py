from __future__ import absolute_import, division, print_function

import inspect
import os
from functools import partial
from math import sqrt

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from odin import backend as bk
from odin import visual as vs
from odin.backend import interpolation
from odin.bay.vi import (GroundTruth, NetworkConfig, RVmeta,
                         VariationalAutoencoder, VariationalPosterior, get_vae,
                         traverse_dims)
from odin.exp import get_current_trainer, get_output_dir, run_hydra
from odin.fuel import IterableDataset, get_dataset
from odin.ml import fast_tsne, fast_umap
from odin.networks import (celeba_networks, dsprites_networks, mnist_networks,
                           shapes3d_networks, shapes3dsmall_networks)
from odin.utils import ArgController, as_tuple, clear_folder
from tensorflow.python import keras
from tqdm import tqdm

try:
  tf.config.experimental.set_memory_growth(
      tf.config.list_physical_devices('GPU')[0], True)
except IndexError:
  pass
tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)

tf.random.set_seed(8)
np.random.seed(8)
sns.set()

# ===========================================================================
# Configuration
# Example:
# python all_vae_test.py vae=betavae ds=dsprites beta=1,10,20 px=bernoulli py=onehot max_iter=100000 -m -j4
# ===========================================================================
OUTPUT_DIR = '/tmp/vae_tests'
batch_size = 32
n_visual_samples = 16

CONFIG = \
r"""
vae:
ds:
qz: mvndiag
beta: 1
gamma: 1
alpha: 10
lamda: 1
zdim: 32
override: False
skip: False
"""


# ===========================================================================
# Helpers
# ===========================================================================
def load_data(name: str):
  ds = get_dataset(name)()
  test = ds.create_dataset(partition='test',
                           inc_labels=1.0 if ds.has_labels else 0.0)
  samples = [
      [i[:n_visual_samples] for i in tf.nest.flatten(x)] for x in test.take(1)
  ][0]
  if ds.has_labels:
    x_samples, y_samples = samples
  else:
    x_samples = samples[0]
    y_samples = None
  return ds, x_samples, y_samples


def to_image(X, grids):
  if X.shape[-1] == 1:  # grayscale image
    X = np.squeeze(X, axis=-1)
  else:  # color image
    X = np.transpose(X, (0, 3, 1, 2))
  nrows, ncols = grids
  fig = vs.plot_figure(nrows=nrows, ncols=ncols, dpi=100)
  vs.plot_images(X, grids=grids)
  image = vs.plot_to_image(fig)
  return image


def evaluate(vae: VariationalAutoencoder, ds: IterableDataset):
  test_u = ds.create_dataset(batch_size=32,
                             drop_remainder=True,
                             partition='test',
                             inc_labels=False)
  test_l = ds.create_dataset(batch_size=32,
                             drop_remainder=True,
                             partition='test',
                             inc_labels=1.0)


def plot_latent_units(mean, std, w):
  # plot the latents and its weights
  fig = plt.figure(figsize=(6, 4), dpi=200)
  ax = plt.gca()
  l1 = ax.plot(mean,
               label='mean',
               linewidth=1.0,
               marker='o',
               markersize=6,
               color='r',
               alpha=0.5)
  l2 = ax.plot(std,
               label='std',
               linewidth=1.0,
               marker='o',
               markersize=6,
               color='g',
               alpha=0.5)
  l3 = ax.plot(w,
               label='weight',
               linewidth=1.0,
               linestyle='--',
               marker='s',
               markersize=6,
               color='b',
               alpha=0.5)
  ax.grid(True)
  ax.legend()
  return vs.plot_to_image(fig)


# ===========================================================================
# Main
# ===========================================================================
@run_hydra(output_dir=OUTPUT_DIR, exclude_keys=['override'])
def main(cfg: dict):
  assert cfg.vae is not None, \
    ('No VAE model given, select one of the following: '
     f"{', '.join(i.__name__.lower() for i in get_vae())}")
  assert cfg.ds is not None, \
    ('No dataset given, select one of the following: '
     'mnist, dsprites, shapes3d, celeba, cortex, newsgroup20, newsgroup5, ...')
  ### paths
  output_dir = get_output_dir()
  model_path = os.path.join(output_dir, 'model')
  if cfg.override:
    clear_folder(output_dir, verbose=True)
  ### load dataset
  ds, x_samples, y_samples = load_data(name=cfg.ds)
  ds_kw = dict(batch_size=batch_size, drop_remainder=True)
  ### prepare model init
  model = get_vae(cfg.vae)
  model_kw = inspect.getfullargspec(model.__init__).args[1:]
  kw = {k: v for k, v in cfg.items() if k in model_kw}
  is_semi_supervised = ds.has_labels and model.is_semi_supervised()
  ### create the model
  network_kw = dict(qz=cfg.qz,
                    zdim=cfg.zdim,
                    activation=tf.nn.leaky_relu,
                    centerize_image=True,
                    is_semi_supervised=is_semi_supervised,
                    skip_generator=cfg.skip,
                    n_channels=x_samples.shape[-1])
  if 'mnist' in cfg.ds:
    fn_networks = mnist_networks
    max_iter = 30000
  elif 'dsprites' in cfg.ds:
    fn_networks = dsprites_networks
    max_iter = 80000
  elif 'shapes3dsmall' in cfg.ds:
    fn_networks = shapes3dsmall_networks
    max_iter = 120000
  elif 'shapes3d' in cfg.ds:
    fn_networks = shapes3d_networks
    max_iter = 150000
  else:
    raise NotImplementedError(
        f'No predefined networks support for dataset {cfg.ds}')
  vae = model(path=model_path, **fn_networks(**network_kw), **kw)
  vae.build((None,) + x_samples.shape[1:])
  vae.load_weights(raise_notfound=False, verbose=True)
  ### prepare evaluation data
  z_samples = vae.sample_prior(sample_shape=n_visual_samples, seed=1)
  if is_semi_supervised:
    train = ds.create_dataset(partition='train', inc_labels=0.1, **ds_kw)
    valid = ds.create_dataset(partition='valid', inc_labels=1.0, **ds_kw)
  else:
    train = ds.create_dataset(partition='train', inc_labels=0., **ds_kw)
    valid = ds.create_dataset(partition='valid', inc_labels=0., **ds_kw)

  ### fit the network
  def callback():
    losses = get_current_trainer().valid_loss
    if losses[-1] <= np.min(losses):
      vae.save_weights(overwrite=True)
    tracking_metrics = dict()
    # show reconstruction image
    P, Q = vae(x_samples, training=True)
    P = as_tuple(P)  # just in case multiple outputs or latents
    Q = as_tuple(Q)
    image_reconstructed = to_image(P[0].mean().numpy(),
                                   grids=(sqrt(n_visual_samples),
                                          sqrt(n_visual_samples)))
    # latents stats
    mean = tf.reduce_mean(tf.concat([q.mean() for q in Q], axis=-1), axis=0)
    std = tf.reduce_mean(tf.concat([q.stddev() for q in Q], axis=-1), axis=0)
    w_d = vae.decoder.trainable_variables[0]
    if w_d.shape.ndims == 2:
      w_d = tf.reduce_sum(w_d, axis=-1)
    else:
      w_d = tf.reduce_sum(w_d, axis=(0, 1, 2))
    image_latents = plot_latent_units(mean, std, w_d)
    # show sampled image
    P = vae.decode(z_samples, training=False)
    P = as_tuple(P)
    image_sampled = to_image(P[0].mean().numpy(),
                             grids=(sqrt(n_visual_samples),
                                    sqrt(n_visual_samples)))
    # tracking the gradients
    all_grads = [(k, v) for k, v in vae.last_metrics.items() if 'grad/' in k]
    encoder_grad = 0
    decoder_grad = 0
    latents_grad = 0
    if len(all_grads) > 0:
      encoder_grad = sum(v for k, v in all_grads if 'encoder' in k)
      decoder_grad = sum(v for k, v in all_grads if 'decoder' in k)
      latents_grad = sum(v for k, v in all_grads if 'latents' in k)
    # latent traverse
    n_indices = 20
    z = tf.concat([q.mean() for q in Q], axis=-1)
    z = traverse_dims(z,
                      feature_indices=np.argsort(std)[:n_indices],
                      min_val=-3.,
                      max_val=3.,
                      n_traverse_points=21,
                      n_random_samples=1,
                      mode='linear',
                      seed=1)
    # support multiple outputs here
    P = vae.decode(z[0] if len(z) == 1 else z, training=False)
    P = as_tuple(P)
    images = P[0].mean().numpy()
    image_traverse = to_image(images,
                              grids=(n_indices,
                                     int(images.shape[0] / n_indices)))
    # create the return metrics
    tracking_metrics = dict(mean=mean,
                            std=std,
                            w_decode=w_d,
                            encoder_grad=encoder_grad,
                            decoder_grad=decoder_grad,
                            latents_grad=latents_grad,
                            noise_units=np.sum(std > 0.9),
                            reconstructed=image_reconstructed,
                            sampled=image_sampled,
                            latents=image_latents,
                            traverse=image_traverse)
    return tracking_metrics

  ### fit
  vae.fit(train,
          valid=valid,
          epochs=-1,
          clipnorm=100,
          max_iter=max_iter,
          valid_freq=1000,
          logging_interval=2,
          skip_fitted=True,
          callback=callback,
          logdir=output_dir,
          compile_graph=True,
          track_gradients=True)

  ### evaluation
  evaluate(vae, ds)


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  main(CONFIG)
