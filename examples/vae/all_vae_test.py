from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
from functools import partial

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from odin import backend as bk
from odin import visual as vs
from odin.backend import interpolation
from odin.bay.vi import (GroundTruth, NetworkConfig, RandomVariable,
                         VariationalAutoencoder, VariationalPosterior, get_vae)
from odin.exp import get_current_trainer, get_output_dir, run_hydra
from odin.fuel import IterableDataset, get_dataset
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
# python all_vae_test.py vae=betavae ds=dsprites beta=1,10,20 max_iter=100000 -m -j4
# ===========================================================================
CONFIG = \
r"""
vae:
ds:
px:
beta: 1
gamma: 6
alpha: 10
zdim: 64
py: onehot
qz: diag
batch_size: 128
max_iter: 25000
override: False
"""


# ===========================================================================
# Helpers
# ===========================================================================
def load_data(name: str, batch_size: int):
  dataset = get_dataset(name)()
  assert dataset.has_labels, f'No labels for given dataset {name}'
  kw = dict(batch_size=batch_size, drop_remainder=True)
  test_l = dataset.create_dataset(partition='test', inc_labels=1.0, **kw)
  sample_images, y = [(x[:16], y[:16]) for x, y in test_l.take(1)][0]
  # inputs structure
  images, labels = tf.data.experimental.get_structure(test_l)
  images_shape = images.shape[1:]
  labels_shape = labels.shape[1:]
  return dataset, (sample_images, y), (images_shape, labels_shape)


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


# ===========================================================================
# Main
# ===========================================================================
@run_hydra(output_dir='/tmp/all_vae', exclude_keys=['max_iter', 'override'])
def main(cfg: dict):
  assert cfg.px is not None, "Output distribution 'px=...' must be given."
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
  ds, (x_samples, y_samples), (x_shape, y_shape) = \
    load_data(name=cfg.ds, batch_size=cfg.batch_size)
  assert ds.has_labels, f"Dataset with name={cfg.ds} has no labels"
  ds_kw = dict(batch_size=int(cfg.batch_size), drop_remainder=True)
  ### the variables
  labels = RandomVariable(y_shape, cfg.py, projection=True, name="Labels")
  latents = RandomVariable(cfg.zdim, cfg.qz, projection=True, name="Latents"),
  observation = RandomVariable(x_shape, cfg.px, projection=True, name="Inputs")
  ### create the model
  model = get_vae(cfg.vae)
  model_kw = inspect.getfullargspec(model.__init__).args[1:]
  kw = {k: v for k, v in cfg.items() if k in model_kw}
  if 'labels' in model_kw:
    kw['labels'] = labels
  vae = model(encoder=NetworkConfig([256, 256, 256], name='Encoder'),
              decoder=NetworkConfig([256, 256, 256], name='Decoder'),
              observation=observation,
              latents=latents,
              path=model_path,
              **kw)
  vae.build((None,) + x_shape)
  vae.load_weights(raise_notfound=False, verbose=True)
  ### prepare evaluation data
  z_samples = vae.sample_prior(sample_shape=16, seed=1)
  if vae.is_semi_supervised:
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
    # posterior
    vp = VariationalPosterior(model=vae,
                              inputs=x_samples,
                              groundtruth=GroundTruth(y_samples),
                              n_samples=1000)
    px = as_tuple(vp.outputs)
    qz = as_tuple(vp.latents)
    # store the histogram
    mean = tf.reduce_mean(qz[0].mean(), axis=0)
    std = tf.reduce_mean(qz[0].stddev(), axis=0)
    # show traverse image
    images = np.concatenate([
        vp.traverse(i, min_val=-3, max_val=3, num=21,
                    mode='linear').outputs[0].mean().numpy()
        for i in np.argsort(std)[:20]
    ])
    image_traverse = to_image(images, grids=(20, int(images.shape[0] / 20)))
    # show sampled image
    px = as_tuple(vae.decode(z_samples, training=False))
    image_sampled = to_image(px[0].mean().numpy(), grids=(4, 4))
    return dict(mean=mean,
                std=std,
                traverse=image_traverse,
                sampled=image_sampled)

  vae.fit(train,
          valid=valid,
          epochs=-1,
          max_iter=int(cfg.max_iter),
          valid_interval=5,
          logging_interval=2,
          skip_fitted=True,
          callback=callback,
          logdir=output_dir,
          compile_graph=True)

  ### evaluation
  evaluate(vae, ds)


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  main(CONFIG)
