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
from odin.bay.vi import (Criticizer, NetworkConfig, RandomVariable,
                         VariationalAutoencoder, get_vae)
from odin.exp import get_output_dir, run_hydra
from odin.fuel import get_dataset
from odin.utils import ArgController, as_tuple
from tensorflow.python import keras
from tqdm import tqdm

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)

tf.random.set_seed(8)
np.random.seed(8)
sns.set()
# ===========================================================================
# Configuration
# TODO: grammarVAE, graphVAE, CycleConsistentVAE, AdaptiveVAE
# vae=betavae,betatcvae,annealedvae,dipvae,infovae,mutualinfovae,factorvae,factor2vae,semifactorvae,semifactor2vae,multitaskvae,multiheadvae
# ds=mnist,dspritesc,celeba -m -ncpu 4
# ===========================================================================
CONFIG = \
r"""
vae:
ds:
zdim: 20
batch_size: 128
max_iter: 12000
px: gaussian
py: onehot
qz: diag
"""


# ===========================================================================
# Helpers
# ===========================================================================
def save_images(pX_Z, name, step, path):
  X = pX_Z.mean().numpy()
  if X.shape[-1] == 1:
    X = np.squeeze(X, axis=-1)
  else:
    X = np.transpose(X, (0, 3, 1, 2))
  fig = vs.plot_figure(nrow=16, ncol=16, dpi=60)
  vs.plot_images(X, fig=fig, title="[%s]#Iter: %d" % (name, step))
  fig.savefig(path, dpi=60)
  plt.close(fig)
  del X


def callback(vae: VariationalAutoencoder):
  name = type(self.model).__name__
  step = int(self.model.step.numpy())
  # reconstructed images
  pX_Z, _ = self.model(self.x_test, training=False)
  save_images(pX_Z, name, step,
              os.path.join(self.output_path, 'reconstruct_%d.png' % step))
  # sampled images
  pX_Z = self.model.decode(self.z_samples, training=False)
  save_images(pX_Z, name, step,
              os.path.join(self.output_path, 'sample_%d.png' % step))
  # learning curves
  self.model.trainer.plot_learning_curves(
      path=os.path.join(self.output_path, 'learning_curves.png'),
      summary_steps=[100, 10],
      dpi=60,
  )


def load_data(name: str, batch_size: int):
  dataset = get_dataset(name)()
  kw = dict(batch_size=batch_size, drop_remainder=True)
  test_l = dataset.create_dataset(partition='test', inc_labels=1.0, **kw)
  sample_images, y = [(x[:16], y[:16]) for x, y in test_l.take(1)][0]
  # inputs structure
  images, labels = tf.data.experimental.get_structure(test_l)
  images_shape = images.shape[1:]
  labels_shape = labels.shape[1:]
  return dataset, (sample_images, y), (images_shape, labels_shape)


# ===========================================================================
# Main
# ===========================================================================
@run_hydra(output_dir='/tmp/all_vae_exp', exclude_keys=['max_iter'])
def main(cfg: dict):
  assert cfg.vae is not None, \
    ('No VAE model given, select one of the following: '
     f"{', '.join(i.__name__.lower() for i in get_vae())}")
  assert cfg.ds is not None, \
    ('No dataset given, select one of the following: '
     'mnist, dsprites, shapes3d, celeba, cortex, newsgroup20, newsgroup5, ...')
  output_dir = get_output_dir()
  ds, (x_samples, y_samples), (x_shape, y_shape) = \
    load_data(name=cfg.ds, batch_size=cfg.batch_size)
  ### create the model
  labels = RandomVariable(y_shape, cfg.py, projection=True, name="Labels")
  latents = RandomVariable(cfg.zdim, cfg.qz, projection=True, name="Latents"),
  inputs = RandomVariable(x_shape, cfg.px, projection=True, name="Inputs")
  # create the model
  model = get_vae(cfg.vae)
  model_kw = inspect.getfullargspec(model.__init__).args[1:]
  kw = {k: v for k, v in cfg.items() if k in model_kw}
  if 'labels' in model_kw:
    kw['labels'] = labels
  model = model(encoder=NetworkConfig([256, 256, 256], name='Encoder'),
                decoder=NetworkConfig([256, 256, 256], name='Decoder'),
                outputs=inputs,
                latents=latents,
                input_shape=x_shape,
                **kw)
  if model.is_semi_supervised:
    train = None
    valid = None
  else:
    train = None
    valid = None
  model.fit(train,
            valid=valid,
            epochs=-1,
            max_iter=int(cfg.max_iter),
            valid_interval=10,
            logging_interval=2,
            log_tag=f"[{cfg.vae}, {cfg.ds}]",
            skip_fitted=True,
            compile_graph=True)


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  main(CONFIG)
