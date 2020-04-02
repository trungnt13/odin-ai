from __future__ import absolute_import, division, print_function

import os
import pickle
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from scipy.stats import describe
from tensorflow.python import keras
from tqdm import tqdm

from odin import backend as bk
from odin import bay, networks
from odin import visual as vs
from odin.backend import interpolation
from odin.bay.vi import autoencoder
from odin.exp import Experimenter, Trainer
from odin.fuel import get_dataset
from odin.utils import ArgController, as_tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

# ===========================================================================
# Configuration
# ===========================================================================
CONFIG = \
r"""
vae: betavae
zdim: 10
zdist: diag
ds: binarizedmnist
conv: False
batch_size: 64
epochs: 200
kw: {}
"""


# ===========================================================================
# Experimenter
# ===========================================================================
class VaeExperimenter(Experimenter):

  def __init__(self):
    super().__init__(save_path='/tmp/vaeexp',
                     config_path=CONFIG,
                     exclude_keys=["epochs", "batch_size", "kw"])

  ####### Utility methods
  def optimize(self, inputs, tape, n_iter, training):
    all_metrics = {}
    total_loss = 0.
    for opt, step in zip(self.optimizers, self.model.train_steps(inputs)):
      loss, metrics = step()
      Trainer.apply_gradients(tape, opt, loss, step.parameters)
      # update metrics and loss
      all_metrics.update(metrics)
      total_loss += loss
      # tape need to be reseted for next
      if tape is not None:
        tape.reset()
    return total_loss, {i: tf.reduce_mean(j) for i, j in all_metrics.items()}

  def callback(self):
    pass

  ####### Experiementer methods
  def on_load_data(self, cfg):
    dataset = get_dataset(cfg.ds)()
    self.is_binary = dataset.is_binary
    train = dataset.create_dataset(partition='train', inc_labels=False)
    valid = dataset.create_dataset(partition='valid', inc_labels=False)
    test = dataset.create_dataset(partition='test', inc_labels=False)
    # sample
    x_valid = [x for x in valid.take(1)][0][:16]
    ### input description
    input_spec = tf.data.experimental.get_structure(train)
    plt.figure(figsize=(12, 12))
    for i, x in enumerate(x_valid.numpy()):
      if x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)
      plt.subplot(4, 4, i + 1)
      plt.imshow(x, cmap='Greys_r' if x.ndim == 2 else None)
      plt.axis('off')
    plt.tight_layout()
    vs.plot_save(os.path.join(self.save_path, '%s.pdf' % cfg.ds))
    ### store
    self.input_dtype = input_spec.dtype
    self.input_shape = input_spec.shape[1:]
    self.train, self.valid, self.test = train, valid, test

  def on_create_model(self, cfg):
    x_rv = bay.RandomVariable(
        event_shape=self.input_shape,
        posterior='bernoulli' if self.is_binary else "gaus",
        projection=False,
        name="Image",
    )
    z_rv = bay.RandomVariable(event_shape=cfg.zdim,
                              posterior=cfg.zdist,
                              name="Latent")
    if cfg.vae in ('mutualinfovae',):
      latent_dim = cfg.zdim * 2
    else:
      latent_dim = cfg.zdim
    encoder, decoder = autoencoder.create_image_autoencoder(
        image_shape=self.input_shape[:-1],
        channels=self.input_shape[-1],
        latent_dim=latent_dim,
        conv=cfg.conv,
        distribution=x_rv.posterior,
    )
    self.model = autoencoder.get_vae(cfg.vae)(outputs=x_rv,
                                              latents=z_rv,
                                              encoder=encoder,
                                              decoder=decoder)
    # maximum two optimizer, in case we use factor VAE
    self.optimizers = [
        tf.optimizers.Adam(learning_rate=0.001,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07,
                           amsgrad=False),
        tf.optimizers.Adam(learning_rate=0.001,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07,
                           amsgrad=False),
    ]

  def on_load_model(self, cfg, path):
    self.on_create_model(cfg)

  def on_train(self, cfg, model_path):
    trainer = Trainer()
    # we use generator here so turn-off autograph
    trainer.fit(self.train.repeat(cfg.epochs),
                optimize=self.optimize,
                valid_ds=self.valid,
                valid_freq=2000,
                callback=self.callback,
                compile_graph=True,
                autograph=False)


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  exp = VaeExperimenter()
  exp.run()
