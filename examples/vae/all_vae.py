from __future__ import absolute_import, division, print_function

import os
import pickle
from functools import partial

import numpy as np
import seaborn as sns
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
from odin.bay.vi import Criticizer, autoencoder
from odin.exp import Experimenter, Trainer
from odin.fuel import get_dataset
from odin.utils import ArgController, as_tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
sns.set()
# ===========================================================================
# Configuration
# TODO: grammarVAE, graphVAE, CycleConsistentVAE, AdaptiveVAE
# vae=betavae,betatcvae,annealedvae,infovae,mutualinfovae,factorvae ds=binarizedmnist,shapes3d -m -ncpu 4
# ===========================================================================
CONFIG = \
r"""
vae: betavae
zdim: 10
zdist: diag
ds: binarizedmnist
conv: False
batch_size: 128
epochs: 200
"""


# ===========================================================================
# Experimenter
# ===========================================================================
class VaeExperimenter(Experimenter):

  def __init__(self):
    super().__init__(save_path='/tmp/vaeexp',
                     config_path=CONFIG,
                     exclude_keys=["epochs", "batch_size"])

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
    # sampled images
    step = int(self.model.step.numpy())
    pX_Z = self.model.decode(self.z_samples, training=False)
    X = pX_Z.mean().numpy()
    if X.shape[-1] == 1:
      X = np.squeeze(X, axis=-1)
    else:
      X = np.transpose(X, (0, 3, 1, 2))
    fig = vs.plot_figure(nrow=16, ncol=16, dpi=80)
    vs.plot_images(X, fig=fig, title="#Iter: %d" % step)
    fig.savefig(os.path.join(self.output_path, 'img_%d.png' % step), dpi=80)
    plt.close(fig)
    del X
    # learning curves
    self.trainer.plot_learning_curves(
        path=os.path.join(self.output_path, 'learning_curves.png'),
        summary_steps=[100, 5],
    )

  ####### Experiementer methods
  def on_load_data(self, cfg):
    dataset = get_dataset(cfg.ds)()
    train = dataset.create_dataset(partition='train', inc_labels=False)
    valid = dataset.create_dataset(partition='valid', inc_labels=False)
    test = dataset.create_dataset(partition='test', inc_labels=False)
    # sample
    x_valid = [x for x in valid.take(1)][0][:16]
    ### input description
    input_spec = tf.data.experimental.get_structure(train)
    fig = plt.figure(figsize=(12, 12))
    for i, x in enumerate(x_valid.numpy()):
      if x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)
      plt.subplot(4, 4, i + 1)
      plt.imshow(x, cmap='Greys_r' if x.ndim == 2 else None)
      plt.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(self.save_path, '%s.pdf' % cfg.ds))
    plt.close(fig)
    ### store
    self.input_dtype = input_spec.dtype
    self.input_shape = input_spec.shape[1:]
    self.train, self.valid, self.test = train, valid, test

  def on_create_model(self, cfg):
    x_rv = bay.RandomVariable(
        event_shape=self.input_shape,
        posterior='bernoulli',
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
    self.z_samples = self.model.sample_prior(16, seed=1)
    self.criticizer = Criticizer(self.model, random_state=1)
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
    assert model_path == self.get_model_path(cfg)
    self.output_path = self.get_output_path(cfg)
    self.trainer = Trainer()
    # just save the first image
    self.callback()
    # we use generator here so turn-off autograph
    # valid_interval will ensure the same frequency for different dataset
    self.trainer.fit(self.train.repeat(cfg.epochs),
                     optimize=self.optimize,
                     valid_ds=self.valid,
                     valid_freq=1,
                     valid_interval=30,
                     logging_interval=2,
                     callback=self.callback,
                     compile_graph=True,
                     autograph=False)


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  exp = VaeExperimenter()
  exp.run()
