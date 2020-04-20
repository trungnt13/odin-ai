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
# vae=betavae,betatcvae,annealedvae,infovae,mutualinfovae,factorvae
# ds=mnist,shapes3d,dsprites,legofaces -m -ncpu 2
# ===========================================================================
CONFIG = \
r"""
vae: betavae
latent_size: 10
ds: mnist
sample_shape: 8
batch_size: 64
epochs: 500
max_iter: 12000
"""


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


# ===========================================================================
# Experimenter
# ===========================================================================
class VaeExperimenter(Experimenter):

  def __init__(self):
    super().__init__(save_path='/tmp/vaeexp',
                     config_path=CONFIG,
                     exclude_keys=["epochs", "batch_size", "max_iter"])

  ####### Utility methods
  def callback(self):
    name = type(self.model).__name__
    step = int(self.model.step.numpy())
    # criticizer
    # self.criticizer.sample_batch()
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

  ####### Experiementer methods
  def on_load_data(self, cfg):
    dataset = get_dataset(cfg.ds)()
    train = dataset.create_dataset(partition='train', inc_labels=False)
    valid = dataset.create_dataset(partition='valid', inc_labels=False)
    test = dataset.create_dataset(partition='test', inc_labels=True)
    # sample
    x_valid = [x for x in valid.take(1)][0][:16]
    self.x_test = [xy[0] for xy in test.take(1)][0][:16]
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
    z_rv = bay.RandomVariable(event_shape=cfg.latent_size,
                              posterior='diag',
                              name="Latent")
    if cfg.vae in ('mutualinfovae',):
      latent_size = cfg.latent_size * 2
    else:
      latent_size = cfg.latent_size
    # create the network
    encoder, decoder = autoencoder.create_image_autoencoder(
        image_shape=self.input_shape,
        latent_size=latent_size,
        distribution=x_rv.posterior)
    # create the model and criticizer
    self.model = autoencoder.get_vae(cfg.vae)(outputs=x_rv,
                                              latents=z_rv,
                                              encoder=encoder,
                                              decoder=decoder)
    self.z_samples = self.model.sample_prior(16, seed=1)
    self.criticizer = Criticizer(self.model, random_state=1)

  def on_load_model(self, cfg, path):
    self.on_create_model(cfg)

  def on_train(self, cfg, model_path):
    self.output_path = self.get_output_path(cfg)
    self.model.fit(self.train,
                   self.valid,
                   optimizer=['adam', 'adam'],
                   learning_rate=0.001,
                   epochs=cfg.epochs,
                   max_iter=cfg.max_iter,
                   sample_shape=cfg.sample_shape,
                   valid_interval=30,
                   logging_interval=2,
                   log_tag="[%s,%s]" % (cfg.vae, cfg.ds),
                   callback=self.callback)


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  exp = VaeExperimenter()
  exp.run()
