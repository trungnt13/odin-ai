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
zdim: 10
ds: binarizedmnist
batch_size: 64
epochs: 200
beta: 100
gamma: 100
vae: betavae
"""


# ===========================================================================
# Experimenter
# ===========================================================================
class VaeExperimenter(Experimenter):

  def __init__(self):
    super().__init__(save_path='/tmp/vaeexp',
                     config_path=CONFIG,
                     exclude_keys=["epochs"])

  ####### Utility methods
  def optimize(self, inputs, tape, n_iter, training):
    pX_Z, qZ_X = self.model(inputs, training=training)
    llk, div = self.model.elbo(inputs, pX_Z, qZ_X)
    llk = tf.expand_dims(llk, axis=0)
    elbo = tf.reduce_mean(llk - div)
    loss = -elbo
    Trainer.apply_gradients(tape, self.optimizer, loss, self.model)
    return loss, dict(llk=tf.reduce_mean(llk), kl=tf.reduce_mean(div))

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
    self.x_rv = bay.RandomVariable(
        event_shape=self.input_shape,
        posterior='bernoulli' if self.is_binary else "gaus",
        name="Image",
    )
    self.z_rv = bay.RandomVariable(event_shape=cfg.zdim,
                                   posterior='diag',
                                   name="Latent")
    self.network = networks.AutoencoderConfig(hidden_dim=64,
                                              nlayers=2,
                                              input_dropout=0.3)

  def on_create_model(self, cfg):
    model = autoencoder.BetaVAE(output=self.x_rv,
                                latent=self.z_rv,
                                config=self.network)
    self.model = model
    self.optimizer = tf.optimizers.Adam(learning_rate=0.001,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-07,
                                        amsgrad=False)

  def on_load_model(self, cfg, path):
    model = autoencoder.BetaVAE(output=self.x_rv,
                                latent=self.z_rv,
                                config=self.network)
    self.model = model

  def on_train(self, cfg, model_path):
    trainer = Trainer()
    trainer.fit(self.train.repeat(cfg.epochs),
                optimize=self.optimize,
                valid_ds=self.valid,
                valid_freq=500,
                callback=self.callback,
                autograph=True)


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  exp = VaeExperimenter()
  exp.run()
