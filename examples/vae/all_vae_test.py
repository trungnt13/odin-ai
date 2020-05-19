from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
from functools import partial

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
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
# vae=betavae,betatcvae,annealedvae,dipvae,infovae,mutualinfovae,factorvae,factor2vae,semifactorvae,semifactor2vae,multitaskvae,multiheadvae
# ds=mnist,dsprites,celeba -m -ncpu 3
# ===========================================================================
CONFIG = \
r"""
vae: betavae
zdim: 10
ds: mnist
batch_size: 64
max_iter: 12000
kwargs: {}
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
class VAEExperimenter(Experimenter):

  def __init__(self):
    super().__init__(save_path='/tmp/vae_exp',
                     config_path=CONFIG,
                     exclude_keys=["batch_size", "max_iter"])

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
    self.dataset = get_dataset(cfg.ds)()
    kw = dict(batch_size=cfg.batch_size, drop_remainder=True)
    self.train_u = self.dataset.create_dataset(partition='train',
                                               inc_labels=False,
                                               **kw)
    self.valid_u = self.dataset.create_dataset(partition='valid',
                                               inc_labels=False,
                                               **kw)
    self.test_u = self.dataset.create_dataset(partition='test',
                                              inc_labels=True,
                                              **kw)
    self.train_l = self.dataset.create_dataset(partition='train',
                                               inc_labels=0.1,
                                               **kw)
    self.valid_l = self.dataset.create_dataset(partition='valid',
                                               inc_labels=1.0,
                                               **kw)
    self.test_l = self.dataset.create_dataset(partition='test',
                                              inc_labels=1.0,
                                              **kw)
    # sample
    self.sample_images, y = [(x[:16], y[:16]) for x, y in self.test_l.take(1)
                            ][0]
    if np.any(np.sum(y, axis=1) > 1):
      if np.any(y > 1):
        self.labels_dist = "nb"  # negative binomial
      else:
        self.labels_dist = "bernoulli"
    else:
      self.labels_dist = "onehot"
    # inputs structure
    images, labels = tf.data.experimental.get_structure(self.train_l)['inputs']
    self.images_shape = images.shape[1:]
    self.labels_shape = labels.shape[1:]

  def on_create_model(self, cfg, model_dir, md5):
    kwargs = dict(
        labels=bay.RandomVariable(self.labels_shape,
                                  self.labels_dist,
                                  True,
                                  name="Labels"),
        factors=bay.RandomVariable(cfg.zdim, 'diag', True, name="Factors"),
    )
    kwargs.update(cfg.kwargs)
    # create the model
    model = autoencoder.get_vae(cfg.vae)
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.getfullargspec(model.__init__).args
    }
    model = model(encoder=cfg.ds,
                  outputs=bay.RandomVariable(self.images_shape,
                                             'bernoulli',
                                             False,
                                             name="Images"),
                  latents=bay.RandomVariable(cfg.zdim,
                                             'diag',
                                             True,
                                             name="Latents"),
                  **kwargs)
    self.model = model
    self.model.load_weights(model_dir)

  def on_train(self, cfg, output_dir, model_dir):
    # self.output_path = self.get_output_path(cfg)
    self.model.fit(
        self.train_l if self.model.is_semi_supervised else self.train_u,
        valid_freq=500,
        epochs=-1,
        max_iter=cfg.max_iter,
        logging_interval=2,
        log_tag=f"[{cfg.vae}, {cfg.ds}]",
        # earlystop_patience=10,
        checkpoint=model_dir)

  def on_eval(self, cfg, output_dir):
    pass
    # self.z_samples = self.model.sample_prior(16, seed=1)
    # self.criticizer = Criticizer(self.model, random_state=1)

  def on_plot(self, cfg, output_dir):
    pass


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  exp = VAEExperimenter()
  exp.run()
