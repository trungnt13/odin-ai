from __future__ import absolute_import, division, print_function

import os
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
zdim: 16
ds: binarizedmnist
batch_size: 64
beta: 100
gamma: 100
"""


# ===========================================================================
# Experimenter
# ===========================================================================
class VaeExperimenter(Experimenter):

  def __init__(self):
    super().__init__(save_path='/tmp/vaeexp', config_path=CONFIG)

  def on_load_data(self, cfg):
    dataset = get_dataset(cfg.ds)()
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
    self.input_shape = input_spec.shape
    self.train, self.valid, self.test = train, valid, test

  def on_create_model(self, cfg):
    pass

  def on_load_model(self, cfg):
    pass

  def on_train(self, cfg, model_path):
    pass


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  exp = VaeExperimenter()
  exp.run()
