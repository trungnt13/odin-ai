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
from odin.bay.vi import (DisentanglementGym, GroundTruth, NetConf, RVmeta,
                         VariationalAutoencoder, VariationalPosterior, get_vae,
                         traverse_dims, DimReduce, Correlation)
from odin.fuel import IterableDataset, get_dataset
from odin.ml import fast_tsne, fast_umap
from odin.networks import get_networks, get_optimizer_info
from odin.training import get_output_dir, run_hydra
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
skip: False
eval: False
"""


# ===========================================================================
# Helpers
# ===========================================================================
def create_gym(dsname: str, vae: VariationalAutoencoder) -> DisentanglementGym:
  gym = DisentanglementGym(dataset=dsname, vae=vae)
  cfg = dict(reconstruction=True,
             latents_sampling=True,
             latents_traverse=True,
             latents_stats=True,
             mig_score=True,
             silhouette_score=True,
             adjusted_rand_score=True)
  gym.set_config(track_gradients=True, mode='train', **cfg)
  gym.set_config(track_gradients=False, mode='valid', **cfg)
  gym.set_config(
      correlation_methods=Correlation.Lasso | Correlation.Importance |
      Correlation.Spearman | Correlation.MutualInfo,
      dimension_reduction=DimReduce.UMAP | DimReduce.TSNE | DimReduce.PCA,
      latents_pairs=Correlation.Lasso | Correlation.MutualInfo,
      elbo=True,
      sap_score=True,
      dci_score=True,
      beta_vae=True,
      factor_vae=True,
      adjusted_mutual_info=True,
      normalized_mutual_info=True,
      mode='test',
      **cfg)
  return gym


# ===========================================================================
# Main
# ===========================================================================
@run_hydra(output_dir=OUTPUT_DIR, exclude_keys=['eval'])
def main(cfg: dict):
  assert cfg.vae is not None, \
    ('No VAE model given, select one of the following: '
     f"{', '.join(i.__name__.lower() for i in get_vae())}")
  assert cfg.ds is not None, \
    ('No dataset given, select one of the following: '
     'mnist, dsprites, shapes3d, celeba, cortex, newsgroup20, newsgroup5, ...')
  ### load dataset
  ds = get_dataset(name=cfg.ds)
  ds_kw = dict(batch_size=batch_size, drop_remainder=True)
  ### path, save the output to the subfolder with dataset name
  output_dir = get_output_dir(subfolder=cfg.ds.lower())
  gym_train_path = os.path.join(output_dir, 'gym_train')
  gym_valid_path = os.path.join(output_dir, 'gym_valid')
  gym_test_path = os.path.join(output_dir, 'gym_test')
  model_path = os.path.join(output_dir, 'model')
  ### prepare model init
  model = get_vae(cfg.vae)
  model_kw = inspect.getfullargspec(model.__init__).args[1:]
  model_kw = {k: v for k, v in cfg.items() if k in model_kw}
  is_semi_supervised = ds.has_labels and model.is_semi_supervised()
  if is_semi_supervised:
    train = ds.create_dataset(partition='train', inc_labels=0.1, **ds_kw)
    valid = ds.create_dataset(partition='valid', inc_labels=1.0, **ds_kw)
  else:
    train = ds.create_dataset(partition='train', inc_labels=0., **ds_kw)
    valid = ds.create_dataset(partition='valid', inc_labels=0., **ds_kw)
  ### create the model
  vae = model(path=model_path,
              **get_networks(cfg.ds,
                             centerize_image=True,
                             is_semi_supervised=is_semi_supervised,
                             skip_generator=cfg.skip),
              **model_kw)
  vae.build((None,) + ds.shape)
  vae.load_weights(raise_notfound=False, verbose=True)
  vae.early_stopping.mode = 'max'
  gym = create_gym(dsname=cfg.ds, vae=vae)

  ### fit the network
  def callback():
    metrics = vae.trainer.last_valid_metrics
    llk = metrics['llk_image'] if 'llk_image' in metrics else metrics[
        'llk_dense_type']
    vae.early_stopping.update(llk)
    signal = vae.early_stopping(verbose=True)
    if signal > 0:
      vae.save_weights(overwrite=True)
    # create the return metrics
    return dict(**gym.train()(prefix='train/', dpi=150),
                **gym.valid()(prefix='valid/', dpi=150))

  ### evaluation
  if cfg.eval:
    vae.load_weights()
    gym.train()
    gym(save_path=gym_train_path, dpi=200, verbose=True)
    gym.valid()
    gym(save_path=gym_valid_path, dpi=200, verbose=True)
    gym.test()
    gym(save_path=gym_test_path, dpi=200, verbose=True)
  ### fit
  else:
    vae.early_stopping.patience = 10
    max_iter, learning_rate = get_optimizer_info(cfg.ds)
    vae.fit(train,
            valid=valid,
            learning_rate=learning_rate,
            epochs=-1,
            clipnorm=100,
            max_iter=max_iter,
            valid_freq=int(0.05 * max_iter),
            logging_interval=2,
            skip_fitted=True,
            callback=callback,
            logdir=output_dir,
            compile_graph=True,
            track_gradients=True)
    vae.early_stopping.plot_losses(
        path=os.path.join(output_dir, 'early_stopping.png'))
    vae.plot_learning_curves(os.path.join(output_dir, 'learning_curves.png'))


# ===========================================================================
# Run the experiment
# ===========================================================================
if __name__ == "__main__":
  main(CONFIG)
