import os
import re
from collections import defaultdict
from functools import partial
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from odin import visual as vs
from odin.bay.helpers import concat_distributions
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.metrics import (Correlation, dci_scores, mutual_info_gap,
                                 separated_attr_predictability)
from odin.bay.vi.posterior import GroundTruth, Posterior
from odin.bay.vi.utils import discretizing, traverse_dims
from odin.ml import DimReduce
from odin.utils import as_tuple
from scipy import stats
from typeguard import typechecked
from typing_extensions import Literal
from odin.utils import uuid

__all__ = [
    'DisentanglementGym',
    'Correlation',
    'DimReduce',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _show_latent_units(mean, std, w):
  # plot the latents and its weights
  fig = plt.figure(figsize=(8, 4), dpi=200)
  ax = plt.gca()
  l1 = ax.plot(mean,
               label='mean',
               linewidth=1.0,
               marker='o',
               markersize=3,
               color='r',
               alpha=0.5)
  l2 = ax.plot(std,
               label='stddev',
               linewidth=1.0,
               marker='^',
               markersize=3,
               color='g',
               alpha=0.5)
  ax.set_ylim(-1.5, 1.5)
  ax.tick_params(axis='y', colors='r')
  ax.grid(True)
  ## plotting the weights
  ax = ax.twinx()
  l3 = ax.plot(w,
               label='weights',
               linewidth=1.0,
               linestyle='--',
               marker='s',
               markersize=3,
               color='b',
               alpha=0.5)
  ax.tick_params(axis='y', colors='b')
  ax.grid(False)
  lines = l1 + l2 + l3
  ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
  return vs.plot_to_image(fig)


def _to_image(X, grids):
  if X.shape[-1] == 1:  # grayscale image
    X = np.squeeze(X, axis=-1)
  else:  # color image
    X = np.transpose(X, (0, 3, 1, 2))
  nrows, ncols = grids
  fig = vs.plot_figure(nrows=nrows, ncols=ncols, dpi=100)
  vs.plot_images(X, grids=grids)
  image = vs.plot_to_image(fig)
  return image


def _save_image(arr, path):
  from PIL import Image
  if hasattr(arr, 'numpy'):
    arr = arr.numpy()
  im = Image.fromarray(arr)
  im.save(path)


def _process_labels(y: tf.Tensor, dsname: str) -> Tuple[np.ndarray, np.ndarray]:
  """Return categorical labels and factors-based label"""
  y_categorical = None
  y_discrete = None
  if dsname == 'mnist':
    y_categorical = tf.argmax(y, axis=-1)
    y_discrete = y
  elif 'celeba' in dsname:
    y = tf.argmax(y, axis=-1)
  elif 'shapes3d' in dsname:
    y_categorical = y[:, 4]
    y_discrete = discretizing(y,
                              n_bins=[10, 10, 10, 8, 4, 15],
                              strategy='uniform')
  elif 'dsprites' in dsname:
    y_categorical = y[:, 2]
    y_discrete = discretizing(y, n_bins=[10, 6, 3, 8, 8], strategy='uniform')
  return tf.cast(y_categorical, tf.int32).numpy(), \
    tf.cast(y_discrete, tf.int32).numpy()


def _predict(ds, vae, dsname):
  px, qz, labels = defaultdict(list), defaultdict(list), []
  for x, y in ds:
    p, q = vae(x, training=False)
    for idx, dist in enumerate(as_tuple(p)):
      px[idx].append(dist)
    for idx, dist in enumerate(as_tuple(q)):
      qz[idx].append(dist)
    labels.append(y)
  qz = {idx: concat_distributions(dist_list) for idx, dist_list in qz.items()}
  px = {idx: concat_distributions(dist_list) for idx, dist_list in px.items()}
  labels, factors = _process_labels(tf.concat(labels, 0), dsname)
  return px, qz, labels, factors


def _plot_correlation(matrix: np.ndarray,
                      factors: List[str],
                      data_type: str,
                      n_top: int = None):
  latents = [f'z{i}' for i in range(matrix.shape[0])]
  vmin = np.min(matrix)
  vmax = np.max(matrix)
  if n_top is not None:
    n_top = int(n_top)
    matrix = (matrix - np.min(matrix, axis=0, keepdims=True)) /\
      (np.max(matrix, axis=0, keepdims=True) - np.min(matrix, axis=0, keepdims=True) + 1e-10)
    top_ids = np.argsort(np.max(matrix, axis=1) -
                         stats.skew(matrix, axis=1))[::-1]
    data = pd.DataFrame([(latents[zi], factors[fj], matrix[zi, fj])
                         for zi in top_ids[:n_top]
                         for fj in range(matrix.shape[1])],
                        columns=['latent', 'factor', data_type])
  else:
    data = pd.DataFrame([(latents[zi], factors[fj], matrix[zi, fj])
                         for zi in range(matrix.shape[0])
                         for fj in range(matrix.shape[1])],
                        columns=['latent', 'factor', data_type])
  g = sns.relplot(data=data,
                  x='factor',
                  y='latent',
                  hue=data_type,
                  hue_norm=(vmin, vmax),
                  size=data_type,
                  size_norm=(vmin, vmax),
                  sizes=(5, 150),
                  height=10,
                  aspect=0.4,
                  palette="vlag")
  g.set_yticklabels(fontsize=8)
  g.set_xticklabels(rotation=-20, fontsize=6)


# ===========================================================================
# Disentanglement Gym
# ===========================================================================
_n_visual = 25


class DisentanglementGym:

  @typechecked
  def __init__(self,
               dataset: Literal['shapes3d', 'shapes3dsmall', 'dsprites',
                                'celeba', 'celebasmall', 'mnist'],
               vae: VariationalAutoencoder,
               allow_exception: bool = True,
               seed: int = 1):
    from odin.fuel import get_dataset
    self.name = dataset
    self.allow_exception = allow_exception
    self.seed = seed
    self.ds = get_dataset(dataset)
    self._train = self.ds.create_dataset(batch_size=32,
                                         partition='train',
                                         inc_labels=True,
                                         shuffle=1000)
    self._valid = self.ds.create_dataset(batch_size=32,
                                         partition='valid',
                                         inc_labels=True,
                                         shuffle=1000)
    self._test = self.ds.create_dataset(batch_size=32,
                                        partition='test',
                                        inc_labels=True,
                                        shuffle=1000)
    self.vae = vae
    self._is_training = False
    self.z_samples = vae.sample_prior(sample_shape=_n_visual, seed=self.seed)
    ## prepare the samples
    self.x_valid, self.y_valid = [
        (x[:_n_visual], y[:_n_visual]) for x, y in self._valid.take(1)
    ][0]
    self.x_test, self.y_test = [
        (x[:_n_visual], y[:_n_visual]) for x, y in self._test.take(1)
    ][0]
    ## default configuration
    self._reconstruction = True
    self._latents_sampling = True
    self._latents_traverse = True
    self._latents_stats = True
    self._track_gradients = True
    self._correlation_methods = None
    self._dimension_reduction = None
    ## quantitative measures
    self._mig_score = False
    self._dci_score = True
    self._sap_score = False
    ## others
    self._traverse_config = dict(
        min_val=-4,
        max_val=4,
        n_traverse_points=25,
        mode='linear',
    )
    ## dimension reduction algorithm
    try:
      import umap
      from odin.ml import fast_umap
      self.dim_reduce = partial(fast_umap,
                                n_components=2,
                                random_state=self.seed)
    except ImportError:
      from odin.ml import fast_tsne
      self.dim_reduce = partial(fast_tsne,
                                n_components=2,
                                random_state=self.seed)

  @typechecked
  def set_config(
      self,
      reconstruction: Optional[bool] = None,
      latents_sampling: Optional[bool] = None,
      latents_traverse: Optional[bool] = None,
      latents_stats: Optional[bool] = None,
      track_gradients: Optional[bool] = None,
      correlation_methods: Optional[Correlation] = None,
      dimension_reduction: Optional[DimReduce] = None,
      mig_score: Optional[bool] = None,
      dci_score: Optional[bool] = None,
      sap_score: Optional[bool] = None,
      ucs_score: Optional[bool] = None,
  ) -> 'DisentanglementGym':
    for k, v in locals().items():
      if hasattr(self, f'_{k}') and v is not None:
        setattr(self, f'_{k}', v)
    return self

  @typechecked
  def set_traverse_config(
      self,
      min_val: float = -4.,
      max_val: float = 4.,
      n_traverse_points: int = 25,
      mode='linear',
  ) -> 'DisentanglementGym':
    self._traverse_config = dict(
        min_val=min_val,
        max_val=max_val,
        n_traverse_points=n_traverse_points,
        mode=mode,
    )
    return self

  @property
  def is_training(self) -> bool:
    return self._is_training

  def train(self) -> 'DisentanglementGym':
    """Enable the training mode"""
    self._is_training = True
    return self

  def eval(self) -> 'DisentanglementGym':
    """Enable the evaluation mode"""
    self._is_training = False
    return self

  @property
  def _is_predict(self) -> bool:
    return (self._dimension_reduction is not None or
            self._correlation_methods is not None or self._mig_score or
            self._dci_score or self._sap_score)

  def __call__(self, save_path: Optional[str] = None) -> Dict[str, Any]:
    try:
      return self._call_safe(save_path=save_path)
    except Exception as e:
      if self.allow_exception:
        raise e
      else:
        print(e)
      return {}

  def _call_safe(self, save_path=None):
    unique_key = uuid(20)
    vae = self.vae
    grids = (int(sqrt(_n_visual)), int(sqrt(_n_visual)))
    outputs = dict()
    ## training mode
    if self._is_training:
      x, y = self.x_valid, self.y_valid
      ds = self._valid
    ## evaluation mode
    else:
      x, y = self.x_test, self.y_test
      ds = self._test
    ## prepare
    P, Q = vae(x, training=False)
    P, Q = as_tuple(P), as_tuple(Q)
    z_mean = tf.reduce_mean(tf.concat([q.mean() for q in Q], axis=-1), axis=0)
    z_std = tf.reduce_mean(tf.concat([q.stddev() for q in Q], axis=-1), axis=0)
    ## reconstruction
    if self._reconstruction:
      px = P[0]
      image_reconstructed = _to_image(px.mean().numpy(), grids=grids)
      outputs['reconstruction'] = image_reconstructed
    ## latents stats
    if self._latents_stats:
      w_d = vae.decoder.trainable_variables[0]
      if w_d.shape.ndims == 2:  # dense weights
        w_d = tf.reduce_sum(w_d, axis=-1)
      else:  # convolution weights
        w_d = tf.reduce_sum(w_d, axis=(0, 1, 2))
      image_latents = _show_latent_units(z_mean, z_std, w_d)
      outputs['latents_stats'] = image_latents
    ## latents traverse
    if self._latents_traverse:
      n_indices = 20
      z = tf.concat([q.mean() for q in Q], axis=-1)
      z = traverse_dims(z,
                        feature_indices=np.argsort(z_std)[:n_indices],
                        n_random_samples=1,
                        seed=self.seed,
                        **self._traverse_config)
      P = as_tuple(vae.decode(z[0] if len(z) == 1 else z, training=False))
      images = P[0].mean().numpy()
      image_traverse = _to_image(images,
                                 grids=(n_indices,
                                        int(images.shape[0] / n_indices)))
      outputs['latents_traverse'] = image_traverse
    ## latents sampling
    if self._latents_sampling:
      P = as_tuple(vae.decode(self.z_samples, training=False))
      image_sampled = _to_image(P[0].mean().numpy(), grids=grids)
      outputs['latents_sampled'] = image_sampled
    ## latents clusters
    if self._is_predict:
      px, qz, labels, factors = _predict(
          (ds.take(32) if self._is_training else ds.take(64)), vae, self.name)
      # correlation
      if self._correlation_methods is not None:
        for method in self._correlation_methods:
          name = method.name.lower()
          for z_idx, z in qz.items():
            matrix = method(z.mean(), factors, cache_key=unique_key)
            _plot_correlation(matrix,
                              factors=self.ds.labels,
                              data_type=name,
                              n_top=len(self.ds.labels) * 2)
            outputs[f'{name}{z_idx}'] = vs.plot_to_image(plt.gcf())
      # dimension reduction
      if self._dimension_reduction is not None:
        for method in self._dimension_reduction:
          name = method.name.lower()
          for z_idx, z in qz.items():
            z = method(z.mean().numpy(), n_components=2)
            fig = plt.figure(figsize=(8, 8), dpi=150)
            vs.plot_scatter(z, color=labels, size=10.0, alpha=0.6, grid=False)
            outputs[f'{name}{z_idx}'] = vs.plot_to_image(fig)
      # disentangling scores
      if self._mig_score:
        for z_idx, z in qz.items():
          z = z.sample(10)
          z = tf.reshape(z, (-1, z.shape[-1]))
          outputs[f'mig{z_idx}'] = mutual_info_gap(z, np.tile(factors, (10, 1)))
      if self._sap_score:
        for z_idx, z in qz.items():
          outputs[f'sap{z_idx}'] = separated_attr_predictability(
              z.mean(), factors)
      if self._dci_score:
        for z_idx, z in qz.items():
          outputs[f'dci{z_idx}'] = np.mean(
              dci_scores(z.mean(), factors, cache_key=unique_key))
    ## save outputs
    if save_path is not None:
      # create the folder
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      elif not os.path.isdir(save_path):
        raise ValueError(f"'{save_path}' is not a directory")
      # save the images
      for k, v in outputs.items():
        if (hasattr(v, 'shape') and v.shape.ndims == 4 and v.dtype == tf.uint8):
          if v.shape[0] == 1:
            v = tf.squeeze(v, 0)
          _save_image(v, os.path.join(save_path, f'{k}.png'))
    return outputs

  def __str__(self):
    s = 'DisentanglementGym:\n'
    for k, v in sorted(self.__dict__.items()):
      if re.match(r'\_\w*', k):
        s += f' {k[1:]}: {v}\n'
    return s[:-1]
