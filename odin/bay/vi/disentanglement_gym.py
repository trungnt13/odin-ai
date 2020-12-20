import os
import re
from collections import defaultdict
from functools import partial
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from odin import visual as vs
from odin.bay.distributions import Batchwise
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.metrics import (Correlation, beta_vae_score, dci_scores,
                                 factor_vae_score, mutual_info_gap,
                                 separated_attr_predictability)
from odin.bay.vi.posterior import GroundTruth, Posterior
from odin.bay.vi.utils import discretizing, traverse_dims
from odin.fuel import get_dataset
from odin.ml import DimReduce, fast_kmeans
from odin.utils import as_tuple, uuid
from scipy import stats
from six import string_types
from sklearn import metrics
from tqdm import tqdm
from typeguard import typechecked
from typing_extensions import Literal

__all__ = [
    'DisentanglementGym',
    'Correlation',
    'DimReduce',
]

TrainingMode = Literal['train', 'valid', 'test']


# ===========================================================================
# Helpers
# ===========================================================================
def _plot_latent_units(mean, std, w):
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
  return fig


def _to_image(X, grids, dpi):
  if X.shape[-1] == 1:  # grayscale image
    X = np.squeeze(X, axis=-1)
  else:  # color image
    X = np.transpose(X, (0, 3, 1, 2))
  nrows, ncols = grids
  fig = vs.plot_figure(nrows=nrows, ncols=ncols)
  vs.plot_images(X, grids=grids)
  image = vs.plot_to_image(fig, dpi=dpi)
  return image


def _save_image(arr, path):
  from PIL import Image
  if hasattr(arr, 'numpy'):
    arr = arr.numpy()
  im = Image.fromarray(arr)
  im.save(path)


def _process_labels(y: tf.Tensor, dsname: str,
                    labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
  """Return categorical labels and factors-based label"""
  y_categorical = None
  y_discrete = None
  if dsname in ('mnist', 'fashionmnist', 'cifar10', 'cifar100'):
    y_categorical = tf.argmax(y, axis=-1)
    y_discrete = y
    names = [labels[i] for i in range(10)]
  elif 'celeba' in dsname:
    y = tf.argmax(y, axis=-1)
    raise NotImplementedError
  elif 'shapes3d' in dsname:
    y_categorical = y[:, 2]
    y_discrete = discretizing(y,
                              n_bins=[15, 8, 4, 10, 10, 10],
                              strategy='uniform')
    names = ['cube', 'cylinder', 'sphere', 'round']
  elif 'dsprites' in dsname:
    y_categorical = y[:, 2]
    y_discrete = discretizing(y, n_bins=[10, 6, 3, 8, 8], strategy='uniform')
    names = ['square', 'ellipse', 'heart']
  else:
    raise RuntimeError(f'No support for dataset: {dsname}')
  return np.asarray([names[int(i)] for i in y_categorical]), \
    tf.cast(y_discrete, tf.int32).numpy()


def _predict(data, vae, dsname, labels, verbose):
  px, qz, py = defaultdict(list), defaultdict(list), []
  if verbose:
    data = tqdm(data, desc=f'{dsname} predicting')
  for x, y in data:
    p, q = vae(x, training=False)
    for idx, dist in enumerate(as_tuple(p)):
      px[idx].append(dist)
    for idx, dist in enumerate(as_tuple(q)):
      qz[idx].append(dist)
    py.append(y)
  if verbose:
    data.clear()
    data.close()
  qz = {
      idx: Batchwise(dist_list, name=f'latent{idx}')
      for idx, dist_list in qz.items()
  }
  px = {
      idx: Batchwise(dist_list, name=f'output{idx}')
      for idx, dist_list in px.items()
  }
  py = tf.concat(py, 0)
  y_categorical, y_factors = _process_labels(py, dsname=dsname, labels=labels)
  return px, qz, y_categorical, y_factors, py.numpy()


def _plot_correlation(matrix: np.ndarray,
                      factors: List[str],
                      data_type: str,
                      n_top: int = None):
  # matrix of shape `[n_latents, n_factors]`
  latents = [f'z{i}' for i in range(matrix.shape[0])]
  n_latents, n_factors = matrix.shape
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
                  x='latent',
                  y='factor',
                  hue=data_type,
                  hue_norm=(vmin, vmax),
                  size=data_type,
                  size_norm=(vmin, vmax),
                  sizes=(20, 300),
                  height=4,
                  aspect=n_latents / n_factors,
                  palette="Blues")
  g.set_yticklabels(fontsize=10)
  g.set_xticklabels(fontsize=10)
  return plt.gcf()


def _plot_latents_pairs(
    z: np.ndarray,
    f: np.ndarray,
    correlation: np.ndarray,
    labels: List[str],
):
  n_latents, n_factors = correlation.shape
  # binomial_coefficient(n=n_factors, k=2)
  n_pairs = n_factors * (n_factors - 1) // 2
  ## find the best latents for each labels
  f2z = {
      f_idx: z_idx for f_idx, z_idx in enumerate(np.argmax(correlation, axis=0))
  }
  ## plotting each pairs
  ncol = 2
  nrow = n_pairs
  fig = plt.figure(figsize=(ncol * 3.5, nrow * 3))
  c = 1
  styles = dict(size=10,
                alpha=0.8,
                color='bwr',
                cbar=True,
                cbar_nticks=5,
                cbar_ticks_rotation=0,
                cbar_fontsize=8,
                fontsize=10,
                grid=False)
  for f1 in range(n_factors):
    for f2 in range(f1 + 1, n_factors):
      z1 = f2z[f1]
      z2 = f2z[f2]
      vs.plot_scatter(x=z[:, z1],
                      y=z[:, z2],
                      val=f[:, f1].astype(np.float32),
                      xlabel=f'Z{z1}',
                      ylabel=f'Z{z2}',
                      cbar_title=labels[f1],
                      ax=(nrow, ncol, c),
                      **styles)
      vs.plot_scatter(x=z[:, z1],
                      y=z[:, z2],
                      val=f[:, f2].astype(np.float32),
                      xlabel=f'Z{z1}',
                      ylabel=f'Z{z2}',
                      cbar_title=labels[f2],
                      ax=(nrow, ncol, c + 1),
                      **styles)
      c += 2
  plt.tight_layout()
  return fig


# ===========================================================================
# Disentanglement Gym
# ===========================================================================
_n_visual = 25


class DisentanglementGym:
  """Disentanglement Gym

  Parameters
  ----------
  dataset : {'shapes3d', 'shapes3dsmall', 'dsprites', 'dspritessmall', 'celeba', 'celebasmall', 'mnist'}
      name of the data
  vae : VariationalAutoencoder
      instance of `VariationalAutoencoder`
  max_valid_samples : int, optional
      maximum number of samples used for validation, by default 2000
  max_test_samples : int, optional
      maximum number of samples used for testing, by default 20000
  batch_size : int, optional
      batch size, by default 64
  allow_exception : bool, optional
      if False ignore all exception while running, by default True
  seed : int, optional
      seed for random state and reproducibility, by default 1
  """

  @typechecked
  def __init__(self,
               dataset: Literal['shapes3d', 'shapes3dsmall', 'dsprites',
                                'dspritessmall', 'celeba', 'celebasmall',
                                'fashionmnist', 'mnist', 'cifar10', 'cifar100'],
               vae: VariationalAutoencoder,
               max_valid_samples: int = 2000,
               max_test_samples: int = 20000,
               batch_size: int = 64,
               allow_exception: bool = True,
               seed: int = 1):
    self.name = dataset
    self.allow_exception = allow_exception
    self.seed = seed
    self.ds = get_dataset(dataset)
    self._batch_size = int(batch_size)
    self._max_valid_samples = int(max_valid_samples)
    self._max_test_samples = int(max_test_samples)
    ## set seed is importance for comparable results
    tf.random.set_seed(1)
    self._train = self.ds.create_dataset(batch_size=batch_size,
                                         partition='train',
                                         inc_labels=True,
                                         shuffle=1000)
    self._valid = self.ds.create_dataset(batch_size=batch_size,
                                         partition='valid',
                                         inc_labels=True,
                                         shuffle=1000)
    self._test = self.ds.create_dataset(batch_size=batch_size,
                                        partition='test',
                                        inc_labels=True,
                                        shuffle=1000)
    self.vae = vae
    self.z_samples = vae.sample_prior(sample_shape=_n_visual, seed=self.seed)
    ## prepare the samples
    self.x_train, self.y_train = [
        (x[:_n_visual], y[:_n_visual]) for x, y in self._train.take(1)
    ][0]
    self.x_valid, self.y_valid = [
        (x[:_n_visual], y[:_n_visual]) for x, y in self._valid.take(1)
    ][0]
    self.x_test, self.y_test = [
        (x[:_n_visual], y[:_n_visual]) for x, y in self._test.take(1)
    ][0]
    ## default configuration
    self._reconstruction = True
    self._elbo = False
    self._latents_sampling = True
    self._latents_traverse = True
    self._latents_stats = True
    self._track_gradients = True
    self._latents_pairs = None
    self._correlation_methods = None
    self._dimension_reduction = None
    ## unsupervised clustering score
    self._silhouette_score = False
    self._adjusted_rand_score = False
    self._normalized_mutual_info = False
    self._adjusted_mutual_info = False
    ## quantitative measures
    self._mig_score = False
    self._dci_score = False
    self._sap_score = False
    self._factor_vae = False
    self._beta_vae = False
    self.setup = dict(train={}, valid={}, test={})
    self._mode = 'train'
    self.data_info = dict(train=(self._train, self.x_train, self.y_train),
                          valid=(self._valid, self.x_valid, self.y_valid),
                          test=(self._test, self.x_test, self.y_test))
    ## others
    self._traverse_config = dict(
        min_val=-2,
        max_val=2,
        n_traverse_points=15,
        mode='linear',
    )

  @typechecked
  def set_config(
      self,
      reconstruction: bool = False,
      elbo: bool = False,
      latents_sampling: bool = False,
      latents_traverse: bool = False,
      latents_stats: bool = False,
      track_gradients: bool = False,
      latents_pairs: Optional[Correlation] = None,
      correlation_methods: Optional[Correlation] = None,
      dimension_reduction: Optional[DimReduce] = None,
      mig_score: bool = False,
      dci_score: bool = False,
      sap_score: bool = False,
      factor_vae: bool = False,
      beta_vae: bool = False,
      silhouette_score: bool = False,
      adjusted_rand_score: bool = False,
      normalized_mutual_info: bool = False,
      adjusted_mutual_info: bool = False,
      mode: Union[TrainingMode,
                  List[TrainingMode]] = ('train', 'valid', 'test'),
  ) -> 'DisentanglementGym':
    """Set configuration for Disentanglement Gym"""
    kw = dict(locals())
    mode = kw.pop('mode')
    if mode == 'all':
      for k, v in kw.items():
        if hasattr(self, f'_{k}'):
          setattr(self, f'_{k}', v)
    else:
      for m in as_tuple(mode, t=string_types):
        setup = self.setup[m]
        for k, v in kw.items():
          if hasattr(self, f'_{k}'):
            setup[f'_{k}'] = v
    return self

  @typechecked
  def set_traverse_config(
      self,
      min_val: float = -2.,
      max_val: float = 2.,
      n_traverse_points: int = 15,
      mode: Literal['linear', 'quantile', 'gaussian'] = 'linear',
  ) -> 'DisentanglementGym':
    self._traverse_config = dict(
        min_val=min_val,
        max_val=max_val,
        n_traverse_points=n_traverse_points,
        mode=mode,
    )
    return self

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def mode(self) -> TrainingMode:
    """The training mode: 'train', 'valid', 'test' """
    return self._mode

  def train(self) -> 'DisentanglementGym':
    """Enable the training mode"""
    self._mode = 'train'
    for k, v in self.setup['train'].items():
      setattr(self, k, v)
    return self

  def valid(self) -> 'DisentanglementGym':
    self._mode = 'valid'
    for k, v in self.setup['valid'].items():
      setattr(self, k, v)
    return self

  def test(self) -> 'DisentanglementGym':
    """Enable the evaluation mode"""
    self._mode = 'test'
    for k, v in self.setup['test'].items():
      setattr(self, k, v)
    return self

  def _is_predict(self) -> bool:
    return (self._dimension_reduction is not None or
            self._correlation_methods is not None or self._mig_score or
            self._latents_pairs or self._dci_score or self._sap_score or
            self._beta_vae or self._factor_vae)

  def _is_clustering(self) -> bool:
    return (self._adjusted_rand_score or self._adjusted_mutual_info or
            self._normalized_mutual_info or self._silhouette_score)

  def __call__(self,
               save_path: Optional[str] = None,
               remove_saved_image: bool = True,
               dpi: int = 150,
               prefix: str = '',
               verbose: bool = False) -> Dict[str, Any]:
    """Run the disentanglement evaluation protocol

    Parameters
    ----------
    save_path : Optional[str], optional
        path to a folder for saving image files, by default None
    remove_saved_image : bool, optional
        if True don't return saved image, by default True
    dpi : int, optional
        dot-per-inch for saving image, by default 150
    verbose : bool, optional
        logging, by default False

    Returns
    -------
    Dict[str, Any]
        dictionary of monitoring metrics, could be image, number, vector, string, etc
    """
    try:
      return self._call_safe(save_path=save_path,
                             remove_saved_image=remove_saved_image,
                             dpi=dpi,
                             prefix=prefix,
                             verbose=verbose)
    except Exception as e:
      if self.allow_exception:
        raise e
      else:
        print(e)
      return {}

  def _call_safe(self, save_path, remove_saved_image, dpi, prefix, verbose):
    unique_key = uuid(20)
    vae = self.vae
    grids = (int(sqrt(_n_visual)), int(sqrt(_n_visual)))
    outputs = dict()
    ds, x, y = self.data_info[self.mode]
    ## n_samples
    n_score_samples = (10000 if self.mode == 'test' else 5000)
    n_samples = (self._max_test_samples
                 if self.mode == 'test' else self._max_valid_samples)
    n_batches = int(n_samples / self.batch_size)
    is_semi_supervised = type(vae).is_semi_supervised
    ## prepare
    P, Q = vae(x, training=False)
    P, Q = as_tuple(P), as_tuple(Q)
    z_mean = tf.reduce_mean(tf.concat([q.mean() for q in Q], axis=-1), axis=0)
    z_std = tf.reduce_mean(tf.concat([q.stddev() for q in Q], axis=-1), axis=0)
    outputs['latents/mean'] = z_mean
    outputs['latents/stddev'] = z_std
    ## reconstruction
    if self._reconstruction:
      px = P[0]
      image_reconstructed = _to_image(px.mean().numpy(), grids=grids, dpi=dpi)
      outputs['original'] = _to_image(x, grids=grids, dpi=dpi)
      outputs['reconstruction'] = image_reconstructed
    ## ELBO
    if self._elbo:
      elbo_llk = defaultdict(list)
      elbo_kl = defaultdict(list)
      for inputs, _ in ds.take(n_batches):
        llk, kl = vae.elbo_components(inputs, training=False)
        for k, v in llk.items():
          elbo_llk[k].append(v)
        for k, v in kl.items():
          elbo_kl[k].append(v)
      elbo_llk = {
          k: tf.reduce_mean(tf.concat(v, axis=0)) for k, v in elbo_llk.items()
      }
      elbo_kl = {
          k: tf.reduce_mean(tf.concat(v, axis=0)) for k, v in elbo_kl.items()
      }
      outputs.update(elbo_llk)
      outputs.update(elbo_kl)
    ## latents statsss
    if self._latents_stats:
      w_d = vae.decoder.trainable_variables[0]
      if w_d.shape.ndims == 2:  # dense weights
        w_d = tf.reduce_sum(w_d, axis=-1)
      else:  # convolution weights
        w_d = tf.reduce_sum(w_d, axis=(0, 1, 2))
      image_latents = vs.plot_to_image(_plot_latent_units(z_mean, z_std, w_d),
                                       dpi=dpi)
      outputs['latents_stats'] = image_latents
    ## latents traverse
    if self._latents_traverse:
      z = tf.concat([q.mean() for q in Q], axis=-1)
      sorted_indices = np.argsort(z_std)[:20]  # only top 20
      z = traverse_dims(z,
                        feature_indices=sorted_indices,
                        n_random_samples=1,
                        seed=self.seed,
                        **self._traverse_config)
      P = as_tuple(vae.decode(z[0] if len(z) == 1 else z, training=False))
      images = P[0].mean().numpy()
      n_indices = len(sorted_indices)  # do it here, just in case zdim < 20
      image_traverse = _to_image(images,
                                 grids=(n_indices,
                                        int(images.shape[0] / n_indices)),
                                 dpi=dpi)
      outputs['latents_traverse'] = image_traverse
    ## latents sampling
    if self._latents_sampling:
      P = as_tuple(vae.decode(self.z_samples, training=False))
      image_sampled = _to_image(P[0].mean().numpy(), grids=grids, dpi=dpi)
      outputs['latents_sampled'] = image_sampled
    ## latents clusters
    if self._is_predict():
      ds_pred = ds.take(n_batches)
      px, qz, labels, factors, py = _predict(ds_pred,
                                             vae,
                                             self.name,
                                             verbose=verbose,
                                             labels=self.ds.labels)
      qz_mean = {idx: q.mean().numpy() for idx, q in qz.items()}
      # qz_sample = {
      #     idx: q.sample(seed=self.seed).numpy() for idx, q in qz.items()
      # }
      # latents pairs
      if self._latents_pairs is not None:
        for method in self._latents_pairs:
          name = method.name.lower()
          for z_idx, z in qz_mean.items():
            matrix = method(z, factors, cache_key=unique_key, verbose=verbose)
            _plot_latents_pairs(z=z,
                                f=py,
                                correlation=matrix,
                                labels=self.ds.labels)
            outputs[f'pairs_{name}{z_idx}'] = vs.plot_to_image(plt.gcf(),
                                                               dpi=dpi)
      # correlation
      if self._correlation_methods is not None:
        for method in self._correlation_methods:
          name = method.name.lower()
          for z_idx, z in qz_mean.items():
            matrix = method(z, factors, cache_key=unique_key, verbose=verbose)
            _plot_correlation(matrix,
                              factors=self.ds.labels,
                              data_type=name,
                              n_top=len(self.ds.labels) * 2)
            outputs[f'{name}{z_idx}'] = vs.plot_to_image(plt.gcf(), dpi=dpi)
      # dimension reduction
      if self._dimension_reduction is not None:
        from odin.ml import fast_tsne, fast_umap, fast_pca
        for method in self._dimension_reduction:
          name = method.name.lower()
          for z_idx, z in qz_mean.items():
            _x, _y = z, labels
            if method == DimReduce.TSNE:
              _x, _y = _x[:10000], _y[:10000]  # maximum 10000 data points
            _x = method(_x, n_components=2, exaggeration_iter=80)
            fig = plt.figure(figsize=(8, 8), dpi=150)
            vs.plot_scatter(_x, color=_y, size=10.0, alpha=0.6, grid=False)
            outputs[f'{name}{z_idx}'] = vs.plot_to_image(fig, dpi=dpi)
      # clustering scores
      if self._is_clustering():
        for z_idx, z in qz_mean.items():
          labels_pred = fast_kmeans(z,
                                    n_clusters=len(np.unique(labels)),
                                    n_init=200,
                                    random_state=self.seed).predict(z)
          if self._adjusted_rand_score:
            outputs[f'ari{z_idx}'] = metrics.adjusted_rand_score(
                labels, labels_pred)
          if self._adjusted_mutual_info:
            outputs[f'ami{z_idx}'] = metrics.adjusted_mutual_info_score(
                labels, labels_pred)
          if self._normalized_mutual_info:
            outputs[f'nmi{z_idx}'] = metrics.normalized_mutual_info_score(
                labels, labels_pred)
          if self._silhouette_score:
            outputs[f'asw{z_idx}'] = metrics.silhouette_score(
                z, labels, random_state=self.seed)
      # disentangling scores
      if self._mig_score:
        for z_idx, z in qz.items():
          z = z.sample(10)
          z = tf.reshape(z, (-1, z.shape[-1]))
          outputs[f'mig{z_idx}'] = mutual_info_gap(z, np.tile(factors, (10, 1)))
      if self._sap_score:
        for z_idx, z in qz_mean.items():
          outputs[f'sap{z_idx}'] = separated_attr_predictability(z, factors)
      if self._dci_score:
        for z_idx, z in qz_mean.items():
          outputs[f'dci{z_idx}'] = np.mean(
              dci_scores(z, factors, cache_key=unique_key, verbose=verbose))
      if self._beta_vae:
        for z_idx, z in qz.items():
          outputs[f'betavae{z_idx}'] = beta_vae_score(representations=z,
                                                      factors=factors,
                                                      n_samples=n_score_samples,
                                                      verbose=verbose)
      if self._factor_vae:
        for z_idx, z in qz.items():
          outputs[f'factorvae{z_idx}'] = factor_vae_score(
              representations=z,
              factors=factors,
              n_samples=n_score_samples,
              verbose=verbose)
    ## track the gradients
    if vae.trainer is not None and self._track_gradients:
      all_grads = [(k, v)
                   for k, v in vae.trainer.last_train_metrics.items()
                   if 'grad/' in k]
      encoder_grad = 0
      decoder_grad = 0
      latents_grad = 0
      if len(all_grads) > 0:
        outputs['grad/encoder'] = sum(
            tf.linalg.norm(v) for k, v in all_grads if 'encoder' in k)
        outputs['grad/decoder'] = sum(
            tf.linalg.norm(v) for k, v in all_grads if 'decoder' in k)
        outputs['grad/latents'] = sum(
            tf.linalg.norm(v) for k, v in all_grads if 'latents' in k)
    ## save outputs
    if save_path is not None:
      # create the folder
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      elif not os.path.isdir(save_path):
        raise ValueError(f"'{save_path}' is not a directory")
      # save the images
      for k, v in list(outputs.items()):
        if (hasattr(v, 'shape') and len(v.shape) == 4 and v.dtype == tf.uint8):
          if v.shape[0] == 1:
            v = tf.squeeze(v, 0)
          img_path = os.path.join(save_path, f'{k}.png')
          _save_image(v, img_path)
          if verbose:
            print('Saved image at:', img_path)
          if remove_saved_image:
            del outputs[k]
      # save the scores
      score_path = os.path.join(save_path, 'scores.txt')
      with open(score_path, 'w') as f:
        for k, v in outputs.items():
          if hasattr(v, 'numpy'):
            v = v.numpy()
          f.write(f'{k}: {v}\n')
        if verbose:
          print('Saved scores at:', score_path)
    ## add prefix and return
    return {f'{prefix}{k}': v for k, v in outputs.items()}

  def __str__(self):
    s = 'DisentanglementGym:\n'
    for k, v in sorted(self.__dict__.items()):
      if re.match(r'\_\w*', k):
        s += f' {k[1:]}: {v}\n'
    for k, v in self.setup.items():
      s += f' Mode="{k}"\n'
      for i, j in v.items():
        s += f'  {i[1:]}: {j}\n'
    return s[:-1]
