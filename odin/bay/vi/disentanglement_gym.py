import contextlib
import random
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union, Sequence, Callable, Any, \
  Iterator, Text

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from six import string_types
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from tensorflow_probability.python.distributions import Distribution, Normal, \
  Bernoulli
from tqdm import tqdm
from typeguard import typechecked
from typing_extensions import Literal

from odin import visual as vs
from odin.backend import TensorType
from odin.bay.distributions import Batchwise, QuantizedLogistic, \
  MixtureQuantizedLogistic
from odin.bay.vi._base import VariationalModel
from odin.bay.vi.autoencoder import VariationalAutoencoder
from odin.bay.vi.losses import total_correlation
from odin.bay.vi.metrics import (Correlation, beta_vae_score, dci_scores,
                                 factor_vae_score, mutual_info_gap,
                                 separated_attr_predictability,
                                 correlation_matrix, mutual_info_estimate,
                                 importance_matrix, relative_strength)
from odin.bay.vi.utils import discretizing
from odin.fuel import get_dataset, ImageDataset
from odin.ml import DimReduce, fast_kmeans
from odin.search import diagonal_linear_assignment
from odin.utils import as_tuple, uuid

__all__ = [
  'GroundTruth',
  'DisentanglementGym',
  'Correlation',
  'DimReduce',
  'plot_latent_stats'
]

DataPartition = Literal['train', 'valid', 'test']
CorrelationMethod = Literal['spearman', 'pearson', 'lasso', 'mi', 'importance']
ConvertFunction = Callable[[List[Distribution]], tf.Tensor]
Axes = Union[None, plt.Axes, Sequence[int], int]
FactorFilter = Union[Callable[[Any], bool],
                     Dict[Union[str, int], int],
                     float, int, str,
                     None]
DatasetNames = Literal['shapes3d', 'shapes3dsmall',
                       'dsprites', 'dspritessmall',
                       'celeba', 'celebasmall',
                       'fashionmnist', 'mnist',
                       'cifar10', 'cifar100', 'svhn',
                       'cortex', 'pbmc',
                       'halfmoons']


def concat_mean(dists: List[Distribution]) -> tf.Tensor:
  return tf.concat([d.mean() for d in dists], -1)


def first_mean(dists: List[Distribution]) -> tf.Tensor:
  return dists[0].mean()


# ===========================================================================
# Helpers
# ===========================================================================
_CACHE = defaultdict(dict)


def _dist(p: Union[Distribution, Sequence[Distribution]]
          ) -> Union[Sequence[Distribution], Distribution]:
  """Convert DeferredTensor back to original Distribution"""
  if isinstance(p, (tuple, list)):
    return [_dist(i) for i in p]
  p: Distribution
  return (p.parameters['distribution']
          if 'deferred_tensor' in str(type(p)) else p)


def _save_image(arr, path):
  from PIL import Image
  if hasattr(arr, 'numpy'):
    arr = arr.numpy()
  im = Image.fromarray(arr)
  im.save(path)


def _prepare_categorical(y: np.ndarray, ds: ImageDataset,
                         return_index: bool = False) -> np.ndarray:
  """Return categorical labels and factors-based label"""
  if ds is None:
    dsname = None
    labels = None
  else:
    dsname = ds.name
    labels = ds.labels
  if hasattr(y, 'numpy'):
    y = y.numpy()
  if dsname is None:  # unknown
    y_categorical = tf.argmax(y, axis=-1)
    names = np.array([f'#{i}' for i in range(y.shape[1])])
  elif dsname in ('mnist', 'fashionmnist', 'cifar10', 'cifar100', 'cortex'):
    y_categorical = tf.argmax(y, axis=-1)
    names = labels
  elif 'celeba' in dsname:
    y_categorical = tf.argmax(y, axis=-1)
    raise NotImplementedError
  elif 'shapes3d' in dsname:
    y_categorical = y[:, 2]
    names = ['cube', 'cylinder', 'sphere', 'round']
  elif 'dsprites' in dsname:
    y_categorical = y[:, 2]
    names = ['square', 'ellipse', 'heart']
  elif 'halfmoons' in dsname:
    y_categorical = y[:, -1]
    names = ['circle', 'square', 'triangle', 'pentagon']
  elif 'pbmc' == dsname:
    names = ['CD4', 'CD8', 'CD45RA', 'CD45RO']
    y_probs = []
    for x in [i for n in names for i, l in zip(y.T, labels) if n == l]:
      x = x[:, np.newaxis]
      gmm = GaussianMixture(n_components=2,
                            covariance_type='full',
                            n_init=2,
                            random_state=1)
      gmm.fit(x)
      y_probs.append(gmm.predict_proba(x)[:, np.argmax(gmm.means_.ravel())])
    y_categorical = np.argmax(np.vstack(y_probs).T, axis=1)
  else:
    raise RuntimeError(f'No support for dataset: {dsname}')
  if return_index:
    return y_categorical
  return np.asarray([names[int(i)] for i in y_categorical])


def _prepare_images(x, normalize=False):
  """if normalize=True, normalize the image to [0, 1], used for the
  reconstructed or generated image, not the original one.
  """
  x = np.asarray(x)
  n_images = x.shape[0]
  if normalize:
    vmin = x.reshape((n_images, -1)).min(axis=1).reshape((n_images, 1, 1, 1))
    vmax = x.reshape((n_images, -1)).max(axis=1).reshape((n_images, 1, 1, 1))
    x = (x - vmin) / (vmax - vmin)
  if x.shape[-1] == 1:  # grayscale image
    x = np.squeeze(x, -1)
  else:  # color image
    x = np.transpose(x, (0, 3, 1, 2))
  return x


def plot_latent_stats(mean,
                      stddev,
                      kld=None,
                      weights=None,
                      ax=None,
                      name='q(z|x)'):
  # === 2. plotting
  ax = vs.to_axis(ax)
  l1 = ax.plot(mean,
               label='mean',
               linewidth=0.5,
               marker='o',
               markersize=3,
               color='r',
               alpha=0.5)
  l2 = ax.plot(stddev,
               label='stddev',
               linewidth=0.5,
               marker='^',
               markersize=3,
               color='g',
               alpha=0.5)
  # ax.set_ylim(-1.5, 1.5)
  ax.tick_params(axis='y', colors='r')
  ax.set_ylabel(f'{name} Mean', color='r')
  ax.grid(True)
  lines = l1 + l2
  ## plotting the weights
  if kld is not None or weights is not None:
    ax = ax.twinx()
  if kld is not None:
    lines += plt.plot(kld,
                      label='KL(q|p)',
                      linestyle='--',
                      color='y',
                      marker='s',
                      markersize=2.5,
                      linewidth=1.0,
                      alpha=0.5)
  if weights is not None:
    l3 = ax.plot(weights,
                 label='weights',
                 linewidth=1.0,
                 linestyle='--',
                 marker='s',
                 markersize=2.5,
                 color='b',
                 alpha=0.5)
    ax.tick_params(axis='y', colors='b')
    ax.grid(False)
    ax.set_ylabel('L2-norm weights', color='b')
    lines += l3
  ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
  ax.grid(alpha=0.5)
  return ax


def _boostrap_sampling(
    model: VariationalModel,
    inputs: List[np.ndarray],
    groundtruth: 'GroundTruth',
    n_samples: int,
    batch_size: int,
    verbose: bool,
    seed: int,
):
  assert inputs.shape[0] == groundtruth.shape[0], \
    ('Number of samples mismatch between inputs and ground-truth, '
     f'{inputs.shape[0]} != {groundtruth.shape[0]}')
  inputs = as_tuple(inputs)
  Xs = [list() for _ in range(len(inputs))]  # inputs
  Zs = []  # latents
  Os = []  # outputs
  indices = []
  n = 0
  random_state = np.random.RandomState(seed=seed)
  prog = tqdm(desc=f'Sampling', total=n_samples, disable=not verbose)
  while n < n_samples:
    batch = min(batch_size, n_samples - n, groundtruth.shape[0])
    if verbose:
      prog.update(batch)
    # factors
    _, ids = groundtruth.sample_factors(n_per_factor=batch,
                                        return_indices=True,
                                        seed=random_state.randint(0, 1e8))
    indices.append(ids)
    # inputs
    inps = []
    for xi, inp in zip(Xs, inputs):
      if tf.is_tensor(inp):
        inp = tf.gather(inp, indices=ids, axis=0)
      else:
        inp = inp[ids]
      xi.append(inp)
      inps.append(inp)
    # latents representation
    z = model.encode(inps[0] if len(inps) == 1 else inps, training=False)
    o = tf.nest.flatten(as_tuple(model.decode(z, training=False)))
    # post-process latents
    z = as_tuple(z)
    if len(z) == 1:
      z = z[0]
    Os.append(o)
    Zs.append(z)
    # update the counter
    n += len(ids)
  # end progress
  prog.clear()
  prog.close()
  # aggregate all data
  Xs = [np.concatenate(x, axis=0) for x in Xs]
  if isinstance(Zs[0], Distribution):
    Zs = Batchwise(Zs, name="Latents")
  else:
    Zs = Blockwise(
      [
        Batchwise(
          [z[zi] for z in Zs],
          name=f"Latents{zi}",
        ) for zi in range(len(Zs[0]))
      ],
      name="Latents",
    )
  Os = [
    Batchwise(
      [j[i] for j in Os],
      name=f"Output{i}",
    ) for i in range(len(Os[0]))
  ]
  indices = np.concatenate(indices, axis=0)
  groundtruth = groundtruth[indices]
  return Xs, groundtruth, Zs, Os, indices


# ===========================================================================
# GroundTruth
# ===========================================================================
def _fast_samples_indices(known: np.ndarray, factors: np.ndarray):
  outputs = [-1] * len(known)
  for k_idx in range(len(known)):
    for f_idx in range(len(factors)):
      if np.array_equal(known[k_idx], factors[f_idx]):
        if outputs[k_idx] < 0:
          outputs[k_idx] = f_idx
        elif bool(random.getrandbits(1)):
          outputs[k_idx] = f_idx
  return outputs


try:
  # with    numba: ~1.3 sec
  # without numba: ~19.3 sec
  # ~15 times faster
  from numba import jit

  _fast_samples_indices = jit(
    _fast_samples_indices,
    # target='cpu',
    cache=False,
    parallel=False,
    nopython=True)
except ImportError:
  pass


def _create_factor_filter(known: FactorFilter,
                          factor_names: List[str]
                          ) -> Callable[[Any], bool]:
  if callable(known):
    return known
  if known is None:
    known = {}
  if isinstance(known, dict):
    known = {
      factor_names.index(k) if isinstance(k, string_types) else int(k): v
      for k, v in known.items()
    }
    return lambda x: all(x[k] == v for k, v in known.items())
  else:
    return lambda x: x == known


class GroundTruth:
  """Discrete factor for disentanglement analysis. If the factors is continuous,
  the values are casted to `int64` For discretizing continuous factor
  `odin.bay.vi.discretizing`

  Parameters
  ----------
  factors : [type]
      `[num_samples, n_factors]`, an Integer array
  factor_names : [type], optional
      None or `[n_factors]`, list of name for each factor, by default None
  categorical : Union[bool, List[bool]], optional
      list of boolean indicate if the given factor is categorical values or
      continuous values.
      This gives significant meaning when trying to visualize
      the factors, by default False.

  Attributes
  ---------
  factor_labels : list of array,
      unique labels for each factor
  factor_sizes : list of Integer,
      number of factor for each factor

  Reference
  ---------
  Google research: https://github.com/google-research/disentanglement_lib

  Raises
  ------
  ValueError
      factors must be a matrix
  """

  def __init__(
      self,
      factors: Union[tf.Tensor, np.ndarray, tf.data.Dataset],
      factor_names: Optional[Sequence[str]] = None,
      categorical: Union[bool, List[bool]] = False,
      n_bins: Optional[Union[int, List[int]]] = None,
      strategy: Literal['uniform', 'quantile', 'kmeans', 'gmm'] = 'uniform',
  ):
    if isinstance(factors, tf.data.Dataset):
      factors = tf.stack([x for x in factors])
    if tf.is_tensor(factors):
      factors = factors.numpy()
    factors = np.atleast_2d(factors)
    if factors.ndim != 2:
      raise ValueError("factors must be a matrix [n_observations, n_factor], "
                       f"but given shape:{factors.shape}")
    # check factors is one-hot encoded
    if np.all(np.sum(factors, axis=-1) == 1):
      factors = np.argmax(factors, axis=1)[:, np.newaxis]
      categorical = True
    n_factors = factors.shape[1]
    # discretizing
    factors_original = np.array(factors)
    n_bins = as_tuple(n_bins, N=n_factors)
    strategy = as_tuple(strategy, N=n_factors, t=str)
    for i, (b, s) in enumerate(zip(n_bins, strategy)):
      if b is not None:
        factors[:, i] = discretizing(factors[:, i][:, np.newaxis],
                                     n_bins=b,
                                     strategy=s).ravel()
    factors = factors.astype(np.int64)
    # factor_names
    if factor_names is None:
      factor_names = [f'F{i}' for i in range(n_factors)]
    else:
      factor_names = [str(i) for i in ([factor_names]
                                       if not isinstance(factor_names,
                                                         (tuple, list,
                                                          np.ndarray))
                                       else factor_names)]
    assert len(factor_names) == n_factors, \
      f'Given {n_factors} but only {len(factor_names)} names'
    # store the attributes
    self._discrete_factors = factors
    self._original_factors = factors_original
    self.discretizer = list(zip(n_bins, strategy))
    self.categorical = as_tuple(categorical, N=n_factors, t=bool)
    self.factor_names = factor_names
    self.unique_values = [np.unique(x) for x in factors.T]
    self.sizes = [len(lab) for lab in self.unique_values]

  def is_categorical(self, factor_index: Union[int, str]) -> bool:
    if isinstance(factor_index, string_types):
      factor_index = self.factor_names.index(factor_index)
    return self.categorical[factor_index]

  def copy(self) -> 'GroundTruth':
    obj = GroundTruth.__new__(GroundTruth)
    obj._discrete_factors = self._discrete_factors
    obj._original_factors = self._original_factors
    obj.discretizer = self.discretizer
    obj.categorical = self.categorical
    obj.factor_names = self.factor_names
    obj.unique_values = self.unique_values
    obj.sizes = self.sizes
    return obj

  def __getitem__(self, key):
    obj = self.copy()
    obj._discrete_factors = obj._discrete_factors[key]
    obj._original_factors = obj._original_factors[key]
    return obj

  @property
  def original_factors(self) -> np.ndarray:
    return self._original_factors

  @property
  def discretized_factors(self) -> np.ndarray:
    return self._discrete_factors

  @property
  def shape(self) -> List[int]:
    return self._discrete_factors.shape

  @property
  def dtype(self) -> np.dtype:
    return self._discrete_factors.dtype

  @property
  def n_factors(self) -> int:
    return self._discrete_factors.shape[1]

  def sample_factors(
      self,
      factor_filter: FactorFilter = None,
      n_per_factor: int = 16,
      replace: bool = False,
      seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a batch of factors with output shape `[num, num_factor]`.

    Parameters
    ----------
    factor_filter : A Dictionary, mapping from factor_names or factor_index to
        factor_value, this establishes a list of known
        factors to sample from the unknown factors.
    n_per_factor : An Integer
        number of samples per factor
    replace : A Boolean
        replacement sample
    seed: int
        random seed

    Return
    ------
    factors : `[num, n_factors]`
        the samples
    indices : list of Integer
        the indices if sampled factors
    """
    factor_filter = _create_factor_filter(factor_filter, self.factor_names)
    # all samples with similar known factors
    samples = [(idx, x[None, :])
               for idx, x in enumerate(self._discrete_factors)
               if factor_filter(x)]
    rand = np.random.RandomState(seed)
    indices = rand.choice(len(samples), size=int(n_per_factor),
                          replace=replace)
    factors = np.vstack([samples[i][1] for i in indices])
    return factors, np.array([samples[i][0] for i in indices])

  def sample_indices_from_factors(self,
                                  factors: np.ndarray,
                                  seed: int = 1) -> np.ndarray:
    """Sample a batch of observations indices given a batch of factors.
      In other words, the algorithm find all the samples with matching factor
      in given batch, then return the indices of those samples.

    Parameters
    ----------
    factors : `[n_samples, n_factors]`
        the factors
    seed : None or `np.random.RandomState`
        the random seed

    Return
    ------
    indices : list of Integer
        indices
    """
    random_state = np.random.RandomState(seed=seed)
    random.seed(random_state.randint(1e8))
    if factors.ndim == 1:
      factors = np.expand_dims(factors, axis=0)
    assert factors.ndim == 2, "Only support matrix as factors."
    return np.array(_fast_samples_indices(factors, self._discrete_factors))

  def __str__(self):
    text = f'GroundTruth: {self._discrete_factors.shape}\n'
    for i, (discretizer, name, labels) in enumerate(
        zip(self.discretizer, self.factor_names, self.unique_values)):
      ftype = 'categorical' if self.categorical[i] else 'continuous'
      text += (f" Factor#{i} type:{ftype} n={len(labels)} "
               f"name:'{name}' discretizer:{discretizer} "
               f"values:[{','.join([str(i) for i in labels])}]\n")
    return text[:-1]


# ===========================================================================
# Disentanglement Gym
# ===========================================================================
class DisentanglementGym(vs.Visualizer):
  """Disentanglement Gym

  Parameters
  ----------
  dataset : str
      name of the dataset
  model : VariationalAutoencoder
      instance of `VariationalAutoencoder`
  batch_size : int, optional
      batch size, by default 64
  seed : int, optional
      seed for random state and reproducibility, by default 1
  """

  @typechecked
  def __init__(
      self,
      model: VariationalModel,
      dataset: Union[None, ImageDataset, DatasetNames, Text] = None,
      train: Optional[Any] = None,
      valid: Optional[Any] = None,
      test: Optional[Any] = None,
      labels_name: Optional[Sequence[str]] = None,
      batch_size: int = 32,
      dpi: int = 200,
      seed: int = 1):
    self.seed = int(seed)
    self.dpi = int(dpi)
    self._batch_size = int(batch_size)
    self.model = model
    # === 1. prepare dataset
    if isinstance(dataset, string_types):
      self.ds = get_dataset(dataset)
      self.dsname = str(dataset).lower().strip()
    elif isinstance(dataset, ImageDataset):
      self.ds = dataset
      self.dsname = dataset.name
    else:
      self.ds = None
      self.dsname = 'unknown'
    if dataset is None:
      self._labels_name = labels_name
      self._data = dict(train=train, valid=valid, test=test)
    else:
      self._labels_name = self.ds.labels
      kw = dict(batch_size=batch_size,
                label_percent=True,
                shuffle=1000,
                seed=seed)
      self._data = dict(
        train=self.ds.create_dataset(partition='train', **kw),
        valid=self.ds.create_dataset(partition='valid', **kw),
        test=self.ds.create_dataset(partition='test', **kw),
      )
    # === 3. attributes
    self._context_setup = False
    self._x_true = None
    self._y_true = None
    self._groundtruth = None
    self._px = None
    self._qz = None
    self._pz = None
    self._cache_key = None

  def _assert_sampled(self):
    assert self._context_setup, 'Call method run_model to produce the samples'

  @property
  def labels_name(self) -> Sequence[str]:
    return np.array([f'#{i} ' for i in range(self.y_true.shape[1])]) \
      if self._labels_name is None else self._labels_name

  @property
  def x_true(self) -> np.ndarray:
    self._assert_sampled()
    return self._x_true

  @property
  def y_true(self) -> np.ndarray:
    self._assert_sampled()
    return self._y_true

  @property
  def groundtruth(self) -> GroundTruth:
    self._assert_sampled()
    if self._groundtruth is None:
      n_bins = None
      factor_names = self.labels_name
      if self.dsname in ['fashionmnist', 'mnist',
                         'cifar10', 'cifar100', 'svhn',
                         'cortex', 'pbmc']:
        categorical = True
        factor_names = 'classes'
      elif self.dsname in ['shapes3d', 'shapes3dsmall']:
        categorical = [False, False, True, False, False, False]
        n_bins = [15, 8, 4, 10, 10, 10]
      elif self.dsname in ['dsprites', 'dspritessmall']:
        categorical = [False, False, True, False, False]
        n_bins = [10, 6, 3, 8, 8]
      elif self.dsname == 'halfmoons':
        categorical = [False, False, False, True]
        n_bins = [10, 10, 10, 4]
      else:
        raise NotImplementedError
      self._groundtruth = GroundTruth(self.y_true,
                                      factor_names=factor_names,
                                      categorical=categorical,
                                      n_bins=n_bins,
                                      strategy='uniform')
    return self._groundtruth

  @property
  def px_z(self) -> List[Batchwise]:
    """reconstruction: p(x|z)"""
    self._assert_sampled()
    return list(self._px)

  @property
  def qz_x(self) -> List[Batchwise]:
    """latents posterior: q(z|x)"""
    self._assert_sampled()
    return list(self._qz)

  @property
  def pz(self) -> List[Batchwise]:
    """latents prior: p(z) or p(z_i|z_j)"""
    self._assert_sampled()
    return list(self._pz)

  @property
  def n_samples(self) -> int:
    self._assert_sampled()
    return self.x_true.shape[0]

  @property
  def n_latents(self) -> int:
    self._assert_sampled()
    return int(sum(
      np.prod((q.batch_shape + q.event_shape)[1:])
      for q in self.qz_x))

  @property
  def n_factors(self) -> int:
    self._assert_sampled()
    return self.y_true.shape[1]

  def get_correlation_matrix(
      self,
      convert_fn: ConvertFunction = first_mean,
      method: CorrelationMethod = 'spearman',
      n_neighbors: int = 3,
      n_cpu: int = 1,
      sort_pairs: bool = False,
  ) -> np.ndarray:
    """Correlation matrix of `latent codes` (row) and `groundtruth factors`
    (column).

    Parameters
    ----------
    convert_fn : Callable
        convert list of Distribution to a Tensor
    method : {'spearman', 'pearson', 'lasso', 'mi', 'importance'}
        method for calculating the correlation,
        'spearman': rank or monotonic correlation
        'pearson': linear correlation
        'lasso': lasso regression
        'mi': mutual information
        'importance': importance matrix estimated by GBTree
        by default 'spearman'
    n_neighbors : int
        number of neighbors for estimating mutual information (only used for
        method = 'mi')
    n_cpu : int
        number of cpu for calculation of mutual information or importance
        matrix
    sort_pairs : bool, optional
        If True, reorganize the row of correlation matrix
        for the best match between code-factor (i.e. the largest diagonal sum).
        Note: the decoding is performed on train matrix, then applied to test
        matrix, by default False

    Returns
    -------
    ndarray
        correlation matrices `[n_latents, n_factors]`, all entries are in `[0, 1]`.
    OrderedDict (optional)
        mapping from best matched: factor index to latent code index.
    """
    z = convert_fn(self.qz_x).numpy()
    f = self.y_true
    if method in ('spearman', 'pearson', 'lasso'):
      mat = correlation_matrix(x1=z,
                               x2=f,
                               method=method,
                               cache_key=self._cache_key,
                               seed=self.seed)
    elif method == 'mi':
      mat = mutual_info_estimate(representations=z,
                                 factors=f,
                                 continuous_latents=True,
                                 continuous_factors=False,
                                 n_neighbors=n_neighbors,
                                 n_cpu=n_cpu,
                                 cache_key=self._cache_key,
                                 seed=self.seed)
    elif method == 'importance':
      mat = importance_matrix(repr_train=z,
                              factor_train=f,
                              test_size=0.4,
                              seed=self.seed,
                              cache_key=self._cache_key,
                              n_jobs=1)[0]
    else:
      raise ValueError(f'No support for correlation method: {method}')
    ## decoding and return
    if sort_pairs:
      ids = diagonal_linear_assignment(mat)
      mat = mat[ids, :]
      return mat, OrderedDict(zip(range(self.n_factors), ids))
    return mat

  @contextlib.contextmanager
  def run_model(self,
                *,
                n_samples: int = -1,
                partition: DataPartition = 'test',
                verbose: bool = True) -> 'DisentanglementGym':
    # === 0. setup
    tf.random.set_seed(self.seed)
    self._context_setup = True
    self._cache_key = uuid(12)
    # === 1. prepare data
    ds = self._data[partition]
    if ds is None:
      raise ValueError(f'No dataset for partition {partition}')
    if not isinstance(ds, tf.data.Dataset):
      if isinstance(ds, TensorType):
        ds = tf.data.Dataset.from_tensor_slices(ds)
      elif isinstance(ds, (tuple, list)):
        ds = tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensor_slices(i)
                                        for i in ds]))
      else:
        raise ValueError(f'No support for dataset type: {type(ds)}')
    if n_samples > 0:
      ds = ds.take(int(np.ceil(n_samples / self._batch_size)))
    structure = tf.data.experimental.get_structure(ds)
    assert len(structure) == 2, \
      f'Dataset must return inputs and target, but given: {structure}'
    progress = tqdm(ds,
                    desc=f"{self.model.name}-{self.dsname}",
                    disable=not verbose)
    # === 2. running
    x_true = []
    y_true = []
    P_xs = []
    Q_zs = []
    P_zs = []
    for x, y in progress:
      P, Q = self.model(x, training=False)
      Q, Q_prior = self.model.get_latents(return_prior=True)
      P = as_tuple(P)
      Q = as_tuple(Q)
      Q_prior = as_tuple(Q_prior)
      x_true.append(x)
      y_true.append(y)
      P_xs.append(_dist(P))
      Q_zs.append(_dist(Q))
      P_zs.append(_dist(Q_prior))
    # for the reconstruction
    n_reconstruction = len(P_xs[0])
    all_px = [Batchwise([x[i] for x in P_xs]) for i in range(n_reconstruction)]
    # latents
    n_latents = len(Q_zs[0])
    all_qz = [Batchwise([z[i] for z in Q_zs]) for i in range(n_latents)]
    all_pz = []
    for i in range(n_latents):
      p = [z[i] for z in P_zs]
      if all(qz.batch_shape == pz.batch_shape
             for qz, pz in zip(all_qz[i].distributions, p)):
        p = Batchwise(p)
      else:
        p = p[0]
      all_pz.append(p)
    # labels
    x_true = tf.concat(x_true, axis=0).numpy()
    y_true = tf.concat(y_true, axis=0).numpy()
    # === 3. save attributes
    self._px = all_px
    self._qz = all_qz
    self._pz = all_pz
    self._x_true = x_true
    self._y_true = y_true
    try:
      import seaborn
      seaborn.set()
    except ImportError:
      plt.rc('axes', axisbelow=True)
    yield self
    self._context_setup = False

  def plot_reconstruction(self,
                          n_images: int = 36,
                          title: str = '') -> plt.Figure:
    self._assert_sampled()
    rand = np.random.RandomState(self.seed)
    n_rows = int(np.sqrt(n_images))
    ids = rand.permutation(self.n_samples)[:n_images]
    org = _prepare_images(self.x_true[ids])
    rec = _prepare_images(self.px_z[0].mean().numpy()[ids], normalize=True)
    fig = plt.figure(figsize=(12, 7), dpi=self.dpi)
    vs.plot_images(org, grids=(n_rows, n_rows), ax=(1, 2, 1),
                   title=f'{title} Original')
    vs.plot_images(rec, grids=(n_rows, n_rows), ax=(1, 2, 2),
                   title=f'{title} Reconstructed '
                         f'(llk:{self.log_likelihood()[0]:.2f})')
    plt.tight_layout()
    self.add_figure(f'reconstruction{title}', fig)
    return fig

  def plot_distortion(self, title: str = '') -> plt.Figure:
    with tf.device('/CPU:0'):
      start = 0
      llk = []
      for px in self.px_z[0].distributions:
        x = self.x_true[start: start + px.batch_shape[0]]
        start += px.batch_shape[0]
        if hasattr(px, 'distribution'):
          px = px.distribution
        if isinstance(px, Bernoulli):
          px = Bernoulli(logits=px.logits)
        elif isinstance(px, Normal):
          px = Normal(loc=px.loc, scale=px.scale)
        elif isinstance(px, QuantizedLogistic):
          px = QuantizedLogistic(loc=px.loc, scale=px.scale,
                                 low=px.low, high=px.high,
                                 inputs_domain=px.inputs_domain,
                                 reinterpreted_batch_ndims=None)
        elif isinstance(px, MixtureQuantizedLogistic):
          raise NotImplementedError
        else:
          raise NotImplementedError
        llk.append(px.log_prob(x))
      # aggregate and statistics
      llk = -np.concatenate(llk, 0)
      mean = np.mean(llk, 0)
      mean_lims = (np.min(mean), np.max(mean))
      std = np.std(llk, 0)
      std_lims = (np.min(std), np.max(std))
      n_channels = llk.shape[-1]

      # helper
      def ax_config(im, ax, lims):
        ax.axis('off')
        ax.margins(0)
        ax.grid(False)
        ticks = np.linspace(lims[0], lims[1], num=5)
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=ticks)
        cbar.ax.set_yticklabels([f'{i:.2f}' for i in ticks])
        cbar.ax.tick_params(labelsize=6, length=2, width=0.5)

      # plotting
      fig = plt.figure(figsize=(2 * 2, n_channels * 2), dpi=self.dpi)
      idx = 1
      for i in range(n_channels):
        # mean
        ax = plt.subplot(n_channels, 2, idx)
        im = ax.pcolormesh(mean[:, :, i], cmap='Spectral',
                           vmin=mean_lims[0], vmax=mean_lims[1],
                           linewidth=0, rasterized=True)
        ax.set_title(rf'$\mu$', fontsize=6)
        ax_config(im, ax, mean_lims)
        ax.set_ylabel(f'Channel{i}')
        idx += 1
        # std
        ax = plt.subplot(n_channels, 2, idx)
        im = ax.pcolormesh(std[:, :, i], cmap='Spectral',
                           vmin=std_lims[0], vmax=std_lims[1],
                           linewidth=0, rasterized=True)
        ax.set_title(rf'$\sigma$', fontsize=6)
        ax_config(im, ax, std_lims)
        idx += 1
      plt.tight_layout()
    self.add_figure(f'distortion{title}', fig)
    return fig

  def plot_latents_stats(self,
                         latent_idx: int = 0,
                         title: str = '') -> plt.Figure:
    from odin.bay import Vamprior
    self._assert_sampled()
    rand = np.random.RandomState(seed=self.seed)
    # === 0. prepare the latents
    qz = self.qz_x[latent_idx]
    pz = self.pz[latent_idx]
    mean = np.mean(qz.mean(), 0)
    stddev = np.mean(qz.stddev(), 0)
    zdims = mean.shape
    # sort by stddev
    ids = np.argsort(stddev)
    mean = mean[ids]
    stddev = stddev[ids]
    # kld
    with tf.device('/CPU:0'):
      q = Normal(qz.mean(), qz.stddev())
      z = q.sample(seed=rand.randint(1e8))
      if isinstance(pz, Vamprior):
        C = pz.C
        p = pz.distribution  # [n_components, zdim]
        p = Normal(loc=p.mean(), scale=p.stddev())
        kld = tf.reduce_mean(
          q.log_prob(z) -
          (tf.reduce_logsumexp(p.log_prob(tf.expand_dims(z, 1)), 1) - C),
          axis=0)
      else:
        p = Normal(pz.mean(), pz.stddev())
        kld = tf.reduce_mean(q.log_prob(z) - p.log_prob(z), axis=0)
    kld = kld.numpy()[ids]
    # === 1. get the weights
    w_d = None
    if hasattr(self.model, 'decoder'):
      w = self.model.decoder.trainable_variables[0]
      if w.shape[:-1] == zdims:
        w_d = w
        if w_d.shape.ndims == 2:  # dense weights
          w_d = tf.linalg.norm(w_d, axis=-1)
        else:  # convolution weights
          w_d = tf.linalg.norm(w_d, axis=(0, 1, 2))
        w_d = w_d.numpy()[ids]
    else:
      for w in self.model.trainable_variables:
        name = w.name
        if w.shape.rank > 0 and w.shape[0] == zdims and '/kernel' in name:
          w_d = tf.linalg.norm(tf.reshape(w, (w.shape[0], -1)), axis=1).numpy()
          break
    # === 2. plotting
    fig = plt.figure(figsize=(np.log2(np.prod(zdims)) + 2, 5), dpi=self.dpi)
    ax = plot_latent_stats(mean, stddev, kld, w_d, plt.gca())
    collapse = int(np.sum(np.abs(stddev - 1.0) <= 0.05))
    ax.set_title(f'{title} Z{latent_idx} '
                 f'{collapse}/{np.prod(zdims)} collapsed '
                 f'(kl:{self.kl_divergence()[latent_idx]:.2f})')
    self.add_figure(f'latents_stats{latent_idx}', fig)
    return fig

  def plot_latents_factors(
      self,
      convert_fn: ConvertFunction = first_mean,
      method: CorrelationMethod = 'spearman',
      n_points: int = 5000,
      title: str = '') -> plt.Figure:
    """Plot pair of `the two most correlated latent dimension` and each
    `factor`.

    Parameters
    ----------
    convert_fn : Callable
        convert list of Distribution to a Tensor
    method : {'spearman', 'pearson', 'lasso', 'mi', 'importance'}
        correlation method
    n_points : int
        number of scatter points
    title : str
        figure title
    """
    self._assert_sampled()
    # === 0. prepare data
    mat = self.get_correlation_matrix(convert_fn, method)
    n_latents, n_factors = mat.shape
    mat = np.abs(mat)
    z = convert_fn(self.qz_x)
    f = self.y_true
    assert z.shape[1] == n_latents, \
      f'z={z.shape} f={f.shape} corr={mat.shape}'
    assert f.shape[1] == n_factors, \
      f'z={z.shape} f={f.shape} corr={mat.shape}'
    assert z.shape[0] == f.shape[0], \
      f'z={z.shape} f={f.shape} corr={mat.shape}'
    labels = self.ds.labels
    # === 1. shuffling
    rand = np.random.RandomState(seed=self.seed)
    ids = rand.permutation(z.shape[0])
    z = np.asarray(z)[ids][:int(n_points)]
    f = np.asarray(f)[ids][:int(n_points)]
    ## find the best latents for each labels
    f2z = {f_idx: z_idx
           for f_idx, z_idx in enumerate(np.argmax(mat, axis=0))}
    ## special cases
    selected_labels = set(labels)
    n_pairs = len(selected_labels) * (len(selected_labels) - 1) // 2
    ## plotting each pairs
    ncol = 2
    nrow = n_pairs
    fig = plt.figure(figsize=(ncol * 3.5, nrow * 3), dpi=self.dpi)
    c = 1
    styles = dict(size=10, alpha=0.8, color='bwr', cbar=True, cbar_nticks=5,
                  cbar_ticks_rotation=0, cbar_fontsize=8, fontsize=10,
                  grid=False)
    for f1 in range(n_factors):
      for f2 in range(f1 + 1, n_factors):
        if (labels[f1] not in selected_labels or
            labels[f2] not in selected_labels):
          continue
        name1 = labels[f1]
        name2 = labels[f2]
        z1 = f2z[f1]  # best for f1
        z2 = f2z[f2]  # best for f2
        vs.plot_scatter(x=z[:, z1],
                        y=z[:, z2],
                        val=f[:, f1].astype(np.float32),
                        xlabel=f"Z#{z1} - best for '{name1}'",
                        ylabel=f"Z#{z2} - best for '{name2}'",
                        title=f"Colored by: '{name1}'",
                        ax=(nrow, ncol, c),
                        **styles)
        vs.plot_scatter(x=z[:, z1],
                        y=z[:, z2],
                        val=f[:, f2].astype(np.float32),
                        xlabel=f"Z#{z1} - best for '{name1}'",
                        ylabel=f"Z#{z2} - best for '{name2}'",
                        title=f"Colored by: '{name2}'",
                        ax=(nrow, ncol, c + 1),
                        **styles)
        c += 2
    plt.tight_layout()
    if len(title) > 0:
      plt.suptitle(f"{title}")
      plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    self.add_figure(f'pairs_{method}{title}', fig)
    return fig

  def plot_latents_tsne(
      self,
      convert_fn: ConvertFunction = concat_mean,
      y_true: Optional[np.ndarray] = None,
      n_points: int = 2000,
      use_umap: bool = False,
      title: str = '') -> plt.Figure:
    self._assert_sampled()
    z = convert_fn(self.qz_x)
    if hasattr(z, 'numpy'):
      z = z.numpy()
    y = self.y_true if y_true is None else y_true
    if hasattr(y, 'numpy'):
      y = y.numpy()
    y = _prepare_categorical(y, self.ds)
    # shuffling
    fig = plt.figure(figsize=(8, 8), dpi=self.dpi)
    rand = np.random.RandomState(seed=self.seed)
    ids = rand.permutation(z.shape[0])[:n_points]
    if use_umap:
      z = DimReduce.UMAP(z[ids],
                         random_state=rand.randint(1e8))
    else:
      z = DimReduce.TSNE(z[ids],
                         random_state=rand.randint(1e8),
                         framework='sklearn')
    y = y[ids]
    # update title
    scores = self.clustering_score()
    new_title = f"{title} " \
                f"{' '.join([f'{k}:{v:.2f}' for k, v in scores.items()])}"
    # plot
    ax = plt.gca()
    vs.plot_scatter(x=z[:, 0], y=z[:, 1], grid=False, legend_ncol=2,
                    size=12.0, alpha=0.8, color=y, title=new_title, ax=ax)
    self.add_figure(f"latents_{'umap' if use_umap else 'tsne'}{title}",
                    fig)
    return fig

  def plot_latents_traverse(
      self,
      factors: FactorFilter = None,
      n_traverse_points: int = 31,
      n_top_latents: int = 10,
      min_val: Optional[float] = -3,
      max_val: Optional[float] = 3,
      mode: Literal['linear', 'quantile', 'gaussian'] = 'linear',
      seed: int = 1,
      title: str = '') -> plt.Figure:
    self._assert_sampled()
    if min_val is None or max_val is None:
      # should be max here
      stddev = np.max(self.qz_x[0].stddev(), 0)
      if max_val is None:
        max_val = 3 * stddev
      if min_val is None:
        min_val = -3 * stddev
    n_top_latents = min(self.n_latents, n_top_latents)
    factors, idx = self.groundtruth.sample_factors(
      factor_filter=factors,
      n_per_factor=1,
      seed=seed)
    factors, idx = factors[0], idx[0]
    x = self.x_true[idx:idx + 1]
    images, top_latents = self.model.sample_traverse(
      x,
      min_val=min_val,
      max_val=max_val,
      n_best_latents=n_top_latents,
      n_traverse_points=n_traverse_points,
      mode=mode)
    images = as_tuple(images)[0]
    images = _prepare_images(images.mean().numpy(), normalize=True)
    ## plotting
    fig = plt.figure(figsize=(1.5 * n_traverse_points, 1.5 * n_top_latents),
                     dpi=self.dpi)
    vs.plot_images(images, grids=(n_top_latents, n_traverse_points),
                   ax=plt.gca())
    plt.tight_layout()
    plt.title(f'{title} Mode={mode} Factors={factors} Latents={top_latents}',
              fontsize=12)
    self.add_figure(f'traverse_{mode}{title}', fig)
    return fig

  def plot_latents_sampling(self,
                            n_images: int = 36,
                            title: str = '') -> plt.Figure:
    n_rows = int(np.sqrt(n_images))
    images = self.model.sample_observation(n=n_images, seed=self.seed)
    images = as_tuple(images)[0]
    images = _prepare_images(images.mean().numpy(), normalize=True)
    fig = plt.figure(figsize=(5, 5), dpi=self.dpi)
    vs.plot_images(images, grids=(n_rows, n_rows), title='Sampled')
    if len(title) > 0:
      plt.suptitle(f"{title}")
      plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    self.add_figure(f'latents_sampling{title}', fig)
    return fig

  def plot_correlation(
      self,
      convert_fn: ConvertFunction = first_mean,
      method: CorrelationMethod = 'spearman',
      n_top_latents: Optional[int] = None,
      sorting: Literal['match', 'stddev', 'total'] = 'match',
      title: str = '') -> plt.Figure:
    """Plot correlation matrix between latents and factors"""
    if n_top_latents is None:
      n_top_latents = self.n_factors
    # [n_latents, n_factors]
    mat = self.get_correlation_matrix(convert_fn=convert_fn, method=method)
    if n_top_latents < 0:
      n_top_latents = mat.shape[0]
    mat = np.abs(mat)
    # normalize
    vmin = np.min(mat)
    vmax = np.max(mat)
    mat = (mat - vmin) / (vmax - vmin)
    # [n_factors, n_latents]
    mat = mat.T
    # sorting top latents
    if sorting == 'match':
      latent_ids = diagonal_linear_assignment(mat)[:n_top_latents]
    elif sorting == 'stddev':
      stddev = np.concatenate([np.mean(q.stddev(), 0) for q in self.qz_x], 0)
      latent_ids = np.argsort(stddev)[:n_top_latents]  # smaller better
      latent_ids = latent_ids[diagonal_linear_assignment(mat[:, latent_ids])]
    elif sorting == 'total':
      latent_ids = np.argsort(np.sum(mat, 0))[::-1][:n_top_latents]
      latent_ids = latent_ids[diagonal_linear_assignment(mat[:, latent_ids])]
    else:
      raise NotImplementedError(f'No support for sorting method: {sorting}')
    mat = mat[:, latent_ids]
    n_factors, n_latents = mat.shape
    # plotting
    fig = plt.figure(figsize=(n_latents, n_factors), dpi=self.dpi)
    ax = plt.gca()
    ax.set_facecolor('dimgrey')
    for i, row in enumerate(mat):
      row_max = np.argmax(row)
      for j, col in enumerate(row):
        if j == row_max:
          color = 'salmon'
        else:
          color = 'white' if i != j else 'silver'
        point = plt.Circle((j + 1, i + 1),
                           radius=col * 0.35,
                           color=color)
        ax.add_patch(point)
        # ax.scatter(j, i, s=col * s, marker='o',
        #            c='white' if i != j else 'red')
    ax.grid(True, linewidth=0.5, color='gray', alpha=0.5)
    # x-axis
    ax.set_xticks(np.arange(n_latents + 1) + 1)
    ax.set_xticklabels([f'{i}' for i in latent_ids] + [''], fontsize=8)
    ax.set_xlabel('Latents')
    # y-axis
    ax.set_yticks(np.arange(n_factors + 1) + 1)
    ax.set_yticklabels(list(self.labels_name) + [''], fontsize=8)
    ax.set_ylabel('Factors')
    ax.set_title(f'{title} Diagonal {method} score: {sum(np.diag(mat)):.2f}')
    self.add_figure(f'{method}_{sorting}{title}', fig)
    return fig

  def plot_histogram_disentanglement(
      self,
      convert_fn: ConvertFunction = first_mean,
      n_bins_per_factors: int = 15,
      n_bins_per_latents: int = 80,
      method: CorrelationMethod = 'spearman',
      title: str = '') -> plt.Figure:
    """ To illustrate the disentanglement of the codes, the codes' histogram
    bars are colored by the value of factors.
    """
    ### prepare styled plot
    styles = dict(fontsize=12,
                  cbar_horizontal=False,
                  bins_color=int(n_bins_per_factors),
                  bins=int(n_bins_per_latents),
                  color='bwr',
                  cbar=False,
                  alpha=0.8)
    # get all relevant factors
    factor_indices = np.arange(self.n_factors)
    ### correlation
    corr = self.get_correlation_matrix(convert_fn, method=method)
    ### sorting the latents
    latent_indices = diagonal_linear_assignment(np.abs(corr), nan_policy=0)
    latent_indices = latent_indices[:len(factor_indices)]
    corr = corr[:, latent_indices]
    ### prepare the factors
    F = self.y_true[:, factor_indices]
    factor_names = np.asarray(self.labels_name)[factor_indices]
    # codes
    Z = convert_fn(self.qz_x).numpy()
    latent_names = np.array([f'Z{i}' for i in range(Z.shape[1])])
    Z = Z[:, latent_indices]
    latent_names = latent_names[latent_indices]
    ### create the figure
    n_row = F.shape[1]
    n_col = Z.shape[1] + 1
    fig = plt.figure(figsize=(n_col * 2.8, n_row * 3), dpi=self.dpi)
    count = 1
    for fidx, (f, fname) in enumerate(zip(F.T, factor_names)):
      # the first plot show how the factor clustered
      ax, _, _ = vs.plot_histogram(x=f,
                                   color_val=f,
                                   ax=(n_row, n_col, count),
                                   title=f"{fname}",
                                   **styles)
      ax.set_facecolor('silver')
      ax.tick_params(axis='y', labelleft=False)
      count += 1
      # the rest of the row show how the codes align with the factor
      for zidx, (score, z,
                 zname) in enumerate(zip(corr[fidx], Z.T, latent_names)):
        text = "*" if fidx == zidx else ""
        ax, _, _ = vs.plot_histogram(
          x=z,
          color_val=f,
          ax=(n_row, n_col, count),
          title=f"{text}{fname}-{zname} (${score:.2f}$)",
          bold_title=True if fidx == zidx else False,
          **styles)
        ax.tick_params(axis='y', labelleft=False)
        if fidx == zidx:
          ax.set_facecolor('silver')
        count += 1
    ### fine tune the plot
    fig.suptitle(f"{method} {title}", fontsize=20)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    self.add_figure(f'histogram_{method}{title}', fig)
    return fig

  def plot_interpolation(
      self,
      factor1: FactorFilter,
      factor2: Optional[FactorFilter] = None,
      n_points: int = 10,
      latent_idx: int = 0,
      strategy: Literal['mixing', 'dropout', 'noise'] = 'mixing',
      title: str = '') -> Tuple[plt.Figure, plt.Figure]:
    tf.random.set_seed(self.seed)
    f1, idx1 = self.groundtruth.sample_factors(factor1,
                                               n_per_factor=1,
                                               seed=self.seed)
    f1, idx1 = f1[0], idx1[0]
    if strategy == 'mixing':
      f2, idx2 = self.groundtruth.sample_factors(factor2,
                                                 n_per_factor=1,
                                                 seed=self.seed)
      f2, idx2 = f2[0], idx2[0]
    x1 = self.x_true[idx1:idx1 + 1]
    x2 = self.x_true[idx2:idx2 + 1]
    _, q1 = self.model(x1)
    _, q2 = self.model(x2)

    # applying interpolation
    images = defaultdict(list)
    latents = []
    alpha = []
    for a in np.linspace(0.01, 0.99, num=n_points):
      alpha.append(a)
      # mixing
      x = x2 * a + (1. - a) * x1
      images['mixing'].append(x)
      p, q = self.model(x)
      images['mixing_rec'].append(as_tuple(p)[0].mean())
      latents.append(q)
      # latent interpolation
      z = [z2 * a + (1 - a) * z1 for z1, z2 in zip(as_tuple(q1), as_tuple(q2))]
      if not isinstance(q1, (tuple, list)):
        z = z[0]
      p = self.model.decode(z)
      images['mixing_latents'].append(as_tuple(p)[0].mean())
    # === 1. plot latents
    ids = np.argsort(as_tuple(q1)[latent_idx].stddev().numpy().ravel())

    def stats(dist):
      d = as_tuple(dist)[latent_idx]
      mean = d.mean().numpy()[0]
      std = d.stddev().numpy()[0]
      return mean[ids], std[ids]

    fig_latents = plt.figure(figsize=(8, 5))
    m, s = stats(q1)
    plt.plot(m, color='green', label=r'$\mu_{x_1}$', marker='s', markersize=2,
             linewidth=0.5, alpha=0.5)
    plt.plot(s, color='green', label=r'$\sigma_{x_1}$', linestyle='--',
             linewidth=0.6, alpha=0.9)
    m, s = stats(q2)
    plt.plot(m, color='red', label=r'$\mu_{x_2}$', marker='s', markersize=2,
             linewidth=0.5, alpha=0.5)
    plt.plot(s, color='red', label=r'$\sigma_{x_2}$', linestyle='--',
             linewidth=0.6, alpha=0.9)
    cmap = plt.get_cmap('binary')
    for i, qz in enumerate(latents):
      m, s = stats(qz)
      a = alpha[i]
      color = cmap(a)
      line = plt.plot(m, marker='.', markersize=4 + a * 10, linewidth=0,
                      markeredgewidth=0, label=rf'$\mu$-{a:.2f}',
                      color=color, alpha=0.6)
      plt.plot(s, linewidth=0.5, alpha=0.5, label=rf'$\sigma$-{a:.2f}',
               color=line[0].get_color())
    plt.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle(f"{title}")
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    # === 2. plot images
    n_row = len(images)
    n_col = n_points
    fig_reconstruction = plt.figure(figsize=(1.5 * n_col, 1.5 * n_row),
                                    dpi=self.dpi)
    count = 1
    for row, (name, imgs) in enumerate(images.items()):
      for col, img in enumerate(imgs):
        ax = plt.subplot(n_row, n_col, count)
        img = _prepare_images(img, normalize=True)[0]
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.tick_params('both',
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
        ax.grid(False)
        if col == 0:
          ax.set_ylabel(name)
        if row == 0:
          ax.set_title(f'a={alpha[col]:.2f}')
        count += 1
    plt.tight_layout()
    plt.suptitle(f"{title}")
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    # save figures
    self.add_figure(f'interp_reconst{title}', fig_reconstruction)
    self.add_figure(f'interp_latents{title}', fig_latents)
    return fig_reconstruction, fig_latents

  def mig_score(self,
                convert_fn: ConvertFunction = concat_mean,
                n_bins_per_latent: int = 10) -> float:
    """Mutual Information Gap

    Parameters
    ----------
    convert_fn : Callable
        convert list of distribution to single Tensor
    n_bins_per_latent : int, optional
        number of bins for discretizing the latents, by default 10

    Returns
    -------
    float
        mutual information gap score
    """
    self._assert_sampled()
    z = convert_fn(self.qz_x).numpy()
    if n_bins_per_latent > 1:
      z = discretizing(z, independent=True, n_bins=n_bins_per_latent,
                       strategy='uniform', seed=self.seed)
    f = self.groundtruth.discretized_factors
    return mutual_info_gap(z, f)

  def sap_score(self, convert_fn: ConvertFunction = concat_mean) -> float:
    """Separated attributes predictability"""
    self._assert_sampled()
    z = convert_fn(self.qz_x).numpy()
    f = self.groundtruth.discretized_factors
    return separated_attr_predictability(z, f)

  def dci_score(self, convert_fn: ConvertFunction = concat_mean) -> float:
    """Disentanglement completeness and informativeness"""
    self._assert_sampled()
    z = convert_fn(self.qz_x).numpy()
    f = self.groundtruth.discretized_factors
    return dci_scores(z, f, cache_key=self._cache_key)

  def betavae_score(self,
                    latent_idx: int = 0,
                    n_samples: int = 10000) -> float:
    self._assert_sampled()
    f = self.groundtruth.discretized_factors
    return beta_vae_score(representations=self.qz_x[latent_idx],
                          factors=f,
                          n_samples=n_samples,
                          seed=self.seed,
                          verbose=True)

  def factorvae_score(self,
                      latent_idx: int = 0,
                      n_samples: int = 10000) -> float:
    self._assert_sampled()
    f = self.groundtruth.discretized_factors
    return factor_vae_score(representations=self.qz_x[latent_idx],
                            factors=f,
                            n_samples=n_samples,
                            seed=self.seed,
                            verbose=True)

  def clustering_score(
      self, convert_fn: ConvertFunction = concat_mean) -> Dict[str, float]:
    self._assert_sampled()
    if 'clustering' not in _CACHE[self._cache_key]:
      z = convert_fn(self.qz_x).numpy()
      y_true = _prepare_categorical(self.y_true, self.ds, return_index=True)
      y_pred = fast_kmeans(z,
                           n_clusters=len(np.unique(y_true)),
                           n_init=200,
                           random_state=self.seed).predict(z)
      _CACHE[self._cache_key]['clustering'] = dict(
        ari=metrics.adjusted_rand_score(y_true, y_pred),
        ami=metrics.adjusted_mutual_info_score(y_true, y_pred),
        nmi=metrics.normalized_mutual_info_score(y_true, y_pred),
        asw=metrics.silhouette_score(z, y_true, random_state=self.seed)
      )
    return _CACHE[self._cache_key]['clustering']

  def relative_disentanglement_strength(
      self,
      convert_fn: ConvertFunction = concat_mean,
      method: CorrelationMethod = 'spearman') -> float:
    """Relative strength for both axes of correlation matrix.
    Basically, is the mean of normalized maximum correlation per code, and
    per factor.

    Return
    ------
    a scalar - higher is better
    """
    corr_matrix = self.get_correlation_matrix(convert_fn, method=method)
    return relative_strength(corr_matrix)

  def total_correlation(self, latent_idx: int = 0) -> float:
    """ Estimation of total correlation based on fitted Gaussian

    Return
    ------
    total correlation estimation - smaller is better
    """
    # memory complexity O(n*n*d), better do it on CPU
    with tf.device('/CPU:0'):
      qz = self.qz_x[latent_idx]
      return float(total_correlation(qz.sample(seed=self.seed), qz).numpy())

  def elbo(self) -> float:
    from odin.bay.helpers import kl_divergence
    self._assert_sampled()
    with tf.device('/CPU:0'):
      if 'elbo' not in _CACHE[self._cache_key]:
        llk = {f'llk{i}': p.log_prob(tf.convert_to_tensor(x))
               for i, (p, x) in enumerate(zip(self.px_z,
                                              [self.x_true, self.y_true]))}
        kl = {f'kl{i}': kl_divergence(q, p, analytic=False)
              for i, (q, p) in enumerate(zip(self.qz_x, self.pz))}
        _CACHE[self._cache_key]['elbo'] = float(
          np.mean(self.model.elbo(llk, kl)))
    return _CACHE[self._cache_key]['elbo']

  def log_likelihood(self) -> Sequence[float]:
    """Conditional log likelihood `p(x|z)`"""
    self._assert_sampled()
    with tf.device('/CPU:0'):
      if 'llk' not in _CACHE[self._cache_key]:
        llk = []
        for i, (p, x) in enumerate(zip(self.px_z, [self.x_true, self.y_true])):
          x = tf.convert_to_tensor(x)
          llk.append(np.mean(p.log_prob(x)))
        _CACHE[self._cache_key]['llk'] = tuple(llk)
      return _CACHE[self._cache_key]['llk']

  def accuracy_score(self) -> Sequence[float]:
    """Measure the accuracy of the reconstruction and the original data,
    for continuous examples, use threshold value 0.5"""
    self._assert_sampled()
    with tf.device('/CPU:0'):
      if 'acc' not in _CACHE[self._cache_key]:
        acc = []
        for i, (p, x) in enumerate(zip(self.px_z, [self.x_true, self.y_true])):
          true = np.asarray(x)
          if 'Relaxed' in str(type(p.distributions[0])):
            pred = np.concatenate(
              [d.distribution.probs_parameter() for d in p.distributions], 0)
          else:
            try:
              pred = p.mean().numpy()
            except:
              pred = p.mode().numpy()
          true = np.reshape(true, (true.shape[0], -1))
          pred = np.reshape(pred, (pred.shape[0], -1))
          if np.all(np.sum(true, -1) == 1.):
            true = np.argmax(true, -1)
            pred = np.argmax(pred, -1)
          else:
            true = np.ravel(np.round(true))
            pred = np.ravel(np.round(pred))
          acc.append(np.sum(true == pred) / len(true))
        _CACHE[self._cache_key]['acc'] = tuple(acc)
      return _CACHE[self._cache_key]['acc']

  def kl_divergence(self) -> Sequence[float]:
    self._assert_sampled()
    rand = np.random.RandomState(seed=self.seed)
    with tf.device('/CPU:0'):
      if 'kl' not in _CACHE[self._cache_key]:
        kl = []
        for i, (q, p) in enumerate(zip(self.qz_x, self.pz)):
          z = q.sample(seed=rand.randint(1e8))
          kl.append(
            float(tf.reduce_mean(q.log_prob(z) - p.log_prob(z)).numpy()))
        _CACHE[self._cache_key]['kl'] = tuple(kl)
      return _CACHE[self._cache_key]['kl']

  def active_units(self) -> Sequence[int]:
    self._assert_sampled()
    with tf.device('/CPU:0'):
      au = []
      for q, p in zip(self.qz_x, self.pz):
        q_std = np.mean(q.stddev(), axis=0).ravel()
        p_std = p.stddev()
        if p_std.shape.rank > 1:
          p_std = np.mean(p_std, axis=0).ravel()
        au.append(np.sum(np.abs(q_std - p_std) >= 0.05 * p_std))
      return tuple(au)
