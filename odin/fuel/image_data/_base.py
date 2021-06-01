import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing_extensions import Literal

from odin.backend.types_helpers import DataType, LabelType
from odin.fuel.dataset_base import IterableDataset, get_partition, Partition
from odin.utils import as_tuple
from odin.utils.cache_utils import get_cache_path
from tensorflow.python.data import Dataset


def _extract_labeled_examples(ds, n_labeled: int,
                              normalize_method=None
                              ) -> Tuple[tf.Tensor, tf.Tensor]:
  x_labeled, y_labeled = [], []
  for m, (x, y) in tqdm(ds.batch(1024),
                        desc=f'Extracting {n_labeled} labeled examples'):
    x_labeled.append(tf.boolean_mask(x, m, 0))
    y_labeled.append(tf.boolean_mask(y, m, 0))
  x_labeled = tf.concat(x_labeled, 0)
  if normalize_method is not None:
    x_labeled = normalize_method(tf.cast(x_labeled, tf.float32))
  y_labeled = tf.concat(y_labeled, 0)
  return x_labeled, y_labeled


class ImageDataset(IterableDataset):

  def __init__(self):
    super(ImageDataset, self).__init__()
    self.train = None
    self.valid = None
    self.test = None
    self._label_type = None

  @property
  def binarized(self) -> bool:
    raise NotImplementedError

  @property
  def data_type(self) -> DataType:
    return 'image'

  @property
  def shape(self) -> List[int]:
    raise NotImplementedError

  @property
  def label_type(self) -> LabelType:
    if not hasattr(self, '_label_type') or self._label_type is None:
      y = tf.concat([i for _, i in self.train.take(10)], axis=0).numpy()
      if np.all(np.sum(y, axis=-1) == 1.) or \
          (y.ndim == 1 and 'int' in str(y.dtype)):
        self._label_type = 'categorical'
      elif np.all(np.unique(y) == (0., 1.)):
        self._label_type = 'binary'
      else:
        self._label_type = 'factor'
    self._label_type: LabelType
    return self._label_type

  def sample_images(self,
                    save_path: Optional[str] = None,
                    dpi: int = 200,
                    n_samples: int = 25,
                    partition: Partition = 'train',
                    seed: int = 1):
    """ Sample a subset of image from training set """
    n = int(np.sqrt(n_samples))
    assert n * n == n_samples, "Sqrt of n_samples is not an integer"
    train = self.create_dataset(batch_size=n_samples,
                                partition=str(partition),
                                label_percent=1.0)
    train: tf.data.Dataset
    # prepare the data
    images = []
    labels = []
    for data in train.take(10):
      if len(data) >= 2:
        X, y = data[:2]
      else:
        X = data[0]
        y = None
      images.append(X)
      if y is not None:
        labels.append(y)
    rand = np.random.RandomState(seed=seed)
    idx = rand.choice(10)
    images = images[idx].numpy()
    labels = labels[idx].numpy() if len(labels) > 0 else None
    # check labels type
    labels_type = 'multinomial'
    if np.all(np.unique(labels) == [0., 1.]):
      labels_type = 'binary'
    # plot and save the figure
    if save_path is not None:
      plot_images = images
      if plot_images.shape[-1] == 1:
        plot_images = np.squeeze(plot_images, axis=-1)
      from matplotlib import pyplot as plt
      fig = plt.figure(figsize=(16, 16))
      for i in range(n_samples):
        plt.subplot(n, n, i + 1)
        img = plot_images[i]
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.axis('off')
        if labels is not None:
          if labels_type == 'binary':
            y = [
              str(j) for j in self.labels[np.array(labels[i], dtype=np.bool)]
            ]
            lab = ('\n'.join(y) + '\n') if len(y) > 1 else (y[0] + ' ')
          else:
            lab = '\n'.join(
              ["%s=%s" % (l, str(j)) for l, j in zip(self.labels, labels[i])])
            lab += '\n'
          plt.title(f"{lab}", fontsize=6)
      plt.tight_layout()
      fig.savefig(save_path, dpi=int(dpi))
      plt.close(fig)
    return images

  def normalize255(self, image: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(image / 255., 1e-6, 1. - 1e-6)

  def normalize(self, image: tf.Tensor,
                normalize: Literal['probs', 'tanh', 'raster']) -> tf.Tensor:
    if self.binarized:
      if 'raster' in normalize:
        image = tf.clip_by_value(image * 255., 0.0, 255.0)
      elif 'tanh' in normalize:
        image = tf.clip_by_value(image * 2.0 - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)
    else:
      image = tf.clip_by_value(image, 0.0, 255.0)
      if 'probs' in normalize:
        image = self.normalize255(image)
      elif 'tanh' in normalize:
        image = tf.clip_by_value(image / 255. * 2.0 - 1.0, -1.0 + 1e-6,
                                 1.0 - 1e-6)
    return image

  def _build_stratified_map(self, partition) -> Dict[int, List[int]]:
    name = f'_{self.name}_{partition}'
    path = os.path.join(get_cache_path(), name)
    if not os.path.exists(path):
      ds = get_partition(partition,
                         train=self.train,
                         valid=self.valid,
                         test=self.test)
      y_map = defaultdict(list)
      for i, (_, y) in enumerate(ds):
        y_map[np.argmax(y) if y.shape.ndims > 0 else y.numpy()].append(i)
      with open(path, 'wb') as f:
        pickle.dump(y_map, f)
      setattr(self, name, y_map)
    if not hasattr(self, name):
      with open(path, 'rb') as f:
        setattr(self, name, pickle.load(f))
    return getattr(self, name)

  def _get_labeled(self, partition, n_labeled, seed):
    pass

  def create_dataset(
      self,
      partition: Partition = 'train',
      *,
      batch_size: Optional[int] = 64,
      drop_remainder: bool = False,
      shuffle: int = 1000,
      cache: Optional[str] = '',
      prefetch: Optional[int] = tf.data.AUTOTUNE,
      parallel: Optional[int] = tf.data.AUTOTUNE,
      label_percent: Union[bool, int, float] = 0.0,
      oversample_ratio: Union[bool, float] = 0.0,
      fixed_oversample: bool = False,
      normalize: Literal['probs', 'tanh', 'raster'] = 'probs',
      seed: int = 1,
  ) -> tf.data.Dataset:
    """Create `tensorflow.data.Dataset` for the loaded dataset

    Parameters
    ----------
    partition : {'train', 'valid', 'test'}
        [description], by default 'train'
    batch_size : Optional[int], optional
        [description], by default 100
    drop_remainder : bool, optional
        [description], by default False
    shuffle : int, optional
        [description], by default 1000
    cache : Optional[str], optional
        [description], by default ''
    prefetch : Optional[int], optional
        [description], by default tf.data.AUTOTUNE
    parallel : Optional[int], optional
        [description], by default tf.data.AUTOTUNE
    label_percent : Union[bool, int, float], optional
        If `1.0` or `True`, return both image and label, i.e. supervised task.
        If `0.0`or `False`, only image is returned, i.e. unsupervised task.
        If a scalar in `(0, 1)`, it indicate the percent of labelled data,
        i.e. semi-supervised task.
        If an integer `> 1`, exact number of labelled samples, by default `0.0`
    oversample_ratio : Union[bool, float], optional
        a float number within the range `[0, 1]`, indicate the ratio between
        unlabel and label data in minibatch.
        If `0` or `False`, use the default label-unlabel ratio.
        If `1` or `True`, repeat all the label data every minibatch.
        Otherwise, the number is the percent of labeled data for each minibatch,
        by default 0.0.
    fixed_oversample : bool
        if `True`, the amount of labeled sample remain the same in each
        minibatch after oversampling, by default `True`
    normalize : Literal['probs', 'tanh', 'raster']
        image normalizing method, by default 'probs'
    seed : int, optional
        [description], by default 1

    Returns
    -------
    If `0. < label_percent < 1.`, return a dictionary
      image - `(tf.float32, (None, 28, 28, 1))`
      label - `(tf.float32, (None, 10))`
      mask  - `(tf.bool, (None, 1))`
    if `label_percent = 0`, return single image
    if `label_percent = 1`, return tuple of (image, label)
    """
    ds = get_partition(partition,
                       train=self.train,
                       valid=self.valid,
                       test=self.test)
    ######## check labels available
    struct = as_tuple(tf.data.experimental.get_structure(ds))
    has_labels = False
    if len(struct) == 1:
      label_percent = 0.0
    else:
      has_labels = True
    label_percent = float(label_percent)
    assert 0. <= oversample_ratio <= 1., \
      f'oversample_ratio must be in [0, 1] given: {oversample_ratio}'
    # which task
    task = 'unsupervised'
    if label_percent == 1.0:
      task = 'supervised'
    elif 0. < label_percent < 1. or label_percent > 1.:
      task = 'semi'
    ######## prepare the labeled data
    rand = np.random.RandomState(seed=seed)
    length = tf.data.experimental.cardinality(ds).numpy()
    x_labeled, y_labeled, mask_labeled, ds_supervised = [], [], None, None
    if task == 'semi':
      n_labeled = int(label_percent * length \
                        if 0. < label_percent < 1. else int(label_percent))
      n_unlabeled = length - n_labeled
      n_per_classes = int(n_labeled / len(self.labels))
      # for binary labels we could do stratified sampling
      if self.label_type == 'categorical':
        y_map = self._build_stratified_map(partition)
        labeled_ids = np.stack([
          rand.choice(v, size=n_per_classes, replace=False)
          for k, v in y_map.items()
        ])
        is_labeled = np.full((length,), False, dtype=np.bool)
        is_labeled[labeled_ids] = True
      # just pseudo-random sampling
      else:
        is_labeled = np.array(
          [True] * n_labeled + [False] * (length - n_labeled))
        rand.shuffle(is_labeled)
      # add labeling flag to the dataset
      ds = tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(is_labeled), ds))
      # repeat the label data in every minibatch
      if oversample_ratio == 1.0:
        x_labeled, y_labeled = _extract_labeled_examples(
          ds, n_labeled=n_labeled,
          normalize_method=partial(self.normalize, normalize=normalize))
        if y_labeled.shape.ndims == 1:
          y_labeled = tf.one_hot(y_labeled, len(self.labels))
        mask_labeled = tf.cast(tf.ones([x_labeled.shape[0]]), tf.bool)
        ds = ds.filter(lambda i, x: tf.logical_not(i))
      # mixing the label into minibatch
      elif oversample_ratio > 0.0:
        # for some reason sample_from_datasets significantly slowed down
        # if we sample from a single dataset that splitted into two by
        # filtering, and one of which is repeated
        # (e.g. 7000 samples/s dropped down to 1000 samples/s)
        x_labeled, y_labeled = _extract_labeled_examples(
          ds, n_labeled=n_labeled, normalize_method=None)
        mask_labeled = tf.cast(tf.ones([x_labeled.shape[0]]), tf.bool)
        ds_supervised = tf.data.Dataset.from_tensor_slices(
          (mask_labeled, (x_labeled, y_labeled)))
        n_repeat = int(
          np.ceil(oversample_ratio * n_unlabeled / (1 - oversample_ratio) /
                  n_labeled))
        ds_supervised = ds_supervised.shuffle(
          min(n_labeled, 1000),
          seed=seed,
          reshuffle_each_iteration=True,
        ).repeat(n_repeat)
        ds_unsupervised = ds.filter(lambda i, x: tf.logical_not(i))
        # only sampling if not fixed amount of labels
        if not fixed_oversample:
          ds = tf.data.experimental.sample_from_datasets(
            [ds_unsupervised, ds_supervised],
            weights=[1. - oversample_ratio, oversample_ratio],
            seed=seed)
        else:
          ds = ds_unsupervised
      # default ratio
      else:
        fixed_oversample = False
    ######## other cases
    elif task == 'unsupervised':
      ds = ds.map(lambda *x: (False, x))
    elif task == 'supervised':
      ds = ds.map(lambda *x: (True, x))
    else:
      raise ValueError(f'Unknown task type "{task}".')

    def _process(mask, data):
      images = tf.cast(data[0], tf.float32)
      # normalize the image
      images = self.normalize(images, normalize)
      if has_labels:
        labels = data[1]
        # covert to one-hot
        if len(labels.shape) == 1:
          labels = tf.one_hot(labels, len(self.labels))
      # unsupervised task
      if task == 'unsupervised':
        return images
      # supervised task
      elif task == 'supervised':
        return images, labels
      # semi-supervised task
      if oversample_ratio == 1.0:
        return images, x_labeled, y_labeled
      X_sup = tf.boolean_mask(images, mask, 0)
      y_sup = tf.boolean_mask(labels, mask, 0)
      X_uns = tf.boolean_mask(images, tf.logical_not(mask), 0)
      return X_uns, X_sup, y_sup

    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(buffer_size=int(shuffle),
                      seed=seed,
                      reshuffle_each_iteration=True)
    # for mixing unsupervised and supervised data
    if task == 'semi' and fixed_oversample:
      if shuffle is not None and shuffle > 0:
        ds_supervised = ds_supervised.shuffle(buffer_size=int(shuffle),
                                              seed=seed,
                                              reshuffle_each_iteration=True)
      if batch_size is not None and batch_size > 0:
        n_sup = int(np.ceil(batch_size * oversample_ratio))
        batch_size = batch_size - n_sup
        ds_supervised = ds_supervised.batch(n_sup,
                                            drop_remainder=drop_remainder)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
      ds = tf.data.Dataset.zip((ds, ds_supervised))

      def merge_semi(uns, sup):
        m_uns, (x_uns, y_uns) = uns
        m_sup, (x_sup, y_sup) = sup
        return tf.concat([m_uns, m_sup], 0), (tf.concat([x_uns, x_sup], 0),
                                              tf.concat([y_uns, y_sup], 0))

      ds = ds.map(merge_semi)
    # process as normal
    elif batch_size is not None and batch_size > 0:
      ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    # map cache and prefetch
    ds = ds.map(_process, num_parallel_calls=parallel)
    if cache is not None:
      ds = ds.cache(filename=str(cache))
    if prefetch is not None:
      ds = ds.prefetch(buffer_size=prefetch)
    ds: tf.data.Dataset
    return ds

  def __str__(self):
    name = self.__class__.__name__
    return (f"<{name} {self.shape} X:{self.data_type} "
            f"y:{self.label_type} {self.labels}>")
