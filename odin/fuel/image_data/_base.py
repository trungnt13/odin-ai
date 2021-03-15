import os
from typing import Optional, Union, List, Dict
from typing_extensions import Literal
import pickle
from functools import partial

import numpy as np
import tensorflow as tf
from odin.utils import as_tuple
from collections import defaultdict
from tqdm import tqdm

from odin.fuel.dataset_base import IterableDataset, get_partition
from odin.utils.cache_utils import get_cache_path


def _extract_labeled_examples(ds, normalize_method=None):
  x_labeled, y_labeled = [], []
  for m, (x, y) in tqdm(ds.batch(1024), desc='Extracting labeled examples'):
    x_labeled.append(tf.boolean_mask(x, m, 0))
    y_labeled.append(tf.boolean_mask(y, m, 0))
  x_labeled = tf.concat(x_labeled, 0)
  if normalize_method is not None:
    x_labeled = normalize_method(tf.cast(x_labeled, tf.float32))
  y_labeled = tf.concat(y_labeled, 0)
  return x_labeled, y_labeled


class ImageDataset(IterableDataset):

  @property
  def binarized(self) -> bool:
    raise NotImplementedError

  @classmethod
  def data_type(self) -> str:
    return 'image'

  @property
  def shape(self) -> List[int]:
    raise NotImplementedError

  @property
  def is_binary_labels(self) -> bool:
    if not hasattr(self, '_is_binary_labels'):
      self._is_binary_labels = True
      for _, y in self.train.take(10):
        if y.shape.ndims == 0 and 'int' in str(y.dtype):
          continue
        if np.any(np.unique(y) != [0., 1.]):
          self._is_binary_labels = False
    return self._is_binary_labels

  def sample_images(self,
                    save_path=None,
                    dpi=200,
                    n_samples=25,
                    partition='train',
                    seed=1):
    """ Sample a subset of image from training set """
    n = int(np.sqrt(n_samples))
    assert n * n == n_samples, "Sqrt of n_samples is not an integer"
    train = self.create_dataset(batch_size=n_samples,
                                partition=str(partition),
                                label_percent=0.5)
    # prepare the data
    images = []
    labels = []
    mask = []
    for data in train.take(10):
      if isinstance(data, dict):
        X, y = data['inputs']
        mask.append(data['mask'])
      elif isinstance(data, (tuple, list)):
        if len(data) >= 2:
          X, y = data[:2]
        else:
          X = data[0]
          y = None
      else:
        X = data
        y = None
      images.append(X)
      if y is not None:
        labels.append(y)
    rand = np.random.RandomState(seed=seed)
    idx = rand.choice(10)
    images = images[idx].numpy()
    labels = labels[idx].numpy() if len(labels) > 0 else None
    mask = mask[idx].numpy().ravel() if len(mask) > 0 else None
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
          m = True if mask is None else mask[i]
          plt.title("%s[Mask:%s]" % (lab, m), fontsize=6)
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
      partition: Literal['train', 'valid', 'test'] = 'train',
      *,
      batch_size: Optional[int] = 100,
      drop_remainder: bool = False,
      shuffle: int = 1000,
      cache: Optional[str] = '',
      prefetch: Optional[int] = tf.data.AUTOTUNE,
      parallel: Optional[int] = tf.data.AUTOTUNE,
      label_percent: float = 0.0,
      label_weight: float = 0.0,
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
    label_percent : Union[bool, float], optional
        If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.
        , by default False
    label_weight : float, optional
        a float number within the range `[0, 1]`. If `label_weight=0`, use the
        default label-unlabel ratio. If `label_weight=1`, repeat all the label
        data every minibatch. Otherwise, the number is the percent of labeled
        data for each minibatch, by default 0.0.
    normalize : Literal[, optional
        [description], by default 'probs'
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
    assert 0. <= label_weight <= 1., \
      f'label_weight must be in [0, 1] given: {label_weight}'
    ######## prepare the labeled data
    rand = np.random.RandomState(seed=seed)
    length = tf.data.experimental.cardinality(ds).numpy()
    x_labeled, y_labeled, mask_labeled, ds_labeled = [], [], None, None
    if 0. < label_percent < 1. or label_percent > 1.:
      n_labeled = int(label_percent * length \
        if 0. < label_percent < 1. else int(label_percent))
      n_unlabeled = length - n_labeled
      n_per_classes = int(n_labeled / len(self.labels))
      # for binary labels we could do stratified sampling
      if self.is_binary_labels:
        y_map = self._build_stratified_map(partition)
        labeled_ids = np.stack([
            rand.choice(v, size=n_per_classes, replace=False)
            for k, v in y_map.items()
        ])
        labeled = np.full((length,), False, dtype=np.bool)
        labeled[labeled_ids] = True
      # just pseudo-random sampling
      else:
        labeled = np.array([True] * n_labeled + [False] * (length - n_labeled))
        rand.shuffle(labeled)
      ds = tf.data.Dataset.zip(
          (tf.data.Dataset.from_tensor_slices(labeled), ds))
      # repeat the label data in every minibatch
      if label_weight == 1.0:
        x_labeled, y_labeled = _extract_labeled_examples(
            ds, partial(self.normalize, normalize=normalize))
        if y_labeled.shape.ndims == 1:
          y_labeled = tf.one_hot(y_labeled, len(self.labels))
        mask_labeled = tf.cast(tf.ones([x_labeled.shape[0]]), tf.bool)
        ds = ds.filter(lambda i, x: tf.logical_not(i))
      # mixing the label into minibatches
      elif label_weight > 0.0:
        # for some reason sample_from_datasets significantly slowed down
        # if we sample from a single dataset that splitted into two by filtering,
        # and one of which is repeated
        # (e.g. 7000 samples/s dropped down to 1000 samples/s)
        x_labeled, y_labeled = _extract_labeled_examples(ds, None)
        mask_labeled = tf.cast(tf.ones([x_labeled.shape[0]]), tf.bool)
        ds_labeled = tf.data.Dataset.from_tensor_slices(
            (mask_labeled, (x_labeled, y_labeled)))
        n_repeat = int(
            np.ceil(label_weight * n_unlabeled / (1 - label_weight) /
                    n_labeled))
        ds_labeled = ds_labeled.shuffle(
            min(n_labeled, 1000),
            seed=seed,
            reshuffle_each_iteration=True,
        ).repeat(n_repeat)
        ds_unlabeled = ds.filter(lambda i, x: tf.logical_not(i))
        ds = tf.data.experimental.sample_from_datasets(
            [ds_unlabeled, ds_labeled],
            weights=[1. - label_weight, label_weight],
            seed=seed)
    ######## other cases
    elif label_percent == 0:
      ds = ds.map(lambda *x: (False, x))
    elif label_percent == 1:
      ds = ds.map(lambda *x: (True, x))

    def _process(mask, data):
      images = tf.cast(data[0], tf.float32)
      # normalize the image
      images = self.normalize(images, normalize)
      if has_labels:
        labels = data[1]
        # covert to one-hot
        if len(labels.shape) == 1:
          labels = tf.one_hot(labels, len(self.labels))
        labels = labels * tf.cast(tf.expand_dims(mask, -1), tf.float32)
      if label_percent == 0:
        return images
      elif label_percent == 1:
        return images, labels
      if label_weight == 1.0:
        images = tf.concat([images, x_labeled], axis=0)
        labels = tf.concat([labels, y_labeled], axis=0)
        mask = tf.concat([mask, mask_labeled], axis=0)
      return dict(inputs=(images, labels), mask=mask)

    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(buffer_size=int(shuffle),
                      seed=seed,
                      reshuffle_each_iteration=True)
    if batch_size is not None:
      ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(_process, num_parallel_calls=parallel)
    if cache is not None:
      ds = ds.cache(filename=str(cache))
    if prefetch is not None:
      ds = ds.prefetch(buffer_size=prefetch)
    return ds
