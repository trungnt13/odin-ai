import os
from typing import Optional, Union, List, Dict
from typing_extensions import Literal
import pickle

import numpy as np
import tensorflow as tf
from odin.utils import as_tuple
from collections import defaultdict

from odin.fuel.dataset_base import IterableDataset, get_partition
from odin.utils.cache_utils import get_cache_path


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
                                inc_labels=0.5)
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

  def normalize_255(self, image: tf.Tensor) -> tf.Tensor:
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
        image = self.normalize_255(image)
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

  def create_dataset(
      self,
      partition: Literal['train', 'valid', 'test'] = 'train',
      *,
      batch_size: Optional[int] = 100,
      batch_labeled_ratio: float = 0.5,
      drop_remainder: bool = False,
      shuffle: int = 1000,
      cache: Optional[str] = '',
      prefetch: Optional[int] = tf.data.AUTOTUNE,
      parallel: Optional[int] = tf.data.AUTOTUNE,
      inc_labels: Union[bool, float] = False,
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
    batch_labeled_ratio : float, optional
        [description], by default 0.5
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
    inc_labels : Union[bool, float], optional
        If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.
        , by default False
    normalize : Literal[, optional
        [description], by default 'probs'
    seed : int, optional
        [description], by default 1

    Returns
    -------
    tf.data.Dataset
      image - `(tf.float32, (None, 28, 28, 1))`
      label - `(tf.float32, (None, 10))`
      mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
    where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    ds = get_partition(partition,
                       train=self.train,
                       valid=self.valid,
                       test=self.test)
    struct = as_tuple(tf.data.experimental.get_structure(ds))
    has_labels = False
    ids = None
    if len(struct) == 1:
      inc_labels = False
    else:
      has_labels = True
      ids = tf.range(self.n_labels, dtype=tf.float32)
    inc_labels = float(inc_labels)
    ## prepare the labeled data
    rand = np.random.RandomState(seed=seed)
    length = tf.data.experimental.cardinality(ds).numpy()
    if 0. < inc_labels < 1. or inc_labels > 1.:
      n_labeled = int(inc_labels * length \
        if 0. < inc_labels < 1. else int(inc_labels))
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
      # oversampling the labeled
      ds_labeled = ds.filter(lambda i, x: i)
      ds_labeled = ds_labeled.shuffle(min(1000, n_labeled), seed=seed\
        ).repeat()
      # sampling from unlabled (False) and labeled (True) data
      ds = ds.filter(lambda i, x: tf.logical_not(i)\
        ).repeat()
      # from collections import Counter # this is for debugging
      # print(Counter([i for _, (_, i) in ds.as_numpy_iterator()]))
      ds = tf.data.experimental.sample_from_datasets(
          [ds, ds_labeled],
          weights=[1. - batch_labeled_ratio, batch_labeled_ratio],
          seed=seed)
      # .repeat().take(n_unlabeled)
    elif inc_labels == 0:
      ds = ds.map(lambda *x: (False, x))
    elif inc_labels == 1:
      ds = ds.map(lambda *x: (True, x))

    def _process(mask, data):
      image = tf.cast(data[0], tf.float32)
      # normalize the image
      image = self.normalize(image, normalize)
      if has_labels:
        label = tf.cast(data[1], tf.float32)
        # covert to one-hot
        if len(label.shape) == 0:
          label = tf.cast(ids == label, tf.float32)
          label = label * tf.cast(mask, tf.float32)
      if inc_labels == 0:
        return image
      elif inc_labels == 1:
        return image, label
      return dict(inputs=(image, label), mask=mask)

    ds = ds.map(_process, parallel)
    if cache is not None:
      ds = ds.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(int(shuffle), seed=seed, reshuffle_each_iteration=True)
    if batch_size is not None:
      ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds
