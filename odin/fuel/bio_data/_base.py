import base64
import os
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from scipy import sparse

from odin.fuel.dataset_base import IterableDataset, get_partition
from odin.utils.crypto import md5_checksum


class BioDataset(IterableDataset):

  @property
  def var_names(self):
    return self.xvar

  @property
  def name(self):
    raise NotImplementedError

  @property
  def n_labels(self):
    return self.y.shape[1]

  @property
  def labels(self):
    return self.yvar

  @property
  def shape(self):
    return tuple(self.x.shape[1:])

  @property
  def is_binary(self):
    return False

  def create_dataset(self,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=None,
                     partition='train',
                     inc_labels=False,
                     seed=1) -> tf.data.Dataset:
    for attr in ('x', 'y', 'xvar', 'yvar', 'train_ids', 'valid_ids',
                 'test_ids'):
      assert hasattr(self, attr)
    ids = get_partition(partition,
                        train=self.train_ids,
                        valid=self.valid_ids,
                        test=self.test_ids)
    x = self.x[ids]
    y = self.y[ids]
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def _process(*data):
      if inc_labels:
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=data, mask=mask)
      return data

    ds = tf.data.Dataset.from_tensor_slices(x)
    if inc_labels > 0.:
      ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(y)))
    ds = ds.map(_process, parallel)
    if cache is not None:
      ds = ds.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(int(shuffle))
    ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds
