import base64
import os
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from scipy import sparse

from odin.fuel.dataset_base import IterableDataset, get_partition
from odin.utils.crypto import md5_checksum


def _tensor(x):
  x = x.astype(np.float32)
  if isinstance(x, sparse.spmatrix):
    x = tf.SparseTensor(indices=sorted(zip(*x.nonzero())),
                        values=x.data,
                        dense_shape=x.shape)
  return tf.data.Dataset.from_tensor_slices(x)


class BioDataset(IterableDataset):

  def __init__(self):
    super().__init__()
    self.x = None
    self.y = None
    self.xvar = None
    self.yvar = None
    self.train_ids = None
    self.valid_ids = None
    self.test_ids = None

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
    for attr in ('x', 'y', 'xvar', 'yvar'):
      assert hasattr(self, attr)
      assert getattr(self, attr) is not None
    # split train, valid, test data
    if not hasattr(self, 'train_ids') or self.train_ids is None:
      rand = np.random.RandomState(seed=1)
      n = self.x.shape[0]
      ids = rand.permutation(n)
      self.train_ids = ids[:int(0.85 * n)]
      self.valid_ids = ids[int(0.85 * n):int(0.9 * n)]
      self.test_ids = ids[int(0.9 * n):]
    ids = get_partition(partition,
                        train=self.train_ids,
                        valid=self.valid_ids,
                        test=self.test_ids)
    is_sparse_x = isinstance(self.x, sparse.spmatrix)
    is_sparse_y = isinstance(self.y, sparse.spmatrix)
    x = _tensor(self.x[ids])
    y = _tensor(self.y[ids])
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def _process(*data):
      data = list(data)
      if is_sparse_x:
        data[0] = tf.sparse.to_dense(data[0])
      if is_sparse_y and len(data) > 1:
        data[1] = tf.sparse.to_dense(data[1])
      data = tuple(data)
      if inc_labels:
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=data, mask=mask)
      return data

    if inc_labels > 0.:
      ds = tf.data.Dataset.zip((x, y))
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
