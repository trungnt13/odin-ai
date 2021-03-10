import base64
import os
from typing import Dict, List, Optional, Union
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from odin.fuel.dataset_base import IterableDataset, get_partition
from odin.utils.crypto import md5_checksum
from scipy import sparse


def _tensor(x):
  x = x.astype(np.float32)
  if isinstance(x, sparse.spmatrix):
    x = tf.SparseTensor(indices=sorted(zip(*x.nonzero())),
                        values=x.data,
                        dense_shape=x.shape)
  return tf.data.Dataset.from_tensor_slices(x)


class GeneDataset(IterableDataset):

  def __init__(self):
    super().__init__()
    self.x = None
    self.y = None
    self.xvar = None
    self.yvar = None
    self.train_ids = None
    self.valid_ids = None
    self.test_ids = None

  @classmethod
  def data_type(cls) -> str:
    return 'gene'

  def __str__(self):
    return (f"<{self.__class__.__name__} "
            f"X:{self.x.shape} y:{self.y.shape} "
            f"labels:{self.labels}>")

  @property
  def genes_dictionary(self) -> Dict[int, str]:
    attr_name = '_gene_dictionary'
    if not hasattr(self, attr_name):
      setattr(self, attr_name,
              {idx: name for idx, name in enumerate(self.var_names)})
    return getattr(self, attr_name)

  @property
  def vocabulary(self) -> Dict[int, str]:
    return self.genes_dictionary

  @property
  def vocabulary_size(self) -> int:
    return len(self.vocabulary)

  @property
  def var_names(self) -> List[str]:
    return self.xvar

  @property
  def name(self) -> str:
    raise NotImplementedError

  @property
  def n_labels(self) -> int:
    return self.y.shape[1]

  @property
  def labels(self) -> List[str]:
    return self.yvar

  @property
  def shape(self) -> List[int]:
    return tuple(self.x.shape[1:])

  @property
  def is_binary(self) -> bool:
    return False

  def create_dataset(self,
                     partition: str = 'train',
                     *,
                     batch_size: Optional[int] = 32,
                     drop_remainder: bool = False,
                     shuffle: Optional[int] = 1000,
                     prefetch: int = tf.data.experimental.AUTOTUNE,
                     cache: str = '',
                     parallel: Optional[int] = None,
                     inc_labels: bool = False,
                     seed: int = 1) -> tf.data.Dataset:
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
      return data[0] if len(data) == 1 else data

    ds = x
    if inc_labels > 0.:
      ds = tf.data.Dataset.zip((x, y))
    if cache is not None:
      ds = ds.cache(str(cache))
    ds = ds.map(_process, parallel)
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(int(shuffle), seed=seed, reshuffle_each_iteration=True)
    if batch_size is not None:
      ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds
