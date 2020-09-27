from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from typing_extensions import Literal


# ===========================================================================
# Helpers
# ===========================================================================
def get_partition(part,
                  train=None,
                  valid=None,
                  test=None,
                  unlabeled=None,
                  unlabelled=None,
                  **kwargs):
  r""" A function for automatically select the right data partition """
  part = str(part).lower().strip()
  ret = None
  if 'train' in part:
    ret = train
  elif 'valid' in part:
    ret = valid
  elif 'test' in part:
    ret = test
  elif 'unlabeled' in part or 'unlabelled' in part:
    ret = unlabeled if unlabelled is None else unlabelled
  for k, v in kwargs.items():
    if part == str(k).strip().lower():
      ret = v
      break
  if ret is None:
    raise ValueError("No data for partition with name: '%s'" % part)
  return ret


class IterableDataset:

  @property
  def data_type(self) -> Literal['image', 'audio', 'text', 'gene']:
    raise NotImplementedError

  @property
  def name(self) -> str:
    return self.__class__.__name__.lower()

  @property
  def has_labels(self) -> bool:
    return self.n_labels > 0

  @property
  def n_labels(self) -> int:
    return len(self.labels)

  @property
  def labels_indices(self) -> Dict[str, int]:
    if not hasattr(self, "_labels_indices"):
      self._labels_indices = {j: i for i, j in enumerate(self.labels)}
    return self._labels_indices

  @property
  def labels(self) -> List[str]:
    return np.array([])

  @property
  def shape(self) -> List[int]:
    raise NotImplementedError()

  @property
  def is_binary(self) -> bool:
    raise NotImplementedError()

  def create_dataset(self,
                     batch_size: int = 64,
                     drop_remainder: bool = False,
                     shuffle: int = 1000,
                     prefetch: int = tf.data.experimental.AUTOTUNE,
                     cache: str = '',
                     parallel: Optional[int] = None,
                     partition: Literal['train', 'valid', 'test',
                                        'unlabelled'] = 'train',
                     inc_labels: bool = False,
                     seed: int = 1) -> tf.data.Dataset:
    raise NotImplementedError()
