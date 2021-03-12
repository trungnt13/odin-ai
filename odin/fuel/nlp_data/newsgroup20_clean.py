import os
import pickle
import urllib
from typing import Dict, List, Tuple, Union, Optional
from typing_extensions import Literal

import numpy as np
import scipy as sp
import tensorflow as tf
from odin.fuel.dataset_base import IterableDataset, get_partition


def _download_newsgroup20(
    data_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[int, str]]:
  root_path = "https://github.com/akashgit/autoencoding_vi_for_topic_models/raw/9db556361409ecb3a732f99b4ef207aeb8516f83/data/20news_clean"
  file_template = "{split}.txt.npy"
  filename = [
      ("vocab.pkl", os.path.join(root_path, "vocab.pkl")),
      ("train", os.path.join(root_path, file_template.format(split="train"))),
      ("test", os.path.join(root_path, file_template.format(split="test"))),
  ]
  data = {}
  for name, url in filename:
    filepath = os.path.join(data_dir, name)
    # download
    if not os.path.exists(filepath):
      print(f"Download file {filepath}")
      urllib.request.urlretrieve(url, filepath)
    # load
    if '.pkl' in name:
      with open(filepath, 'rb') as f:
        words_to_idx = pickle.load(f)
      n_words = len(words_to_idx)
      data[name.split(".")[0]] = words_to_idx
    else:
      x = np.load(filepath, allow_pickle=True, encoding="latin1")[:-1]
      n_documents = x.shape[0]
      indices = np.array([(row_idx, column_idx)
                          for row_idx, row in enumerate(x)
                          for column_idx in row])
      sparse_matrix = sp.sparse.coo_matrix(
          (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
          shape=(n_documents, n_words),
          dtype=np.float32)
      sparse_matrix = sparse_matrix.tocsr()
      data[name] = sparse_matrix
  vocabulary = {idx: word for word, idx in words_to_idx.items()}
  return data, vocabulary


class Newsgroup20_clean(IterableDataset):

  def __init__(self, path="~/tensorflow_datasets/newsgroup20_clean"):
    super().__init__()
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
      os.makedirs(path)
    data, self._vocab = _download_newsgroup20(path)
    # split into train and valid
    train = data['train']
    rand = np.random.RandomState(seed=1)
    ids = rand.permutation(train.shape[0])
    start = int(0.1 * train.shape[0])
    self.valid = train[ids[:start]]
    self.train = train[ids[start:]]
    self.test = data['test']

  @classmethod
  def data_type(cls) -> str:
    return 'text'

  @property
  def vocabulary(self) -> Dict[int, str]:
    return self._vocab

  @property
  def vocabulary_size(self) -> int:
    return len(self.vocabulary)

  @property
  def shape(self) -> List[int]:
    return (self.train.shape[1],)

  @property
  def binarized(self) -> bool:
    return False

  def create_dataset(self,
                     partition: Literal['train', 'valid', 'test'] = 'train',
                     *,
                     batch_size: Optional[int] = 32,
                     drop_remainder: bool = False,
                     shuffle: int = 1000,
                     cache: Optional[str] = '',
                     prefetch: Optional[int] = tf.data.experimental.AUTOTUNE,
                     parallel: Optional[int] = tf.data.experimental.AUTOTUNE,
                     label_percent: Union[bool, float] = False,
                     seed: int = 1) -> tf.data.Dataset:
    x = get_partition(partition,
                      train=self.train,
                      valid=self.valid,
                      test=self.test)
    x = tf.SparseTensor(indices=sorted(zip(*x.nonzero())),
                        values=x.data,
                        dense_shape=x.shape)
    x = tf.data.Dataset.from_tensor_slices(x)
    if cache is not None:
      x = x.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      x = x.shuffle(int(shuffle), seed=seed, reshuffle_each_iteration=True)
    if batch_size is not None:
      x = x.batch(batch_size, drop_remainder)
    x = x.map(lambda y: tf.cast(tf.sparse.to_dense(y), tf.float32))
    if prefetch is not None:
      x = x.prefetch(prefetch)
    return x
