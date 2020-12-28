import gzip
import os
from urllib.request import urlretrieve
from typing import Optional, Union
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from odin.fuel.dataset_base import get_partition
from odin.fuel.image_data._base import ImageDataset
from odin.utils import md5_checksum, one_hot
from odin.utils.net_utils import download_and_extract


class BinarizedMNIST(ImageDataset):
  """BinarizedMNIST"""

  def __init__(self, normalize: bool = False):
    self.train, self.valid, self.test = tfds.load(
        name='binarized_mnist',
        split=['train', 'validation', 'test'],
        read_config=tfds.ReadConfig(shuffle_seed=seed,
                                    shuffle_reshuffle_each_iteration=True),
        as_supervised=False)
    self._normalize = bool(normalize)

  @property
  def shape(self):
    return (28, 28, 1)

  def create_dataset(self,
                     partition: Literal['train', 'valid', 'test'] = 'train',
                     *,
                     batch_size: Optional[int] = 32,
                     drop_remainder: bool = False,
                     shuffle: int = 1000,
                     cache: Optional[str] = '',
                     prefetch: Optional[int] = tf.data.experimental.AUTOTUNE,
                     parallel: Optional[int] = tf.data.experimental.AUTOTUNE,
                     inc_labels: Union[bool, float] = False,
                     seed: int = 1) -> tf.data.Dataset:
    """
    Parameters
    -----------
    partition : {'train', 'valid', 'test'}
    inc_labels : a Boolean or Scalar. If True, return both image and label,
      otherwise, only image is returned.
      If a scalar is provided, it indicate the percent of labelled data
      in the mask.

    Return
    -------
    tensorflow.data.Dataset :
      image - `(tf.float32, (None, 28, 28, 1))`
      label - `(tf.float32, (None, 10))`
      mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
    where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    ds = get_partition(partition,
                       train=self.train,
                       valid=self.valid,
                       test=self.test)
    struct = tf.data.experimental.get_structure(ds)
    if len(struct) == 1:
      inc_labels = False
    ids = tf.range(self.n_labels, dtype=tf.float32)
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def _process_dict(data):
      image = tf.cast(data['image'], tf.float32)
      if self._normalize:
        image = self.normalize_255(image)
      if inc_labels:
        label = tf.cast(data['label'], tf.float32)
        if len(label.shape) == 0:  # covert to one-hot
          label = tf.cast(ids == label, tf.float32)
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=(image, label), mask=mask)
        return image, label
      return image

    def _process_tuple(*data):
      image = tf.cast(data[0], tf.float32)
      if self._normalize:
        image = self.normalize_255(image)
      if inc_labels:
        label = tf.cast(data[1], tf.float32)
        if len(label.shape) == 0:  # covert to one-hot
          label = tf.cast(ids == label, tf.float32)
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=(image, label), mask=mask)
        return image, label
      return image

    if cache is not None:
      ds = ds.cache(str(cache))
    ds = ds.map(_process_dict if isinstance(struct, dict) else _process_tuple,
                parallel)
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(int(shuffle), seed=seed, reshuffle_each_iteration=True)
    if batch_size is not None:
      ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds


class MNIST(BinarizedMNIST):
  """Original MNIST from Yann Lecun:

      - 55000 examples for train,
      - 5000 for valid, and
      - 10000 for test
  """
  URL = dict(
      X_train=r"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
      y_train=r"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
      X_test=r"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
      y_test=r"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
  )

  MD5 = r"8ba71f60dccd53a0b68bfe41ed4cdf9c"

  def __init__(self,
               normalize: bool = True,
               path: str = '~/tensorflow_datasets/mnist'):
    self._normalize = bool(normalize)
    path = os.path.abspath(os.path.expanduser(path))
    save_path = os.path.join(path, 'mnist.npz')
    if not os.path.exists(path):
      os.makedirs(path)
    assert os.path.isdir(path)

    ## check exist processed file
    all_data = None
    if os.path.exists(save_path):
      if not os.path.isfile(save_path):
        raise ValueError("path to %s must be a file" % save_path)
      if md5_checksum(save_path) != MNIST.MD5:
        print("Miss match MD5 remove file at: ", save_path)
        os.remove(save_path)
      else:
        all_data = np.load(save_path)
    ## download and extract
    if all_data is None:
      from tqdm import tqdm

      def dl_progress(count, block_size, total_size):
        kB = block_size * count / 1024.
        prog.update(kB - prog.n)

      read32 = lambda b: np.frombuffer(
          b, dtype=np.dtype(np.uint32).newbyteorder('>'))[0]

      all_data = {}
      for name, url in MNIST.URL.items():
        basename = os.path.basename(url)
        zip_path = os.path.join(path, basename)
        prog = tqdm(desc="Downloading %s" % basename, unit='kB')
        urlretrieve(url, zip_path, dl_progress)
        prog.clear()
        prog.close()
        with gzip.open(zip_path, "rb") as f:
          magic = read32(f.read(4))
          if magic not in (2051, 2049):
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, zip_path))
          n = read32(f.read(4))
          # images
          if 'X_' in name:
            rows = read32(f.read(4))
            cols = read32(f.read(4))
            buf = f.read(rows * cols * n)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(n, rows, cols, 1)
          # labels
          else:
            buf = f.read(n)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = one_hot(data, 10)
          all_data[name] = data
      np.savez_compressed(save_path, **all_data)
    ## split train, valid, test
    rand = np.random.RandomState(seed=1)
    ids = rand.permutation(all_data['X_train'].shape[0])
    X_train = all_data['X_train'][ids]
    y_train = all_data['y_train'][ids]
    X_valid = X_train[:5000]
    y_valid = y_train[:5000]
    X_train = X_train[5000:]
    y_train = y_train[5000:]
    X_test = all_data['X_test']
    y_test = all_data['y_test']
    to_ds = lambda images, labels: tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(images),
         tf.data.Dataset.from_tensor_slices(labels)))
    self.train = to_ds(X_train, y_train)
    self.valid = to_ds(X_valid, y_valid)
    self.test = to_ds(X_test, y_test)

  @property
  def labels(self):
    return np.array([str(i) for i in range(10)])

  @property
  def shape(self):
    return (28, 28, 1)


class MNIST255(MNIST):

  def __init__(self, path: str = '~/tensorflow_datasets/mnist'):
    super().__init__(normalize=False, path=path)


class BinarizedAlphaDigits(BinarizedMNIST):
  """Binary 20x16 digits of '0' through '9' and capital 'A' through 'Z'.
  39 examples of each class. """

  def __init__(self):
    import tensorflow_datasets as tfds
    self._normalize = False
    self.train, self.valid, self.test = tfds.load(
        name='binary_alpha_digits',
        split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'],
        as_supervised=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed,
                                    shuffle_reshuffle_each_iteration=True),
        shuffle_files=True,
    )

  @property
  def shape(self):
    return (20, 16, 1)


class FashionMNIST(BinarizedMNIST):

  def __init__(self, normalize: bool = True, seed: int = 1):
    self._normalize = normalize
    self.train, self.valid, self.test = tfds.load(
        name='fashion_mnist',
        split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'],
        as_supervised=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed,
                                    shuffle_reshuffle_each_iteration=True),
        shuffle_files=True,
        with_info=False,
    )

  @property
  def labels(self):
    return np.array([
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ])

  @property
  def shape(self):
    return (28, 28, 1)


class FashionMNIST255(FashionMNIST):

  def __init__(self):
    super().__init__(normalize=False)
