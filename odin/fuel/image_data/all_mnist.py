import gzip
import os
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from odin.fuel.image_data._base import ImageDataset
from odin.utils import md5_checksum, one_hot


class BinarizedMNIST(ImageDataset):
  """ BinarizedMNIST """

  def __init__(self):
    self.train, self.valid, self.test = tfds.load(
        name='binarized_mnist',
        split=['train', 'validation', 'test'],
        read_config=tfds.ReadConfig(shuffle_seed=1,
                                    shuffle_reshuffle_each_iteration=True),
        as_supervised=False)
    process = lambda x: x['image']
    self.train = self.train.map(process)
    self.valid = self.valid.map(process)
    self.test = self.test.map(process)

  @property
  def binarized(self):
    return True

  @property
  def shape(self):
    return (28, 28, 1)


class MNIST(ImageDataset):
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

  def __init__(self, path: str = '~/tensorflow_datasets/mnist'):
    path = os.path.abspath(os.path.expanduser(path))
    save_path = os.path.join(path, 'mnist.npz')
    if not os.path.exists(path):
      os.makedirs(path)
    assert os.path.isdir(path)

    ## check exist processed file
    all_data = None
    if os.path.exists(save_path):
      if not os.path.isfile(save_path):
        raise ValueError(f"path to {save_path} must be a file")
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
  def binarized(self):
    return False

  @property
  def shape(self):
    return (28, 28, 1)


class BinarizedAlphaDigits(BinarizedMNIST):
  """Binary 20x16 digits of '0' through '9' and capital 'A' through 'Z'.
  39 examples of each class. """

  def __init__(self):
    self.train, self.valid, self.test = tfds.load(
        name='binary_alpha_digits',
        split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'],
        as_supervised=True,
        read_config=tfds.ReadConfig(shuffle_seed=1,
                                    shuffle_reshuffle_each_iteration=True),
        shuffle_files=True,
    )

  @property
  def binarized(self):
    return True

  @property
  def shape(self):
    return (20, 16, 1)

  @property
  def labels(self):
    return np.array([str(i) for i in range(10)] +
                    [chr(i) for i in range(65, 91)])


# ===========================================================================
# Fashion MNIST
# ===========================================================================
class FashionMNIST(ImageDataset):

  def __init__(self, seed: int = 1):
    self.train, self.valid, self.test = tfds.load(
        name='fashion_mnist',
        split=['train[:55000]', 'train[55000:]', 'test'],
        as_supervised=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed,
                                    shuffle_reshuffle_each_iteration=True),
        shuffle_files=True,
        with_info=False,
    )

  @property
  def binarized(self):
    return False

  @property
  def labels(self):
    return np.array([
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ])

  @property
  def shape(self):
    return (28, 28, 1)


# ===========================================================================
# SVHN
# ===========================================================================
class SVHN(ImageDataset):

  def __init__(self, inc_extra: bool = False):
    self.train, self.valid, self.test, self.extra = tfds.load(
        name='svhn_cropped',
        split=['train[:95%]', 'train[95%:]', 'test', 'extra'],
        read_config=tfds.ReadConfig(shuffle_seed=1,
                                    shuffle_reshuffle_each_iteration=True),
        as_supervised=True,
    )
    self.inc_extra = inc_extra
    if inc_extra:
      self.train = self.train.concatenate(self.extra)

  @property
  def binarized(self):
    return False

  @property
  def shape(self):
    return (32, 32, 3)

  @property
  def labels(self):
    return np.array([str(i) for i in range(10)])
