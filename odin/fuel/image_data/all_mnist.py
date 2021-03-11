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

  def __init__(self):
    self.train, self.valid, self.test = tfds.load(
        name='mnist',
        split=['train[:55000]', 'train[55000:]', 'test'],
        read_config=tfds.ReadConfig(shuffle_seed=1,
                                    shuffle_reshuffle_each_iteration=True),
        as_supervised=True)

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
