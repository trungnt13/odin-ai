from __future__ import absolute_import, division, print_function

import os
import pickle
import shutil
import tarfile
from urllib.request import urlretrieve
from typing import Optional, Union
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from odin.fuel.image_data._base import ImageDataset
from odin.fuel.dataset_base import get_partition
from odin.utils import get_file, md5_checksum, md5_folder, one_hot


class CIFAR(ImageDataset):
  r""" CIFAR10 """

  URL = {
      10: r"https://www.cs.toronto.edu/%7Ekriz/cifar-10-python.tar.gz",
      100: r"https://www.cs.toronto.edu/%7Ekriz/cifar-100-python.tar.gz"
  }

  MD5 = {
      10: r"c58f30108f718f92721af3b95e74349a",
      100: r"eb9058c3a382ffc7106e4002c42a8d85"
  }

  MD5_EXTRACT = {
      10: r"341026eedb2822e04c43dfb5a62e1d19",
      100: r"fb755dd51de7edcbd1a5f794883159d0"
  }

  DIR_NAME = {10: "cifar-10-batches-py", 100: "cifar-100-python"}

  def __init__(self, version, path="~/tensorflow_datasets/cifar"):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
      os.makedirs(path)
    version = int(version)
    assert version in (10, 100), "Only support CIFAR-10 and CIFAR-100"
    ## download and extract
    url = CIFAR.URL[version]
    basename = os.path.basename(url)
    zip_path = os.path.join(path, basename)
    if os.path.exists(
        zip_path) and md5_checksum(zip_path) != CIFAR.MD5[version]:
      os.remove(zip_path)
    if not os.path.exists(zip_path):
      from tqdm import tqdm
      prog = tqdm(desc=f"Downloading file '{basename}'", unit="kB")

      def dl_progress(count, block_size, total_size):
        kB = count * block_size / 1024.
        prog.update(kB - prog.n)

      urlretrieve(url, zip_path, reporthook=dl_progress)
      prog.clear()
      prog.close()
    # extract
    data_dir = os.path.join(path, CIFAR.DIR_NAME[version])
    if os.path.exists(
        data_dir) and md5_folder(data_dir) != CIFAR.MD5_EXTRACT[version]:
      shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
      with tarfile.open(zip_path, "r:gz") as f:
        print("Extract zip file to ")
        f.extractall(path)
    ## load data
    X_train = []
    y_train = []
    y_train_coarse = []
    X_test = []
    y_test = []
    y_test_coarse = []
    for i in os.listdir(data_dir):
      if '.' not in i:
        with open(os.path.join(data_dir, i), 'rb') as f:
          data = pickle.load(f, encoding='bytes')
          if b'batch_label' not in data:  # metadata
            continue
          # labels
          if b"labels" in data:
            lab = data[b'labels']
          elif b"fine_labels" in data:
            lab = data[b'fine_labels']
          lab_coarse = data[
              b'coarse_labels'] if b'coarse_labels' in data else []
          # store the data
          if b'test' in data[b'batch_label'] or 'test' in i:
            X_test.append(data[b'data'])
            y_test += lab
            y_test_coarse += lab_coarse
          else:
            X_train.append(data[b'data'])
            y_train += lab
            y_train_coarse += lab_coarse

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)
    self.X_test = np.concatenate(X_test, axis=0)
    self.y_test = np.array(y_test)
    self.X_valid = X_train[:5000]
    self.y_valid = y_train[:5000]
    self.X_train = X_train[5000:]
    self.y_train = y_train[5000:]
    if len(y_train_coarse) > 0:
      y_train_coarse = np.array(y_train_coarse)
      self.y_valid_coarse = y_train_coarse[:5000]
      self.y_train_coarse = y_train_coarse[5000:]
      self.y_test_coarse = np.array(y_test_coarse)

  @property
  def is_binary(self):
    return False

  @property
  def shape(self):
    return (32, 32, 3)

  def create_dataset(self,
                     partition: Literal['train', 'valid', 'test'] = 'train',
                     *,
                     batch_size: int = 32,
                     drop_remainder: bool = False,
                     shuffle: int = 1000,
                     cache: Optional[str] = '',
                     prefetch: Optional[int] = tf.data.experimental.AUTOTUNE,
                     parallel: Optional[int] = tf.data.experimental.AUTOTUNE,
                     inc_labels: Union[bool, float] = False,
                     seed: int = 1) -> tf.data.Dataset:
    r"""
    Arguments:
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean or Scalar. If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.

    Return :
      tensorflow.data.Dataset :
        image - `(tf.float32, (None, 28, 28, 1))`
        label - `(tf.float32, (None, 10))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    X, y = get_partition(partition,
                         train=(self.X_train, self.y_train),
                         valid=(self.X_valid, self.y_valid),
                         test=(self.X_test, self.y_test))
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)
    assert X.shape[0] == y.shape[0]
    X = np.reshape(X, (-1, 3, 32, 32))
    X = np.transpose(X, (0, 2, 3, 1))
    y = one_hot(y, self.n_labels)

    def _process(*data):
      image = tf.cast(data[0], tf.float32)
      image = self.normalize_255(image)
      if inc_labels:
        label = tf.cast(data[1], tf.float32)
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=(image, label), mask=mask)
        return image, label
      return image

    ds = tf.data.Dataset.from_tensor_slices(X)
    if inc_labels > 0.:
      ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(y)))
    ds = ds.map(_process, parallel)
    if cache is not None:
      ds = ds.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(int(shuffle), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds


# ===========================================================================
# Shortcuts
# ===========================================================================
class CIFAR10(CIFAR):

  def __init__(self, path="~/tensorflow_datasets/cifar"):
    super().__init__(10, path=path)

  @property
  def labels(self):
    return np.array([
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    ])


class CIFAR100(CIFAR):

  def __init__(self, coarse_labels=False, path="~/tensorflow_datasets/cifar"):
    super().__init__(100, path=path)
    self._coarse_labels = bool(coarse_labels)
    if self._coarse_labels:
      self.y_train = self.y_train_coarse
      self.y_valid = self.y_valid_coarse
      self.y_test = self.y_test_coarse

  @property
  def labels(self):
    if self._coarse_labels:
      y = [
          'aquatic_mammals', 'fish', 'flowers', 'food_containers',
          'fruit_and_vegetables', 'household_electrical_devices',
          'household_furniture', 'insects', 'large_carnivores',
          'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
          'large_omnivores_and_herbivores', 'medium_mammals',
          'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
          'trees', 'vehicles_1', 'vehicles_2'
      ]
    else:
      y = [
          'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
          'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
          'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra',
          'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
          'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
          'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
          'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
          'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
          'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
          'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
          'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
          'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
          'willow_tree', 'wolf', 'woman', 'worm'
      ]
    return np.array(y)


class CIFAR20(CIFAR100):

  def __init__(self, path="~/tensorflow_datasets/cifar"):
    super().__init__(coarse_labels=True, path=path)
