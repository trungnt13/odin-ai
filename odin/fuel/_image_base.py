import gzip
import os
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

from odin.utils import md5_checksum, one_hot
from odin.utils.net_utils import download_and_extract


# ===========================================================================
# Helpers
# ===========================================================================
def _partition(part,
               train=None,
               valid=None,
               test=None,
               unlabeled=None,
               unlabelled=None, **kwargs):
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


class ImageDataset:

  @property
  def name(self):
    return self.__class__.__name__.lower()

  def sample_images(self,
                    save_path=None,
                    dpi=120,
                    n_samples=25,
                    partition='train',
                    seed=1):
    r""" Sample a subset of image from training set """
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

  def normalize_255(self, image):
    return tf.clip_by_value(image / 255., 1e-6, 1. - 1e-6)

  @property
  def n_labels(self):
    return len(self.labels)

  @property
  def labels(self):
    return np.array([])

  @property
  def shape(self):
    raise NotImplementedError()

  @property
  def is_binary(self):
    raise NotImplementedError()

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
    raise NotImplementedError()


# ===========================================================================
# Dataset
# ===========================================================================
class BinarizedMNIST(ImageDataset):
  r""" BinarizedMNIST """

  def __init__(self):
    import tensorflow_datasets as tfds
    self.train, self.valid, self.test = tfds.load(
        name='binarized_mnist',
        split=['train', 'validation', 'test'],
        as_supervised=False)

  @property
  def is_binary(self):
    return True

  @property
  def shape(self):
    return (28, 28, 1)

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
    ds = _partition(partition,
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
      if not self.is_binary:
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
      if not self.is_binary:
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

    ds = ds.map(_process_dict if isinstance(struct, dict) else _process_tuple,
                parallel)
    if cache is not None:
      ds = ds.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(int(shuffle))
    ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds


class MNIST(BinarizedMNIST):
  r""" MNIST
  55000 examples for train, 5000 for valid, and 10000 for test
  """
  URL = dict(
      X_train=r"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
      y_train=r"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
      X_test=r"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
      y_test=r"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
  )

  MD5 = r"8ba71f60dccd53a0b68bfe41ed4cdf9c"

  def __init__(self, path='~/tensorflow_datasets/mnist'):
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
  def is_binary(self):
    return False

  @property
  def shape(self):
    return (28, 28, 1)


class BinarizedAlphaDigits(BinarizedMNIST):
  r""" Binary 20x16 digits of '0' through '9' and capital 'A' through 'Z'.
  39 examples of each class. """

  def __init__(self):
    import tensorflow_datasets as tfds
    self.train, self.valid, self.test = tfds.load(
        name='binary_alpha_digits',
        split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'],
        as_supervised=True,
        shuffle_files=True,
    )

  @property
  def shape(self):
    return (20, 16, 1)
