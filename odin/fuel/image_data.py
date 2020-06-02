import glob
import hashlib
import os
import shutil
import tarfile
import time
import urllib
import zipfile
from functools import partial
from numbers import Number

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from bigarray import MmapArray, MmapArrayWriter
from odin.fuel._image_base import (MNIST, BinarizedAlphaDigits, BinarizedMNIST,
                                   ImageDataset, _partition)
from odin.fuel._image_cifar import CIFAR10, CIFAR20, CIFAR100
from odin.fuel._image_lego_faces import LegoFaces
from odin.fuel._image_synthesize import YDisentanglement
from odin.utils import as_tuple, batching, get_datasetpath, one_hot
from odin.utils.crypto import md5_checksum
from odin.utils.net_utils import download_and_extract, download_google_drive


# ===========================================================================
# Datasets
# ===========================================================================
class CelebA(ImageDataset):
  r""" The dataset must be manually downloaded from Google Drive:
    https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFuzTm8

  The following files are required:
    - img_align_celeba.zip
    - list_attr_celeba.txt
    - list_eval_partition.txt

  Argument:
    path : path to the folder contained the three files.
    image_size : (optional) an Integer.
      The smallest dimension of a downsampled image. if `None`, original size
      (218, 178) is kept.
    square_image : a Boolean. If True, crop the downsampled image to a square.

  Attributes:
    train_files, valid_files, test_files : `numpy.ndarray`, list of path
      to images shape `[218, 178, 3]`
    train_attr, valid_attr, test_attr : `numpy.ndarray`, 40 attributes
      for each images

  Reference:
    Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, (2015).
      Deep Learning Face Attributes in the Wild. Proceedings of International
      Conference on Computer Vision (ICCV)
  """

  def __init__(self,
               path="~/tensorflow_datasets/celeb_a",
               image_size=64,
               square_image=True):
    self.image_size = image_size
    self.square_image = bool(square_image)
    path = os.path.abspath(os.path.expanduser(path))
    self.path = path
    assert os.path.isdir(path), "'%s' must be a directory" % path
    zip_path = os.path.join(path, 'img_align_celeba.zip')
    attr_path = os.path.join(path, 'list_attr_celeba.txt')
    attr_cache = attr_path + '.npz'
    part_path = os.path.join(path, 'list_eval_partition.txt')
    for i in [zip_path, attr_path, part_path]:
      assert os.path.exists(i), "'%s' must exists" % i
    ### read the attr
    if not os.path.exists(attr_cache):
      with open(attr_path, 'r') as f:
        text = [[i for i in line.strip().split(' ') if len(i) > 0] for line in f
               ]
        header = np.array(text[1])
        attributes = np.array(text[2:])[:, 1:]
      with open(attr_cache, 'wb') as f:
        np.savez(f, header=header, attributes=attributes.astype(np.int8))
    else:
      with open(attr_cache, 'rb') as f:
        data = np.load(f)
        header = data['header']
        attributes = data['attributes']
    self._header = header
    attributes = attributes.astype(np.float32)
    ### read the partition
    with open(part_path, 'r') as f:
      partition = np.array([line.strip().split(' ') for line in f])[:, 1:]
    ### extracting the data
    image_path = os.path.join(path, 'img_align_celeba')
    image_files = glob.glob(image_path + "/*.jpg", recursive=True)
    if len(image_files) != 202599:
      print("Extracting 202599 image files ...")
      with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path)
      image_files = glob.glob(image_path + '/*.jpg', recursive=True)
    image_files = np.array(sorted(image_files))
    assert image_files.shape[0] == attributes.shape[0] == partition.shape[0]
    ### splitting the data
    attributes = attributes.astype('float32')
    train_files, train_attr = [], []
    valid_files, valid_attr = [], []
    test_files, test_attr = [], []
    for path, attr, part in zip(image_files, attributes, partition):
      part = int(part[0])
      if part == 0:  # train
        train_files.append(path)
        train_attr.append(attr)
      elif part == 1:  # valid
        valid_files.append(path)
        valid_attr.append(attr)
      else:  # test
        test_files.append(path)
        test_attr.append(attr)
    ### store the attributes
    self.train_files = np.array(train_files)
    self.valid_files = np.array(valid_files)
    self.test_files = np.array(test_files)
    self.train_attr = np.array(train_attr)
    self.valid_attr = np.array(valid_attr)
    self.test_attr = np.array(test_attr)

  @property
  def original_shape(self):
    return (218, 178, 3)

  @property
  def shape(self):
    if self.image_size is None:
      return self.original_shape
    h, w = self.original_shape[:2]
    image_size = int(self.image_size)
    return (image_size if self.square_image else int(float(image_size) / w * h),
            image_size, 3)

  @property
  def is_binary(self):
    return False

  @property
  def labels(self):
    return self._header

  def create_dataset(self,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     partition='train',
                     inc_labels=True,
                     seed=1) -> tf.data.Dataset:
    r""" The default argument will downsize and crop the image to square size
    (64, 64)

    Arguments:
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean or Scalar. If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.

    Return :
      tensorflow.data.Dataset :
        image - `(tf.float32, (None, 64, 64, 3))`
        label - `(tf.float32, (None, 40))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    image_shape = self.original_shape
    image_size = self.image_size
    if image_size is not None:
      image_size = int(image_size)
      height = int(float(image_size) / image_shape[1] * image_shape[0])
      # offset_height, offset_width, target_height, target_width
      crop_offset = ((height - image_size) // 2, 0, image_size, image_size)
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def read(path):
      img = tf.io.decode_jpeg(tf.io.read_file(path))
      img.set_shape(image_shape)
      img = tf.cast(img, tf.float32)
      img = self.normalize_255(img)
      if image_size is not None:
        img = tf.image.resize(img, (height, image_size),
                              preserve_aspect_ratio=True,
                              antialias=False)
        if self.square_image:
          img = tf.image.crop_to_bounding_box(img, *crop_offset)
      return img

    def mask(image, label):
      mask = gen.uniform(shape=(1,)) < inc_labels
      return dict(inputs=(image, label), mask=mask)

    ### select partition
    images, attrs = _partition(
        partition,
        train=(self.train_files, self.train_attr),
        valid=(self.valid_files, self.valid_attr),
        test=(self.test_files, self.test_attr),
    )
    # convert [-1, 1] to [0., 1.]
    attrs = (attrs + 1.) / 2
    images = tf.data.Dataset.from_tensor_slices(images).map(read, parallel)
    if inc_labels:
      attrs = tf.data.Dataset.from_tensor_slices(attrs)
      images = tf.data.Dataset.zip((images, attrs))
      if 0. < inc_labels < 1.:  # semi-supervised mask
        images = images.map(mask)

    if cache is not None:
      images = images.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      images = images.shuffle(int(shuffle))
    images = images.batch(batch_size, drop_remainder)
    if prefetch is not None:
      images = images.prefetch(prefetch)
    return images


class Shapes3D(ImageDataset):
  r""" The dataset must be manually downloaded from GCS at
    https://console.cloud.google.com/storage/browser/3d-shapes

  All the factors are:
    ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
  Values per factor:
    'floor_hue': 10
    'wall_hue': 10
    'object_hue': 10
    'scale': 8
    'shape': 4
    'orientation': 15
  Pixel range [0, 255]

  Arguments:
    path: path to the "3dshapes.h5" downloaded from GCS
    cache_dir: path for storing the processed and memory-mapped numpy array
    seed: random seed when splitting the dataset into train, valid and test

  Reference:
    Burgess, Chris and Kim, Hyunjik (2018). 3D Shapes Dataset.
      https://github.com/deepmind/3dshapes-dataset
  """

  def __init__(self,
               path='~/tensorflow_datasets/3dshapes.h5',
               cache_dir=None,
               seed=8):
    path = os.path.abspath(os.path.expanduser(path))
    assert os.path.exists(path), "Path to file %s must exists" % path
    self.path = path
    if cache_dir is None:
      cache_dir = os.path.dirname(path)
    if not os.path.exists(cache_dir):
      os.mkdir(cache_dir)
    image_path = os.path.join(cache_dir, '3dshapes.images')
    label_path = os.path.join(cache_dir, '3dshapes.labels')
    # ====== read the dataset and cache it again ====== #
    if not os.path.exists(image_path) or not os.path.exists(label_path):
      import h5py
      with h5py.File(path, 'r') as dataset:
        images = dataset['images']
        labels = dataset['labels']
        with MmapArrayWriter(image_path,
                             shape=images.shape,
                             dtype=images.dtype,
                             remove_exist=True) as img, \
          MmapArrayWriter(label_path,
                          shape=labels.shape,
                          dtype=labels.dtype,
                          remove_exist=True) as lab:
          for start, end in tqdm(list(batching(8000, n=images.shape[0])),
                                 desc="Caching data"):
            img.write(images[start:end])
            lab.write(labels[start:end])
    # ====== load the data ====== #
    self.images = MmapArray(image_path)
    self.factors = MmapArray(label_path)
    # ====== split the dataset ====== #
    rand = np.random.RandomState(seed=seed)
    n = len(self.images)
    ids = rand.permutation(n)
    # train:85% valid:5% test:10%
    self.train_indices = ids[:int(0.85 * n)]
    self.valid_indices = ids[int(0.85 * n):int(0.9 * n)]
    self.test_indices = ids[int(0.9 * n):]

  @property
  def labels(self):
    return np.array([
        'floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'
    ])

  @property
  def is_binary(self):
    return False

  @property
  def shape(self):
    return (64, 64, 3)

  def create_dataset(self,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     partition='train',
                     inc_labels=True,
                     seed=1) -> tf.data.Dataset:
    r""" Create tensorflow dataset for train, valid and test
      The images are normalized in range [-1, 1]

    Arguments:
      batch_size: A tf.int64 scalar tf.Tensor, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: A tf.bool scalar tf.Tensor, representing whether the
        last batch should be dropped in the case it has fewer than batch_size
        elements; the default behavior is not to drop the smaller batch.
      shuffle: A tf.int64 scalar tf.Tensor, representing the number of elements
        from this dataset from which the new dataset will sample.
        If `None` or smaller or equal 0, turn off shuffling
      prefetch:  A tf.int64 scalar tf.Tensor, representing the maximum number
        of elements that will be buffered when prefetching.
      cache: A tf.string scalar tf.Tensor, representing the name of a directory
        on the filesystem to use for caching elements in this Dataset. If a
        filename is not provided, the dataset will be cached in memory.
        If `None`, turn off caching
      parallel: A tf.int32 scalar tf.Tensor, representing the number elements
        to process asynchronously in parallel. If not specified, elements will
        be processed sequentially. If the value `tf.data.experimental.AUTOTUNE`
        is used, then the number of parallel calls is set dynamically based
        on available CPU.
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean or Scalar. If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.

    Return :
      tensorflow.data.Dataset :
        image - `(tf.float32, (None, 64, 64, 3))`
        label - `(tf.float32, (None, 6))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    # both images and labels, note: a tuple must be used here
    types = (tf.float32, tf.float32)
    shapes = (tf.TensorShape(self.images.shape[1:]),
              tf.TensorShape(self.factors.shape[1:]))
    if not inc_labels:
      types = types[0]
      shapes = shapes[0]
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def gen_data(indices):
      for i in indices:
        img = self.images[i]
        img = tf.cast(img, tf.float32)
        img = tf.clip_by_value(img / 255., 1e-6, 1. - 1e-6)
        if inc_labels:
          yield img, tf.cast(self.factors[i], dtype=tf.float32)
        else:
          yield img

    def process(*ims):
      r""" Normalizing the image to range [0., 1.] dtype tf.float32"""
      if inc_labels:
        ims, lab = ims
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(tf.shape(ims)[0], 1)) < inc_labels
          return dict(inputs=(ims, lab), mask=mask)
        return ims, lab
      return ims[0]

    ### get the right partition
    indices = _partition(
        partition,
        train=self.train_indices,
        valid=self.valid_indices,
        test=self.test_indices,
    )
    ds = tf.data.Dataset.from_generator(partial(gen_data, indices),
                                        output_types=types,
                                        output_shapes=shapes)
    ds = ds.batch(batch_size, drop_remainder).map(process, parallel)
    if cache is not None:
      ds = ds.cache(str(cache))
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(shuffle)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds

  def validate_data(self, path=None):
    if path is None:
      path = self.path
    import h5py
    with h5py.File(path, 'r') as dataset:
      images1 = dataset['images']
      labels1 = dataset['labels']
      for start, end in tqdm(list(batching(8000, n=self.images.shape[0]))):
        assert np.all(self.images[start:end] == images1[start:end]) and \
          np.all(self.factors[start:end] == labels1[start:end])
    return self


class dSprites(ImageDataset):
  r"""

  Discrete attributes:
    label_orientation: ()
    label_scale: ()
    label_shape: ()
    label_x_position: ()
    label_y_position: ()

  Continuous attributes
    value_orientation: 40 values in [0, 2pi]
    value_scale:  6 values linearly spaced in [0.5, 1]
    value_shape: square, ellipse, heart
    value_x_position:  32 values in [0, 1]
    value_y_position:  32 values in [0, 1]
  """

  def __init__(self, continuous=False):
    super().__init__()
    import tensorflow_datasets as tfds
    self.train, self.valid, self.test = tfds.load(
        "dsprites",
        split=["train[:85%]", "train[85%:90%]", "train[90%:]"],
        shuffle_files=True)
    if continuous:
      self._factors = np.array([
          'value_orientation', 'value_scale', 'value_shape', 'value_x_position',
          'value_y_position'
      ])
    else:
      self._factors = np.array([
          'label_orientation', 'label_scale', 'label_shape', 'label_x_position',
          'label_y_position'
      ])

  @property
  def labels(self):
    return self._factors

  @property
  def is_binary(self):
    return True

  @property
  def shape(self):
    return (64, 64, 1)

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
        image - `(tf.float32, (None, 64, 64, 1))`
        label - `(tf.float32, (None, 5))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    ds = _partition(partition,
                    train=self.train,
                    valid=self.valid,
                    test=self.test)
    factors = self._factors
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def _process(data):
      image = tf.cast(data['image'], tf.float32)
      if inc_labels:
        label = tf.convert_to_tensor([data[i] for i in factors],
                                     dtype=tf.float32)
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=(image, label), mask=mask)
        return image, label
      return image

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


class dSpritesC(dSprites):
  r""" Same as dSprites but the factors labels are non-negative continuous
  values """

  def __init__(self):
    super().__init__(continuous=True)


class MultidSprites(object):
  r""" https://github.com/deepmind/multi_object_datasets """

  def __init__(self,
               path='~/tensorflow_datasets/3dshapes.h5',
               cache_dir=None,
               seed=8):
    pass


class STL10(ImageDataset):
  r""" Overview
   - 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey,
      ship, truck.
   - Images are 96x96 pixels, color.
   - 500 training images (10 pre-defined folds), 800 test images per class.
   - 100000 unlabeled images for unsupervised learning. These examples are
      extracted from a similar but broader distribution of images.
      For instance, it contains other types of animals (bears, rabbits, etc.)
      and vehicles (trains, buses, etc.) in addition to the ones in the
      labeled set.
   - Images were acquired from labeled examples on ImageNet.

  Reference:
    Adam Coates, Honglak Lee, Andrew Y. Ng . "An Analysis of Single Layer
      Networks in Unsupervised Feature Learning". AISTATS, 2011.

  Link:
    http://ai.stanford.edu/~acoates/stl10
  """
  URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
  MD5_DOWNLOAD = r"91f7769df0f17e558f3565bffb0c7dfb"
  MD5_EXTRACT = r"559636c835853bf1aca295ab34f5ad9e"
  IMAGE_SHAPE = (3, 96, 96)

  def __init__(self, path="~/tensorflow_datasets/stl10", image_size=64):
    self.path, self.extract_path = download_and_extract(
        path,
        STL10.URL,
        extract=True,
        md5_download=STL10.MD5_DOWNLOAD,
        md5_extract=STL10.MD5_EXTRACT)
    ### read all the images
    self.bin_files = {
        name.split('.')[0]: os.path.join(self.extract_path, name)
        for name in os.listdir(self.extract_path)
        if '.bin' in name
    }
    with open(os.path.join(self.extract_path, "class_names.txt"), 'r') as f:
      self.class_names = np.array([line.strip() for line in f])
    self.image_size = int(image_size)

  @property
  def labels(self):
    return np.array([
        'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey',
        'ship', 'truck'
    ])

  @property
  def is_binary(self):
    return False

  @property
  def shape(self):
    return (self.image_size, self.image_size, 3)

  def create_dataset(self,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     partition='train',
                     inc_labels=True,
                     seed=1) -> tf.data.Dataset:
    r"""
    Arguments:
      partition : {'train', 'train_labelled', 'valid', 'test', 'unlabelled'}
        - 'train' : combination of both train and unlablled
        - 'train-labelled' : only the train data
      inc_labels : a Boolean or Scalar. If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.

    Return :
      tensorflow.data.Dataset :
        image - `(tf.float32, (None, 64, 64, 3))`
        label - `(tf.float32, (None, 10))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    image_size = self.image_size
    if isinstance(image_size, Number) and image_size == 96:
      image_size = None
    ### select partition
    images_path, labels_path = _partition(
        partition,
        train=((self.bin_files['train_X'], self.bin_files['unlabeled_X']),
               self.bin_files['train_y']),
        train_labelled=(self.bin_files['train_X'], self.bin_files['train_y']),
        test=(self.bin_files['test_X'], self.bin_files['test_y']),
        unlabeled=(self.bin_files['unlabeled_X'], None),
        unlabelled=(self.bin_files['unlabeled_X'], None),
    )

    X = [
        np.reshape(np.fromfile(path, dtype=np.uint8), (-1,) + STL10.IMAGE_SHAPE)
        for path in tf.nest.flatten(images_path)
    ]
    is_unlabelled = (labels_path is None)
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)
    # load the labels
    if inc_labels:
      if is_unlabelled:
        y = [np.zeros(shape=(X[0].shape[0], self.n_labels), dtype=np.float32)]
      else:
        y = np.fromfile(labels_path, dtype=np.uint8) - 1
        y = [one_hot(y, self.n_labels).astype(np.float32)]
        if len(X) == 2:  # combined of both train and unlablled set
          y.append(
              np.zeros(shape=(X[1].shape[0], self.n_labels), dtype=np.float32))
      assert len(y) == len(X)

    ### read and resize the data
    def resize(img):
      img = tf.cast(img, tf.float32)
      img = self.normalize_255(img)
      img = tf.transpose(img, perm=(2, 1, 0))
      if image_size is not None:
        img = tf.image.resize(img, (image_size, image_size),
                              preserve_aspect_ratio=True,
                              antialias=False)
      return img

    def masking(image, label):
      mask = tf.logical_and(
          gen.uniform(shape=(1,)) < inc_labels,
          tf.reduce_sum(label) > 0.)
      return dict(inputs=(image, label), mask=mask)

    ### processing
    datasets = None
    must_masking = any(np.all(i == 0.) for i in y)
    for x_i, y_i in zip(X, y if inc_labels else X):
      images = tf.data.Dataset.from_tensor_slices(x_i).map(resize, parallel)
      if inc_labels:
        labels = tf.data.Dataset.from_tensor_slices(y_i)
        images = tf.data.Dataset.zip((images, labels))
        if 0. < inc_labels < 1. or must_masking:  # semi-supervised mask
          images = images.map(masking)
      datasets = images if datasets is None else datasets.concatenate(images)
    # cache data
    if cache is not None:
      datasets = datasets.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      datasets = datasets.shuffle(int(shuffle) * len(X))
    datasets = datasets.batch(batch_size, drop_remainder)
    if prefetch is not None:
      datasets = datasets.prefetch(prefetch)
    # return
    return datasets
