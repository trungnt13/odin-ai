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
from odin.fuel._image_data1 import (BinarizedAlphaDigits, BinarizedMNIST,
                                    ImageDataset, _partition)
from odin.utils import (as_tuple, batching, get_all_files, get_datasetpath,
                        one_hot)
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

  def __init__(self, path="~/tensorflow_datasets/celeb_a"):
    path = os.path.abspath(os.path.expanduser(path))
    self.path = path
    assert os.path.isdir(path), "'%s' must be a directory" % path
    zip_path = os.path.join(path, 'img_align_celeba.zip')
    attr_path = os.path.join(path, 'list_attr_celeba.txt')
    part_path = os.path.join(path, 'list_eval_partition.txt')
    for i in [zip_path, attr_path, part_path]:
      assert os.path.exists(i), "'%s' must exists" % i
    # ====== read the attr ====== #
    with open(attr_path, 'r') as f:
      text = [[i for i in line.strip().split(' ') if len(i) > 0] for line in f]
      self._header = np.array(text[1])
      attributes = np.array(text[2:])[:, 1:]
    # ====== read the partition ====== #
    with open(part_path, 'r') as f:
      partition = np.array([line.strip().split(' ') for line in f])[:, 1:]
    # ====== extracting the data ====== #
    image_path = os.path.join(path, 'img_align_celeba')
    image_files = get_all_files(image_path,
                                filter_func=lambda x: '.jpg' == x[-4:])
    if len(image_files) != 202599:
      with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path)
      image_files = get_all_files(image_path,
                                  filter_func=lambda x: '.jpg' == x[-4:])
    image_files = np.array(sorted(image_files))
    assert image_files.shape[0] == attributes.shape[0] == partition.shape[0]
    # ====== splitting the data ====== #
    attributes = attributes.astype('float32')
    train_files, train_attr = [], []
    valid_files, valid_attr = [], []
    test_files, test_attr = [], []
    for path, attr, part in zip(image_files, attributes, partition):
      part = int(part[0])
      if part == 0:
        train_files.append(path)
        train_attr.append(attr)
      elif part == 1:
        valid_files.append(path)
        valid_attr.append(attr)
      else:
        test_files.append(path)
        test_attr.append(attr)
    self.train_files = np.array(train_files)
    self.valid_files = np.array(valid_files)
    self.test_files = np.array(test_files)
    self.train_attr = np.array(train_attr)
    self.valid_attr = np.array(valid_attr)
    self.test_attr = np.array(test_attr)

  @property
  def shape(self):
    return (218, 178, 3)

  @property
  def is_binary(self):
    return False

  @property
  def attribute_name(self):
    return self._header

  def create_dataset(self,
                     batch_size=64,
                     image_size=64,
                     square_image=True,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     partition='train',
                     inc_labels=True) -> tf.data.Dataset:
    r""" Data

    Arguments:
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean. If True, return both image and label, otherwise,
        only image is returned.

    Return:
      train, valid, test: `tensorflow.data.Dataset`
         - image `(tf.float32, (64, 64, 3))`
         - label `(tf.float32, (40,))`
    """
    image_shape = [218, 178, 3]
    if image_size is not None:
      image_size = int(image_size)
      height = int(float(image_size) / image_shape[1] * image_shape[0])
      # offset_height, offset_width, target_height, target_width
      crop_offset = ((height - image_size) // 2, 0, image_size, image_size)

    def read(path):
      img = tf.io.decode_jpeg(tf.io.read_file(path))
      img.set_shape(image_shape)
      img = tf.clip_by_value(tf.cast(img, tf.float32) / 255., 1e-6, 1. - 1e-6)
      if image_size is not None:
        img = tf.image.resize(img, (height, image_size),
                              preserve_aspect_ratio=True,
                              antialias=False)
        if square_image:
          img = tf.image.crop_to_bounding_box(img, *crop_offset)
      return img

    ### select partition
    images, attrs = _partition(
        partition,
        train=(self.train_files, self.train_attr),
        valid=(self.valid_files, self.valid_attr),
        test=(self.test_files, self.test_attr),
    )
    images = tf.data.Dataset.from_tensor_slices(images).map(read, parallel)
    if inc_labels:
      attrs = tf.data.Dataset.from_tensor_slices(self.train_attr)

    if cache is not None:
      images = images.cache(str(cache))
    # return both image and attributes
    if inc_labels:
      images = tf.data.Dataset.zip((images, attrs))
    # shuffle must be called after cache
    if shuffle is not None:
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
    'floor_hue': 10,
    'wall_hue': 10,
    'object_hue': 10,
    'scale': 8,
    'shape': 4,
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
    self.labels = MmapArray(label_path)
    # ====== split the dataset ====== #
    rand = np.random.RandomState(seed=seed)
    n = len(self.images)
    ids = rand.permutation(n)
    # train:70% valid:10% test:20%
    self.train_indices = ids[:int(0.7 * n)]
    self.valid_indices = ids[int(0.7 * n):int(0.8 * n)]
    self.test_indices = ids[int(0.8 * n):]

  @property
  def is_binary(self):
    return False

  @property
  def shape(self):
    return (64, 64, 3)

  @staticmethod
  def process(*ims):
    r""" Normalizing the image to range [0., 1.] dtype tf.float32"""
    if len(ims) == 2:
      ims, lab = ims
      return tf.clip_by_value(tf.cast(ims, tf.float32) / 255., 1e-6,
                              1. - 1e-6), lab
    ims = ims[0]
    if len(ims.shape) == 4:
      return tf.clip_by_value(tf.cast(ims, tf.float32) / 255., 1e-6, 1. - 1e-6)
    return ims

  def create_dataset(self,
                     batch_size=128,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     partition='train',
                     inc_labels=True) -> tf.data.Dataset:
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
      inc_labels : a Boolean. If True, return both image and label, otherwise,
        only image is returned.

    Return:
      train, valid, test: `tensorflow.data.Dataset`
         - image `(tf.float32, (64, 64, 3))`
         - label `(tf.float32, (6,))`
    """
    # both images and labels, note: a tuple must be used here
    types = (tf.uint8, tf.float32)
    shapes = (tf.TensorShape(self.images.shape[1:]),
              tf.TensorShape(self.labels.shape[1:]))
    if not inc_labels:
      types = types[0]
      shapes = shapes[0]

    def gen(indices):
      for i in indices:
        if inc_labels:
          yield self.images[i], tf.cast(self.labels[i], dtype=tf.float32)
        else:
          yield self.images[i]

    base_dataset = lambda ids: tf.data.Dataset.from_generator(
        partial(gen, indices=ids), output_types=types, output_shapes=shapes
    ).batch(batch_size, drop_remainder).map(Shapes3D.process, parallel)
    ### get the right partition
    indices = _partition(
        partition,
        train=self.train_indices,
        valid=self.valid_indices,
        test=self.test_indices,
    )
    ds = base_dataset(indices)
    if cache is not None:
      ds = ds.cache(str(cache))
    if shuffle is not None:
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
          np.all(self.labels[start:end] == labels1[start:end])
    return self


class MultidSprites(object):
  r""" https://github.com/deepmind/multi_object_datasets """

  def __init__(self,
               path='~/tensorflow_datasets/3dshapes.h5',
               cache_dir=None,
               seed=8):
    pass


class LegoFaces(object):
  r""" https://www.echevarria.io/blog/lego-face-vae/index.html """

  def __init__(self, path):
    super().__init__()


class SLT10(ImageDataset):
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
  MD5_DOWNLOAD = "91f7769df0f17e558f3565bffb0c7dfb"
  MD5_EXTRACT = "6d49c882f94d0659c0aea2ac58068e9c"
  IMAGE_SHAPE = (3, 96, 96)

  def __init__(self, path="~/slt10"):
    self.path, self.extract_path = download_and_extract(
        path,
        SLT10.URL,
        extract=True,
        md5_download=SLT10.MD5_DOWNLOAD,
        md5_extract=SLT10.MD5_EXTRACT)
    ### read all the images
    self.bin_files = {
        name.split('.')[0]: os.path.join(self.extract_path, name)
        for name in os.listdir(self.extract_path)
        if '.bin' in name
    }
    with open(os.path.join(self.extract_path, "class_names.txt"), 'r') as f:
      self.class_names = np.array([line.strip() for line in f])

  @property
  def is_binary(self):
    return False

  @property
  def shape(self):
    return (96, 96, 3)

  def create_dataset(self,
                     batch_size=64,
                     image_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     partition='train',
                     inc_labels=True) -> tf.data.Dataset:
    r"""
    Arguments:
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean. If True, return both image and label, otherwise,
        only image is returned.

    Return :
      train, test, unlabeled : `tensorflow.data.Dataset`
        image - `(tf.float32, (64, 64, 3))`
        label - `(tf.float32, (10,))`
    """
    if isinstance(image_size, Number) and image_size == 96:
      image_size = None
    ### select partition
    images_path, labels_path = _partition(
        partition,
        train=(self.bin_files['train_X'], self.bin_files['train_y']),
        test=(self.bin_files['test_X'], self.bin_files['test_y']),
        unlabeled=(self.bin_files['unlabeled_X'], None),
    )
    X = np.reshape(np.fromfile(images_path, dtype=np.uint8),
                   (-1,) + SLT10.IMAGE_SHAPE)
    if labels_path is None:  # unlabled data
      inc_labels = False
    if inc_labels:
      y = np.fromfile(labels_path, dtype=np.uint8) - 1
      y = one_hot(y, len(self.class_names)).astype(np.float32)
    ### read and resize the data
    def resize(img):
      img = tf.clip_by_value(tf.cast(img, tf.float32) / 255., 1e-6, 1. - 1e-6)
      img = tf.transpose(img, perm=(1, 2, 0))
      if image_size is not None:
        img = tf.image.resize(img, (image_size, image_size),
                              preserve_aspect_ratio=False,
                              antialias=False)
      return img

    ### processing
    images = tf.data.Dataset.from_tensor_slices(X).map(resize, parallel)
    if inc_labels:
      labels = tf.data.Dataset.from_tensor_slices(y)
    if cache is not None:
      images = images.cache(str(cache))
    # return both image and attributes
    if inc_labels:
      images = tf.data.Dataset.zip((images, labels))
    # shuffle must be called after cache
    if shuffle is not None:
      images = images.shuffle(int(shuffle))
    images = images.batch(batch_size, drop_remainder)
    if prefetch is not None:
      images = images.prefetch(prefetch)
    return images
