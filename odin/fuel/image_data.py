import hashlib
import os
import zipfile
from functools import partial

import numpy as np
import requests
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from odin.utils import as_tuple, batching, get_all_files, get_datasetpath
from odin.utils.crypto import md5_checksum


def download_file_from_google_drive(id, destination, chunk_size=32 * 1024):
  r""" Original code for dowloading the dataset:
  https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download """
  URL = "https://docs.google.com/uc?export=download"
  URL = "https://drive.google.com/uc?id="
  session = requests.Session()

  response = session.get(URL, params={'id': id}, stream=True)

  token = None
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      token = value
  if token:
    params = {'id': id, 'confirm': token}
    response = session.get(URL, params=params, stream=True)

  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size),
                      total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc=destination):
      if chunk:  # filter out keep-alive new chunks
        f.write(chunk)


class CelebA(object):
  r""" The dataset must be manually downloaded from Google Drive:
    https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFuzTm8

  The following files are required:
    - img_align_celeba.zip
    - list_attr_celeba.txt
    - list_eval_partition.txt

  Argument:
    path : path to the folder contained the three files.

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
                     return_mode=0):
    r"""
    Arguments:
      return_mode: An Integer. Determine which data will be returned
          - 0 : only return the images `(tf.uint8, (64, 64, 3))`
          - 1 : only return the labels `(tf.float64, (6,))`
          - otherwise return both images and labels
    """
    image_shape = [218, 178, 3]
    if image_size is not None:
      image_size = int(image_size)
      height = int(float(image_size) / image_shape[1] * image_shape[0])
      # offset_height, offset_width, target_height, target_width
      crop_offset = ((height - image_size) // 2, 0, image_size, image_size)

    if cache is not None:
      cache = as_tuple(cache, N=3, t=str)

    def read(path):
      img = tf.io.decode_jpeg(tf.io.read_file(path))
      img.set_shape(image_shape)
      img = tf.clip_by_value(tf.cast(img, tf.float32) / 255., 1e-6, 1. - 1e-6)
      if image_size is not None:
        img = tf.image.resize(img, (height, image_size),
                              preserve_aspect_ratio=True,
                              antialias=True)
        if square_image:
          img = tf.image.crop_to_bounding_box(img, *crop_offset)
      return img

    if return_mode == 1:
      train = tf.data.Dataset.from_tensor_slices(self.train_attr)
      valid = tf.data.Dataset.from_tensor_slices(self.valid_attr)
      test = tf.data.Dataset.from_tensor_slices(self.test_attr)
    else:
      train = tf.data.Dataset.from_tensor_slices(self.train_files).map(
          read, parallel)
      valid = tf.data.Dataset.from_tensor_slices(self.valid_files).map(
          read, parallel)
      test = tf.data.Dataset.from_tensor_slices(self.test_files).map(
          read, parallel)

    datasets = [train, valid, test]
    if cache is not None:
      datasets = [ds.cache(path) for ds, path in zip(datasets, cache)]
    # return both image and attributes
    if return_mode != 0:
      datasets = [
          tf.data.Dataset.zip((img, tf.data.Dataset.from_tensor_slices(att)))
          for img, att in zip(datasets, (self.train_attr, self.valid_attr,
                                         self.test_attr))
      ]
    # shuffle must be called after cache
    if shuffle is not None:
      datasets[0] = datasets[0].shuffle(int(shuffle))
    datasets = [ds.batch(batch_size, drop_remainder) for ds in datasets]
    if prefetch is not None:
      datasets = [ds.prefetch(prefetch) for ds in datasets]
    return tuple(datasets)


class Shapes3D(object):
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
    from bigarray import MmapArrayWriter, MmapArray
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
    self.train_indices = ids[:int(0.6 * n)]
    self.valid_indices = ids[int(0.6 * n):int(0.8 * n)]
    self.test_indices = ids[int(0.8 * n):]

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
                     return_mode=0):
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
      return_mode: An Integer. Determine which data will be returned
          - 0 : only return the images `(tf.uint8, (64, 64, 3))`
          - 1 : only return the labels `(tf.float64, (6,))`
          - otherwise return both images and labels
    """
    if cache is not None:
      cache = as_tuple(cache, N=3, t=str)
    if return_mode == 0:  # only images
      types = tf.uint8
      shapes = tf.TensorShape(self.images.shape[1:])
    elif return_mode == 1:  # only labels
      types = tf.float64
      shapes = tf.TensorShape(self.labels.shape[1:])
    else:  # both images and labels, note: a tuple must be used here
      types = (tf.uint8, tf.float64)
      shapes = (tf.TensorShape(self.images.shape[1:]),
                tf.TensorShape(self.labels.shape[1:]))

    def gen(indices):
      for i in indices:
        if return_mode == 0:
          yield self.images[i]
        elif return_mode == 1:
          yield self.labels[i]
        else:
          yield self.images[i], self.labels[i]

    base_dataset = lambda ids: tf.data.Dataset.from_generator(
        partial(gen, indices=ids), output_types=types, output_shapes=shapes
    ).batch(batch_size, drop_remainder).map(Shapes3D.process, parallel)

    train = base_dataset(self.train_indices)
    valid = base_dataset(self.valid_indices)
    test = base_dataset(self.test_indices)
    if cache is not None:
      train = train.cache(cache[0])
      valid = valid.cache(cache[1])
      test = test.cache(cache[2])
    if shuffle is not None:
      train = train.shuffle(shuffle)
    if prefetch is not None:
      train = train.prefetch(prefetch)
      valid = valid.prefetch(prefetch)
      test = test.prefetch(prefetch)
    return train, valid, test

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
