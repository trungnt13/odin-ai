import hashlib
import os
import zipfile
from functools import partial

import numpy as np
import requests
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from odin.utils import batching, get_datasetpath
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


class Shapes3D(object):
  r"""
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
      return tf.cast(ims, tf.float32) / 255., lab
    ims = ims[0]
    if len(ims.shape) == 4:
      return tf.cast(ims, tf.float32) / 255.
    return ims

  @staticmethod
  def deprocess(ims):
    r""" De-normalizing the image to range [0, 255] dtype tf.uint8"""
    return tf.cast(ims * 255., tf.uint8)

  def create_dataset(self,
                     batch_size=128,
                     drop_remainder=False,
                     shuffle=None,
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
      train = train.cache(cache)
      valid = valid.cache(cache)
      test = test.cache(cache)
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
