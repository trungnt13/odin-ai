import os
from functools import partial
from typing import Optional

import numpy as np
import tensorflow as tf
from bigarray import MmapArray, MmapArrayWriter
from odin.fuel.image_data._base import ImageDataset, get_partition
from odin.utils import batching
from tqdm import tqdm


# ===========================================================================
# Shapes 3D
# ===========================================================================
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

  Parameters
  -----------
  path: path to the "3dshapes.h5" downloaded from GCS
  cache_dir: path for storing the processed and memory-mapped numpy array
  seed: random seed when splitting the dataset into train, valid and test

  Reference
  ----------
  Burgess, Chris and Kim, Hyunjik (2018). 3D Shapes Dataset.
      https://github.com/deepmind/3dshapes-dataset
  """

  def __init__(self,
               path: str = '~/tensorflow_datasets/3dshapes.h5',
               cache_dir: Optional[str] = None,
               image_size: Optional[int] = None,
               seed: int = 1):
    path = os.path.abspath(os.path.expanduser(path))
    assert os.path.exists(path), f"Path to file {path} must exists"
    self.path = path
    if cache_dir is None:
      cache_dir = os.path.dirname(path)
    if not os.path.exists(cache_dir):
      os.mkdir(cache_dir)
    if image_size is None:
      suffix = ''
    else:
      image_size = int(image_size)
      suffix = str(image_size)
    image_path = os.path.join(cache_dir, f'3dshapes{suffix}.images')
    label_path = os.path.join(cache_dir, f'3dshapes.labels')
    ## read the dataset and cache it again
    if not os.path.exists(image_path) or not os.path.exists(label_path):
      import h5py
      with h5py.File(path, 'r') as dataset:
        images = dataset['images']
        labels = dataset['labels']
        shape = (images.shape if image_size is None else
                 (images.shape[0], image_size, image_size, 3))
        with MmapArrayWriter(image_path,
                             shape=shape,
                             dtype=np.uint8,
                             remove_exist=True) as img, \
          MmapArrayWriter(label_path,
                          shape=labels.shape,
                          dtype=labels.dtype,
                          remove_exist=True) as lab:
          for start, end in tqdm(list(batching(25000, n=images.shape[0])),
                                 desc="Caching data"):
            x, y = images[start:end], labels[start:end]
            if image_size is not None:
              x = tf.image.resize(x,
                                  method=tf.image.ResizeMethod.BILINEAR,
                                  size=(image_size, image_size),
                                  preserve_aspect_ratio=True,
                                  antialias=True).numpy().astype(np.uint8)
            img.write(x)
            lab.write(y)
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
    self.image_size = 64 if image_size is None else image_size

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
    indices = get_partition(
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
      ds = ds.shuffle(int(shuffle), seed=seed, reshuffle_each_iteration=True)
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


class Shapes3DMedium(Shapes3D):
  """ Shapes3D dataset with downsampled image (48, 48, 3) """

  def __init__(self,
               path='~/tensorflow_datasets/3dshapes.h5',
               cache_dir=None,
               seed=8):
    super().__init__(path=path, cache_dir=cache_dir, image_size=48, seed=seed)


class Shapes3DSmall(Shapes3D):
  """ Shapes3D dataset with downsampled image (28, 28, 3) """

  def __init__(self,
               path='~/tensorflow_datasets/3dshapes.h5',
               cache_dir=None,
               seed=8):
    super().__init__(path=path, cache_dir=cache_dir, image_size=28, seed=seed)


# ===========================================================================
# dSprites
# ===========================================================================
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
    ds = get_partition(partition,
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
      ds = ds.shuffle(int(shuffle), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds


class dSpritesC(dSprites):
  r""" Same as dSprites but the factors labels are non-negative continuous
  values """

  def __init__(self):
    super().__init__(continuous=True)
