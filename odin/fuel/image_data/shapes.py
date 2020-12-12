import os
from functools import partial
from typing import Optional, Union
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from bigarray import MmapArray, MmapArrayWriter
from odin.fuel.image_data._base import ImageDataset
from odin.fuel.dataset_base import get_partition
from odin.utils import batching
from tqdm import tqdm
from typeguard import typechecked

__all__ = [
    'Shapes3D',
    'Shapes3DSmall',
    'dSprites',
    'dSpritesSmall',
]


# ===========================================================================
# Shapes 3D
# ===========================================================================
class _ShapeDataset(ImageDataset):

  @typechecked
  def __init__(self,
               name: Literal['dsprites', 'shapes3d'],
               image_size: int = 64,
               continuous: bool = True,
               seed: int = 1):
    super().__init__()
    import tensorflow_datasets as tfds
    tf.random.set_seed(seed)
    try:

      self.train, self.valid, self.test = tfds.load(
          name,
          split=["train[:85%]", "train[85%:90%]", "train[90%:]"],
          read_config=tfds.ReadConfig(shuffle_seed=seed,
                                      shuffle_reshuffle_each_iteration=True),
          shuffle_files=True)
    except Exception as e:
      import traceback
      traceback.print_exc()
      print('Try `ulimit -n 100000`')
      exit()
    self.image_size = image_size
    self.dsname = name
    if continuous:
      self._prefix = 'value_'
      self._continuous_labels = True
    else:
      self._prefix = 'label_'
      self._continuous_labels = False
    if name == 'dsprites':
      self._factors = np.array(
          ['orientation', 'scale', 'shape', 'x_position', 'y_position'])
      self.n_channels = 1
    else:
      self._factors = np.array([
          'orientation', 'scale', 'shape', 'floor_hue', 'wall_hue', 'object_hue'
      ])
      self.n_channels = 3

  @property
  def labels(self):
    return self._factors

  @property
  def is_binary(self):
    return False

  @property
  def shape(self):
    return (self.image_size, self.image_size, self.n_channels)

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
    """

    Parameters
    ----------
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
        image - `(tf.float32, (None, 64, 64, 1))`
        label - `(tf.float32, (None, 5))` for dSprites and  `(None, 6)` for Shapes3D
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    ds = get_partition(partition,
                       train=self.train,
                       valid=self.valid,
                       test=self.test)
    factors = [f'{self._prefix}{i}' for i in self._factors]
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def _process(data):
      image = tf.cast(data['image'], tf.float32)
      # normalize the image
      if self.dsname == 'shapes3d':
        image = self.normalize_255(image)
      else:
        image = tf.clip_by_value(image, 1e-6, 1. - 1e-6)
      # resize the image
      if self.image_size != 64:
        image = tf.image.resize(image, (self.image_size, self.image_size),
                                method=tf.image.ResizeMethod.BILINEAR,
                                preserve_aspect_ratio=True,
                                antialias=True)
      # process the labels
      if inc_labels:
        # dSprites shapes attribute is encoded as [1, 2, 3], should be [0, 1, 2]
        label = []
        for name in factors:
          fi = data[name]
          if self._continuous_labels:
            if self.dsname == 'dsprites':
              if 'orientation' in name:  # orientation
                fi = fi - np.pi
              elif 'shape' in name:  # shape
                fi = fi - 1
            else:
              if 'orientation' in name:  # orientation
                fi = fi / 30. * np.pi
          label.append(fi)
        label = tf.convert_to_tensor(label, dtype=tf.float32)
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


class Shapes3D(_ShapeDataset):
  """ The dataset must be manually downloaded from GCS at
    https://console.cloud.google.com/storage/browser/3d-shapes

  All the factors are:
    ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
  Values per factor:
    'orientation': 15
    'scale': 8
    'shape': 4
    'floor_hue': 10
    'wall_hue': 10
    'object_hue': 10
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
               image_size: int = 64,
               continuous: bool = True,
               seed: int = 1):
    super().__init__(name='shapes3d',
                     image_size=image_size,
                     continuous=continuous,
                     seed=seed)


class Shapes3DSmall(Shapes3D):
  """Shapes3D dataset with downsampled image (28, 28, 3) """

  def __init__(self, continuous: bool = True, seed: int = 1):
    super().__init__(image_size=28, continuous=continuous, seed=seed)


# ===========================================================================
# dSprites
# ===========================================================================
class dSprites(_ShapeDataset):
  """dSprites dataset with continuous non-negative attributes values
  by defaults.

  Discrete attributes:
    label_orientation: ()
    label_scale: ()
    label_shape: ()
    label_x_position: ()
    label_y_position: ()

  Continuous attributes
    value_orientation: 40 values in [-pi, pi]
    value_scale:  6 values linearly spaced in [0.5, 1]
    value_shape: square, ellipse, heart
    value_x_position:  32 values in [0, 1]
    value_y_position:  32 values in [0, 1]
  """

  def __init__(self,
               image_size: int = 64,
               continuous: bool = True,
               seed: int = 1):
    super().__init__(name='dsprites',
                     image_size=image_size,
                     continuous=continuous,
                     seed=seed)


class dSpritesSmall(dSprites):
  """dSprites dataset with downsampled image (28, 28, 1) """

  def __init__(self, continuous: bool = True, seed: int = 1):
    super().__init__(image_size=28, continuous=continuous, seed=seed)
