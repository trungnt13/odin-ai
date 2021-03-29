from typing import Optional, Union

import numpy as np
import tensorflow as tf
from typeguard import typechecked
from typing_extensions import Literal

import tensorflow_datasets as tfds
from odin.fuel.dataset_base import get_partition
from odin.fuel.image_data._base import ImageDataset

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
    try:
      self.train, self.valid, self.test = tfds.load(
          name,
          split=["train[:90%]", "train[90%:95%]", "train[95%:]"],
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

    ### convert to tensorflow dataset
    factors = [f'{self._prefix}{i}' for i in self._factors]

    @tf.function
    def process(data):
      image = tf.cast(data['image'], tf.float32)
      if self.dsname == 'dsprites':
        image = image * 255.0
      ## resize the image
      if self.image_size != 64:
        image = tf.image.resize(image, (self.image_size, self.image_size),
                                method=tf.image.ResizeMethod.BILINEAR,
                                preserve_aspect_ratio=True,
                                antialias=True)
      # resize sampling would result numerical unstable values
      image = tf.clip_by_value(image, 0.0, 255.0)
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
      return image, label

    self.train = self.train.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    self.valid = self.valid.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    self.test = self.test.map(process, num_parallel_calls=tf.data.AUTOTUNE)

  @property
  def labels(self):
    return self._factors

  @property
  def binarized(self) -> bool:
    return False

  @property
  def shape(self):
    return self.image_size, self.image_size, self.n_channels


# ===========================================================================
# Shapes3D
# ===========================================================================
class Shapes3D(_ShapeDataset):
  """ The dataset must be manually downloaded from GCS at
    https://console.cloud.google.com/storage/browser/3d-shapes

  Recommend to run `ulimit -n 100000` before download this dataset

  All the factors are:
      ['orientation', 'scale', 'shape', 'floor_hue', 'wall_hue', 'object_hue']

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
  """Shapes3D dataset with downsampled image (32, 32, 3) """

  def __init__(self, continuous: bool = True, seed: int = 1):
    super().__init__(image_size=32, continuous=continuous, seed=seed)


# ===========================================================================
# dSprites
# ===========================================================================
class dSprites(_ShapeDataset):
  """dSprites dataset with continuous non-negative attributes values
  by defaults.

  All factors are:
      ['orientation', 'scale', 'shape', 'x_position', 'y_position']

  Recommend to run `ulimit -n 100000` before download this dataset

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
