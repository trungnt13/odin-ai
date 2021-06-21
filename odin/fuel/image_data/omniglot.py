from typing import Optional

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from odin.fuel.image_data._base import ImageDataset


class Omniglot(ImageDataset):
  """ Omniglot dataset """

  def __init__(self, image_size: Optional[int] = 28, seed: int = 1):
    train, valid, test = tfds.load(
        name='omniglot',
        split=['train[:90%]', 'train[90%:]', 'test'],
        read_config=tfds.ReadConfig(shuffle_seed=seed,
                                    shuffle_reshuffle_each_iteration=True),
        as_supervised=True,
    )

    if image_size is None:
      image_size = 105
    image_size = int(image_size)
    if image_size != 105:

      @tf.function
      def resize(x, y):
        x = tf.image.resize(x,
                            size=(image_size, image_size),
                            method=tf.image.ResizeMethod.BILINEAR,
                            preserve_aspect_ratio=True,
                            antialias=True)
        y = tf.cast(y, dtype=tf.float32)
        return x, y

      train = train.map(resize, tf.data.AUTOTUNE)
      valid = valid.map(resize, tf.data.AUTOTUNE)
      test = test.map(resize, tf.data.AUTOTUNE)

    self.train = train
    self.valid = valid
    self.test = test
    self._image_size = image_size

  @property
  def binarized(self):
    return False

  @property
  def shape(self):
    return (self._image_size, self._image_size, 3)

  @property
  def labels(self):
    """  50 different alphabets. """
    return np.array([str(i) for i in range(1623)])
