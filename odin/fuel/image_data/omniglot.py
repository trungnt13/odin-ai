import numpy as np
from typing import Optional
import tensorflow as tf
import tensorflow_datasets as tfds
from odin.fuel.image_data.all_mnist import BinarizedMNIST


class Omniglot(BinarizedMNIST):
  """ Omniglot dataset """

  def __init__(self, image_size: Optional[int] = 28):
    train, valid, test = tfds.load(name='omniglot',
                                   split=['train[:80%]', 'train[80%:]', 'test'],
                                   read_config=tfds.ReadConfig(
                                       shuffle_seed=1,
                                       shuffle_reshuffle_each_iteration=True),
                                   as_supervised=True)

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
    self._image_size = int(28)
    self._normalize = True

  @property
  def shape(self):
    return (self._image_size, self._image_size, 3)

  @property
  def labels(self):
    """  50 different alphabets. """
    return np.array([str(i) for i in range(1623)])
