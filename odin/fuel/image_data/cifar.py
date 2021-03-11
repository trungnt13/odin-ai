import numpy as np
from typing_extensions import Literal

import tensorflow_datasets as tfds
from odin.fuel.image_data._base import ImageDataset


def _quantisize(images, levels=256):
  """" Quantization code from
  `https://github.com/larsmaaloee/BIVA/blob/master/data/cifar10.py` """
  images = images / 255.
  return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')


class CIFAR(ImageDataset):
  """ CIFAR10 """

  def __init__(self,
               version: Literal[10, 20, 100],
               quantize_bits: int = 8,
               seed: int = 1):
    assert version in (10, 20, 100), \
      "Only support CIFAR-10, CIFAR-20 and CIFAR-100"
    self.version = version
    if version == 10:
      dsname = 'cifar10'
    else:
      dsname = 'cifar100'
    self.train, self.valid, self.test = tfds.load(
        name=dsname,
        split=['train[:48000]', 'train[48000:]', 'test'],
        # as_supervised=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed,
                                    shuffle_reshuffle_each_iteration=True),
        shuffle_files=True,
        with_info=False,
    )
    if version in (10, 100):
      process = lambda dat: (dat['image'], dat['label'])
    elif version == 20:
      process = lambda dat: (dat['image'], dat['coarse_label'])
    self.train = self.train.map(process)
    self.valid = self.valid.map(process)
    self.test = self.test.map(process)

  @property
  def binarized(self):
    return False

  @property
  def shape(self):
    return (32, 32, 3)

  @property
  def labels(self):
    if self.version == 10:
      y = [
          'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck'
      ]
    elif self.version == 20:
      y = [
          'aquatic_mammals', 'fish', 'flowers', 'food_containers',
          'fruit_and_vegetables', 'household_electrical_devices',
          'household_furniture', 'insects', 'large_carnivores',
          'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
          'large_omnivores_and_herbivores', 'medium_mammals',
          'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
          'trees', 'vehicles_1', 'vehicles_2'
      ]
    else:
      y = [
          'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
          'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
          'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra',
          'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
          'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
          'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
          'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
          'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
          'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
          'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
          'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
          'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
          'willow_tree', 'wolf', 'woman', 'worm'
      ]
    return np.array(y)


# ===========================================================================
# Shortcuts
# ===========================================================================
class CIFAR10(CIFAR):

  def __init__(self):
    super().__init__(10)


class CIFAR100(CIFAR):

  def __init__(self):
    super().__init__(100)

class CIFAR20(CIFAR):

  def __init__(self):
    super().__init__(20)
