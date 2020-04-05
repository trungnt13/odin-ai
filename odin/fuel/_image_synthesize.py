import os

import numpy as np
import tensorflow as tf

from odin.fuel._image_base import ImageDataset, _partition
from odin.utils.crypto import md5_checksum


class YDisentanglement(ImageDataset):
  r"""
  Attributes :
    number of letter "Y" : an Integer
    xoffset : a Float
    yoffset : a Float
    rotation : a Float (from 0 - 180)
  """
  MD5 = r"19db3f0cc5829a1308a8023930dd61e6"

  def __init__(self, path="/tmp/ydisentanglement.npz"):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(path):
      if not os.path.isfile(path):
        raise ValueError("path to '%s' is a folder, require path to a file" %
                         path)
      if md5_checksum(path) != YDisentanglement.MD5:
        os.remove(path)
    # create new dataset if not exist
    if not os.path.exists(path):
      images_train, attributes_train = YDisentanglement.generate_data(
          training=True)
      images_test, attributes_test = YDisentanglement.generate_data(
          training=False)
      with open(path, 'wb') as f:
        np.savez(f,
                 images_train=images_train,
                 attributes_train=attributes_train,
                 images_test=images_test,
                 attributes_test=attributes_test)
      print(md5_checksum(path))

    with open(path, 'rb') as f:
      data = np.load(f)
      self.images_train = data['images_train']
      self.attributes_train = data['attributes_train']
      self.images_test = data['images_test']
      self.attributes_test = data['attributes_test']

  @property
  def is_binary(self):
    return True

  @property
  def shape(self):
    return (48, 48, 1)

  def create_dataset(self,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=None,
                     partition='train',
                     inc_labels=False) -> tf.data.Dataset:
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
    images, attributes = _partition(partition,
                                    train=(self.images_train,
                                           self.attributes_train),
                                    test=(self.images_test,
                                          self.attributes_test))
    images = tf.data.Dataset.from_tensor_slices(images)
    attributes = tf.data.Dataset.from_tensor_slices(attributes)
    ds = tf.data.Dataset.zip((images, attributes)) if inc_labels else images

    def _process(*data):
      image = tf.expand_dims(tf.cast(data[0], tf.float32), -1)
      if inc_labels:
        label = tf.cast(data[1], tf.float32)
        return image, label
      return image

    ds = ds.map(_process)
    if cache is not None:
      ds = ds.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None:
      ds = ds.shuffle(int(shuffle))
    ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds

  @staticmethod
  def generate_data(num=16, image_path=None, training=True, seed=1):
    from PIL import Image, ImageDraw, ImageColor, ImageChops
    size = 48
    resample = Image.BICUBIC
    org = Image.new("1", (size, size))
    images = []
    attributes = []
    rand = np.random.RandomState(seed)
    ## different configuraiton for training and testing
    if training:
      all_text = ("Y", "YY", "YYYY")
      rotation_range = (0, 180)
    else:
      rotation_range = (180, 360)
      all_text = ("YYY",)

    def _to_offset(o, mode):
      # only moving horizontal and diagonal
      # (let see if the model could extrapolate to vertial movement)
      if training:
        x, y = (o, 0) if mode == 0 else (o, o)
      else:
        x, y = 0, o
      return x, y

    ## test
    for text in all_text:
      img = org.copy()
      draw = ImageDraw.Draw(img)
      w, h = draw.textsize(text)
      draw.text([(size - w) / 2, (size - h) / 2], text, fill=1)
      del draw
      for offset in np.linspace(-(num - 1) // 2, (num - 1) // 2,
                                num=num,
                                endpoint=True):
        offset = int(offset)
        if offset == 0:
          i1 = img
        else:
          mode = rand.randint(0, 2)
          xoffset, yoffset = _to_offset(offset, mode)
          i1 = ImageChops.offset(img, xoffset=xoffset, yoffset=yoffset)
        # rotation
        for rotation in np.linspace(*rotation_range, num=num, endpoint=False):
          if rotation > 0:
            i2 = i1.rotate(rotation, resample=resample)
          else:
            i2 = i1
          images.append(np.array(i2).astype(np.uint8))
          attributes.append((len(text), xoffset, yoffset, rotation))
    # final data
    images = np.stack(images)
    attributes = np.array(attributes)
    ## save image
    if image_path is not None:
      from tqdm import tqdm
      n = int(np.ceil(np.sqrt(images.shape[0])))
      fig = plt.figure(figsize=(18, 18), dpi=80)
      for i, img in tqdm(list(enumerate(images))):
        ax = plt.subplot(n, n, i + 1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
      fig.tight_layout()
      fig.savefig(image_path, dpi=80)
    return images, attributes
