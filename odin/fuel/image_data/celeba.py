import glob
import os
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from odin.fuel.image_data._base import ImageDataset, get_partition
from tqdm import tqdm


class CelebA(ImageDataset):
  r"""The dataset must be manually downloaded from Google Drive:
    https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFuzTm8

  The following files are required:
    - img_align_celeba.zip
    - list_attr_celeba.txt
    - list_eval_partition.txt

  Parameters
  -----------
  path : path to the folder contained the three files.
  image_size : (optional) an Integer.
    The smallest dimension of a downsampled image. if `None`, original size
    (218, 178) is kept.
  square_image : a Boolean. If True, crop the downsampled image to a square.

  Attributes
  -----------
  train_files, valid_files, test_files : `numpy.ndarray`, list of path
    to images shape `[218, 178, 3]`
  train_attr, valid_attr, test_attr : `numpy.ndarray`, 40 attributes
    for each images

  Reference
  ----------
  Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, (2015).
      Deep Learning Face Attributes in the Wild. Proceedings of International
      Conference on Computer Vision (ICCV)
  """

  def __init__(self,
               path: str = "~/tensorflow_datasets/celeb_a",
               image_size: Optional[int] = 64,
               square_image: bool = True):
    self.image_size = image_size
    self.square_image = bool(square_image)
    path = os.path.abspath(os.path.expanduser(path))
    self.path = path
    assert os.path.isdir(path), f"'{path}' must be a directory"
    zip_path = os.path.join(path, 'img_align_celeba.zip')
    attr_path = os.path.join(path, 'list_attr_celeba.txt')
    attr_cache = attr_path + '.npz'
    part_path = os.path.join(path, 'list_eval_partition.txt')
    for i in [zip_path, attr_path, part_path]:
      assert os.path.exists(i), f"'{i}' must exists"
    ### read the attr
    if not os.path.exists(attr_cache):
      with open(attr_path, 'r') as f:
        text = [[i for i in line.strip().split(' ') if len(i) > 0] for line in f
               ]
        header = np.array(text[1])
        attributes = np.array(text[2:])[:, 1:]
      with open(attr_cache, 'wb') as f:
        np.savez(f, header=header, attributes=attributes.astype(np.int8))
    else:
      with open(attr_cache, 'rb') as f:
        data = np.load(f)
        header = data['header']
        attributes = data['attributes']
    self._header = header
    attributes = attributes.astype(np.float32)
    ### read the partition
    with open(part_path, 'r') as f:
      partition = np.array([line.strip().split(' ') for line in f])[:, 1:]
    ### extracting the data
    image_path = os.path.join(path, 'img_align_celeba')
    image_files = glob.glob(image_path + "/*.jpg", recursive=True)
    if len(image_files) != 202599:
      print("Extracting 202599 image files ...")
      with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path)
      image_files = glob.glob(image_path + '/*.jpg', recursive=True)
    image_files = np.array(sorted(image_files))
    assert image_files.shape[0] == attributes.shape[0] == partition.shape[0]
    ### splitting the data
    attributes = attributes.astype('float32')
    train_files, train_attr = [], []
    valid_files, valid_attr = [], []
    test_files, test_attr = [], []
    for path, attr, part in zip(image_files, attributes, partition):
      part = int(part[0])
      if part == 0:  # train
        train_files.append(path)
        train_attr.append(attr)
      elif part == 1:  # valid
        valid_files.append(path)
        valid_attr.append(attr)
      else:  # test
        test_files.append(path)
        test_attr.append(attr)
    ### store the attributes
    self.train_files = np.array(train_files)
    self.valid_files = np.array(valid_files)
    self.test_files = np.array(test_files)
    self.train_attr = np.array(train_attr)
    self.valid_attr = np.array(valid_attr)
    self.test_attr = np.array(test_attr)

  @property
  def original_shape(self):
    return (218, 178, 3)

  @property
  def shape(self):
    if self.image_size is None:
      return self.original_shape
    h, w = self.original_shape[:2]
    image_size = int(self.image_size)
    return (image_size if self.square_image else int(float(image_size) / w * h),
            image_size, 3)

  @property
  def is_binary(self):
    return False

  @property
  def labels(self):
    return self._header

  def create_dataset(self,
                     batch_size=32,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     partition='train',
                     inc_labels=True,
                     seed=1) -> tf.data.Dataset:
    r""" The default argument will downsize and crop the image to square size
    (64, 64)

    Arguments:
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean or Scalar. If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.

    Return :
      tensorflow.data.Dataset :
        image - `(tf.float32, (None, 64, 64, 3))`
        label - `(tf.float32, (None, 40))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    image_shape = self.original_shape
    image_size = self.image_size
    if image_size is not None:
      image_size = int(image_size)
      height = int(float(image_size) / image_shape[1] * image_shape[0])
      # offset_height, offset_width, target_height, target_width
      crop_offset = ((height - image_size) // 2, 0, image_size, image_size)
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def read(path):
      img = tf.io.decode_jpeg(tf.io.read_file(path))
      img.set_shape(image_shape)
      img = tf.cast(img, tf.float32)
      img = self.normalize_255(img)
      if image_size is not None:
        img = tf.image.resize(img, (height, image_size),
                              method=tf.image.ResizeMethod.BILINEAR,
                              preserve_aspect_ratio=True,
                              antialias=True)
        if self.square_image:
          img = tf.image.crop_to_bounding_box(img, *crop_offset)
      return img

    def mask(image, label):
      mask = gen.uniform(shape=(1,)) < inc_labels
      return dict(inputs=(image, label), mask=mask)

    ### select partition
    images, attrs = get_partition(
        partition,
        train=(self.train_files, self.train_attr),
        valid=(self.valid_files, self.valid_attr),
        test=(self.test_files, self.test_attr),
    )
    # convert [-1, 1] to [0., 1.]
    attrs = (attrs + 1.) / 2
    images = tf.data.Dataset.from_tensor_slices(images).map(read, parallel)
    if inc_labels:
      attrs = tf.data.Dataset.from_tensor_slices(attrs)
      images = tf.data.Dataset.zip((images, attrs))
      if 0. < inc_labels < 1.:  # semi-supervised mask
        images = images.map(mask)

    if cache is not None:
      images = images.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      images = images.shuffle(int(shuffle),
                              seed=seed,
                              reshuffle_each_iteration=True)
    images = images.batch(batch_size, drop_remainder)
    if prefetch is not None:
      images = images.prefetch(prefetch)
    return images


class CelebASmall(CelebA):
  """Downsampled CelebA dataset with image size (28, 28)"""

  def __init__(self, path="~/tensorflow_datasets/celeb_a", square_image=True):
    super().__init__(path=path, image_size=28, square_image=square_image)


class CelebABig(CelebA):
  """Original CelebA dataset with image size (178, 178)"""

  def __init__(self, path="~/tensorflow_datasets/celeb_a", square_image=True):
    super().__init__(path=path, image_size=None, square_image=square_image)
