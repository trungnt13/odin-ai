import glob
import os
import zipfile
from typing import Optional, Union, List
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from odin.fuel.image_data._base import ImageDataset
from odin.fuel.dataset_base import get_partition
from tqdm import tqdm


class CelebA(ImageDataset):
  """The dataset must be manually downloaded from Google Drive:
    https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFuzTm8

  The following files are required:
    - img_align_celeba.zip
    - list_attr_celeba.txt
    - list_eval_partition.txt

  The attributes take value of -1 or 1, it is normalized to 0 and 1 as binary
  values

  There are 162770 images for training, 19867 validation images, and 19962 test images.

  Parameters
  -----------
  path : path to the folder contained the three files.
      - 'img_align_celeba.zip'
      - 'list_attr_celeba.txt'
      - 'list_eval_partition.txt'
  image_size : (optional) an Integer.
      The smallest dimension of a downsampled image. if `None`, original size
      (218, 178) is kept.
  square_image : bool
      If True, crop the downsampled image to a square.
  labels : List[str]
      only provided attributes are presented as labels


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

  def __init__(
      self,
      path: str = "~/tensorflow_datasets/celeb_a",
      image_size: Optional[int] = 64,
      square_image: bool = True,
      labels: List[str] = [
          'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair',
          'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Eyeglasses',
          'Heavy_Makeup', 'Male', 'No_Beard', 'Pale_Skin', 'Receding_Hairline',
          'Smiling', 'Wavy_Hair', 'Wearing_Earrings', 'Young'
      ],
  ):
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
    ### extra filtering of the attributes
    ids = [i for i, name in enumerate(labels) if name in self._header]
    self._header = self._header[ids]
    self.train_attr = np.array(train_attr)[:, ids]
    self.valid_attr = np.array(valid_attr)[:, ids]
    self.test_attr = np.array(test_attr)[:, ids]
    ### convert to tensorflow dataset
    image_shape = self.original_shape
    image_size = self.image_size
    if image_size is not None:
      image_size = int(image_size)
      height = int(float(image_size) / image_shape[1] * image_shape[0])
      # offset_height, offset_width, target_height, target_width
      crop_offset = ((height - image_size) // 2, 0, image_size, image_size)

    @tf.function
    def _read(path):
      img = tf.io.decode_jpeg(tf.io.read_file(path))
      img.set_shape(image_shape)
      ## resize the image
      if image_size is not None:
        img = tf.image.resize(img, (height, image_size),
                              method=tf.image.ResizeMethod.BILINEAR,
                              preserve_aspect_ratio=True,
                              antialias=True)
        if self.square_image:
          img = tf.image.crop_to_bounding_box(img, *crop_offset)
      ## normalize the image
      img = tf.clip_by_value(tf.cast(img, tf.float32), 0.0, 255.)
      return img

    def _to_tfds(images_path, attributes) -> tf.data.Dataset:
      images = tf.data.Dataset.from_tensor_slices(images_path)
      images = images.map(_read, num_parallel_calls=tf.data.AUTOTUNE)
      attributes = (attributes + 1.) / 2
      attributes = tf.data.Dataset.from_tensor_slices(attributes)
      return tf.data.Dataset.zip((images, attributes))

    self.train = _to_tfds(self.train_files, self.train_attr)
    self.valid = _to_tfds(self.valid_files, self.valid_attr)
    self.test = _to_tfds(self.test_files, self.test_attr)

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
  def binarized(self):
    return False

  @property
  def labels(self):
    return self._header


class CelebASmall(CelebA):
  """Downsampled CelebA dataset with image size (28, 28)"""

  def __init__(self, path="~/tensorflow_datasets/celeb_a", square_image=True):
    super().__init__(path=path, image_size=28, square_image=square_image)


class CelebABig(CelebA):
  """Original CelebA dataset with image size (178, 178)"""

  def __init__(self, path="~/tensorflow_datasets/celeb_a", square_image=True):
    super().__init__(path=path, image_size=None, square_image=square_image)
