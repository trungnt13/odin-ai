"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
LICENSE: https://github.com/fchollet/keras/blob/master/LICENSE
Original code: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
"""
from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.ndimage as ndi

from PIL import Image

from odin.utils import as_tuple


def apply_transform(x,
                    transform_matrix,
                    fill_mode='nearest'):
  """Apply the image transformation specified by a matrix.

  # Arguments
      x: 2D numpy array, single image.
      transform_matrix: Numpy array specifying the geometric transformation.
      channel_axis: Index of axis for channels in the input tensor.
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
      cval: Value used for points outside the boundaries
          of the input if `mode='constant'`.

  # Returns
      The transformed version of the input.
  """
  x = np.rollaxis(x, 2, 0)
  final_affine_matrix = transform_matrix[:2, :2]
  final_offset = transform_matrix[:2, 2]
  channel_images = [ndi.interpolation.affine_transform(
      x_channel,
      final_affine_matrix,
      final_offset,
      order=0,
      mode=fill_mode,
      cval=0.) for x_channel in x]
  x = np.stack(channel_images, axis=0)
  x = np.rollaxis(x, 0, 2 + 1)
  return x


def transform_matrix_offset_center(matrix, x, y):
  o_x = float(x) / 2 + 0.5
  o_y = float(y) / 2 + 0.5
  offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
  reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
  transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
  return transform_matrix


def rotate(x, rg, fill_mode='nearest'):
  """Performs a random rotation of a Numpy image tensor.

  # Arguments
      x: Input tensor. Must be 3D.
      rg: Rotation range, in degrees.
          can be negative or positive
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).

  # Returns
      Rotated Numpy image tensor.
  """
  theta = np.pi / 180 * rg
  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])

  h, w = x.shape[0], x.shape[1]
  transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
  x = apply_transform(x, transform_matrix, fill_mode)
  return x


def shift(x, wrg, hrg, fill_mode='nearest'):
  """Performs a random spatial shift of a Numpy image tensor.

  # Arguments
      x: Input tensor. Must be 3D.
      wrg: Width shift range, as a float fraction of the width.
          can be negative or positive
      hrg: Height shift range, as a float fraction of the height.
          can be negative or positive
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).

  # Returns
      Shifted Numpy image tensor.
  """
  h, w = x.shape[0], x.shape[1]
  tx = hrg * h
  ty = wrg * w
  translation_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])

  transform_matrix = translation_matrix  # no need to do offset
  x = apply_transform(x, transform_matrix, fill_mode)
  return x


def zoom(x, zoom_width, zoom_height, fill_mode='nearest'):
  """Performs a random spatial zoom of a Numpy image tensor.

  # Arguments
      x: Input tensor. Must be 3D.
      zoom_range: Tuple of floats; zoom range for width and height.
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).

  # Returns
      Zoomed Numpy image tensor.

  # Raises
      ValueError: if `zoom_range` isn't a tuple.
  """
  if zoom_width == 1 and zoom_height == 1:
    zx, zy = 1, 1
  else:
    zx, zy = np.random.uniform(zoom_width, zoom_height, 2)
  zoom_matrix = np.array([[zx, 0, 0],
                          [0, zy, 0],
                          [0, 0, 1]])

  h, w = x.shape[0], x.shape[1]
  transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
  x = apply_transform(x, transform_matrix, fill_mode)
  return x


def shear(x, intensity, fill_mode='nearest'):
  """Performs a random spatial shear of a Numpy image tensor.

  # Arguments
      x: Input tensor. Must be 3D.
      intensity: Transformation intensity.
          can be negative or positive
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
      cval: Value used for points outside the boundaries
          of the input if `mode='constant'`.

  # Returns
      Sheared Numpy image tensor.
  """
  shear = intensity
  shear_matrix = np.array([[1, -np.sin(shear), 0],
                           [0, np.cos(shear), 0],
                           [0, 0, 1]])

  h, w = x.shape[0], x.shape[1]
  transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
  x = apply_transform(x, transform_matrix, fill_mode)
  return x


def one_hot(y, nb_classes):
  a = np.zeros((len(y), nb_classes), 'uint8')
  a[np.arange(len(y)), y] = 1
  return a


def read(path, grayscale=False, crop=None, scale=None, target_size=None,
         transpose=None, resample_mode=2):
  """
  Parameters
  ----------
  grayscale: bool
      force to convert Image to grayscale or not
  crop: 4-tuple of int
       (left, upper, right, lower)
  scale: 2-tuple of float
      (factor of width, factor of height) of output images
  target_size: 2-tuple of int
      desire size for image (image will be padded if the size
      mis-match)
  transpose: int, or list of int
      if a list of int is provided, will return a list of images
      <0: Do nothing
      0: PIL.Image.FLIP_LEFT_RIGHT
      1: PIL.Image.FLIP_TOP_BOTTOM
      2: PIL.Image.ROTATE_90
      3: PIL.Image.ROTATE_180
      4: PIL.Image.ROTATE_270
      5: PIL.Image.TRANSPOSE
  resample_mode: int
      0 = PIL.Image.NEAREST: use nearest neighbour
      1 = PIL.Image.LANCZOS: a high-quality downsampling filter
      2 = PIL.Image.BILINEAR: linear interpolation
      3 = PIL.Image.BICUBIC: cubic spline interpolation
  Return
  ------
  image: (height, width, channel) if RGB image
         (height, width) if Grayscale image

  Example
  -------
  >>> x = image.read('/Users/trungnt13/tmp/1090.jpg', grayscale=True,
  ...                scale=0.4, target_size=(800, 900),
  ...                transpose=(-1, 0, 1, 2, 3, 4))
  ... # return a list of 6 images with size=(800, 900)
  >>> print([i.shape for i in x])
  ... # [(900, 800), (900, 800), (900, 800), (900, 800), (900, 800), (900, 800)]
  >>> plt.subplot(1, 6, 1)
  >>> plt.imshow(x[0], cmap='gray')
  >>> plt.subplot(1, 6, 2)
  >>> plt.imshow(x[1], cmap='gray')
  >>> plt.subplot(1, 6, 3)
  >>> plt.imshow(x[2], cmap='gray')
  >>> plt.subplot(1, 6, 4)
  >>> plt.imshow(x[3], cmap='gray')
  >>> plt.subplot(1, 6, 5)
  >>> plt.imshow(x[4], cmap='gray')
  >>> plt.subplot(1, 6, 6)
  >>> plt.imshow(x[5], cmap='gray')
  >>> plt.show(block=True)

  Benchmarks
  ----------
  RESAMPLE_MODE = ANTIALIAS:
   * scale and target_size: 0.019858 (sec)
   * Only scale: 0.009354 (sec)
   * NO scale or target_size: 0.002620 (sec)
  RESAMPLE_MODE = BILINEAR:
   * scale and target_size: 0.012858 (sec)
   * Only scale: 0.006643 (sec)
   * NO scale or target_size: 0.002419 (sec)
  """
  img = Image.open(path, mode="r")
  if grayscale:
    img = img.convert('L')
  else:  # Ensure 3 channel even when loaded image is grayscale
    img = img.convert('RGB')
  # ====== crop ====== #
  if crop:
    img = img.crop(crop)
  # ====== scale ====== #
  if scale and scale != 1.:
    size = img.size
    scale = as_tuple(scale, len(size), t=float)
    scale = tuple([int(i * j) for i, j in zip(scale, size)])
    img = img.resize(scale, resample=resample_mode)
  # ====== return X ====== #
  X = []
  transpose = (-1,) if transpose is None else as_tuple(transpose, t=int)
  # ====== traverse each transposed images ====== #
  orig_img = img
  for t in transpose:
    img = orig_img.transpose(t) if t >= 0 else orig_img
    # ====== target_size ====== #
    if target_size:
      size = img.size
      target_size = as_tuple(target_size, 2, int)
      # image is bigger than the target size
      scale = [j / i for i, j in zip(size, target_size)]
      # print(size, target_size, scale)
      # resize smaller or bigger and preserve the ratio
      if any(i < 1 for i in scale) or all(i > 1 for i in scale):
        scale = min(scale)
        scale = [int(round(scale * i)) for i in size]
        img = img.resize(scale, resample=resample_mode)
      # centerlize the image
      size = img.size
      if size != target_size:
        new_img = Image.new('L' if grayscale else 'RGB',
                            target_size,   # A4 at 72dpi
                            0 if grayscale else (0, 0, 0))  # White
        x = 0 if size[0] == target_size[0] else (target_size[0] - size[0]) // 2
        y = 0 if size[1] == target_size[1] else (target_size[1] - size[1]) // 2
        new_img.paste(img, (x, y))
        img.close()
        img = new_img
    X.append(np.asarray(img))
    img.close()
  # ====== end ====== #
  orig_img.close()
  # X = [x.transpose(-1, 0, 1) if x.ndim == 3 else x for x in X]
  return X[0] if len(X) == 1 else X
