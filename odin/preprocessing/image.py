from __future__ import print_function, division, absolute_import

import numpy as np
from PIL import Image

from odin.utils import as_tuple


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
