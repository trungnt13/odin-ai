from __future__ import absolute_import, division, print_function

import os

import numpy as np


class Animation(object):
  r""" This class tracking the changes in image using Gif animation

  Arguments:
    figsize : tuple of Integer. Given the width and height of the figure.
  """

  def __init__(self, figsize=None):
    super().__init__()
    from matplotlib import pyplot as plt
    if figsize is not None:
      self.fig = plt.figure(figsize=figsize)
    else:
      self.fig = plt.figure()
    self.artists = []
    self.axes = []

  def __len__(self):
    return len(self.artists)

  def plot_spectrogram(self, spec, cmap='magma'):
    r"""
    Arguments:
      spec: 3D Tensor, a minibatch of spectrogram in (T, D) format
    """
    assert len(spec.shape) == 3, "spec must be 3-D tensor."
    n = int(np.ceil(np.sqrt(spec.shape[0])))
    if len(self.axes) == 0:
      self.axes = [
          self.fig.add_subplot(n, n, i + 1) for i in range(spec.shape[0])
      ]
    imgs = []
    for i, ax in enumerate(self.axes):
      x = spec[i, :, :]
      if hasattr(x, 'numpy'):
        x = x.numpy()
      # transpose to time(x)-frequency(y)
      im = ax.pcolorfast(x.T, cmap=cmap)
      ax.axis('off')
      imgs.append(im)
    self.artists.append(imgs)
    return self

  def plot_images(self, images, grayscale=False):
    r"""
    Arguments:
      images: 3D or 4D Tensor
      grayscale: A Boolean. The images are grayscale images, if 3D tensor is
        provided, grayscale automatically switch to True
    """
    assert len(images.shape) == 4 or len(images.shape) == 3, \
      "Only support 3D or 4D batched-images."
    if len(images.shape) == 3:
      grayscale = True
    elif len(images.shape) == 4 and images.shape[-1] == 1:
      grayscale = True
      images = images[:, :, :, 0]
    n = int(np.ceil(np.sqrt(images.shape[0])))
    if len(self.axes) == 0:
      self.axes = [
          self.fig.add_subplot(n, n, i + 1) for i in range(images.shape[0])
      ]
    imgs = []
    for i, ax in enumerate(self.axes):
      im = ax.imshow(images[i, :, :], cmap='gray') if grayscale else \
        ax.imshow(images[i, :, :])  # channel last
      ax.axis('off')
      imgs.append(im)
    self.artists.append(imgs)
    return self

  def save(self,
           path='/tmp/tmp.gif',
           save_freq=None,
           writer='imagemagick',
           clear_folder=False,
           dpi=None,
           interval=200,
           repeat_delay=1200,
           repeat=False):
    r"""
    path : path to 'gif' or 'png' file, if a folder is given, write the
      animation to multiple 'png' files.
    save_freq : None or Integer. If given, only save the animation at given
      frequency, determined by number of artists stored.
    writer: 'ffmpeg', 'pillow', 'imagemagick', None
    """
    if len(self.artists) <= 1:
      return self
    if save_freq is not None:
      if len(self.artists) % int(save_freq) != 0:
        return self
    # ====== save to Animation ====== #
    import matplotlib.animation as animation
    if os.path.isdir(path):
      if clear_folder:
        for f in os.listdir(path):
          f = os.path.join(path, f)
          if os.path.isfile(f):
            os.remove(f)
      path = os.path.join(path, 'image.png')
    ani = animation.ArtistAnimation(self.fig,
                                    self.artists,
                                    interval=interval,
                                    repeat_delay=repeat_delay,
                                    repeat=repeat,
                                    blit=True)
    ani.save(path, writer=writer, dpi=dpi)
    return self
