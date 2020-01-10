from __future__ import absolute_import, division, print_function

import numpy as np


class Animation(object):

  def __init__(self, figsize=None):
    super().__init__()
    from matplotlib import pyplot as plt
    if figsize is not None:
      self.fig = plt.figure(figsize=figsize)
    else:
      self.fig = plt.figure()
    self.artists = []
    self.axes = []

  def plot_images(self, images, grayscale=False):
    from matplotlib import pyplot as plt
    assert len(images.shape) == 4 or len(images.shape) == 3, \
      "Only support 3D or 4D batched-images."
    if len(images.shape) == 3:
      grayscale = True
    elif len(images.shape) == 4 and images.shape[-1] == 1:
      grayscale = True
      images = images[:, :, :, 0]
    n = int(np.ceil(np.sqrt(images.shape[0])))
    if len(self.axes) == 0:
      self.axes = [plt.subplot(n, n, i + 1) for i in range(images.shape[0])]
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
           dpi=None,
           interval=200,
           repeat_delay=1200,
           repeat=True):
    r"""
    save_freq : None or Integer. If given, only save the animation at given
      frequency, determined by number of artists stored.
    writer: 'ffmpeg', 'pillow', 'imagemagick', None
    """
    if save_freq is not None:
      if len(self.artists) % int(save_freq) != 0:
        return self
    import matplotlib.animation as animation
    ani = animation.ArtistAnimation(self.fig,
                                    self.artists,
                                    interval=interval,
                                    repeat_delay=repeat_delay,
                                    repeat=repeat,
                                    blit=True)
    ani.save(path, writer=writer, dpi=dpi)
    return self
