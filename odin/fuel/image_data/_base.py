import numpy as np
import tensorflow as tf
from odin.fuel.dataset_base import IterableDataset


class ImageDataset(IterableDataset):

  @property
  def data_type(self) -> str:
    return 'image'

  def sample_images(self,
                    save_path=None,
                    dpi=120,
                    n_samples=25,
                    partition='train',
                    seed=1):
    r""" Sample a subset of image from training set """
    n = int(np.sqrt(n_samples))
    assert n * n == n_samples, "Sqrt of n_samples is not an integer"
    train = self.create_dataset(batch_size=n_samples,
                                partition=str(partition),
                                inc_labels=0.5)
    # prepare the data
    images = []
    labels = []
    mask = []
    for data in train.take(10):
      if isinstance(data, dict):
        X, y = data['inputs']
        mask.append(data['mask'])
      elif isinstance(data, (tuple, list)):
        if len(data) >= 2:
          X, y = data[:2]
        else:
          X = data[0]
          y = None
      else:
        X = data
        y = None
      images.append(X)
      if y is not None:
        labels.append(y)
    rand = np.random.RandomState(seed=seed)
    idx = rand.choice(10)
    images = images[idx].numpy()
    labels = labels[idx].numpy() if len(labels) > 0 else None
    mask = mask[idx].numpy().ravel() if len(mask) > 0 else None
    # check labels type
    labels_type = 'multinomial'
    if np.all(np.unique(labels) == [0., 1.]):
      labels_type = 'binary'
    # plot and save the figure
    if save_path is not None:
      plot_images = images
      if plot_images.shape[-1] == 1:
        plot_images = np.squeeze(plot_images, axis=-1)
      from matplotlib import pyplot as plt
      fig = plt.figure(figsize=(16, 16))
      for i in range(n_samples):
        plt.subplot(n, n, i + 1)
        img = plot_images[i]
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.axis('off')
        if labels is not None:
          if labels_type == 'binary':
            y = [
                str(j) for j in self.labels[np.array(labels[i], dtype=np.bool)]
            ]
            lab = ('\n'.join(y) + '\n') if len(y) > 1 else (y[0] + ' ')
          else:
            lab = '\n'.join(
                ["%s=%s" % (l, str(j)) for l, j in zip(self.labels, labels[i])])
            lab += '\n'
          m = True if mask is None else mask[i]
          plt.title("%s[Mask:%s]" % (lab, m), fontsize=6)
      plt.tight_layout()
      fig.savefig(save_path, dpi=int(dpi))
      plt.close(fig)
    return images

  def normalize_255(self, image):
    return tf.clip_by_value(image / 255., 1e-6, 1. - 1e-6)


