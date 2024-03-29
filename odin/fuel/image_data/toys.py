import os
from multiprocessing import cpu_count
from typing import List

import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

from ._base import ImageDataset


def make_halfmoons(n_samples_per_factors=200,
                   image_size=64,
                   marker_size=12.,
                   seed=1,
                   n_cpu=4):
  from matplotlib import pyplot as plt
  from odin.utils import MPI
  from tqdm import tqdm

  rand = np.random.RandomState(seed=seed)
  shapes = ['o', 's', '^', 'p']
  shapes_to_idx = {v: k for k, v in enumerate(shapes)}
  colors = np.linspace(0.0, 1.0, num=10)
  n_factors = len(shapes) * len(colors)
  n_samples = n_samples_per_factors * n_factors

  shapes = np.tile(shapes, [int(n_samples / len(shapes))])
  colors = np.tile(colors, [int(n_samples / len(colors))])
  rand.shuffle(shapes)
  rand.shuffle(colors)
  # === 1. Generate data
  x, y = datasets.make_moons(n_samples=n_samples,
                             shuffle=True,
                             noise=.05,
                             random_state=rand.randint(1e8))
  x_min = np.min(x, 0, keepdims=True)
  x_max = np.max(x, 0, keepdims=True)
  x = (x - x_min) / (x_max - x_min) * 2. - 1.

  # === 2. Helper
  dpi = 200
  cmap = plt.get_cmap('coolwarm')

  def create_image(ids: List[int]):
    all_x = []
    all_y = []
    for i in ids:
      fig = plt.figure(figsize=(image_size / dpi, image_size / dpi),
                       dpi=dpi,
                       facecolor="black",
                       frameon=True)
      ax = plt.gca()
      ax.set_facecolor('black')
      ax.scatter(x[i, 0], x[i, 1],
                 s=marker_size,
                 marker=shapes[i],
                 color=cmap(colors[i]),
                 antialiased=True,
                 edgecolors='none')
      ax.set_xlim([-1.2, 1.2])
      ax.set_ylim([-1.2, 1.2])
      ax.axis('off')
      ax.margins(0)
      fig.tight_layout(pad=0)
      # convert to array
      fig.canvas.draw()
      img = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
      img = np.reshape(img, (image_size, image_size, 3))
      # img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
      plt.close(fig)
      # save data
      all_x.append(np.expand_dims(img, 0))
      all_y.append(
        [x[i, 0], x[i, 1], y[i], colors[i] * 2. - 1., shapes_to_idx[shapes[i]]])
    return np.concatenate(all_x, 0), np.vstack(all_y)

  # === 2. Generate images
  jobs = list(range(n_samples))
  progress = tqdm(total=n_samples, unit='images')
  X = []
  Y = []
  for img, lab in MPI(jobs, create_image, ncpu=n_cpu, batch=100):
    progress.update(img.shape[0])
    X.append(img)
    Y.append(lab)
  progress.clear()
  progress.close()
  return np.concatenate(X, 0), np.concatenate(Y, 0)


class HalfMoons(ImageDataset):
  """Half Moons data but instead of position as features, we save the images
  of each data points include some factor of variations:

  The factors are:
    - x position [-1, 1]
    - y position [-1, 1]
    - labels [0, 1]
    - colors [-1, 1] cmap 'coolwarm' (10 linearly spaced values)
    - shapes [0 'circle', 1 'square', 2 'triangle', 3 'pentagon']

  There are 4000 images,
  i.e. 100 images for each combination of color and shape

  The tensorflow datasets don't include the labels as factors

  Raw numpy arrays (included the labels) could be accessed via attributes:
    - x_train, y_train
    - x_valid, y_valid
    - x_test, y_test

  """

  def __init__(self,
               path: str = '~/.halfmoons',
               n_cpu: int = -1,
               seed: int = 1):
    super(HalfMoons, self).__init__()
    self.image_size = 64
    if n_cpu <= 0:
      n_cpu = cpu_count() - 1
    path = os.path.abspath(os.path.expanduser(path))
    if '.npz' not in path.lower():
      path = f'{path}.npz'
    if not os.path.exists(path):
      X, Y = make_halfmoons(image_size=self.image_size, n_cpu=n_cpu, seed=seed)
      np.savez_compressed(path, X=X, Y=Y)
    else:
      data = np.load(path)
      X = data['X']
      Y = data['Y']
    Y = Y.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,
                                                        random_state=seed,
                                                        stratify=Y[:, 2])
    x_train, x_valid, y_train, y_valid = train_test_split(
      x_train, y_train, train_size=0.9,
      random_state=seed,
      stratify=y_train[:, 2])
    # numpy datasets
    self.x_train = x_train
    self.x_valid = x_valid
    self.x_test = x_test
    self.y_train = y_train
    self.y_valid = y_valid
    self.y_test = y_test
    # remove the labels
    y_train = y_train[:, [0, 1, 3, 4]]
    y_valid = y_valid[:, [0, 1, 3, 4]]
    y_test = y_test[:, [0, 1, 3, 4]]
    # tensorflow datasets
    self.train = tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(x_train),
      tf.data.Dataset.from_tensor_slices(y_train)))
    self.valid = tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(x_valid),
      tf.data.Dataset.from_tensor_slices(y_valid)))
    self.test = tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(x_test),
      tf.data.Dataset.from_tensor_slices(y_test)))

  @property
  def binarized(self) -> bool:
    return False

  @property
  def shape(self) -> List[int]:
    return [self.image_size, self.image_size, 3]

  @property
  def labels(self) -> List[str]:
    return ['x', 'y', 'color', 'shape']
