import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from odin import visual as vs
from odin.fuel import (CelebABig, CelebASmall, Shapes3DSmall, dSpritesSmall,
                       dSprites, ImageDataset, get_all_dataset, SVHN, CIFAR10)
from odin.utils import as_tuple
from tqdm import tqdm
from collections import Counter

sns.set()

ds = CIFAR10()
p = []
it = ds.create_dataset('train',
                       label_percent=100,
                       oversample_ratio=0.1,
                       normalize='raster',
                       drop_remainder=False).take(1404)
all_labels = []
all_unlabels = []
total = 0
for data in tqdm(it.as_numpy_iterator()):
  x, y = data['inputs']
  m = data['mask'].ravel()
  p.append(np.sum(m) / m.shape[0])
  y_labeled = y[m]
  y_unlabeled = y[np.logical_not(m)]
  all_labels += np.argmax(y_labeled, axis=-1).tolist()
  all_unlabels += np.argmax(y_unlabeled, axis=-1).tolist()
  total += x.shape[0]
print('Total:', total)
print('Labeled:', Counter(all_labels))
print('Unlabeled:', Counter(all_unlabels))
print('p=', np.mean(p))
## plot images
plt.figure(figsize=(15, 15), dpi=150)
for i, (img, lab, mask) in enumerate(zip(x[:25], y, m)):
  plt.subplot(5, 5, i + 1)
  plt.imshow(img.astype(np.uint8))
  plt.title(ds.labels[np.argmax(lab)] if mask else str(lab), fontsize=6)
  plt.axis('off')
plt.tight_layout()
vs.plot_save()

for ds in get_all_dataset('image'):
  print(ds)
  if ds in (Shapes3DSmall, dSpritesSmall, CelebABig, CelebASmall):
    continue
  ds = ds()
  ds: ImageDataset
  # first test
  for partition in ('train', 'valid', 'test'):
    print(' ', partition)
    for normalize in ('probs', 'tanh', 'raster'):
      print('  ', normalize)
      x = ds.create_dataset(partition,
                            label_percent=True,
                            normalize=normalize,
                            drop_remainder=True)
      for data in x.shuffle(1000).take(10):
        data = as_tuple(data)
        if len(data) == 2:
          img, lab = data
        else:
          img = data[0]
          lab = None
        img = img.numpy()
        if normalize == 'probs':
          assert np.all(img >= 0.0) and np.all(img <= 1.0)
        elif normalize == 'tanh':
          assert np.all(img >= -1.0) and np.all(img <= 1.0) and np.any(
              img < 0.0)
        elif normalize == 'raster':
          assert np.all(img >= 0.0) and np.all(img <= 255.0) and np.any(
              img > 1.0)
      ## save images
      image = img[:9]
      labels = [None] * 9 if lab is None else lab[:9].numpy()
      if image.shape[-1] == 1:
        image = np.squeeze(image, -1)
      if normalize == 'raster':
        image = image.astype(np.uint8)
      elif normalize == 'tanh':
        image = (image + 1.) / 2.
      plt.figure(figsize=(5, 5), dpi=150)
      for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image[i], cmap='Greys' if image[i].ndim == 2 else None)
        plt.title(str(labels[i]), fontsize=6)
        plt.axis('off')
      plt.tight_layout()
      plt.suptitle(f'{partition}_{normalize}')
  vs.plot_save(f'/tmp/data_{ds.name.lower()}.pdf', verbose=True)
