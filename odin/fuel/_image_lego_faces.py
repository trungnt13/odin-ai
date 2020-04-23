import glob
import inspect
import os
import re
import shutil
import warnings
import zipfile
from collections import Counter
from functools import partial
from itertools import chain
from numbers import Number
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from six import string_types
from urllib3 import PoolManager
from urllib3.exceptions import InsecureRequestWarning

from odin.fuel._image_base import ImageDataset, _partition
from odin.utils.crypto import md5_folder
from odin.utils.mpi import MPI


# ===========================================================================
# Helpers
# ===========================================================================
def _resize(img, image_size, outpath=None):
  name = ""
  if isinstance(img, string_types):
    from PIL import Image
    name = os.path.basename(img)
    img = Image.open(img, mode="r")
  elif isinstance(img, np.ndarray):
    from PIL import Image
    img = Image.fromarray(img)
  img = img.resize((int(image_size), int(image_size)))
  # save the output image
  if outpath is not None:
    outpath = os.path.join(outpath, name)
    img.save(outpath, "JPEG", quality=90)
    del img
    img = outpath
  else:
    arr = np.array(img)
    del img
    img = arr
  return img


def scrap_lego_faces(metadata, path, resize=64, n_processes=4):
  r""" This function does not filter out bad images """
  from tqdm import tqdm
  from PIL import Image

  def _download_image(meta, conn):
    part_id, desc = meta
    desc = desc.replace("Minifigure, ", "")
    return_path = []
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=InsecureRequestWarning)
      response = conn.request(
          "GET",
          f"https://www.bricklink.com/v2/catalog/catalogitem.page?P={part_id}",
          preload_content=False)
      img_url = re.search(
          rf"\bimg\.bricklink\.com\/ItemImage\/[A-Z]+\/[0-9]+\/{part_id}\.png\b",
          str(response.read(), 'utf-8'),
      )
      if img_url is not None:
        img_url = img_url.group(0)
        img_response = conn.request("GET",
                                    f"https://{img_url}",
                                    preload_content=False)
        image_path = f"{path}/{part_id}"
        # convert to jpg with white background
        image = Image.open(img_response).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        del background
        width, height = image.size
        ratio = width / height
        # split the image
        if ratio >= 1.6 or part_id:
          im = np.array(image)
          M = im.shape[0]
          N = im.shape[1] // 2
          halves = [
              im[x:x + M, y:y + N]
              for x in range(0, im.shape[0], M)
              for y in range(0, im.shape[1], N)
          ]
          image = [Image.fromarray(half, "RGB") for half in halves[:2]]
        else:
          image = [image]
        # crop to square image
        for idx, im in enumerate(image):
          width, height = im.size
          new_len = min(width, height)
          left = (width - new_len) / 2
          top = (height - new_len) / 2
          right = (width + new_len) / 2
          bottom = (height + new_len) / 2
          im = im.crop((left, top, right, bottom))
          # resize the image
          if resize is not None:
            im = im.resize((int(resize), int(resize)))
          # save image
          out = image_path + ('.jpg' if idx == 0 else ('_%d.jpg' % idx))
          im.save(out, "JPEG", quality=90)
          return_path.append(out)
          del im
    return return_path

  conn = PoolManager(
      num_pools=2,
      headers={
          "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0"
      },
      maxsize=100,
      cert_reqs='CERT_NONE')
  all_images = []
  for image_path in tqdm(MPI(
      jobs=list(zip(metadata["Number"].values, metadata["Name"].values)),
      func=partial(_download_image, conn=conn),
      ncpu=max(1, int(n_processes)),
      batch=1,
  ),
                         desc="Download lego faces",
                         unit="image",
                         total=metadata.shape[0]):
    all_images += image_path
  return np.array(all_images)


def _process_desc(s):
  s = s.replace("Minifigure, Head", "")
  s = s.lower()
  s = [i.strip() for i in s.split('-')]
  if len(s) > 2:
    s = [' '.join(s[:-1]), s[-1]]
  s = s[0]
  return s


# ===========================================================================
# Heuristic classification of factors
# ===========================================================================
ATTRIBUTES = [
    ('female', 'beauty'),
    ('male', 'beard', 'moustache', 'goatee', 'stubble', 'sideburns', 'angular'),
    # beard
    'beard',
    'moustache',
    'goatee',
    'stubble',
    # themes
    'exo',  # exo-force lego
    'hp',  # harry porter
    'potc',  # pirate of the Caribiean
    'lotr',  # lord of the ring
    'sw',  # star war
    'chima',
    'evil',
    'skull',
    'batman',
    'nba',
    'jack',
    'joker',
    ('skywalker', 'anakin'),
    'tiger',
    'robin',
    'darth',
    'pilot',
    'captain',
    'lion',
    'alien',
    # different smiling
    ('dimple', 'dimples'),
    ('smile', 'smiling'),
    'grin',
    # eyebrow
    'unibrow',
    # glasses
    'glasses',
    'sunglasses',
    'goggles',
    'eyepatch',
    # faceware
    'balaclava',
    ('tattoo', 'tattoos'),
    'fur',
    'mask',
    'metal',
    'paint',
    ('scar', 'scars'),
    'freckles',
    'wrinkles',
    'sweat',
    # headwear
    'headband',
    'headset',
    # teeth
    'fangs',
    ('wink', 'winking'),
    # hairs
    'curly',
    'bushy',
    'messy',
    # emotions
    'angry',
    'determined',
    ('scowl', 'scowling'),
    'happy',
    'neutral',
    # others
    'emmet',
    # colors
    'white',
    'brown',
    'red',
    'black',
    'orange',
    'green',
    'gray',
    'azure',
    # open mouth
    ['openmouth', lambda s: 'open' in s and 'mouth' in s and 'closed' not in s]
]


def _extract_factors(images, descriptions, freq_threshold: int = 10):
  from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
  descriptions = [[
      w
      for w in re.sub(r"[^a-zA-Z\d\s:]", "", desc).strip().split(" ")
      if len(w) > 0 and w not in ENGLISH_STOP_WORDS
  ]
                  for desc in descriptions]
  # filtering by number of occurences
  count = Counter(chain(*descriptions))
  count = {i: j for i, j in count.items() if j > freq_threshold}
  descriptions = [[w for w in desc if w in count] for desc in descriptions]
  # extract the factors
  factors = np.array([[
      a in desc if isinstance(a, string_types) else \
        (a[1](desc) if isinstance(a, list) else any(i in desc for i in a))
      for a in ATTRIBUTES
  ]
                      for desc in descriptions],
                     dtype=np.float32)
  ids = np.arange(len(factors))[np.sum(factors, axis=-1) == 0]
  assert len(ids) == 0, "Some images have zero factors: %s" % \
    '; '.join('%s-%s' % (os.path.basename(images[i]), descriptions[i]) for i in ids)
  # just plot images for each word
  # from matplotlib import pyplot as plt
  # from PIL import Image
  # count = [
  #     word
  #     for word, _ in sorted(count.items(), key=lambda x: x[1], reverse=True)
  # ]
  # for word in count:
  #   ids = [(img, i) for img, i in zip(images, descriptions) if word in i]
  #   np.random.shuffle(ids)
  #   print(word, ":", len(ids))
  #   fig = plt.figure(figsize=(10, 10))
  #   for i in range(9):
  #     if i >= len(ids):
  #       break
  #     plt.subplot(3, 3, i + 1)
  #     img, desc = ids[i]
  #     plt.imshow(np.array(Image.open(img)))
  #     plt.title(' '.join(desc), fontsize=6)
  #     plt.axis('off')
  #   plt.tight_layout()
  #   fig.savefig('/tmp/tmp_%s.png' % word, dpi=80)
  #   plt.close(fig)
  return factors


# ===========================================================================
# Main classes
# ===========================================================================
class LegoFaces(ImageDataset):
  r""" All credits to
  Links:
    https://www.echevarria.io/blog/lego-face-vae/index.html
    https://github.com/iechevarria/lego-face-VAE

  To remove all image with background, set `background_threshold=0`
  """

  METADATA = r"https://raw.githubusercontent.com/iechevarria/lego-face-VAE/master/dataset_scripts/minifig-heads.csv"
  DATASET = r"https://github.com/iechevarria/lego-face-VAE/raw/master/dataset.zip"
  MD5 = r"2ea2f858cbbed72e1a7348676921a3ac"

  def __init__(self,
               path="~/tensorflow_datasets/lego_faces",
               image_size=64,
               background_threshold=255):
    super().__init__()
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
      os.makedirs(path)
    ### download metadata
    meta_path = os.path.join(path, 'meta.csv')
    if not os.path.exists(meta_path):
      print("Download lego faces metadata ...")
      meta_path, _ = urlretrieve(url=LegoFaces.METADATA, filename=meta_path)
    import pandas as pd
    metadata = pd.read_csv(meta_path)
    metadata = metadata[metadata["Category Name"] == "Minifigure, Head"]
    ### check downloaded images
    image_folder = os.path.join(path, "dataset")
    if os.path.exists(image_folder):
      if md5_folder(image_folder) != LegoFaces.MD5:
        shutil.rmtree(image_folder)
    ### download data
    zip_path = os.path.join(path, "dataset.zip")
    if not os.path.exists(zip_path):
      print("Download zip lego faces dataset ...")
      zip_path, _ = urlretrieve(url=LegoFaces.DATASET, filename=zip_path)
    if not os.path.exists(image_folder):
      with zipfile.ZipFile(zip_path, mode="r") as f:
        print("Extract all lego faces images ...")
        f.extractall(path)
    ### load all images, downsample if necessary
    images = glob.glob(image_folder + '/*.jpg', recursive=True)
    if image_size != 128:
      image_folder = image_folder + '_%d' % int(image_size)
      if not os.path.exists(image_folder):
        os.mkdir(image_folder)
      if len(os.listdir(image_folder)) != len(images):
        shutil.rmtree(image_folder)
        os.mkdir(image_folder)
        from tqdm import tqdm
        images = [
            i for i in tqdm(MPI(jobs=images,
                                func=partial(_resize,
                                             image_size=image_size,
                                             outpath=image_folder),
                                ncpu=3,
                                batch=1),
                            total=len(images),
                            desc="Resizing images to %d" % image_size)
        ]
      else:
        images = glob.glob(image_folder + '/*.jpg', recursive=True)
    ### extract the heuristic factors
    metadata = {
        part_id: desc
        for part_id, desc in zip(metadata["Number"], metadata["Name"])
    }
    images_desc = {}
    for path in images:
      name = os.path.basename(path)[:-4]
      if name in metadata:
        desc = metadata[name]
      else:
        name = name.split('_')
        desc = metadata[name[0]]
      images_desc[path] = _process_desc(desc)
    ### tokenizing the description
    from PIL import Image

    def imread(p):
      img = Image.open(p, mode='r')
      arr = np.array(img, dtype=np.uint8)
      del img
      return arr

    self.image_size = image_size
    self.images = np.stack(
        [i for i in MPI(jobs=images, func=imread, ncpu=2, batch=1)])
    self.factors = _extract_factors(list(images_desc.keys()),
                                    list(images_desc.values()))
    ### remove images with background
    ids = np.array([
        True if np.min(i) <= int(background_threshold) else False
        for i in self.images
    ])
    self.images = self.images[ids]
    self.factors = self.factors[ids]
    ### split the dataset
    rand = np.random.RandomState(seed=1)
    n = len(self.images)
    ids = rand.permutation(n)
    self.train = (self.images[:int(0.8 * n)], self.factors[:int(0.8 * n)])
    self.valid = (self.images[int(0.8 * n):int(0.9 * n)],
                  self.factors[int(0.8 * n):int(0.9 * n)])
    self.test = (self.images[int(0.9 * n):], self.factors[int(0.9 * n):])

  @property
  def labels(self):
    return np.array(
        [a if isinstance(a, string_types) else a[0] for a in ATTRIBUTES])

  @property
  def shape(self):
    return (self.image_size, self.image_size, 3)

  @property
  def is_binary(self):
    return False

  def create_dataset(self,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=None,
                     partition='train',
                     inc_labels=False,
                     seed=1) -> tf.data.Dataset:
    r"""
    Arguments:
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean or Scalar. If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.

    Return :
      tensorflow.data.Dataset :
        image - `(tf.float32, (None, 64, 64, 3))`
        label - `(tf.float32, (None, 66))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    X, y = _partition(partition,
                      train=self.train,
                      valid=self.valid,
                      test=self.test)
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def _process(*data):
      image = tf.cast(data[0], tf.float32)
      image = self.normalize_255(image)
      if inc_labels:
        label = tf.cast(data[1], tf.float32)
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return image, label, mask
        return image, label
      return image

    ds = tf.data.Dataset.from_tensor_slices(X)
    if inc_labels:
      ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(y)))
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
