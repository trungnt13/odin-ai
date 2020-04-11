import glob
import os
import re
import shutil
import warnings
import zipfile
from collections import Counter
from functools import partial
from urllib.request import urlretrieve

import numpy as np
from urllib3 import PoolManager
from urllib3.exceptions import InsecureRequestWarning

from odin.fuel._image_base import ImageDataset
from odin.utils.crypto import md5_folder
from odin.utils.mpi import MPI


# ===========================================================================
# Helpers
# ===========================================================================
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


def _extract_factors(description: str, freq_threshold: int = 2):
  from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
  from sklearn.decomposition import PCA
  from sklearn.manifold import TSNE
  from odin.ml import Transformer
  from odin import visual as vs
  from matplotlib import pyplot as plt
  import seaborn as sns
  sns.set()
  bert = Transformer()
  description = [
      i for i in re.sub(r"[^a-zA-Z\d\s:]", "", description).strip().split(" ")
      if len(i) > 1
  ]
  count = Counter(description)
  count = {
      i: j
      for i, j in count.items()
      if j > freq_threshold and i not in ENGLISH_STOP_WORDS
  }
  text = list(count.keys())
  embedding = np.mean(bert.transform(text, max_len=1), axis=1)
  embedding = TSNE().fit_transform(
      PCA(n_components=None).fit_transform(embedding))
  fig = plt.figure(figsize=(12, 12))
  for (x, y), s in zip(embedding, text):
    plt.scatter(x, y, s=1, alpha=0.001)
    plt.text(x,
             y,
             s,
             verticalalignment='center',
             horizontalalignment='center',
             fontsize=4,
             color='red',
             alpha=0.6)
  fig.savefig('/tmp/tmp.png', dpi=180)


# ===========================================================================
# Main classes
# ===========================================================================
class LegoFaces(ImageDataset):
  r""" All credits to
  Links:
    https://www.echevarria.io/blog/lego-face-vae/index.html
    https://github.com/iechevarria/lego-face-VAE
  """

  METADATA = r"https://raw.githubusercontent.com/iechevarria/lego-face-VAE/master/dataset_scripts/minifig-heads.csv"
  DATASET = r"https://github.com/iechevarria/lego-face-VAE/raw/master/dataset.zip"
  MD5 = r"2ea2f858cbbed72e1a7348676921a3ac"

  def __init__(self, path="~/tensorflow_datasets/lego_faces"):
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
    ### load all images
    images = glob.glob(image_folder + '/*.jpg', recursive=True)
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
    factors = _extract_factors(' '.join(images_desc.values()))
