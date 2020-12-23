import base64
import os
import zipfile
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from scipy import sparse

from odin.fuel.bio_data._base import BioDataset
from odin.utils import one_hot
from odin.utils.crypto import md5_checksum

_URL = [
    r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/counts_mel.RData",
    r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/cellData_mel.RData",
]

__all__ = ['Melanoma', 'Forebrain', 'Insilico', 'BreastTumor', 'Leukemia']


class Melanoma(BioDataset):
  """melanoma ATAC data from (Bravo González-Blas, et al. 2019)

  References
  ----------
  Bravo González-Blas, C. et al. cisTopic: cis-regulatory topic modeling
      on single-cell ATAC-seq data. Nat Methods 16, 397–400 (2019).
  Verfaillie, A. et al. Decoding the regulatory landscape of melanoma
      reveals TEADS as regulators of the invasive cell state.
      Nat Commun 6, (2015).
  """

  def __init__(self, path: str = "~/tensorflow_datasets/melanoma_atac"):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
      os.makedirs(path)
    ### download data
    data = {}
    for url in _URL:
      fname = os.path.basename(url)
      fpath = os.path.join(path, fname)
      if not os.path.exists(fpath):
        print(f"Downloading file: {fname} ...")
        urlretrieve(url, filename=fpath)
      data[fname.split(".")[0]] = fpath
    ### load data
    try:
      import rpy2.robjects as robjects
      from rpy2.robjects import pandas2ri
      from rpy2.robjects.conversion import localconverter
      robjects.r['options'](warn=-1)
      robjects.r("library(Matrix)")
      pandas2ri.activate()
    except ImportError:
      raise ImportError("Require package 'rpy2' for reading Rdata file.")
    loaded_data = {}
    for k, v in data.items():
      robjects.r['load'](v)
      x = robjects.r[k]
      if k == "counts_mel":
        with localconverter(robjects.default_converter + pandas2ri.converter):
          # dgCMatrix
          x = sparse.csr_matrix((x.slots["x"], x.slots["i"], x.slots["p"]),
                                shape=tuple(robjects.r("dim")(x))[::-1],
                                dtype=np.float32)
      else:
        x = robjects.conversion.rpy2py(x)
      loaded_data[k] = x
    ### post-processing
    x = loaded_data['counts_mel']
    labels = []
    for i, j in zip(loaded_data["cellData_mel"]['cellLine'],
                    loaded_data["cellData_mel"]['LineType']):
      labels.append(i + '_' + j.split("-")[0])
    labels = np.array(labels)
    labels = np.array(labels)
    labels_name = {name: i for i, name in enumerate(sorted(set(labels)))}
    labels = one_hot(np.array([labels_name[i] for i in labels]),
                     len(labels_name))
    ### assign the data
    self.x = x
    self.y = labels
    self.xvar = np.array([f"Region{i + 1}" for i in range(x.shape[1])])
    self.yvar = np.array(list(labels_name.keys()))


# ===========================================================================
# More datasets
# ===========================================================================
def _load_scale_dataset(path, dsname):
  url = str(
      base64.decodebytes(
          b'aHR0cHM6Ly9haS1kYXRhc2V0cy5zMy5hbWF6b25hd3MuY29tL3NjYWxlX2RhdGFzZXRzLnppcA==\n'
      ), 'utf-8')
  md5 = r"5fc7c52108220e30a04f033e355716c0"
  path = os.path.abspath(os.path.expanduser(path))
  if not os.path.exists(path):
    os.makedirs(path)
  filename = os.path.basename(url)
  filepath = os.path.join(path, filename)
  # download
  if not os.path.exists(filepath):
    print(f"Downloading {url} ...")
    urlretrieve(url, filename=filepath)
  # extract
  zip_path = os.path.join(path, 'scale_datasets')
  if not os.path.exists(zip_path):
    with zipfile.ZipFile(filepath, "r") as f:
      f.extractall(path)
  # load
  cell = np.load(os.path.join(zip_path, f"{dsname}_cell"))
  labels = np.load(os.path.join(zip_path, f"{dsname}_labels"))
  peak = np.load(os.path.join(zip_path, f"{dsname}_peak"))
  x = sparse.load_npz(os.path.join(zip_path, f"{dsname}_x"))
  ids = {key: i for i, key in enumerate(sorted(set(labels)))}
  labels = one_hot(np.array([ids[i] for i in labels]), len(ids))
  return x, labels, peak, np.array(list(ids.keys()))


class Forebrain(BioDataset):

  def __init__(self, path: str = "~/tensorflow_datasets/scale_atac"):
    self.x, self.y, self.xvar, self.yvar = _load_scale_dataset(
        path=path, dsname="forebrain")


class Insilico(BioDataset):

  def __init__(self, path: str = "~/tensorflow_datasets/scale_atac"):
    self.x, self.y, self.xvar, self.yvar = _load_scale_dataset(
        path=path, dsname="insilico")


class BreastTumor(BioDataset):

  def __init__(self, path: str = "~/tensorflow_datasets/scale_atac"):
    self.x, self.y, self.xvar, self.yvar = _load_scale_dataset(
        path=path, dsname="breast_tumor")


class Leukemia(BioDataset):

  def __init__(self, path: str = "~/tensorflow_datasets/scale_atac"):
    self.x, self.y, self.xvar, self.yvar = _load_scale_dataset(
        path=path, dsname="leukemia")
