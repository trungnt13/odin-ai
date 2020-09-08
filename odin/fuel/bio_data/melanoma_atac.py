import base64
import os
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from scipy import sparse

from odin.fuel.bio_data._base import BioDataset
from odin.utils import one_hot

_URL = [
    r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/counts_mel.RData",
    r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/cellData_mel.RData",
]


class MelanomaATAC(BioDataset):
  r""" melanoma ATAC data from (Bravo González-Blas, et al. 2019)

  Reference:
    Bravo González-Blas, C. et al. cisTopic: cis-regulatory topic modeling
      on single-cell ATAC-seq data. Nat Methods 16, 397–400 (2019).
    Verfaillie, A. et al. Decoding the regulatory landscape of melanoma
      reveals TEADS as regulators of the invasive cell state.
      Nat Commun 6, (2015).
  """

  def __init__(self, path="~/tensorflow_datasets/melanoma_atac"):
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
