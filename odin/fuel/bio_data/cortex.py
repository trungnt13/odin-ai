import base64
import os
import zipfile
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse

from odin.fuel.bio_data._base import GeneDataset


def _load_single_cell_data(url, path):
  path = os.path.abspath(os.path.expanduser(path))
  if not os.path.exists(path):
    path = os.makedirs(path)
  url = str(base64.decodebytes(url), 'utf-8')
  zip_path = os.path.join(path, os.path.basename(url))
  name = os.path.basename(zip_path).replace('.zip', '')
  extracted_path = os.path.join(path, name)
  # download
  if not os.path.exists(zip_path):
    urlretrieve(filename=zip_path, url=url)
  # extract
  if not os.path.isdir(extracted_path):
    with zipfile.ZipFile(open(zip_path, 'rb')) as f:
      f.extractall(path)
  # load data
  with open(os.path.join(extracted_path, 'X'), 'rb') as f:
    X = sparse.load_npz(f)
  with open(os.path.join(extracted_path, 'y'), 'rb') as f:
    y = sparse.load_npz(f)
  with open(os.path.join(extracted_path, 'var_names'), 'rb') as f:
    var_names = np.load(f, allow_pickle=True)
  with open(os.path.join(extracted_path, 'labels'), 'rb') as f:
    labels = np.load(f, allow_pickle=True)
  # store data
  x = X
  if isinstance(x, (sparse.coo_matrix, sparse.dok_matrix)):
    x = x.tocsr()
  y = y
  if isinstance(y, (sparse.coo_matrix, sparse.dok_matrix)):
    y = y.tocsr()
  xvar = var_names
  yvar = labels
  return x, y, xvar, yvar


class Cortex(GeneDataset):

  def __init__(self, path="~/tensorflow_datasets/cortex"):
    super().__init__()
    url = b'aHR0cHM6Ly9haS1kYXRhc2V0cy5zMy5hbWF6b25hd3MuY29tL2NvcnRleC56aXA=\n'
    self.x, self.y, self.xvar, self.yvar = _load_single_cell_data(url=url,
                                                                  path=path)

  @property
  def name(self):
    return f"cortex"
