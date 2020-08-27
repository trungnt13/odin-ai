import base64
import os
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from scipy import sparse

from odin.fuel.bio_data._base import BioDataset
from odin.utils.crypto import md5_checksum


class PBMC(BioDataset):
  _URL = {
      '5k':
          b'aHR0cHM6Ly9haS1kYXRhc2V0cy5zMy5hbWF6b25hd3MuY29tL3BibWM1ay5ucHo=\n',
      '10k':
          b'aHR0cHM6Ly9haS1kYXRhc2V0cy5zMy5hbWF6b25hd3MuY29tL3BibWMxMGsubnB6\n'
  }

  def __init__(self, dataset='5k', path="~/tensorflow_datasets/pbmc"):
    path = os.path.abspath(os.path.expanduser(path))
    self.dsname = dataset
    if not os.path.exists(path):
      os.makedirs(path)
    url = str(base64.decodebytes(PBMC._URL[str(dataset).lower().strip()]),
              'utf-8')
    name = os.path.basename(url)
    filename = os.path.join(path, name)
    urlretrieve(url,
                filename=filename,
                reporthook=lambda blocknum, bs, size: None)
    ### load the data
    data = np.load(filename, allow_pickle=True)
    self.x = data['x'].tolist().todense().astype(np.float32)
    self.y = data['y'].tolist().todense().astype(np.float32)
    assert md5_checksum(self.x) == data['xmd5'].tolist(), \
      "MD5 for transcriptomic data mismatch"
    assert md5_checksum(self.y) == data['ymd5'].tolist(), \
      "MD5 for proteomic data mismatch"
    self.xvar = data['xvar']
    self.yvar = data['yvar']
    self.pairs = data['pairs']
    ### split train, valid, test data
    rand = np.random.RandomState(seed=1)
    n = self.x.shape[0]
    ids = rand.permutation(n)
    self.train_ids = ids[:int(0.85 * n)]
    self.valid_ids = ids[int(0.85 * n):int(0.9 * n)]
    self.test_ids = ids[int(0.9 * n):]

  @property
  def name(self):
    return f"pbmc{self.dsname}"
