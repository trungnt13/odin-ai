# embryos

import numpy as np
from scipy import sparse

from odin.fuel.bio_data._base import BioDataset
from odin.fuel.bio_data.cortex import _load_single_cell_data


class HumanEmbryos(BioDataset):

  def __init__(self, path="~/tensorflow_datasets/human_embryos"):
    super().__init__()
    url = b'aHR0cHM6Ly9haS1kYXRhc2V0cy5zMy5hbWF6b25hd3MuY29tL2VtYnJ5by56aXA=\n'
    self.x, self.y, self.xvar, self.yvar = _load_single_cell_data(url=url,
                                                                  path=path)
    # split train, valid, test data
    rand = np.random.RandomState(seed=1)
    n = self.x.shape[0]
    ids = rand.permutation(n)
    self.train_ids = ids[:int(0.85 * n)]
    self.valid_ids = ids[int(0.85 * n):int(0.9 * n)]
    self.test_ids = ids[int(0.9 * n):]

  @property
  def name(self):
    return f"cortex"
