from __future__ import absolute_import, division, print_function

from contextlib import contextmanager

from odin.bay.vi.data_utils import Factor
from odin.bay.vi.variational_autoencoder import VariationalAutoencoder


class Criticizer():

  def __init__(self, vae: VariationalAutoencoder):
    super().__init__()
    assert isinstance(vae, VariationalAutoencoder)
    self._vae = vae
