import inspect
from typing import Optional, Type

from odin.bay.random_variable import RVconf
from odin.networks import NetConf
from odin.bay.vi.autoencoder.auxiliary_vae import *
from odin.bay.vi.autoencoder.beta_vae import *
from odin.bay.vi.autoencoder.conditional_vae import *
from odin.bay.vi.autoencoder.deterministic import *
from odin.bay.vi.autoencoder.dip_vae import *
from odin.bay.vi.autoencoder.factor_vae import *
from odin.bay.vi.autoencoder.hierarchical_vae import *
from odin.bay.vi.autoencoder.hyperbolic_vae import *
from odin.bay.vi.autoencoder.info_vae import *
from odin.bay.vi.autoencoder.irm_vae import *
from odin.bay.vi.autoencoder.lda_vae import *
from odin.bay.vi.autoencoder.multitask_vae import *
from odin.bay.vi.autoencoder.self_supervised_vae import *
from odin.bay.vi.autoencoder.stochastic_vae import *
from odin.bay.vi.autoencoder.vamprior import *
from odin.bay.vi.autoencoder.variational_autoencoder import *
from odin.bay.vi.autoencoder.vq_vae import *
from odin.bay.vi.autoencoder.semafo_vae import *
from six import string_types
from odin.bay.vi.autoencoder.two_stage_vae import *


def get_vae(name: str = None) -> Type[VariationalAutoencoder]:
  """Get VAE model by name"""
  if not isinstance(name, string_types):
    if inspect.isclass(name):
      name = name.__name__
    else:
      name = type(name).__name__
  name = str(name).strip().lower()
  vae = None
  for key, val in globals().items():
    if inspect.isclass(val) and issubclass(val, VariationalAutoencoder):
      if name == key.lower():
        vae = val
        break
  if vae is None:
    raise ValueError(f"Cannot find VAE with name '{name}'")
  return vae


def get_all_vae() -> List[Type[VariationalAutoencoder]]:
  """Return all available VAE models"""
  all_vae = []
  for key, val in globals().items():
    if inspect.isclass(val) and issubclass(val, VariationalAutoencoder):
      all_vae.append(val)
  return sorted(all_vae, key=lambda i: i.__name__)
