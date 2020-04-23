from odin.bay.vi.autoencoder.beta_vae import *
from odin.bay.vi.autoencoder.dip_vae import *
from odin.bay.vi.autoencoder.factor_vae import *
from odin.bay.vi.autoencoder.info_vae import *
from odin.bay.vi.autoencoder.lda_vae import *
from odin.bay.vi.autoencoder.networks import *
from odin.bay.vi.autoencoder.self_supervised_vae import *
from odin.bay.vi.autoencoder.semi_supervised_vae import *
from odin.bay.vi.autoencoder.stochastic_vae import *
from odin.bay.vi.autoencoder.variational_autoencoder import *
from odin.bay.vi.autoencoder.vq_vae import *


def get_vae(name=None) -> VariationalAutoencoder:
  import inspect
  from six import string_types
  if not isinstance(name, string_types):
    if inspect.isclass(name):
      name = name.__name__
    else:
      name = type(name).__name__
  name = str(name).strip().lower()
  all_vae = []
  vae = None
  for key, val in globals().items():
    if inspect.isclass(val) and issubclass(val, VariationalAutoencoder):
      if name in key.lower():
        vae = val
        break
      else:
        all_vae.append(val)
  if vae is None:
    if name == 'nonetype':
      return sorted(all_vae, key=lambda cls: cls.__name__)
    raise ValueError("Cannot find VAE with name '%s', all VAE are: %s" %
                     (name, ", ".join([i.__name__ for i in all_vae])))
  return vae
