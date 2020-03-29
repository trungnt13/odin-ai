from odin.bay.vi.autoencoder.beta_vae import *
from odin.bay.vi.autoencoder.factor_vae import *
from odin.bay.vi.autoencoder.grammar_vae import *
from odin.bay.vi.autoencoder.variational_autoencoder import *
from odin.bay.vi.autoencoder.dip_vae import *
from odin.networks import (ConvNetwork, DeconvNetwork, DenseNetwork,
                           NetworkConfig)


def create_image_encoder_decoder(image_shape=(64, 64),
                                 channels=3,
                                 activation='relu',
                                 projection_dim=256,
                                 latent_dim=10,
                                 distribution='gaus',
                                 name="Image"):
  r""" Initialized the Convolutional encoder and decoder often used in
  Disentangled VAE literatures

  Arguments:
  """
  from odin.bay import parse_distribution
  distribution, _ = parse_distribution(distribution)
  n_params = int(tf.reduce_prod(distribution.params_size(1)).numpy())

  encoder = ConvNetwork(filters=[32, 32, 64, 64],
                        kernel_size=[4, 4, 4, 4],
                        strides=[2, 2, 2, 2],
                        batchnorm=False,
                        activation=activation,
                        end_layers=[
                            keras.layers.Flatten(),
                            keras.layers.Dense(projection_dim,
                                               activation='linear')
                        ],
                        input_shape=image_shape + (channels,),
                        name="%sEncoder" % name)
  encoder_shape = encoder.layers[-3].output.shape[1:]
  decoder = DeconvNetwork(filters=[64, 32, 32, channels * n_params],
                          kernel_size=[4, 4, 4, 4],
                          strides=[2, 2, 2, 2],
                          activation=[activation] * 3 + ['linear'],
                          batchnorm=False,
                          start_layers=[
                              keras.layers.Dense(256, activation='relu'),
                              keras.layers.Dense(int(np.prod(encoder_shape)),
                                                 activation='relu'),
                              keras.layers.Reshape(encoder_shape),
                          ],
                          end_layers=[keras.layers.Flatten()],
                          input_shape=(latent_dim,),
                          name="%sDecoder" % name)
  return encoder, decoder


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
