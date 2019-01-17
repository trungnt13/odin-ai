
from six import add_metaclass
from abc import ABCMeta, abstractproperty

from odin import backend as K
from odin.utils import flatten_list
from odin.nnet.base import Dense
from odin.nnet.models.base import Model

@add_metaclass(ABCMeta)
class _BNFbase(Model):

  @abstractproperty
  def nb_layers(self):
    raise NotImplementedError

  def get_input_info(self):
    w0 = self.get_loaded_param('w0')
    nb_ceps = w0.shape[1]
    return {'X': ((None, nb_ceps), 'float32')}

  def _initialize(self):
    param_names = flatten_list([('b%d' % i, 'w%d' % i)
                                for i in range(self.nb_layers)])
    weights = self.get_loaded_param(param_names)
    # ====== create ====== #
    layers = []
    for i in range(self.nb_layers):
      b = weights[i * 2].ravel()
      W = weights[i * 2 + 1].T
      num_units = b.shape[0]
      if i == self.nb_layers - 1:
        name = 'Bottleneck'
        nonlinearity = K.linear
      else:
        name = "Layer%d" % (i + 1)
        nonlinearity = K.relu
      op = Dense(num_units=num_units, W_init=W, b_init=b,
                 activation=nonlinearity, name=name)
      layers.append(op)
    self.layers = layers

  def _apply(self, X):
    # ====== apply ====== #
    for op in self.layers[:-1]:
      X = op(X)
      X = K.renorm_rms(X, axis=1)
    return self.layers[-1](X)


class BNF_1024_MFCC39(_BNFbase):
  """ Bottleneck fully connected network (1024-units):
  + Feature: 39-D MFCCs with 13-D+detla+ddelta)
     - sample_rate = 8000
     - filter_lo_edge = 100
     - filter_hi_edge = 4000
     - num_cepstral_coefs = 13
     - frame_length = 0.025
     - frame_shift = 0.010
     - preemphasis_coef = 0.97
     - num_channels = 40
     - num_fft_points = 512
     - window_type = hamm
     - spectrum_type = mag
     - compression_type = log
     - NO RASTA filter applied on MFCC
  + Input features must be normalized: (x - mean) / std
  + Context size: 21
  + Nonlinearity: Relu
  + Renorm: True (scales the data such that RMS is 1.0,
           performed after the activation)
  NOTE: the last layer (the bottleneck) is linear activated, and no renorm.
  """

  @property
  def nb_layers(self):
    return 5


class BNF_2048_MFCC39(_BNFbase):
  """ Bottleneck fully connected network (2048-units):
  + Feature: 39-D MFCCs with 13-D+detla+ddelta)
     - sample_rate = 8000
     - filter_lo_edge = 100
     - filter_hi_edge = 4000
     - num_cepstral_coefs = 13
     - frame_length = 0.025
     - frame_shift = 0.010
     - preemphasis_coef = 0.97
     - num_channels = 40
     - num_fft_points = 512
     - window_type = hamm
     - spectrum_type = mag
     - compression_type = log
     - NO RASTA filter applied on MFCC
  + Input features must be normalized: (x - mean) / std
  + Context size: 21
  + Nonlinearity: Relu
  + Renorm: True (scales the data such that RMS is 1.0,
           performed after the activation)
  NOTE: the last layer (the bottleneck) is linear activated, and no renorm.
  """
  @property
  def nb_layers(self):
    return 6


class BNF_2048_MFCC40(_BNFbase):
  """ Bottleneck fully connected network (2048-units):
    + Feature: static 40-D MFCCs
       - sample_rate = 8000
       - filter_lo_edge = 100
       - filter_hi_edge = 4000
       - num_cepstral_coefs = 40
       - frame_length = 0.025
       - frame_shift = 0.010
       - preemphasis_coef = 0.97
       - num_channels = 40
       - num_fft_points = 512
       - window_type = hamm
       - spectrum_type = mag
       - compression_type = log
       - NO RASTA filter applied on MFCC
    + Input features must be normalized: (x - mean) / std
    + Context size: 21
    + Nonlinearity: Relu
    + Renorm: True (scales the data such that RMS is 1.0,
              performed after the activation)
   NOTE: the last layer (the bottleneck) is linear activated, and no renorm.
  """
  @property
  def nb_layers(self):
    return 6
