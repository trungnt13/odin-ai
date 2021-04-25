from typing import Sequence, Optional, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.initializers.initializers_v2 import Initializer
from tensorflow_probability.python.distributions import Distribution
from typing_extensions import Literal

from odin.backend import atleast_2d
from odin.backend.types_helpers import Activation, Optimizer
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.layers import MVNDiagLatents
from odin.utils import as_tuple


class TwoStageVAE(BetaVAE):
  """ Two Stage VAE

  References
  ----------
  Dai, B., Wipf, D., 2019. Diagnosing and Enhancing VAE Models.
      arXiv:1903.05789 [cs, stat].
  Source code: https://github.com/daib13/TwoStageVAE
  """

  def __init__(
      self,
      second_units: Sequence[int] = (1024, 1024, 1024),
      second_zdim: Optional[int] = None,
      activation: Activation = tf.nn.relu,
      initializer: Optional[Initializer] = None,
      auto_train_2stages: bool = True,
      ratio_2stages: int = 3,
      **kwargs):
    """
    Parameters
    ----------
    second_units : Sequence[int]
        list of hidden units for encoder and decoder of the second stage
    second_zdim : Optional[int]
        number of latents units for second stage, by default, the same
        number of units as first stage latents.
    activation : Activation
        activation function for hidden layers of the second stage
    auto_train_2stages : bool
        if True, automatically train the second stage with double the
        number of iteration right after the first stage training,
        i.e. single call to fit function will involve the two stage
        subsequently
    ratio_2stages : int
        if stage-1 is trained for `n` steps, then stage-2 is trained for
        `ratio_2stages * n` steps.
    """
    super(TwoStageVAE, self).__init__(**kwargs)
    # True = stage-1 (vanilla VAE training)
    # False = stage-2 (second VAE training)
    self._train_first_stage = True
    self._eval_first_stage = True
    self.ratio_2stages = int(ratio_2stages)
    self._auto_train_2stages = bool(auto_train_2stages)
    # === 1. prepare the inputs
    zdim = sum(np.prod(z.event_shape) for z in as_tuple(self.latents))
    if second_zdim is None:
      second_zdim = zdim
    x_inp = keras.Input([zdim])
    z_inp = keras.Input([second_zdim])
    kw = dict(activation=activation,
              kernel_initializer=initializer)
    # === 2. encoder
    x_out = x_inp
    for i, units in enumerate(second_units):
      x_out = tf.keras.layers.Dense(units, name=f'Encoder2_{i}', **kw)(x_out)
    x_out = tf.keras.layers.Concatenate(-1)([x_inp, x_out])
    self.encoder2 = keras.Model(inputs=x_inp, outputs=x_out, name='Encoder2')
    # === 3. decoder
    z_out = z_inp
    for i, units in enumerate(second_units[::-1]):
      z_out = tf.keras.layers.Dense(units, name=f'Decoder2_{i}', **kw)(z_out)
    z_out = tf.keras.layers.Concatenate(-1)([z_inp, z_out])
    self.decoder2 = keras.Model(inputs=z_inp, outputs=z_out, name='Decoder2')
    # === 4. latents and outputs
    self.latents2 = MVNDiagLatents(units=second_zdim, name='Latents2')
    self.latents2(self.encoder2.output)
    self.observation2 = MVNDiagLatents(units=zdim, name='Observation2')
    self.observation2(self.decoder2.output)

  def sample_prior(self, n: int = 1, seed: int = 1,
                   two_stage: bool = True) -> tf.Tensor:
    if two_stage:
      z = atleast_2d(self.latents2.sample(sample_shape=n, seed=seed))
      z = self.decoder2(z, training=False)
      qz = self.observation2(z, trainable=False)
      return tf.convert_to_tensor(qz)
    else:
      return atleast_2d(self.latents.sample(sample_shape=n, seed=seed))

  def sample_observation(self, n: int = 1, seed: int = 1,
                         training: bool = False,
                         two_stage: bool = True) -> Distribution:
    z = self.sample_prior(n, seed=seed, two_stage=two_stage)
    return self.decode(z, training=training)

  def set_train_stage(self, stage: Literal[1, 2]) -> 'TwoStageVAE':
    """Set the training stage of two stage VAE:

      - 1: training vanillaVAE
      - 2: training the second VAE converge to the true manifold
    """
    self._train_first_stage = stage == 1
    return self

  def set_eval_stage(self, stage: Literal[1, 2]) -> 'TwoStageVAE':
    """Set the evaluation stage of two stage VAE:

      - 1: evaluate using latents from vanilla VAE
      - 2: use the enhanced latents from two stage VAE
    """
    self._eval_first_stage = stage == 1
    return self

  def encode(self,
             inputs,
             training=None,
             mask=None,
             only_encoding=False,
             **kwargs) -> Distribution:
    if only_encoding:
      return super(TwoStageVAE, self).encode(inputs,
                                             training=training,
                                             mask=mask,
                                             only_encoding=True,
                                             **kwargs)
    qz_x = super(TwoStageVAE, self).encode(inputs,
                                           training=training,
                                           mask=mask,
                                           **kwargs)
    if not self._eval_first_stage:
      qz_x = self.observation2(
        self.decoder2(
          self.latents2(
            self.encoder2(qz_x, training=training),
            training=training),
          training=training),
        training=training)
    return qz_x

  def encode_two_stages(
      self,
      inputs,
      training=None,
      mask=None) -> Tuple[Distribution, Distribution, Distribution]:
    """Encode three distributions:

      - `q(z|x)` : latents of the vanilla VAE
      - `q(u|z)` : latents of the second stage
      - `q(z|u)` : the enhanced latents after the second stage

    The three distributions have the same `batch_shape` and `event_shape`
    """
    qz_x = self.encode(inputs, training=training, mask=mask)
    qu_z = self.latents2(self.encoder2(qz_x, training=training),
                         training=training)
    qz_u = self.observation2(self.decoder2(qu_z, training=training),
                             training=training)
    return qz_x, qu_z, qz_u

  def elbo_components2(self, inputs, training=None, mask=None, **kwargs):
    qz = tf.stop_gradient(
      self.encode(inputs, training=training, mask=mask, **kwargs))
    z = tf.convert_to_tensor(qz)
    h_e = self.encoder2(z, training=training)
    qz2 = self.latents2(h_e, training=training)
    h_d = self.decoder2(qz2, training=training)
    pz = self.observation2(h_d, training=training)
    return {f'llk_{self.latents.name}': pz.log_prob(z)}, \
           {f'kl_latents2': qz2.KL_divergence(analytic=self.analytic,
                                              reverse=self.reverse,
                                              free_bits=self.free_bits)}

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    if self._train_first_stage:
      return super().elbo_components(inputs, training=training, mask=mask,
                                     **kwargs)
    else:
      return self.elbo_components2(inputs, training=training, mask=mask,
                                   **kwargs)

  def fit(self, train, *, valid=None, **kwargs) -> 'TwoStageVAE':
    super(TwoStageVAE, self).fit(train, valid=valid, **kwargs)
    if self._train_first_stage and self._auto_train_2stages:
      self.set_train_stage(2)
      for k in ['on_batch_end', 'on_valid_end', 'optimizer', 'learning_rate']:
        kwargs.pop(k, None)
      super(TwoStageVAE, self).fit(
        train,
        epochs=kwargs.pop('epochs', -1) * self.ratio_2stages,
        max_iter=kwargs.pop('max_iter', 1000) * self.ratio_2stages,
        **kwargs)
      self.set_train_stage(1)
    return self
