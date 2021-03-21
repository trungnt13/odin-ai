from typing import List, Union, Tuple, Dict, Sequence, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow_probability.python.distributions import NOT_REPARAMETERIZED, \
  Distribution

from odin.backend.interpolation import Interpolation, linear
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import AnnealingVAE
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.bay.vi.utils import prepare_ssl_inputs
from odin.bay.layers import DistributionDense
from odin.utils import as_tuple


# ===========================================================================
# helpers
# ===========================================================================
def _get_llk_y(py, y, mask, alpha):
  llk_y = 0.
  # if labels is provided,
  # p(y|z) is q(y|z) here since we use the same network for both
  if len(y) > 0:
    # support only 1 labels set provided
    if isinstance(y, (tuple, list)):
      y = y[0]
    llk_y = py.log_prob(y)
    if mask is not None:
      llk_y = tf.cond(
        tf.reduce_any(mask),
        true_fn=lambda: tf.transpose(
          tf.boolean_mask(tf.transpose(llk_y), mask, axis=0)),
        false_fn=lambda: 0.,
      )
    llk_y = tf.reduce_mean(alpha * llk_y)
    llk_y = tf.cond(tf.abs(llk_y) < 1e-8,
                    true_fn=lambda: 0.,
                    false_fn=lambda: llk_y)
  return llk_y


class SplitVAE(AnnealingVAE):

  def __init__(self, n_split: int = 3, name: str = 'SplitVAE', **kwargs):
    super(SplitVAE, self).__init__(name=name, **kwargs)
    zdim = sum(int(np.prod(z.event_shape)) for z in as_tuple(self.latents))
    units = [zdim // n_split for i in range(n_split - 1)]
    units.append(zdim - sum(units))
    del self._latents
    self._latents = [
      RVconf(units[i], 'mvndiag', projection=True,
             name=f'latents{i}').create_posterior()
      for i in range(n_split)]
    self.encoder.track_outputs = True
    self.flatten = keras.layers.Flatten()
    self.project = keras.layers.Dense(256)

  def encode(self,
             inputs,
             training=None,
             mask=None,
             only_encoding=False,
             **kwargs):
    h = super(SplitVAE, self).encode(inputs, training=training, mask=mask,
                                     only_encoding=True, **kwargs)
    skip = []
    for layer, outputs in h._last_outputs:
      if isinstance(layer, (keras.layers.Dense, Conv)):
        skip.append(self.flatten(outputs))
    h = self.project(tf.concat(skip, -1))
    if only_encoding:
      return h
    return tuple(
      [qz(h, training=training, mask=mask, sample_shape=self.sample_shape)
       for qz in self.latents])

  def decode(self,
             latents,
             training=None,
             mask=None,
             only_decoding=False,
             **kwargs):
    latents = tf.concat(latents, 1)
    return super().decode(latents, training=training, mask=mask,
                          only_decoding=only_decoding, **kwargs)


class SemafoBase(AnnealingVAE):

  def __init__(
      self,
      labels: RVconf = RVconf(10, 'onehot', projection=True, name="digits"),
      alpha: float = 10.0,
      mi_coef: Union[float, Interpolation] = linear(vmin=0.1,
                                                    vmax=0.05,
                                                    length=20000),
      reverse_mi: bool = False,
      steps_without_mi: int = 1000,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._separated_steps = False
    self.labels = _parse_layers(labels)
    self._mi_coef = mi_coef
    self.alpha = alpha
    self.steps_without_mi = int(steps_without_mi)
    self.reverse_mi = bool(reverse_mi)

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    # don't condition on the labels, only accept inputs
    X = X[0]
    qz_x = super().encode(X, training=training, mask=None, **kwargs)
    return qz_x

  def decode(self, latents, training=None, mask=None, **kwargs):
    px_z = super().decode(latents, training, mask, **kwargs)
    py_z = self.predict_factors(latents=latents, training=training, mask=mask)
    return as_tuple(px_z) + (py_z,)

  def predict_factors(self,
                      inputs=None,
                      latents=None,
                      training=None,
                      mask=None,
                      **kwargs) -> Distribution:
    """Return the predictive distribution of the factors (a.k.a labels) 
    `p(y|z)`"""
    if inputs is not None:
      latents = self.encode(inputs, training=training, mask=mask, **kwargs)
    elif latents is None:
      raise ValueError("Either 'inputs' or 'latents' must be provided")
    py_z = self.labels(tf.concat(as_tuple(latents), axis=-1),
                       training=training,
                       mask=mask)
    return py_z

  @property
  def mi_coef(self):
    if isinstance(self._mi_coef, Interpolation):
      step = tf.maximum(0.,
                        tf.cast(self.step - self.steps_without_mi, tf.float32))
      return self._mi_coef(step)
    return tf.constant(self._mi_coef, dtype=self.dtype)

  @classmethod
  def is_hierarchical(cls) -> bool:
    return False

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  def train_steps(self, inputs, training=None, mask=None, name='', **kwargs):
    if not self._separated_steps:
      yield from super().train_steps(inputs, training=training, mask=mask,
                                     name=name, **kwargs)
    else:
      X, y, mask = prepare_ssl_inputs(inputs, mask, n_unsupervised_inputs=1)
      X = X[0]
      mask = tf.reshape(mask, (-1,))
      X_u = tf.boolean_mask(X, tf.logical_not(mask), axis=0)
      # === 1. supervised steps
      if len(y) > 0:
        X_l = tf.boolean_mask(X, mask, axis=0)
        y_l = tf.boolean_mask(y[0], mask, axis=0)
        yield from super().train_steps(inputs=[X_l, y_l],
                                       mask=None,
                                       training=training,
                                       name=f'{name}labeled',
                                       **kwargs)
        # === 2. unsupervised steps
        yield from super().train_steps(inputs=X_u,
                                       mask=None,
                                       training=training,
                                       name=f'{name}unlabeled',
                                       **kwargs)


# ===========================================================================
# SemafoVAE
# ===========================================================================
class SemafoVAE(SemafoBase):
  """A semaphore is a variable or abstract data type used to control access to
  a common resource by multiple processes and avoid critical section problems in
  a concurrent system

  For MNIST, `mi_coef` could be from 0.1 to 0.5.
  For dSprites, CelebA and Shapes3D, `mi_coef=0.1` has been tested.

  It is also seems that the choice of `mi_coef` is crucial, regardless
  the percentage of labels, or the networks design.

  It is also crucial to get the right value at the beginning of training
  (i.e. small enough)

  Parameters
  ----------
  reverse_mi : bool
      if True, minimize `D_kl(p(y|z)||q(y|z))`, otherwise, `D_kl(q(y|z)||p(y|z))`
  steps_without_mi : int
      number of step without mutual information maximization which allows
      the network to fit better encoder. Gradients backpropagated to encoder
      often NaNs if optimize all objective from beginning, default: 1000

  SemafoVAE  [Callback#50001]:
  llk_x:-73.33171081542969
  llk_y:-0.9238954782485962
  acc_y:0.7268000245094299

  Without autoregressive
  llk_x:-72.9976577758789
  llk_y:-0.7141319513320923
  acc_y:0.8095999956130981

  Idea: set mi_coef = 1. / alpha
  """

  def __init__(self, name: str = 'SemafoVAE', **kwargs):
    super().__init__(name=name, **kwargs)

  def _mi_loss(self,
               Q: Sequence[Distribution],
               py_z: Distribution,
               training: Optional[bool] = None,
               mask: Optional[bool] = None,
               which_latents_sampling: Optional[List[int]] = None,
               ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    ## sample the prior
    batch_shape = Q[0].batch_shape
    if which_latents_sampling is None:
      which_latents_sampling = list(range(len(Q)))
    z_prime = [q.KL_divergence.prior.sample(batch_shape)
               if i in which_latents_sampling else
               tf.stop_gradient(tf.convert_to_tensor(q))
               for i, q in enumerate(Q)]
    if len(z_prime) == 1:
      z_prime = z_prime[0]
    ## decoding
    px = self.decode(z_prime, training=training)[0]
    if px.reparameterization_type == NOT_REPARAMETERIZED:
      x = px.mean()
    else:
      x = tf.convert_to_tensor(px)
    # should not stop gradient here, generator need to be updated
    # x = tf.stop_gradient(x)
    Q_prime = self.encode(x, training=training)
    qy_z = self.predict_factors(latents=Q_prime, training=training)
    ## y ~ p(y|z), stop gradient here is important to prevent the encoder
    # updated twice this significantly increase the stability, otherwise,
    # encoder and latents often get NaNs gradients
    if self.reverse_mi:  # D_kl(p(y|z)||q(y|z))
      y_samples = tf.stop_gradient(py_z.sample())
      Dkl = py_z.log_prob(y_samples) - qy_z.log_prob(y_samples)
    else:  # D_kl(q(y|z)||p(y|z))
      y_samples = tf.stop_gradient(qy_z.sample())
      Dkl = qy_z.log_prob(y_samples) - py_z.log_prob(y_samples)
    ## only calculate MI for unsupervised data
    mi_mask = tf.logical_not(mask)  # TODO: tf.reduce_any(mi_mask)
    mi_y = tf.reduce_mean(tf.boolean_mask(Dkl, mask=mi_mask, axis=0))
    ## mutual information (we want to maximize this, hence, add it to the llk)
    if training:
      mi_y = tf.cond(
        self.step >= self.steps_without_mi,
        true_fn=lambda: self.mi_coef * mi_y,
        false_fn=lambda: tf.stop_gradient(mi_y),
      )
    else:
      mi_y = tf.stop_gradient(mi_y)
    ## this value is just for monitoring
    mi_z = []
    for q, z in zip(as_tuple(Q_prime), as_tuple(z_prime)):
      mi = tf.reduce_mean(tf.stop_gradient(q.log_prob(z)))
      mi = tf.cond(tf.math.is_nan(mi),
                   true_fn=lambda: 0.,
                   false_fn=lambda: tf.clip_by_value(mi, -1e8, 1e8))
      mi_z.append(mi)
    return mi_y, mi_z

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    P, Q = self.last_outputs
    px_z = P[:-1]
    py_z = P[-1]
    Q = as_tuple(Q)  # q(z|x)
    ## supervised loss
    llk[f"llk_{self.labels.name}"] = _get_llk_y(py_z, y, mask, self.alpha)
    ## MI objective
    mi_y, mi_z = self._mi_loss(Q, py_z, training=training, mask=mask)
    llk[f'mi_{self.labels.name}'] = mi_y
    for z, mi in zip(as_tuple(self.latents), mi_z):
      llk[f'mi_{z.name}'] = mi
    return llk, kl


class RemafoVAE(SemafoVAE):
  """ semafoVAE with reversed KL for the mutual information

  (i.e. minimize `D_kl(p(y|z)||q(y|z))`)
  """

  def __init__(self, name: str = 'RemafoVAE', **kwargs):
    super().__init__(reverse_mi=True, name=name, **kwargs)


# ===========================================================================
# Hierarchical latents model
# ===========================================================================
class semafod(SemafoVAE):
  """Semafo VAE with double latents"""

  def __init__(self, name: str = 'semafod', **kwargs):
    super().__init__(name=name, **kwargs)
    # zdim = int(np.prod(self.latents.event_shape))
    zdim = int(np.prod(self.labels.event_shape))
    self.latents_y = RVconf(zdim, 'mvndiag', projection=True,
                            name=f'{self.latents.name}_y').create_posterior()

  @classmethod
  def is_hierarchical(cls) -> bool:
    return True

  def predict_factors(self,
                      inputs=None,
                      latents=None,
                      training=None,
                      mask=None,
                      **kwargs) -> Distribution:
    """Return the predictive distribution of the factors (a.k.a labels)
    `p(y|z)`"""
    if isinstance(latents, (tuple, list)):
      latents = latents[-1]
    return super().predict_factors(inputs=inputs, latents=latents,
                                   training=training, mask=mask, **kwargs)

  def encode(self, inputs, training=None, mask=None, only_encoding=False,
             **kwargs):
    h = super().encode(inputs, training=training, mask=mask,
                       only_encoding=True, **kwargs)
    qz1_x = self.latents(h, training=training, mask=mask,
                         sample_shape=self.sample_shape)
    qz2_x = self.latents_y(h, training=training, mask=mask,
                           sample_shape=self.sample_shape)
    return qz1_x, qz2_x

  def decode(self, latents, training=None, mask=None, **kwargs):
    qz1_x, qz2_x = latents
    py_z = self.predict_factors(latents=qz2_x, training=training, mask=mask)
    px_z = super(AnnealingVAE, self).decode(tf.concat(latents, axis=-1),
                                            training, mask, **kwargs)
    return as_tuple(px_z) + (py_z,)

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super(AnnealingVAE, self).elbo_components(X[0],
                                                        mask=mask,
                                                        training=training)
    P, Q = self.last_outputs
    kl[f'kl_{self.latents_y.name}'] = self.beta * Q[-1].KL_divergence(
      analytic=self.analytic, free_bits=self.free_bits, reverse=self.reverse)
    py_z = P[-1]
    ## supervised loss
    llk[f"llk_{self.labels.name}"] = _get_llk_y(py_z, y, mask, self.alpha)
    ## MI objective
    mi_y, mi_z = self._mi_loss(Q, py_z, training=training, mask=mask,
                               which_latents_sampling=[1])
    ## maximizing the MI
    llk[f'mi_{self.labels.name}'] = mi_y
    for z, mi in zip(as_tuple(self.latents), mi_z):
      llk[f'mi_{z.name}'] = mi
    return llk, kl

  class semafod(SemafoVAE):
    """Semafo VAE with double latents"""

    def __init__(self, name: str = 'semafod', **kwargs):
      super().__init__(name=name, **kwargs)
      zdim = int(np.prod(self.latents.event_shape))
      zdim = int(np.prod(self.labels.event_shape))
      self.latents_y = RVconf(zdim, 'mvndiag', projection=True,
                              name=f'{self.latents.name}_y').create_posterior()

    @classmethod
    def is_hierarchical(cls) -> bool:
      return True

    def predict_factors(self,
                        inputs=None,
                        latents=None,
                        training=None,
                        mask=None,
                        **kwargs) -> Distribution:
      """Return the predictive distribution of the factors (a.k.a labels)
      `p(y|z)`"""
      if isinstance(latents, (tuple, list)):
        latents = latents[-1]
      return super().predict_factors(inputs=inputs, latents=latents,
                                     training=training, mask=mask, **kwargs)

    def encode(self, inputs, training=None, mask=None, only_encoding=False,
               **kwargs):
      h = super().encode(inputs, training=training, mask=mask,
                         only_encoding=True, **kwargs)
      if only_encoding:
        return h
      qz1_x = self.latents(h, training=training, mask=mask,
                           sample_shape=self.sample_shape)
      qz2_x = self.latents_y(h, training=training, mask=mask,
                             sample_shape=self.sample_shape)
      return qz1_x, qz2_x

    def decode(self, latents, training=None, mask=None, **kwargs):
      qz1_x, qz2_x = latents
      py_z = self.predict_factors(latents=qz2_x, training=training, mask=mask)
      px_z = super(AnnealingVAE, self).decode(tf.concat(latents, axis=-1),
                                              training, mask, **kwargs)
      return as_tuple(px_z) + (py_z,)

    def elbo_components(self, inputs, training=None, mask=None):
      ## unsupervised ELBO
      X, y, mask = prepare_ssl_inputs(inputs, mask=mask,
                                      n_unsupervised_inputs=1)
      if mask is not None:
        mask = tf.reshape(mask, (-1,))
      llk, kl = super(AnnealingVAE, self).elbo_components(X[0],
                                                          mask=mask,
                                                          training=training)
      P, Q = self.last_outputs
      kl[f'kl_{self.latents_y.name}'] = self.beta * Q[-1].KL_divergence(
        analytic=self.analytic, free_bits=self.free_bits, reverse=self.reverse)
      py_z = P[-1]
      ## supervised loss
      llk[f"llk_{self.labels.name}"] = _get_llk_y(py_z, y, mask, self.alpha)
      ## MI objective
      mi_y, mi_z = self._mi_loss(Q, py_z, training=training, mask=mask,
                                 which_latents_sampling=[1])
      ## maximizing the MI
      llk[f'mi_{self.labels.name}'] = mi_y
      for z, mi in zip(as_tuple(self.latents), mi_z):
        llk[f'mi_{z.name}'] = mi
      return llk, kl


class semafoh(semafod):
  """Semafo VAE with double hierarchical latents"""

  def __init__(self, name: str = 'semafoh', **kwargs):
    super().__init__(name=name, **kwargs)

  def encode(self, inputs, training=None, mask=None, only_encoding=False,
             **kwargs):
    h = super(SemafoVAE, self).encode(inputs, training=training, mask=mask,
                                      only_encoding=True, **kwargs)
    if only_encoding:
      return h
    qz1_x = self.latents(h, training=training, mask=mask,
                         sample_shape=self.sample_shape)
    qz2_x = self.latents_y(tf.concat([h, qz1_x], -1),
                           training=training,
                           mask=mask,
                           sample_shape=self.sample_shape)
    return qz1_x, qz2_x


# ===========================================================================
# Separated step for supervised and unsupervised examples
# Having separated networks for p(y|z) and q(y|z) does not work.
# ===========================================================================
class semafos(SemafoBase):
  """Semafo with multiple steps training"""

  def __init__(self, name='semafos', **kwargs):
    super().__init__(name=name, **kwargs)
    self._separated_steps = True

  def encode(self, inputs, training=None, mask=None, **kwargs):
    inputs = as_tuple(inputs)
    if len(inputs) == 1:
      inputs = inputs[0]
      labels = None
    else:
      inputs, labels = inputs
    qz_x = super().encode(inputs, training=training, mask=None, **kwargs)
    # hidden trick to pass the labels to decode method
    qz_x._labeled = labels
    return qz_x

  def decode(self, latents, training=None, mask=None, **kwargs):
    py_z = self.predict_factors(latents=latents, training=training, mask=mask)
    # if labeled data is provided, use them in p(x|y,z)
    if hasattr(latents, '_labeled') and latents._labeled is not None:
      y = latents._labeled
    else:
      y = tf.stop_gradient(tf.convert_to_tensor(py_z))
    h = tf.concat(as_tuple(latents) + (y,), axis=-1)
    px_z = super(AnnealingVAE, self).decode(h, training, mask, **kwargs)
    return as_tuple(px_z) + (py_z,)

  def elbo_components(self, inputs, training=None, mask=None):
    inputs = as_tuple(inputs)
    if len(inputs) == 1:
      X, y = inputs[0], None
      is_supervised = False
    else:
      X, y = inputs
      is_supervised = True
    # === 0. ELBO
    llk, kl = super(AnnealingVAE, self).elbo_components(inputs,
                                                        training=training,
                                                        mask=mask)
    P, Q = self.last_outputs
    P = as_tuple(P)
    py_z = P[-1]  # p(y|z_l)
    # === 1. Supervised case E_q(z|x)[log p(y|z)]
    if is_supervised:
      llk[f'llk_{self.labels.name}'] = self.alpha * py_z.log_prob(y)
    # === 2. Unsupervised case E_q(z|x)[D_kl(q(y|z)||p(y|z))]
    else:
      qz_x = Q
      batch_shape = qz_x.batch_shape_tensor()
      assert isinstance(qz_x, Distribution), \
        f'Only support single latents, but given {Q}'
      z = qz_x.KL_divergence.prior.sample(batch_shape)
      px_z = self.decode(z, training=training)[0]
      px_z: Distribution
      x = px_z.mean() \
        if px_z.reparameterization_type == NOT_REPARAMETERIZED else \
        tf.convert_to_tensor(px_z)
      qy_z = self.predict_factors(inputs=x, training=training)
      y = tf.stop_gradient(tf.convert_to_tensor(qy_z))
      D_kl = tf.cond(
        self.step >= self.steps_without_mi,
        true_fn=lambda: self.mi_coef * (qy_z.log_prob(y) - py_z.log_prob(y)),
        false_fn=lambda: tf.zeros(batch_shape, dtype=self.dtype))
      llk[f'mi_{self.labels.name}'] = D_kl
    return llk, kl


class semafosm(semafos):
  """Semafo with separated training steps and multi-task learning"""

  def __init__(self, name='semafosm', **kwargs):
    super().__init__(name=name, **kwargs)

  def encode(self, inputs, training=None, mask=None, **kwargs):
    inputs = as_tuple(inputs)[0]
    return super().encode(inputs, training=training, mask=mask, **kwargs)

  def decode(self, latents, training=None, mask=None, **kwargs):
    px_z = super(AnnealingVAE, self).decode(latents, training=training,
                                            mask=mask, **kwargs)
    py_z = self.predict_factors(latents=latents, training=training, mask=mask)
    return as_tuple(px_z) + (py_z,)


class semafosc(semafos):
  """Semafo with multiple steps training and simple conditioning"""

  def __init__(self, name='semafosc', **kwargs):
    super().__init__(name=name, **kwargs)

  def encode(self, inputs, training=None, mask=None, **kwargs):
    inputs = as_tuple(inputs)[0]
    return super().encode(inputs, training=training, mask=None, **kwargs)

  def decode(self, latents, training=None, mask=None, **kwargs):
    py_z = self.predict_factors(latents=latents, training=training, mask=mask)
    # if labeled data is provided, use them in p(x|y,z)
    y = tf.stop_gradient(tf.convert_to_tensor(py_z))
    h = tf.concat(as_tuple(latents) + (y,), axis=-1)
    px_z = super(AnnealingVAE, self).decode(h, training=training, mask=mask,
                                            **kwargs)
    return as_tuple(px_z) + (py_z,)


# ===========================================================================
# Failed system
# ===========================================================================
class semafop(SemafoVAE):
  """ Semafo VAE minimize directly `D(p(y|z_u)||p(y|z_l))` instead
  `D(q(y|z_u)||p(y|z_l))` """

  def __init__(self, mi_coef=1.0, name: str = 'semafop', **kwargs):
    super().__init__(mi_coef=mi_coef, name=name, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask,
                                    n_unsupervised_inputs=1)
    X = X[0]
    mask = tf.reshape(mask, (-1,))
    X_u = tf.boolean_mask(X, tf.logical_not(mask), axis=0)
    X_l = tf.boolean_mask(X, mask, axis=0)
    y_l = tf.boolean_mask(y[0], mask, axis=0)
    ## supervised
    llk_l, kl_l = super(AnnealingVAE, self).elbo_components(X_l,
                                                            training=training)
    P_l, Q_l = self.last_outputs
    ## unsupervised
    llk_u, kl_u = super(AnnealingVAE, self).elbo_components(X_u,
                                                            training=training)
    P_u, Q_u = self.last_outputs
    ## merge the losses
    llk = {}
    for k, v in llk_l.items():
      llk[k] = tf.concat([v, llk_u[k]], axis=0)
    kl = {}
    for k, v in kl_l.items():
      kl[k] = tf.concat([v, kl_u[k]], axis=0)
    ## supervised loss
    py_z = P_l[-1]
    llk[f"llk_{self.labels.name}"] = tf.reduce_mean(self.alpha *
                                                    py_z.log_prob(y_l))
    ## minimizing D(q(y|z_u)||p(y|z_l)) objective
    # calculate the pair-wise distance between q(y|z) and p(y|z)
    qy_z = P_u[-1]
    y = tf.convert_to_tensor(qy_z)
    tf.assert_equal(
      tf.shape(X_u), tf.shape(X_l),
      'Require number of labeled examples equal unlabeled examples')
    kl[f'kl_{self.labels.name}'] = self.alpha * tf.reduce_mean(
      qy_z.log_prob(y) - py_z.log_prob(y))
    # llk_q = tf.expand_dims(qy_z.log_prob(y), axis=-1)
    # llk_p = py_z.log_prob(tf.expand_dims(y, axis=-2))
    # mi_y = tf.reduce_mean(llk_q - llk_p)
    # kl[f'kl_{self.labels.name}'] = self.mi_coef * mi_y
    ## return
    return llk, kl


class semafot(SemafoVAE):
  """SemafoVAE with tied q(y|z) and p(y|z)
  """

  def __init__(self, name: str = 'semafot', **kwargs):
    super().__init__(name=name, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask,
                                    n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    P, Q = self.last_outputs
    px_z = P[:-1]
    py_z = P[-1]
    Q = as_tuple(Q)  # q(z|x)
    ## supervised loss
    llk[f"llk_{self.labels.name}"] = _get_llk_y(py_z, y, mask, self.alpha)
    ## MI objective
    mi_y, mi_z = self._mi_loss(Q, py_z, training=training, mask=mask)
    llk[f'mi_{self.labels.name}'] = mi_y
    for z, mi in zip(as_tuple(self.latents), mi_z):
      llk[f'mi_{z.name}'] = mi
    return llk, kl


class semafod_old(SemafoVAE):
  """Semafo VAE using generative method for density-ratio estimation of
  `D(q(y|z)||p(y|z))`
  """

  def __init__(self, name: str = 'semafod', **kwargs):
    super().__init__(name=name, **kwargs)
    labels_kw = self.labels.get_config()
    labels_kw['name'] += '_q'
    self.labels_p = self.labels
    self.labels_q = DistributionDense(**labels_kw)

  def build(self, input_shape):
    super().build(input_shape)
    self.labels_q(keras.Input(self.latents.event_shape))
    return self

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super(AnnealingVAE, self).elbo_components(X[0],
                                                        mask=mask,
                                                        training=training)
    P, Q = self.last_outputs
    px_z = P[:-1]
    py_z = P[-1]
    Q = as_tuple(Q)  # q(z|x)
    ## supervised loss
    llk[f"llk_{self.labels.name}"] = _get_llk_y(py_z, y, mask, self.alpha)
    ## MI objective
    self.labels = self.labels_q
    mi_y, mi_z = self._mi_loss(Q, py_z, training=training, mask=mask)
    self.labels = self.labels_p
    ## maximizing the MI
    llk[f'mi_{self.labels.name}'] = mi_y
    for z, mi in zip(as_tuple(self.latents), mi_z):
      llk[f'mi_{z.name}'] = mi
    return llk, kl
