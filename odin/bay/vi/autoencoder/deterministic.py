import tensorflow as tf
from odin.bay.vi.autoencoder.variational_autoencoder import VariationalAutoencoder, LayerCreator
from odin.bay.random_variable import RVmeta
from odin.utils import as_tuple
from odin.bay.layers import DistributionDense, VectorDeterministicLayer, DeterministicLayer


# ===========================================================================
# Helper
# ===========================================================================
def _mse_log_prob(x_pred, x_true):
  return -tf.losses.mean_squared_error(y_true=x_true, y_pred=x_pred.mean())


def _copy_distribution_dense(p: DistributionDense, posterior, posterior_kwargs):
  return DistributionDense(
      event_shape=p.event_shape,
      posterior=posterior,
      posterior_kwargs=posterior_kwargs,
      activation=p.activation,
      use_bias=p.use_bias,
      dropout=p._dropout,
      projection=p.projection,
      flatten_inputs=p.flatten_inputs,
      kernel_initializer=p.kernel_initializer,
      bias_initializer=p.bias_initializer,
      kernel_regularizer=p.kernel_regularizer,
      bias_regularizer=p.bias_regularizer,
      activity_regularizer=p.activity_regularizer,
      kernel_constraint=p.kernel_constraint,
      bias_constraint=p.bias_constraint,
  )


# ===========================================================================
# Main
# ===========================================================================
class _DeterministicLatents(VariationalAutoencoder):
  """The vanilla autoencoder could be interpreted as a Variational Autoencoder with
  `vector deterministic` distribution for latent codes."""

  def __init__(
      self,
      latents: LayerCreator = RVmeta(64,
                                     'vdeterministic',
                                     projection=True,
                                     name="Latents"),
      dropout: float = 0.3,
      activity_l2coeff: float = 0.,
      activity_l1coeff: float = 0.,
      weights_l2coeff: float = 0.,
      weights_l1coeff: float = 0.,
      **kwargs,
  ):
    deterministic_latents = []
    for qz in as_tuple(latents):
      if isinstance(qz, RVmeta):
        qz.posterior = 'vdeterministic'
      elif (isinstance(qz, DistributionDense) and
            qz.posterior != VectorDeterministicLayer):
        qz = _copy_distribution_dense(qz,
                                      posterior=VectorDeterministicLayer,
                                      posterior_kwargs=dict(name='latents'))
      deterministic_latents.append(qz)
    if len(deterministic_latents) == 1:
      deterministic_latents = deterministic_latents[0]
    super().__init__(latents=latents, **kwargs)
    self.dropout = tf.convert_to_tensor(dropout,
                                        dtype_hint=self.dtype,
                                        name='dropout')
    self.activity_l2coeff = float(activity_l2coeff)
    self.activity_l1coeff = float(activity_l1coeff)
    self.weights_l2coeff = float(weights_l2coeff)
    self.weights_l1coeff = float(weights_l1coeff)

  def encode(self,
             inputs,
             training=None,
             mask=None,
             only_encoding=False,
             **kwargs):
    if training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)
    return super().encode(inputs,
                          training=training,
                          mask=mask,
                          only_encoding=only_encoding,
                          **kwargs)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      training=training,
                                      mask=mask,
                                      **kwargs)
    # this make sure no KL is leaking
    kl = {k: 0. for k, v in kl.items()}
    # add L2 and L1 normalization
    P, Q = self.last_outputs
    for layer, qz_x in zip(as_tuple(self.latents), as_tuple(Q)):
      name = layer.name
      if self.activity_l2coeff > 0:
        kl[f'{name}_activityl2'] = self.activity_l2coeff * tf.linalg.norm(qz_x,
                                                                          ord=2)
      if self.activity_l1coeff > 0:
        kl[f'{name}_activityl1'] = self.activity_l1coeff * tf.linalg.norm(qz_x,
                                                                          ord=1)
      if self.weights_l2coeff > 0:
        kl[f'{name}_weightsl2'] = self.weights_l2coeff * tf.linalg.norm(
            layer.trainable_weights[0], ord=2)
      if self.weights_l1coeff > 0:
        kl[f'{name}_weightsl1'] = self.weights_l1coeff * tf.linalg.norm(
            layer.trainable_weights[0], ord=1)
    return llk, kl


class Autoencoder(_DeterministicLatents):
  """Denoising Autoencoder using Mean-squared error loss"""

  def __init__(
      self,
      observation: LayerCreator = RVmeta((28, 28, 1),
                                         'bernoulli',
                                         projection=True,
                                         name='image'),
      name='Autoencoder',
      **kwargs,
  ):
    deterministic_obs = []
    for px in as_tuple(observation):
      if isinstance(px, RVmeta):
        px.posterior = 'deterministic'
        px.kwargs['log_prob'] = _mse_log_prob
        px.kwargs['reinterpreted_batch_ndims'] = len(observation.event_shape)
      elif (isinstance(px, DistributionDense) and
            px.posterior != DeterministicLayer):
        px = _copy_distribution_dense(px,
                                      posterior=DeterministicLayer,
                                      posterior_kwargs=dict(
                                          log_prob=_mse_log_prob, name='MSE'))
      deterministic_obs.append(px)
    if len(deterministic_obs) == 1:
      deterministic_obs = deterministic_obs[0]
    super().__init__(observation=deterministic_obs, name=name, **kwargs)


class DistEncoder(_DeterministicLatents):
  """Distribution Encoder with deterministic latents and modelled outputs
  distribution"""

  def __init__(self, name='DistEncoder', **kwargs):
    super().__init__(name=name, **kwargs)
