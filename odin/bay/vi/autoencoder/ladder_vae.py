import tensorflow as tf
from typing import List, Union
from tensorflow.keras.layers import Layer
from odin.bay.vi.autoencoder.beta_vae import betaVAE, annealedVAE
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.bay.random_variable import RVmeta
from odin.networks import NetConf
from tensorflow_probability.python.distributions import Normal, Independent
from tensorflow_probability.python.layers import DistributionLambda
from odin.backend.interpolation import linear


class LadderMergeDistribution(DistributionLambda):
  """ https://github.com/casperkaae/LVAE/blob/066858a3fb53bb1c529a6f12ae5afb0955722845/run_models.py#L106 """

  def __init__(self, name='LadderMergeDistribution'):
    super().__init__(make_distribution_fn=LadderMergeDistribution.new,
                     name=name)

  @staticmethod
  def new(dists):
    q_e, q_d = dists
    mu_e = q_e.mean()
    mu_d = q_d.mean()
    prec_e = 1 / q_e.variance()
    prec_d = 1 / q_d.variance()
    mu = (mu_e * prec_e + mu_d * prec_d) / (prec_e + prec_d)
    scale = tf.math.sqrt(1 / (prec_e + prec_d))
    dist = Independent(Normal(loc=mu, scale=scale), reinterpreted_batch_ndims=1)
    dist.KL_divergence = q_d.KL_divergence
    return dist


class ladderVAE(betaVAE):
  """ The ladder variational autoencoder

  `X -> E -> E1 -> z1 -> E2 -> z2 -> E3 -> z -> D2 -> z2 -> D1 -> z1 -> D -> X~`

  The return from `encode` method: (z)

  The return from `decode` method: (X~, z2, z1)

  Parameters
  ----------
  ladder_encoder : List[Union[Layer, NetConf]], optional
      the mapping layers between latents in the encoding part
  ladder_decoder : List[Union[Layer, NetConf]], optional
      the mapping layers between latents in the decoding part
  ladder_units : List[Union[Layer, RVmeta]], optional
      number of hidden units for stochastic latents

  References
  ----------
  Sønderby, C.K., Raiko, T., Maaløe, L., Sønderby, S.K., Winther, O., 2016.
    Ladder variational autoencoders, Advances in Neural Information Processing Systems.
    Curran Associates, Inc., pp. 3738–3746.
  https://github.com/casperkaae/LVAE
  """

  def __init__(
      self,
      ladder_encoder: List[Union[Layer, NetConf]] = [
          NetConf([256, 256], batchnorm=True),
          NetConf([256, 256], batchnorm=True)
      ],
      ladder_decoder: List[Union[Layer, NetConf]] = [
          NetConf([256, 256], batchnorm=True),
          NetConf([256, 256], batchnorm=True)
      ],
      ladder_units: List[int] = [32, 32],
      beta=linear(vmin=0., vmax=1., length=1000, delay_in=100),
      **kwargs,
  ):
    super().__init__(beta=beta, **kwargs)
    assert len(ladder_encoder) == len(ladder_decoder) == len(ladder_units)
    self.ladder_encoder = [_parse_layers(i) for i in ladder_encoder]
    self.ladder_decoder = [_parse_layers(i) for i in ladder_decoder]
    self.ladder_latents = [
        _parse_layers(RVmeta(units, 'normal', projection=True, name=f'Z{i}'))
        for i, units in enumerate(ladder_units)
    ]
    self.ladder_merge = LadderMergeDistribution()

  def encode(self,
             inputs,
             training=None,
             mask=None,
             only_encoding=False,
             **kwargs):
    h = self.encoder(inputs, training=training, mask=mask)
    self.last_ladder = []
    for e, z in zip(self.ladder_encoder, self.ladder_latents):
      h = e(tf.convert_to_tensor(h), training=training, mask=mask)
      self.last_ladder.append(z(h))
    return self.latents(h,
                        training=training,
                        mask=mask,
                        sample_shape=self.sample_shape)

  def decode(self,
             latents,
             training=None,
             mask=None,
             only_decoding=False,
             **kwargs):
    h = latents
    outputs = []
    for d, z, qz_e in zip(self.ladder_decoder, self.ladder_latents[::-1],
                          self.last_ladder[::-1]):
      h = d(tf.convert_to_tensor(h), training=training, mask=mask)
      qz_d = z(h, training=training, mask=mask)
      h = self.ladder_merge([qz_e, qz_d])
      outputs.append(h)
    h = self.decoder(tf.convert_to_tensor(h), training=training, mask=mask)
    h = self.observation(h, training=training, mask=mask, **kwargs)
    outputs.append(h)
    return tuple([outputs[-1]] + outputs[:-1])

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    P, Q = self.last_outputs
    for qz, lz in zip(P[1:], self.ladder_latents[::-1]):
      kl[f'kl_{lz.name}'] = self.beta * qz.KL_divergence(analytic=self.analytic,
                                                         reverse=self.reverse,
                                                         sample_shape=None,
                                                         keepdims=True)
    return llk, kl


class stackedVAE():
  pass
