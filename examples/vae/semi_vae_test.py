import inspect
from argparse import Namespace
from typing import List, Sequence, Union, Dict, Any

import tensorflow as tf

from odin.bay.vi import VariationalModel
from odin.fuel import get_dataset, ImageDataset
from odin.networks import get_networks, SequentialNetwork, TrainStep
from odin.utils import as_tuple
from utils import *
from tensorflow.keras.layers import *

IMAGE_SHAPE: List[int] = []
ZDIM: int = 0
DS: ImageDataset = None
N_LABELS: float = 0.


# ===========================================================================
# Helpers
# ===========================================================================
def merge_metrics(*metr: Dict[str, Any]):
  metrics = {}
  for m in metr:
    for k, v in m.items():
      metrics[k] = tf.reduce_mean(v)
  return metrics


# ===========================================================================
# Models
# ===========================================================================
class VIB(VariationalModel):

  def __init__(self, encoder, latents, labels):
    super(VIB, self).__init__()
    self.encoder = encoder
    self.latents = latents
    self.labels = labels

  def encode(self, inputs, training=None):
    h = self.encoder(inputs, training=training)
    return self.latents(h, training=training, sample_shape=self.sample_shape)

  def decode(self, latents, training=None):
    return self.labels(tf.convert_to_tensor(latents), training=training)

  def call(self, inputs, training=None, **kwargs):
    qz = self.encode(inputs, training)
    py = self.decode(qz, training)
    return py, qz

  def train_steps(self, inputs, training=True, name='', *args, **kwargs):
    if len(inputs) == 2:
      x_u = None
      x_s, y = inputs
    else:
      x_u, x_s, y = inputs

    def vib():
      py_s, qz_s = self(x_s, training=True)
      # final
      kl = dict(kl_z=qz_s.KL_divergence(analytic=self.analytic,
                                        free_bits=self.free_bits))
      llk = dict(llk_y=py_s.log_prob(y))
      metrics = merge_metrics(llk, kl)
      elbo = self.elbo(llk, kl)
      return -tf.reduce_mean(elbo), metrics

    yield vib


class IVIB(VIB):

  def __init__(self, decoder, latents, labels, observation):
    super().__init__(None, latents, labels)
    self.encoder = SequentialNetwork([
      Dense(1024), BatchNormalization(), Activation(tf.nn.swish),
      Dense(1024), BatchNormalization(), Activation(tf.nn.swish),
      Dense(1024), BatchNormalization(), Activation(tf.nn.swish),
      Dense(512)
    ])
    self.observation = observation
    self.decoder = decoder

  def build(self, input_shape=None):
    return super().build((None, DS.n_labels))

  def call(self, inputs, training=None, **kwargs):
    inputs = as_tuple(inputs)[-1]
    h_e = self.encoder(inputs, training=training)
    qz = self.latents(h_e, training=training, sample_shape=self.sample_shape)
    h_d = self.decoder(tf.convert_to_tensor(qz), training=training)
    px = self.observation(h_d, training=training)
    return px, qz

  def train_steps(self, inputs, training=True, name='', *args, **kwargs):
    if len(inputs) == 2:
      x_u = None
      x_s, y = inputs
    else:
      x_u, x_s, y = inputs

    def ivib():
      px, qz = self(y, training=True)
      # final
      kl = dict(kl_z=qz.KL_divergence(analytic=self.analytic,
                                      free_bits=self.free_bits))
      llk = dict(llk_x=px.log_prob(x_s))
      metrics = merge_metrics(llk, kl)
      elbo = self.elbo(llk, kl)
      return -tf.reduce_mean(elbo), metrics

    yield ivib


class Multitask(VariationalModel):

  def __init__(self, encoder, decoder, latents, observation, labels):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.latents = latents
    self.labels = labels
    self.observation = observation

  def call(self, inputs, training=None, **kwargs):
    inputs = as_tuple(inputs)[0]
    h_e = self.encoder(inputs, training=training, **kwargs)
    qz_x = self.latents(h_e, training=training,
                        sample_shape=self.sample_shape)
    h_d = self.decoder(qz_x, training=training, **kwargs)
    px_z = self.observation(h_d, training=training, **kwargs)
    py_z = self.labels(qz_x, training=training, **kwargs)
    return (px_z, py_z), qz_x

  def train_steps(self, inputs, training=True, name='', *args, **kwargs):
    x_u, x_s, y_s = inputs

    def elbo():
      alpha = tf.constant(10., dtype=self.dtype, name='alpha')
      (pu_x_z, pu_y_z), qu_z_x = self(x_u, training=training)
      (ps_x_z, ps_y_z), qs_z_x = self(x_s, training=training)
      # === 1. LLK
      llk = dict(llk_xu=pu_x_z.log_prob(x_u),
                 llk_xs=ps_x_z.log_prob(x_s),
                 llk_ys=alpha * ps_y_z.log_prob(y_s))
      # === 2. KL
      kw = dict(analytic=self.analytic, free_bits=self.free_bits)
      kl = dict(kl_zu=qu_z_x.KL_divergence(**kw),
                kl_zs=qs_z_x.KL_divergence(**kw))
      # === 3. ELBO
      elbo = self.elbo(llk, kl)
      metrics = merge_metrics(llk, kl)
      return -tf.reduce_mean(elbo), metrics

    yield elbo


class Semafo(Multitask):

  def train_steps(self, inputs, training=True, name='', *args, **kwargs):
    x_u, x_s, y_s = inputs
    tf.assert_equal(tf.shape(x_u)[0], tf.shape(x_s)[0],
                    'Number of supervised sample must be equal '
                    'number of unsupervised samples.')

    def elbo():
      alpha = tf.constant(10., dtype=self.dtype, name='alpha')
      (pu_x_z, qu_y_z), qu_z_x = self(x_u, training=training)
      (ps_x_z, ps_y_z), qs_z_x = self(x_s, training=training)
      # === 1. LLK
      llk = dict(llk_xu=pu_x_z.log_prob(x_u),
                 llk_xs=ps_x_z.log_prob(x_s),
                 llk_ys=alpha * ps_y_z.log_prob(y_s))
      # === 2. KL
      kw = dict(analytic=self.analytic, free_bits=self.free_bits)
      y_u = tf.convert_to_tensor(qu_y_z)
      kl = dict(kl_zu=qu_z_x.KL_divergence(**kw),
                kl_zs=qs_z_x.KL_divergence(**kw),
                kl_yu=qu_y_z.log_prob(y_u) - ps_y_z.log_prob(y_u))
      # === 3. ELBO
      elbo = self.elbo(llk, kl)
      metrics = merge_metrics(llk, kl)
      return -tf.reduce_mean(elbo), metrics

    yield elbo


# ===========================================================================
# Main
# ===========================================================================
def create_model(args: Namespace) -> VariationalModel:
  networks = get_networks(args.ds, zdim=args.zdim, is_semi_supervised=True)
  name = args.vae
  for k, v in globals().items():
    if (isinstance(v, type) and
        issubclass(v, VariationalModel) and
        name.lower() == k.lower()):
      spec = inspect.getfullargspec(v.__init__)
      networks = {k: v for k, v in networks.items()
                  if k in spec.args or k in spec.kwonlyargs}
      model = v(**networks)
      model.build([None] + IMAGE_SHAPE)
      return model
  raise ValueError(f'Cannot find model with name: {name}')


def evaluate(vae: VariationalModel, args: Namespace):
  vae.load_weights(get_model_path(args), raise_notfound=True, verbose=True)


def main(args: Namespace):
  # === 1. get data and set metadata
  ds = get_dataset(args.ds)
  global IMAGE_SHAPE, ZDIM, DS, N_LABELS
  ZDIM = int(args.zdim)
  IMAGE_SHAPE = list(ds.shape)
  DS = ds
  N_LABELS = float(args.y)
  # === 2. create model
  vae = create_model(args)
  if args.eval:
    vae.load_weights(get_model_path())
    evaluate(vae, args)
  else:
    train(vae, ds, args,
          label_percent=args.y,
          oversample_ratio=0.5)


if __name__ == '__main__':
  set_cfg('/home/trung/exp/multitask')
  main(get_args(dict(y=(float, 100))))
