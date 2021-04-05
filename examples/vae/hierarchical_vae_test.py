import os
from argparse import Namespace
from functools import partial
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow_probability.python.distributions import Distribution, Bernoulli, \
  Independent
from tensorflow_probability.python.layers import DistributionLambda
from tqdm import tqdm

from odin.bay import VariationalAutoencoder, HierarchicalVAE, \
  BiConvLatents
from odin.bay.distributions import QuantizedLogistic
from odin.bay.layers import MVNDiagLatents
from odin.fuel import ImageDataset, get_dataset
from odin.networks import CenterAt0
from odin.networks import resnets as rn
from odin.utils import minibatch
from utils import *

# ===========================================================================
# Constants and helpers
# ===========================================================================
IMAGE_SHAPE = [32, 32, 3]


def find(name: str, layers: List[Layer]) -> Tuple[int, Layer]:
  for idx, l in enumerate(layers):
    if l.name == name:
      return idx, l
  raise ValueError(f'Cannot find layer with name {name}')


# ===========================================================================
# Layers
# * Upsampling and pooling is not recommended
# * Residual make training more stable but promoting posterior collapse
# ===========================================================================
RES_KW = dict(filters_in=64, filters_out=64, expand_ratio=2.,
              batchnorm=False, activation=tf.nn.elu)
Skip = partial(rn.Skip, coef=1.0)
Tonv = partial(Conv2DTranspose,
               # filters=64,
               # kernel_size=3,
               # strides=1,
               padding='same',
               activation=tf.nn.elu,
               kernel_initializer='he_normal')
Conv = partial(Conv2D,
               # filters=64,
               # kernel_size=3,
               # strides=1,
               padding='same',
               activation=tf.nn.elu,
               kernel_initializer='he_normal')


def latents(zdim):
  return MVNDiagLatents(zdim)


def bi_latents(idx, filters, kernel, strides, e_layers, d_layers, name):
  i1, d = find(f'd{idx}', d_layers)
  i2, e = find(f'e{idx}', e_layers)
  d = BiConvLatents(d, encoder=e, filters=filters,
                    kernel_size=kernel, strides=strides,
                    name=name)
  d_layers[i1] = d


def observation(distribution: str):
  if distribution == 'qlogistic':
    n_params = 2
    obs = DistributionLambda(
      lambda params: QuantizedLogistic(
        *[
          # loc
          p if i == 0 else
          # Ensure scales are positive and do not collapse to near-zero
          tf.nn.softplus(p) + tf.cast(tf.exp(-7.), tf.float32)
          for i, p in enumerate(tf.split(params, 2, -1))],
        low=0,
        high=255,
        inputs_domain='sigmoid',
        reinterpreted_batch_ndims=3),
      convert_to_tensor_fn=Distribution.sample,
      name='image')
  elif distribution == 'bernoulli':
    n_params = 1
    obs = DistributionLambda(
      lambda params: Independent(Bernoulli(logits=params, dtype=tf.float32),
                                 len(IMAGE_SHAPE)),
      convert_to_tensor_fn=Distribution.sample,
      name='image')
  else:
    raise NotImplementedError
  return n_params, obs


# === 0. Vanilla VAE

# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x4x2,64x3x1,32x4x2,32x3x1
def model_vae1(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  return VariationalAutoencoder(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# same as vae1, with free_bits=0.5
def model_vae1a(zdim, dist):
  vae = model_vae1(zdim, dist)
  vae: VariationalAutoencoder
  vae.free_bits = 0.5
  return vae


# Use Pooling and Upsampling
# 32x3x1,d,32x3x1,64x3x1,d,64x3x1
# 64x3x1,u,64x3x1,32x3x1,u,32x3x1
def model_vae2(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    AveragePooling2D(),  # (16, 16)
    Conv(32, 3, 1, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    AveragePooling2D(),  # (8, 8)
    Conv(64, 3, 1, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Conv(64, 3, 1, name='d3'),  # (8, 8, 64)
    UpSampling2D(),  # (16, 16)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Conv(32, 3, 1, name='d1'),  # (16, 16, 32)
    UpSampling2D(),  # (32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),
  ]
  return VariationalAutoencoder(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# 32x3x1,32x4x2,64x3x1,64x4x2,128x3x1,128x4x2
# 128x4x2,128x3x1,64x4x2,64x3x1,32x4x2,32x3x1
def model_vae3(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e1'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e2'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e3'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e4'),  # (8, 8, 64)
    Conv(128, 3, 1, name='e5'),  # (8, 8, 128)
    Conv(128, 4, 2, name='e6'),  # (4, 4, 128)
    Flatten(),
    Dense(512, name='encoder_proj'),
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([4, 4, 32]),
    Tonv(128, 4, 2, name='d6'),  # (8, 8, 128)
    Conv(128, 3, 1, name='d5'),  # (8, 8, 128)
    Tonv(64, 4, 2, name='d4'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d3'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d2'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d1'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  return VariationalAutoencoder(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# === 1. Bidirectional Latents VAE

# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,64x8x4,------,32x8x4
def model_bvae1(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# AvgPool encoder, ConvTrans decoder
# 32x3x1,d,32x3x1,64x3x1,d,64x3x1
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,64x8x4,------,32x8x4
def model_bvae1a(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    AveragePooling2D(),  # (16, 16)
    Conv(32, 3, 1, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    AveragePooling2D(),  # (8, 8)
    Conv(64, 3, 1, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# ConvTrans encoder, UpSampling decoder
# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x3x1,u,64x3x1,32x3x1,u,32x3x1
# ------,64x8x4,------,32x8x4
def model_bvae1b(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Conv(64, 3, 1, name='d3'),  # (8, 8, 64)
    UpSampling2D(),  # (16, 16)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Conv(32, 3, 1, name='d1'),  # (16, 16, 32)
    UpSampling2D(),  # (32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# Same as 1b, but use 4x4 kernel for ConvTrans
# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x3x1,u,64x4x1,32x3x1,u,32x4x1
# ------,64x8x4,------,32x8x4
def model_bvae1bi(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Conv(64, 3, 1, name='d3'),  # (8, 8, 64)
    UpSampling2D(),  # (16, 16)
    Conv(64, 4, 1, name='d2'),  # (16, 16, 64)
    Conv(32, 3, 1, name='d1'),  # (16, 16, 32)
    UpSampling2D(),  # (32, 32)
    Conv(32, 4, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# Same as 1b, use extra layer so the latent not adjacent to upsampling
# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x3x1,u,64x4x1,64x3x1,u,32x4x1,32x3x1
# ---------------,64x8x4,--------,32x8x4
def model_bvae1bii(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Conv(64, 3, 1, name='d3'),  # (8, 8, 64)
    UpSampling2D(),  # (16, 16)
    Conv(64, 4, 1),
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    UpSampling2D(),  # (32, 32)
    Conv(32, 4, 1),
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# 64x3x1,d,64x3x1,64x3x1,d,64x3x1 (skip)
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,64x8x4,------,32x8x4
def model_bvae1c(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(64, 3, 1, name='e0'),  # (32, 32, 64)
    AveragePooling2D(),  # (16, 16)
    Skip(Conv(64, 3, 1, name='e1')),  # (16, 16, 64)
    Skip(Conv(64, 3, 1, name='e2')),  # (16, 16, 64)
    AveragePooling2D(),  # (8, 8)
    Skip(Conv(64, 3, 1, name='e3')),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(64, 4, 2, name='d1'),  # (32, 32, 64)
    Conv(64, 3, 1, name='d0'),  # (32, 32, 64)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# same as 1c, but without skip
# 64x3x1,d,64x3x1,64x3x1,d,64x3x1
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,64x8x4,------,32x8x4
def model_bvae1ci(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(64, 3, 1, name='e0'),  # (32, 32, 64)
    AveragePooling2D(),  # (16, 16)
    Conv(64, 3, 1, name='e1'),  # (16, 16, 64)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    AveragePooling2D(),  # (8, 8)
    Conv(64, 3, 1, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(64, 4, 2, name='d1'),  # (32, 32, 64)
    Conv(64, 3, 1, name='d0'),  # (32, 32, 64)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# Use Pooling and Upsampling
# 32x3x1,d,32x3x1,64x3x1,d,64x3x1
# 64x3x1,u,64x3x1,32x3x1,u,32x3x1
# --------,64x8x4,--------,32x8x4
def model_bvae2(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    AveragePooling2D(),  # (16, 16)
    Conv(32, 3, 1, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    AveragePooling2D(),  # (8, 8)
    Conv(64, 3, 1, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Conv(64, 3, 1, name='d3'),  # (8, 8, 64)
    UpSampling2D(),  # (16, 16)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Conv(32, 3, 1, name='d1'),  # (16, 16, 32)
    UpSampling2D(),  # (32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# Use feature map for downsampling latents
# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,4x3x1,------,2x3x1
def model_bvae3(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 4, 3, 1, e_layers, d_layers, 'latents1')
  bi_latents(0, 2, 3, 1, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# Use both feature map and big kernel
# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,16x4x2,------,8x4x2
def model_bvae4(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512, name='encoder_proj')
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512, name='decoder_proj'),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 16, 4, 2, e_layers, d_layers, 'latents1')
  bi_latents(0, 8, 4, 2, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# === 2. Parallel Latents VAE

def model_pvae(zdim, dist):
  n_params, obs = observation(dist)
  return HierarchicalVAE(
    encoder=Sequential(encoder(), 'Encoder'),
    decoder=Sequential(decoder(zdim, n_params), 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# ===========================================================================
# Main
# ===========================================================================
def create_vae(args) -> Tuple[VariationalAutoencoder, ImageDataset]:
  dist = 'bernoulli' if args.ds == 'shapes3d' else 'qlogistic'
  key = f'model_{args.vae}'
  if key not in globals():
    raise ValueError(f'Cannot find model with name: {args.vae}')
  model = globals()[key](args.zdim, dist)
  model: VariationalAutoencoder
  model.build([None] + IMAGE_SHAPE)
  ds = get_dataset(args.ds)
  ds: ImageDataset
  return model, ds


def evaluate(model, ds, args):
  model.load_weights(get_model_path(args),
                     raise_notfound=True,
                     verbose=True)

  test = ds.create_dataset('test', batch_size=32)
  # === 1. marginalized llk
  n_mcmc = 100
  llk = []
  kl = []
  for x in tqdm(test.take(10)):
    qz = model.encode(x, training=False)
    batch_llk = []
    for s, e in minibatch(8, n_mcmc):
      n = e - s
      z = qz.sample(n)
      z = tf.reshape(z, (-1, z.shape[-1]))
      px = model.decode(z, training=False)
      # llk
      batch_llk.append(
        tf.reshape(px.log_prob(tf.tile(x, (n, 1, 1, 1))), (n, -1)))
      # kl
      exit()
    batch_llk = tf.concat(batch_llk, 0)
    llk.append(batch_llk)
  llk = tf.concat(llk, axis=-1)
  print(llk.shape)
  print('LLK:', tf.reduce_mean(tf.reduce_logsumexp(llk, 0)))


def main(args: Namespace):
  assert args.ds in ('shapes3d', 'cifar10', 'svhn')
  model, ds = create_vae(args)
  if args.eval:
    evaluate(model, ds, args)
  else:
    train(model, ds, args)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  set_cfg(os.path.expanduser('~/exp/hierarchical'))
  main(get_args())
