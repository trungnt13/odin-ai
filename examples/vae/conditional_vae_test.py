from __future__ import absolute_import, division, print_function

import collections
import os
import random
import time

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python import keras
from torch import nn
from tqdm import tqdm

from odin import backend as bk
from odin.bay.vi.autoencoder import (conditionalM2VAE, FactorDiscriminator,
                                     ImageNet, create_image_autoencoder)
from odin.bay.vi.utils import marginalize_categorical_labels
from odin.fuel import MNIST, STL10, CelebA, LegoFaces, Shapes3D, dSprites
from odin.networks import (ConditionalEmbedding, ConditionalProjection,
                           RepeaterEmbedding, SkipConnection,
                           get_conditional_embedding, skip_connect)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

ds = MNIST()
train = ds.create_dataset(partition='train', inc_labels=0.5)
test = ds.create_dataset(partition='test', inc_labels=True)
encoder, decoder = create_image_autoencoder(image_shape=(28, 28, 1),
                                            input_shape=(28, 28, 2),
                                            center0=True,
                                            latent_shape=20)
vae = conditionalM2VAE(encoder=encoder,
                       decoder=decoder,
                       conditional_embedding='embed',
                       alpha=0.1 * 10)
vae.fit(train, compile_graph=True, epochs=-1, max_iter=8000, sample_shape=())

x = vae.sample_data(16)
y = vae.sample_labels(16)
m = tf.cast(
    tf.transpose(tf.random.categorical(tf.math.log([[0.5, 0.5]]), y.shape[0])),
    tf.float32)
y = y * m
print(vae.classify(x))

# with label
pX_Z, qZ_X, y = vae([x, y], sample_shape=(2, 3), return_labels=True)
elbo = vae.elbo(x, pX_Z, qZ_X, return_components=False)
print(elbo.shape)

# no label
pX_Z, qZ_X, y = vae(x, sample_shape=(2, 3), return_labels=True)
elbo = vae.elbo(x, pX_Z, qZ_X, return_components=False)
print(elbo.shape)
