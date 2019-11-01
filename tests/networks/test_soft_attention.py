from __future__ import absolute_import, division, print_function

import os

import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from six import add_metaclass
from tensorflow.python import keras

from odin import backend as bk
from odin import networks as net
from odin.visual import plot_figure, plot_save

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
torch.manual_seed(8)

Tq = 8
Tv = 18
dim = 20

x = np.random.rand(12, Tq, dim).astype('float32')
y = np.random.rand(12, Tv, dim).astype('float32')
z = np.random.rand(12, Tv, dim).astype('float32')
label = np.random.randint(0, 2, size=12)
query_mask = np.random.randint(0, 2, size=(12, Tq))
value_mask = np.random.randint(0, 2, size=(12, Tv))

att = net.SoftAttention(attention_type='mul',
                        residual=False,
                        scale_tied=True,
                        input_dim=20,
                        num_heads=0,
                        causal=True,
                        heads_depth=3,
                        heads_norm=0.1,
                        dropout=0.5,
                        temporal_dropout=True,
                        heads_output_mode='mean')

t = att(x, mask=[query_mask, query_mask])
print(t.shape)
print(bk.reduce_sum(t))

t = att(x, y, mask=[query_mask, value_mask], training=True)
print(t.shape)
print(bk.reduce_sum(t))

t = att(x, y, z, mask=[query_mask, value_mask])
print(t.shape)
print(bk.reduce_sum(t))

# print(att.layers)
# print(att.weights)
att = net.HardAttention
