from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32,seed=5218'

import numpy as np
import tensorflow as tf

from odin import fuel as F, nnet as N, backend as K
from odin.utils import ctext
from odin.config import get_rng

# ===========================================================================
# CONST
# ===========================================================================
learning_rate = 0.01
epoch = 15
batch_size = 128

# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.FMNIST.load()
labels = ds['labels']
nb_classes = len(labels)

X_train, y_train = ds['X_train'][:] / 255, ds['y_train']
X_test, y_test = ds['X_test'][:] / 255, ds['y_test']

ids = get_rng().permutation(X_train.shape[0])
X_train = X_train[ids]
y_train = y_train[ids]

X_valid = X_train[:10000]
y_valid = y_train[:10000]

X_train = X_train[10000:]
y_train = y_train[10000:]

print("Labels:", ctext(labels, 'cyan'))
print("Training:", ctext(X_train.shape, 'cyan'))
print("Validation:", ctext(X_valid.shape, 'cyan'))
print("Testing:", ctext(X_test.shape, 'cyan'))
# ====== placeholder ====== #
input_shape = (None,) + X_train.shape[1:]
X = K.placeholder(shape=input_shape, dtype='float32', name='X')
y = K.placeholder(shape=(None,), dtype='float32', name='y')
y_onehot = tf.one_hot(indices=tf.cast(y, 'int32'), depth=nb_classes)
# ===========================================================================
# Create O.D.I.N network
# ===========================================================================
odin_net = N.Sequence([
    N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    N.Conv(num_filters=32, filter_size=5, pad='same', activation=tf.nn.relu),
    N.Pool(pool_size=2, strides=2, mode='max'),

    N.Conv(num_filters=64, filter_size=5, pad='same', activation=tf.nn.relu),
    N.Pool(pool_size=2, strides=2, mode='max'),

    N.Flatten(outdim=2),
    N.Dense(num_units=1024, activation=tf.nn.relu),
    N.Dropout(level=0.2),
    N.Dense(num_units=nb_classes)
], debug=True, name="ODIN_fmnist")
y1_logits = odin_net(X)
y1_probs = tf.nn.softmax(y1_logits)
y1_pred = tf.argmax(y1_probs, axis=-1)
# ===========================================================================
# Create tensorflow network
# ===========================================================================
inputs = N.Dimshuffle(pattern=(0, 1, 2, 'x'))(X)
conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5],
                         padding="same", activation=tf.nn.relu)
#pooling layer 1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#convolution layer 2
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                         padding="same", activation=tf.nn.relu)
#pooling layer 1
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#flatten the output volume of pool2 into a vector
pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
#dense layer
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#dropout regularization
dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=K.is_training())
#logits layer
y2_logits = tf.layers.dense(inputs=dropout, units=nb_classes)
y2_probs = tf.nn.softmax(y2_logits)
y2_pred = tf.argmax(y2_probs, axis=-1)
# ===========================================================================
# Create objectives
# ===========================================================================
#loss
loss1 = tf.losses.softmax_cross_entropy(y_onehot, y1_logits)
loss2 = tf.losses.softmax_cross_entropy(y_onehot, y2_logits)

acc1 = K.metrics.categorical_accuracy(y_onehot, y1_probs)
acc2 = K.metrics.categorical_accuracy(y_onehot, y2_probs)

cm1 = K.metrics.confusion_matrix(y, y1_pred, labels=labels)
cm2 = K.metrics.confusion_matrix(y, y2_pred, labels=labels)
# optimizer
train_op1 = K.optimizers.Adam(lr=learning_rate).minimize(loss1)
train_op2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2)
