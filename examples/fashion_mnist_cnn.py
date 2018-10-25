#
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'gpu,float32,seed=5218'
import timeit

import numpy as np
import tensorflow as tf

from odin import fuel as F, nnet as N, backend as K
from odin import training
from odin.utils import ctext, get_modelpath, get_exppath
from odin.config import get_rng
from odin.ml import evaluate
from odin.visual import plot_save, plot_figure
from odin.stats import freqcount, classification_diagnose
# ===========================================================================
# CONST
# ===========================================================================
learning_rate = 0.01
epoch = 15
batch_size = 128
MODEL_PATH_ODIN = get_modelpath(name='fmnist_odin', override=True)
MODEL_PATH_TF = get_modelpath(name='fmnist_tf', override=True)
FIG_PATH = get_exppath(tag='FMNIST', override=True)

TRAINING_ODIN = True
TRAINING_TF = True
# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.FMNIST_original.load()
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
K.initialize_all_variables()
# ====== create function ====== #
f_pred1 = K.function(inputs=X, outputs=y1_probs, training=False)
f_pred2 = K.function(inputs=X, outputs=y2_probs, training=False)
# ===========================================================================
# Training in ODIN
# ===========================================================================
if TRAINING_ODIN:
  trainer = training.MainLoop(batch_size=batch_size, shuffle_level=0, seed=None,
                              allow_rollback=False, labels=labels, verbose=2)
  trainer.set_train_task(func=K.function(inputs=[X, y], outputs=[loss1, acc1],
                                         updates=train_op1, training=True),
                         data=(X_train, y_train),
                         epoch=epoch, name='train')
  trainer.run()
# ===========================================================================
# Training in Tensorflow
# ===========================================================================
if TRAINING_TF:
  sess = K.get_session()
  #Divide input training set into mini batches of size batch_size.
  #If the total number of training examles is not exactly divisible by batch_size,
  #the last batch will have less number of examples than batch_size.
  total_size = X_train.shape[0]
  number_of_batches = int(total_size / batch_size)
  print("Training:Start")
  for e in range(epoch):
    start_time = timeit.default_timer()
    epoch_cost = 0
    epoch_accuracy = 0
    for i in range(number_of_batches):
      mini_x = X_train[i * batch_size:(i + 1) * batch_size, :, :]
      mini_y = y_train[i * batch_size:(i + 1) * batch_size]
      _, cost = sess.run([train_op2, loss2],
          feed_dict={X: mini_x,
                     y: mini_y,
                     K.is_training(): True})
      train_accuracy = sess.run(acc2,
          feed_dict={X: mini_x,
                     y: mini_y,
                     K.is_training(): False})
      epoch_cost += cost
      epoch_accuracy += train_accuracy
    #If the total number of training examles is not exactly divisible by batch_size,
    #we have one more batch of size (total_size - number_of_batches*batch_size)
    if total_size % batch_size != 0:
      mini_x = X_train[number_of_batches * batch_size:total_size, :, :]
      mini_y = y_train[number_of_batches * batch_size:total_size]
      _, cost = sess.run([train_op2, loss2],
          feed_dict={X: mini_x,
                     y: mini_y,
                     K.is_training(): True})
      train_accuracy = sess.run(acc2,
          feed_dict={X: mini_x,
                     y: mini_y,
                     K.is_training(): False})
      epoch_cost += cost
      epoch_accuracy += train_accuracy
    epoch_cost /= number_of_batches
    if total_size % batch_size != 0:
      epoch_accuracy /= (number_of_batches + 1)
    else:
      epoch_accuracy /= number_of_batches
    print("Epoch: {} Cost: {} accuracy: {} time: {}(s)".format(
        e + 1, np.squeeze(epoch_cost), epoch_accuracy,
        timeit.default_timer() - start_time))
# ===========================================================================
# Evaluation
# ===========================================================================
def plot_diagonose(x, yt, yp, name):
  N = 25
  plot_figure(nrow=N, ncol=N)
  for i, ((true, pred), samples) in enumerate(classification_diagnose(
      X=x, y_true=yt, y_pred=yp, return_list=True, top_n=N, num_samples=N)):
    for j in range(N):
      if j >= len(samples):
        continue
      s = samples[j]
      ax = plt.subplot(N, N, i * N + j + 1)
      ax.imshow(s, cmap=plt.cm.Greys_r)
      ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
      if j == 0:
        ax.set_title('True:%s - Pred:%s' % (labels[true], labels[pred]),
                     fontsize=10)
  plt.subplots_adjust(hspace=0.4, wspace=0.001)
  plot_save(os.path.join(FIG_PATH, name))
# ====== odin ====== #
y1_valid = f_pred1(X_valid)
y1_test = f_pred1(X_test)
evaluate(y_true=y_valid, y_pred_proba=y1_valid, labels=labels, title='ODIN-Valid',
         path=os.path.join(FIG_PATH, 'odin_valid.pdf'))
evaluate(y_true=y_test, y_pred_proba=y1_test, labels=labels, title='ODIN-Test',
         path=os.path.join(FIG_PATH, 'odin_test.pdf'))
plot_diagonose(X_valid, yt=y_valid, yp=y1_valid, name='odin_valid_diag.pdf')
plot_diagonose(X_test, yt=y_test, yp=y1_test, name='odin_test_diag.pdf')
# ====== tensorflow ====== #
y2_valid = f_pred2(X_valid)
y2_test = f_pred2(X_test)
evaluate(y_true=y_valid, y_pred_proba=y2_valid, labels=labels, title='TF-Valid',
         path=os.path.join(FIG_PATH, 'tf_valid.pdf'))
evaluate(y_true=y_test, y_pred_proba=y2_test, labels=labels, title='TF-Test',
         path=os.path.join(FIG_PATH, 'tf_test.pdf'))
plot_diagonose(X_valid, yt=y_valid, yp=y2_valid, name='tf_valid_diag.pdf')
plot_diagonose(X_test, yt=y_test, yp=y2_test, name='tf_test_diag.pdf')
