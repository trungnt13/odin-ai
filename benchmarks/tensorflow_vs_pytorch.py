# This code testing the performance of training simple autoencoder
# in 3 scenarios:
#  * Pytorch auto differantiation
#  * Tensorflow 2.0 eager excution
#  * Tensorlfow 2.0 autograph
# NOTE: the number of epoch must be big enough to provide a
# subjective results since the initialization time of Tensorflow
# is much greater than Pytorch
#
# With default configuration, running on NVIDIA GTX 1080:
#  * Pytorch      [128] total:24.67  avr:0.31  min:0.23  max:0.41
#                 [512] total:7.13   avr:0.09  min:0.07  max:0.12
#  * TF-eager     [128] total:63.47  avr:0.79  min:0.74  max:1.47
#                 [512] total:16.64  avr:0.21  min:0.18  max:0.94
#  * TF-autograph [128] total:13.25  avr:0.17  min:0.09  max:1.49
#                 [512] total:5.27   avr:0.07  min:0.04  max:1.51
# [..] is the batch size
#
# Personal conclusion:
#  * Initialization time of TF need to be improved
#  * TF-eager still an inefficient way to run experiment
#  * TF-autograph is good and easy way, but still get a lot of issues
from __future__ import absolute_import, division, print_function

import multiprocessing as mpi
import os
import time

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(8)


# ====== helper ====== #
def batching(batch_size, n):
  start, end = int(0), int(n)
  batch_size = int(batch_size)
  return ((i, min(i + batch_size, end))
          for i in range(start, end + batch_size, batch_size)
          if i < end)


# ====== configuration ====== #
batch_size = 512
n_epoch = 80
n_dim = 120
network_configs = [n_dim, 1024, 512, 256, 128, 64, 128, 256, 512, 1024, n_dim]
learning_rate = 1e-4
train = np.random.rand(12000, n_dim).astype('float32')


# ===========================================================================
# Benchmark functions
# ===========================================================================
def pytorch_benchmark():
  import torch
  try:
    device = torch.device('cuda')
  except:
    raise RuntimeError("The test require GPU to run!")

  layers = []
  for nin, nout in zip(network_configs, network_configs[1:]):
    layers.append(torch.nn.Linear(nin, nout))
    layers.append(torch.nn.ReLU())
  model = torch.nn.Sequential(*layers)
  print("#Layers:", len(list(model.modules())) - 1)
  model.cuda()
  loss_fn = torch.nn.MSELoss(reduction='sum')

  epoch_time = []
  for e in range(n_epoch):
    total_loss = 0
    start_time = time.time()

    for nbatch, (start, end) in enumerate(batching(batch_size, train.shape[0])):
      x = torch.from_numpy(train[start:end]).float().to(device)
      x_pred = model(x)
      loss = loss_fn(x_pred, x)
      total_loss += loss
      model.zero_grad()
      loss.backward()
      with torch.no_grad():
        for param in model.parameters():
          param -= learning_rate * param.grad

    epoch_time.append(time.time() - start_time)
    total_loss = total_loss.cpu().detach().numpy()
    if (e + 1) % 10 == 0 or e == 0:
      print("Epoch%d:" % (e + 1), '%.2f' % (total_loss / (nbatch + 1)))
  print("[PyTorch] total:%.2f  avr:%.2f  min:%.2f  max:%.2f" %
        (np.sum(epoch_time), np.mean(epoch_time), np.min(epoch_time),
         np.max(epoch_time)))


def tfeager_benchmark():
  import tensorflow as tf
  from tensorflow.python.keras import Sequential
  from tensorflow.python.keras.layers import Dense, Activation

  assert tf.test.is_gpu_available()

  with tf.device("gpu:0"):
    layers = []
    for nunits in network_configs[1:]:
      layers.append(Dense(nunits, activation='linear'))
      layers.append(Activation('relu'))
    model = Sequential(layers)
    print("#Layers:", len(model.layers))
    loss_fn = lambda x, y: tf.reduce_sum(tf.square(x - y))

    epoch_time = []
    for e in range(n_epoch):
      total_loss = 0
      start_time = time.time()

      for nbatch, (start, end) in enumerate(batching(batch_size,
                                                     train.shape[0])):
        x = train[start:end]
        with tf.GradientTape() as grad:
          x_pred = model(x)
          loss = loss_fn(x_pred, x)
        total_loss += loss
        for p, g in zip(model.weights, grad.gradient(loss, model.weights)):
          p.assign_sub(learning_rate * g)

      epoch_time.append(time.time() - start_time)
      if (e + 1) % 10 == 0 or e == 0:
        print("Epoch%d:" % (e + 1), '%.2f' % (total_loss / (nbatch + 1)))
    print("[TF-Eager] total:%.2f  avr:%.2f  min:%.2f  max:%.2f" %
          (np.sum(epoch_time), np.mean(epoch_time), np.min(epoch_time),
           np.max(epoch_time)))


def tfautograph_benchmark():
  import tensorflow as tf
  from tensorflow.python.keras import Sequential
  from tensorflow.python.keras.layers import Dense, Activation

  assert tf.test.is_gpu_available()

  with tf.device("gpu:0"):
    layers = []
    for nunits in network_configs[1:]:
      layers.append(Dense(nunits, activation='linear'))
      layers.append(Activation('relu'))
    model = Sequential(layers)
    print("#Layers:", len(model.layers))
    loss_fn = lambda x, y: tf.reduce_sum(tf.square(x - y))

    @tf.function
    def train_fn(x):
      with tf.GradientTape() as grad:
        x_pred = model(x)
        loss = loss_fn(x_pred, x)
      for p, g in zip(model.weights, grad.gradient(loss, model.weights)):
        p.assign_sub(learning_rate * g)
      return loss

    epoch_time = []
    for e in range(n_epoch):
      total_loss = 0
      start_time = time.time()

      for nbatch, (start, end) in enumerate(batching(batch_size,
                                                     train.shape[0])):
        x = train[start:end]
        loss = train_fn(x)
        total_loss += loss

      epoch_time.append(time.time() - start_time)
      if (e + 1) % 10 == 0 or e == 0:
        print("Epoch%d:" % (e + 1), '%.2f' % (total_loss / (nbatch + 1)))
    print("[TF-Autograph] total:%.2f  avr:%.2f  min:%.2f  max:%.2f" %
          (np.sum(epoch_time), np.mean(epoch_time), np.min(epoch_time),
           np.max(epoch_time)))


# ===========================================================================
# Main code
# ===========================================================================
# for fn in (pytorch_benchmark, tfeager_benchmark, tfautograph_benchmark):
#   p = mpi.Process(target=fn)
#   p.start()
#   p.join()
# pytorch_benchmark()
# tfeager_benchmark()
# tfautograph_benchmark()
