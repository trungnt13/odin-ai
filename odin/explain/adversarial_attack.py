from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator

from odin.traininglain.helpers import _may_add_batch_dim, get_pretrained_model


# it seems adding tf.function don't improve much performance
# @tf.function
def _adversarial_optimizing(model, X, y, X_org, loss_function, l2_norm, l1_norm,
                            learning_rate):
  with tf.GradientTape() as tape:
    tape.watch(X)
    y_pred = model(X)
    loss = loss_function(y, y_pred)
    if l2_norm > 0:
      loss += l2_norm * tf.norm(X - X_org, ord=2)
    if l1_norm > 0:
      loss += l1_norm * tf.norm(X - X_org, ord=1)
  gradients = tape.gradient(loss, X)
  # Normalize the gradients.
  gradients /= tf.math.reduce_std(gradients) + 1e-8
  # gradient descent
  X = X - gradients * learning_rate
  return loss, X


class AdversarialAttack(BaseEstimator):

  def __init__(self,
               model,
               loss_function=tf.losses.sparse_categorical_crossentropy,
               model_kwargs={'include_top': True},
               epoch=80,
               l2_norm=0.0,
               l1_norm=0.0,
               learning_rate=0.01,
               verbose=10):
    super().__init__()
    self.model = get_pretrained_model(model, model_kwargs)
    self.input_shape = self.model.input_shape
    self.dtype = self.model.dtype
    self.loss_function = loss_function
    # training settings
    self.learning_rate = learning_rate
    self.epoch = epoch
    self.l2_norm = l2_norm
    self.l1_norm = l1_norm
    self.verbose = int(verbose)

  def fit(self, X, y):
    X = _may_add_batch_dim(X, self.input_shape)
    X = tf.convert_to_tensor(X, dtype=self.dtype)
    X_org = X
    y = tf.convert_to_tensor(y, dtype=self.model.output.dtype)
    benchmark = []

    for epoch in range(self.epoch):
      start_time = time.time()
      loss, X = _adversarial_optimizing(self.model, X, y, X_org,
                                        self.loss_function, self.l2_norm,
                                        self.l1_norm, self.learning_rate)
      benchmark.append(time.time() - start_time)
      if self.verbose > 0 and (epoch + 1) % self.verbose == 0:
        print("Epoch#%d Loss:%.4f (%.2f sec/epoch)" %
              (epoch + 1, loss, np.mean(benchmark)))
    return X.numpy()
