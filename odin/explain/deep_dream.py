from __future__ import absolute_import, division, print_function

import inspect
import os
import time

import numpy as np
import tensorflow as tf
from six import string_types
from sklearn.base import BaseEstimator
from tensorflow import keras

from odin.traininglain.helpers import _may_add_batch_dim, get_pretrained_model


@tf.function
def _deep_dream_optimizing(model, img, learning_rate, func_reduce):
  with tf.GradientTape() as tape:
    # This needs gradients relative to `img`
    # `GradientTape` only watches `tf.Variable`s by default
    tape.watch(img)
    # add batch dimension
    layer_activations = model(img)
    # calculate activation of each layer
    losses = []
    for act in layer_activations:
      loss = func_reduce(act)
      losses.append(loss)
    loss = tf.reduce_sum(losses)
  # Calculate the gradient of the loss with respect to the pixels of the input image.
  gradients = tape.gradient(loss, img)
  # Normalize the gradients.
  gradients /= tf.math.reduce_std(gradients) + 1e-8
  # update images, note this is gradient ascent, not descent
  img = img + gradients * learning_rate
  return loss, img


class DeepDream(BaseEstimator):
  r"""
  Arguments:
    model : `keras.Model` or String.
      some options from `keras.applications`: 'DenseNet121', 'DenseNet169',
      'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet',
      'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2',
      'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2', 'VGG16', 'VGG19'.
    layers : list of String. Maximizing the activation of layers with
      given name
    loss_func : callable. Loss function for maximizing (i.e. gradient ascent)
  """

  def __init__(self,
               model='InceptionV3',
               layers=['mixed3', 'mixed5'],
               loss_function=tf.reduce_mean,
               epoch=100,
               learning_rate=0.01,
               octave_scale=1.2,
               octave_step=3,
               model_kwargs={
                   'include_top': False,
                   'weights': 'imagenet'
               },
               verbose=10):
    super().__init__()
    # model settings
    self.model = get_pretrained_model(model, model_kwargs)
    self.input_shape = self.model.input_shape
    self.dtype = self.model.dtype
    # reduction function
    assert callable(loss_function), 'loss_function must be callable'
    self.loss_function = loss_function
    self.set_layers(layers)
    # training settings
    self.epoch = int(epoch)
    self.learning_rate = float(learning_rate)
    self.octave_scale = float(octave_scale)
    self.octave_step = int(octave_step)
    self.verbose = int(verbose)

  def set_layers(self, layers):
    layers = tf.nest.flatten(layers)
    name = self.model.name
    name = name + '_' + '_'.join(layers)
    # Maximize the activations of these layers
    layers = [self.model.get_layer(str(name)).output for name in layers]
    # Create the feature extraction model
    self.dream_model = keras.Model(inputs=self.model.input,
                                   outputs=layers,
                                   name=name)
    return self

  def fit(self, X):
    # add batch dimension if necessary
    X = _may_add_batch_dim(X, self.input_shape)
    X = tf.convert_to_tensor(X, dtype=self.dtype)
    base_shape = tf.cast(tf.shape(X)[1:-1], tf.float32)
    benchmark = []

    for step in range(max(self.octave_step, 1)):
      # resize the image
      if self.octave_scale > 1:
        new_shape = tf.cast(base_shape * (self.octave_scale**step), tf.int32)
        print(" * Resize: old_shape=%s -> new_shape=%s" % (X.shape, new_shape))
        X = tf.image.resize(X, new_shape)
      # optimize the resized image
      for epoch in range(self.epoch):
        start_time = time.time()
        loss, X = _deep_dream_optimizing(self.dream_model, X,
                                         self.learning_rate, self.loss_function)
        benchmark.append(time.time() - start_time)
        if self.verbose > 0 and (epoch + 1) % self.verbose == 0:
          print("Octave#%d Epoch#%d Shape:%s Loss:%.4f (%.2f sec/epoch)" %
                (step, epoch + 1, X.shape, loss, np.mean(benchmark)))
    return X.numpy()
