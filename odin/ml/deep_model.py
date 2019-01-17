from __future__ import print_function, division, absolute_import

import time
import pickle

import numpy as np
import tensorflow as tf

from odin.utils import (is_number, uuid, batching, as_tuple, Progbar,
                        one_hot, wprint, ctext, is_string, uuid)
from odin import (backend as K, nnet as N, fuel as F, visual as V)
from odin.backend.role import has_roles, Weight, Bias
from odin.ml.base import BaseEstimator, Evaluable

__all__ = [
    'NeuralClassifier',
    'NeuralRegressor'
]
# ===========================================================================
# Helper
# ===========================================================================
def _read_network_description(network):
  pass

class _NeuralEstimator(BaseEstimator, Evaluable):
  pass

# ===========================================================================
# Main classes
# ===========================================================================
class NeuralRegressor(_NeuralEstimator):
  pass

class NeuralClassifier(_NeuralEstimator):
  """ NeuralNetwork """

  def __init__(self, network,
               l1=0., l2=0., confusion_matrix=True,
               tol=1e-4, patience=3, rollback=True,
               batch_size=256, max_epoch=100, max_iter=None,
               optimizer='adadelta', learning_rate=1.0, class_weights=None,
               dtype='float32', seed=5218, verbose=False,
               path=None, name=None):
    super(NeuralNetworkClassifier, self).__init__()
    if not isinstance(network, N.NNOp):
      raise ValueError("`network` must be instance of odin.nnet.NNOp")
    self._network = network
    self._input_shape = None
    self._output_shape = None
    self._nb_classes = None
    self._dtype = np.dtype(dtype)
    self._class_weights = class_weights
    # ====== flags ====== #
    self.l1 = float(l1)
    self.l2 = float(l2)
    self.confusion_matrix = bool(confusion_matrix)
    # ====== stop training ====== #
    self.tol = float(tol)
    self.patience = int(patience)
    self.rollback = bool(rollback)
    # ====== others ====== #
    self._train_history = []
    self._valid_history = []
    self._rand_state = np.random.RandomState(seed=int(seed))
    self.verbose = int(verbose)
    # ====== others ====== #
    if name is None:
      name = self.__class__.__name__ + uuid()
    self._name = str(name)
    self._path = path

  # ==================== properties ==================== #
  @property
  def network(self):
    return self._network

  @property
  def input_shape(self):
    return self._input_shape

  @property
  def output_shape(self):
    return self._output_shape

  @property
  def name(self):
    return self._name

  @property
  def path(self):
    return self._path

  @property
  def labels(self):
    return tuple(self._labels)

  @property
  def nb_classes(self):
    return self._nb_classes

  @property
  def feat_dim(self):
    return self._feat_dim

  @property
  def is_fitted(self):
    return self._is_fitted

  @property
  def dtype(self):
    return self._dtype

  @property
  def placeholders(self):
    return (self._X, self._y)

  @property
  def parameters(self):
    """ Return list of parameters
    [W, b] if fit_intercept=True
    [W] otherwise
    """
    if self.fit_intercept:
      return [K.get_value(self._model.get('W')),
              K.get_value(self._model.get('b'))]
    else:
      return [K.get_value(self._model.get('W'))]

  def set_parameters(self, W=None, b=None):
    if W is not None:
      K.set_value(self._model.get('W'), W)
    if self.fit_intercept and b is not None:
      K.set_value(self._model.get('b'), b)
    return self

  # ==================== sklearn methods ==================== #
  def _initialize(self, X, y=None):
    with tf.name_scope(self.name):
      # ====== input_shape ====== #
      if self._input_shape is None:
        self._input_shape = X.shape
      elif self.input_shape[1:] != X.shape[1:]:
        raise ValueError("Initialized with input shape: %s, given tensor with shape: %s"
          % (self.input_shape, X.shape))
      # ====== output_shape ====== #
      if self._output_shape is None:
        self._output_shape = y.shape
      elif self.output_shape[1:] != y.shape[1:]:
        raise ValueError("Initialized with output shape: %s, given tensor with shape: %s"
          % (self.output_shape, y.shape))
      # ====== placeholder ====== #
      self._X = K.placeholder(shape=self.input_shape, dtype=self.dtype, name='input')
      self._y = K.placeholder(shape=self.output_shape, dtype=self.dtype, name='output')
      # ====== run the network ====== #
      y_pred_logits = self.network.apply(self._X)
      nb_classes = y_pred_logits.shape.as_list()[-1]
      if len(self._output_shape) == 1:
        y_true = tf.one_hot(indices=tf.cast(self._y, 'int32'),
                            depth=nb_classes)
      elif self._output_shape[-1] != nb_classes:
        raise ValueError("Given %d classes, but output from network has %s classes" %
          (self._output_shape[-1], nb_classes))
      self._nb_classes = nb_classes
      # ====== sigmoid or softmax ====== #
      if nb_classes == 2:
        fn_activation = tf.nn.sigmoid
        fn_loss = tf.losses.sigmoid_cross_entropy
        fn_acc = K.metrics.binary_accuracy
      else:
        fn_activation = tf.nn.softmax
        fn_loss = tf.losses.softmax_cross_entropy
        fn_acc = K.metrics.categorical_accuracy
      y_pred_proba = fn_activation(y_pred_logits)
      # ====== class weight ====== #
      class_weights = np.ones(shape=(nb_classes,), dtype=self.dtype) if self._class_weights is None\
      else as_tuple(self._class_weights, N=self.nb_classes, t=float)
      class_weights = tf.constant(value=class_weights, dtype=self.dtype,
                                 name="class_weights")
      weights = tf.gather(class_weights,
                          tf.cast(self._y, 'int32') if self.nb_classes == 2 else
                          tf.argmax(self._y, axis=-1))
      # ====== objectives ====== #
      cost_train = fn_loss(y_true, logits=y_pred_logits, weights=weights)
      exit()

  def fit(self, X, y=None, cv=None):
    self._initialize(X, y)
