from __future__ import print_function, division, absolute_import

import pickle

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator

from odin.utils import (is_number, uuid, batching, as_tuple, Progbar,
                        one_hot, wprint, ctext)
from odin import (backend as K, nnet as N, fuel as F, visual as V)
from odin.backend.role import has_roles, Weight, Bias


class LogisticRegression(BaseEstimator):
  """ LogisticRegression

  Parameters
  ----------
  nb_classes : int
      number of output classes
  l1 : float (scalar)
      weight for L1 regularization
  l2 : float (scalar)
      weight for L2 regularization
  fit_intercept : bool (default: True)
      Specifies if a constant (a.k.a. bias or intercept) should be
      added to the decision function.
  confusion_matrix : bool (default: True)
      print confusion matrix during fitting and evaluating
  tol : float (default: 1e-4)
      Tolerance for stopping criteria
  patience : int (default: 3)
      how many failed improvement before stopping the training
  rollback : bool (default: True)
      allow the model to rollback to the best checkpoint after
      each generalization degradation
  batch_size : int
      batch size
  """

  def __init__(self, nb_classes, l1=0., l2=0.,
               fit_intercept=True, confusion_matrix=True,
               tol=1e-4, patience=3, rollback=True,
               batch_size=25000, max_epoch=100, max_iter=None,
               optimizer=None,
               class_weight=None, dtype='float32',
               seed=5218, path=None, name=None):
    super(LogisticRegression, self).__init__()
    # ====== basic dimensions ====== #
    if isinstance(nb_classes, (tuple, list, np.ndarray)):
      self._labels = tuple([str(i) for i in nb_classes])
      self._nb_classes = len(nb_classes)
    elif is_number(nb_classes):
      self._labels = tuple([str(i) for i in range(nb_classes)])
      self._nb_classes = int(nb_classes)
    self._feat_dim = None
    self._dtype = np.dtype(dtype)
    # ====== preprocessing class weight ====== #
    if class_weight is None:
      class_weight = np.ones(shape=(self.nb_classes,),
                             dtype=self.dtype)
    elif is_number(class_weight):
      class_weight = np.zeros(shape=(self.nb_classes,),
                              dtype=self.dtype) + class_weight

    self.class_weight = class_weight
    # ====== flags ====== #
    self.l1 = float(l1)
    self.l2 = float(l2)
    self.fit_intercept = bool(fit_intercept)
    self.confusion_matrix = bool(confusion_matrix)
    # ====== internal states ====== #
    self._is_initialized = False
    self._is_fitted = False
    # ====== others ====== #
    if name is None:
      name = uuid(length=8)
      self._name = 'LogisticRegression_%s' % name
    else:
      self._name = str(name)
    self._path = path
    # ====== training ====== #
    self.batch_size = int(batch_size)
    self.max_epoch = max_epoch
    self.max_iter = max_iter
    if optimizer is None:
      optimizer = K.optimizers.Adadelta()
    self._optimizer = optimizer
    # ====== stop training ====== #
    self.tol = float(tol)
    self.patience = int(patience)
    self.rollback = bool(rollback)
    # ====== others ====== #
    self._losses_history = Progbar(target=1,
                                   print_report=True,
                                   print_summary=True)
    self._losses_history.set_summarizer('#iter', lambda x: x[-1] - x[0])
    self._losses_history.set_summarizer('#sample', lambda x: x[-1] - x[0])
    self._losses_history.set_labels(self._labels)
    self._seed = int(seed)
    self._rand_state = np.random.RandomState(seed=self._seed)

  # ==================== pickling ==================== #
  def __getstate__(self):
    pass

  def __setstate__(self, states):
    pass

  # ==================== properties ==================== #
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
  def is_initialized(self):
    return self._is_initialized

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
  def _initialize(self, X):
    # ====== check inputs dimensions ====== #
    if not hasattr(X, 'shape'):
      raise ValueError("`X` must have `shape` attribute.")
    feat_dim = X.shape[1]
    if self.is_initialized:
      if feat_dim != self._feat_dim:
        raise RuntimeError("Feature dimension mismatch %d and %d" %
                           (feat_dim, self.feat_dim))
      return self
    # ====== binary or multi-classes ====== #
    if self.nb_classes == 2:
      out_shape = (None,)
      fn_activation = tf.nn.sigmoid
      fn_loss = tf.losses.sigmoid_cross_entropy
      fn_acc = K.metrics.binary_accuracy
    else:
      out_shape = (None, self.nb_classes)
      fn_activation = tf.nn.softmax
      fn_loss = tf.losses.softmax_cross_entropy
      fn_acc = K.metrics.categorical_accuracy
    # ====== create model ====== #
    self._feat_dim = feat_dim
    self._losses_history.set_name("[LogisticRegression] #in:%d #out:%d" %
                                  (self.feat_dim, self.nb_classes))
    with tf.name_scope(self.name, 'logistic_regression'):
      # inputs
      self._X = K.placeholder(shape=(None, self.feat_dim),
                              dtype=self.dtype,
                              name='%s_input' % self.name)
      self._y = K.placeholder(shape=out_shape,
                              dtype=self.dtype,
                              name='%s_output' % self.name)
      # check the bias
      if is_number(self.fit_intercept):
        b_init = float(self.fit_intercept)
      elif self.fit_intercept is False or \
      self.fit_intercept is None:
        b_init = None
      else:
        b_init = self.fit_intercept
      # create the model and initialize
      self._model = N.Dense(num_units=self.nb_classes,
                      W_init=K.rand.glorot_uniform,
                      b_init=b_init,
                      activation=K.linear)
      y_logits = self._model(self._X)
      y_prob = fn_activation(y_logits)
      # applying class weights
      class_weights = tf.constant(value=self.class_weight,
                                  dtype=self.dtype, name="class_weights")
      weights = tf.gather(class_weights,
                          tf.cast(self._y, 'int32') if self.nb_classes == 2 else
                          tf.argmax(self._y, axis=-1))
      # optimizer
      params = [v for v in self._model.variables
                if has_roles(v, Weight) or has_roles(v, Bias)]
      losses = fn_loss(self._y, y_logits, weights=weights)
      l1_norm = tf.norm(self._model.get('W'), ord='1') if self.l1 > 0. else 0
      l2_norm = tf.norm(self._model.get('W'), ord='2') if self.l2 > 0. else 0
      losses = losses + self.l1 * l1_norm + self.l2 * l2_norm
      acc = fn_acc(self._y, y_prob)
      updates = self._optimizer.get_updates(losses, params)
      norm = self._optimizer.norm
      # create function
      if self.confusion_matrix:
        cm = K.metrics.confusion_matrix(y_true=self._y, y_pred=y_prob,
                                        labels=self.nb_classes)
      self._f_train = K.function(inputs=(self._X, self._y),
                                 outputs=[losses, acc, cm, norm]
                                 if self.confusion_matrix else [losses, acc, norm],
                                 updates=updates,
                                 training=True)
      self._f_score = K.function(inputs=(self._X, self._y),
                                 outputs=[losses, acc, cm]
                                 if self.confusion_matrix else [losses, acc],
                                 training=False)
      self._f_pred_prob = K.function(inputs=self._X,
                                     outputs=y_prob,
                                     training=False)
      self._f_pred_logit = K.function(inputs=self._X,
                                      outputs=y_logits,
                                      training=False)
    return self

  def fit(self, X, y=None):
    self._initialize(X)
    if not hasattr(X, 'shape') or not hasattr(X, '__iter__') or \
    not hasattr(X, '__len__'):
      raise ValueError("`X` must has 'shape', '__len__' and '__iter__' attributes")
    nb_samples = len(X)
    # convert to odin.fuel.Data if possible
    if isinstance(X, (np.ndarray, list, tuple)):
      X = F.as_data(X)
    if isinstance(y, (np.ndarray, list, tuple)):
      y = F.as_data(y)
    # ====== preprocessing ====== #
    if y is None:
      if hasattr(X, 'set_batch'):
        create_it = lambda seed: ((i, j)
                for i, j in X.set_batch(batch_size=self.batch_size, seed=seed))
      elif hasattr(X, '__getitem__'):
        create_it = lambda seed: (X[start:end]
                          for start, end in batching(n=nb_samples,
                                                     batch_size=self.batch_size,
                                                     seed=seed))
      else:
        create_it = lambda seed: (x for x in X)
    else:
      if hasattr(X, 'set_batch') and hasattr(y, 'set_batch'):
        create_it = lambda seed: ((i, j) for i, j in zip(
            X.set_batch(batch_size=self.batch_size, seed=seed),
            y.set_batch(batch_size=self.batch_size, seed=seed)))
      elif hasattr(X, '__getitem__') and hasattr(y, '__getitem__'):
        create_it = lambda seed: ((X[start:end], y[start:end])
          for start, end in batching(n=nb_samples,
                                     batch_size=self.batch_size,
                                     seed=seed))
      else:
        create_it = lambda seed: ((i, j) for i, j in zip(X, y))
    # ====== fitting ====== #
    curr_niter = 0
    curr_nepoch = 0
    curr_nsample = 0
    curr_patience = int(self.patience)
    last_losses = None
    last_checkpoint = None
    best_epoch = None
    is_converged = False
    self._losses_history.target = nb_samples
    while not is_converged:
      curr_nepoch += 1
      seed = self._rand_state.randint(0, 10e8)
      for i, j in create_it(seed):
        curr_niter += 1
        curr_nsample += i.shape[0]
        # one-hot encode if necessary
        if j.ndim == 1 and self.nb_classes > 2:
          j = one_hot(j, nb_classes=self.nb_classes)
        if self.confusion_matrix:
          losses, acc, cm, norm = self._f_train(i, j)
        else:
          losses, acc, norm = self._f_train(i, j)
        # update progress
        self._losses_history['#iter'] = curr_niter
        self._losses_history['#sample'] = curr_nsample
        self._losses_history['Patience'] = curr_patience
        if self.confusion_matrix:
          self._losses_history['ConfusionMatrix'] = cm
        self._losses_history['EntropyLoss'] = losses
        self._losses_history['Accuracy'] = acc
        self._losses_history['GradNorm'] = norm
        self._losses_history.add(i.shape[0])
      # early stopping
      losses = np.mean(self._losses_history.get_report(epoch=-1,
                                                       key='EntropyLoss'))
      if last_checkpoint is None: # first check point
        last_checkpoint = self.parameters
      if last_losses is not None:
        if last_losses - losses <= self.tol: # smaller is better
          curr_patience -= 1
          if self.rollback:
            wprint('[LogisticRegression] Rollback to the best checkpoint '
                   'at epoch: %s' % ctext(best_epoch, 'cyan'))
            self.set_parameters(*last_checkpoint)
        else: # save checkpoint
          last_checkpoint = self.parameters
          best_epoch = curr_nepoch - 1
          if self._path is not None:
            with open(self._path, 'wb') as f:
              pickle.dump(self, f)
      last_losses = losses
      if curr_patience <= 0:
        is_converged = True
      # end the training
      if self.max_iter is not None and \
      curr_niter >= self.max_iter:
        break
      if self.max_epoch is not None and \
      curr_nepoch >= self.max_epoch:
        break
    # ====== reset to best points ====== #
    self.set_parameters(*last_checkpoint)
    self._is_fitted = True

  def predict_proba(self, X):
    """Probability estimates.
    The returned estimates for all classes are ordered by the
    label of classes.
    For a multi_class problem, if multi_class is set to be "multinomial"
    the softmax function is used to find the predicted probability of
    each class.
    Else use a one-vs-rest approach, i.e calculate the probability
    of each class assuming it to be positive using the logistic function.
    and normalize these values across all the classes.
    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
    Returns
    -------
    T : array-like, shape = [n_samples, n_classes]
        Returns the probability of the sample for each class in the model,
        where classes are ordered as they are in ``self.classes_``.
    """
    if not hasattr(self, "coef_"):
      raise NotFittedError("Call fit before prediction")
    calculate_ovr = self.coef_.shape[0] == 1 or self.multi_class == "ovr"
    if calculate_ovr:
      return super(LogisticRegression, self)._predict_proba_lr(X)
    else:
      return softmax(self.decision_function(X), copy=False)

  def predict_log_proba(self, X):
    """Log of probability estimates.
    The returned estimates for all classes are ordered by the
    label of classes.
    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
    Returns
    -------
    T : array-like, shape = [n_samples, n_classes]
        Returns the log-probability of the sample for each class in the
        model, where classes are ordered as they are in ``self.classes_``.
    """
    return np.log(self.predict_proba(X))
