from __future__ import print_function, division, absolute_import

import time
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin.ml.base import BaseEstimator, Evaluable
from odin.backend.role import has_roles, Weight, Bias
from odin import (backend as K, nnet as N, fuel as F, visual as V)
from odin.utils import (is_number, uuid, batching, as_tuple, Progbar,
                        one_hot, wprint, ctext, is_string)

# ===========================================================================
# Helper
# ===========================================================================
def _create_it_func(X, y, batch_size, start, end):
  """ Return a lambda function that create new generator """
  nb_samples = end - start
  create_it = None
  # ====== y is None ====== #
  if y is None:
    if hasattr(X, 'set_batch'):
      create_it = lambda seed: (x for x in X.set_batch(
          batch_size=batch_size,
          start=start, end=end,
          seed=seed))
    elif hasattr(X, '__getitem__'):
      create_it = lambda seed: (X[start:end]
                        for start, end in batching(n=nb_samples,
                                                   batch_size=batch_size,
                                                   start=start, end=end,
                                                   seed=seed))
  # ====== provided y ====== #
  else:
    if hasattr(X, 'set_batch') and hasattr(y, 'set_batch'):
      create_it = lambda seed: ((i, j) for i, j in zip(
          X.set_batch(batch_size=batch_size, start=start, end=end, seed=seed),
          y.set_batch(batch_size=batch_size, start=start, end=end, seed=seed)))
    elif hasattr(X, '__getitem__') and hasattr(y, '__getitem__'):
      create_it = lambda seed: ((X[start:end], y[start:end])
        for start, end in batching(n=nb_samples,
                                   batch_size=batch_size,
                                   start=start, end=end,
                                   seed=seed))
  # ====== exception ====== #
  if create_it is None:
    raise ValueError("`X` and `y` must has attributes 'set_batch' or '__getitem__'")
  return create_it

def _preprocess_xy(x, y, nb_classes):
  if x.ndim > 2:
    x = np.reshape(x, newshape=(x.shape[0], -1))
  if y is not None:
    if y.ndim == 1 and nb_classes > 2:
      y = one_hot(y, nb_classes=nb_classes)
    return x, y
  return x

def _fitting_helper(it, fn, nb_samples, nb_classes, title):
  prog = Progbar(target=nb_samples, print_report=True,
                 print_summary=False, name=title)
  results = None
  start_time = time.time()
  for nb_iter, (x, y) in enumerate(it):
    # ====== preprocessing ====== #
    x, y = _preprocess_xy(x, y, nb_classes)
    # ====== post-processing results ====== #
    if results is None:
      results = list(fn(x, y))
    else:
      for idx, r in enumerate(fn(x, y)):
        results[idx] += r
    # ====== update progress ====== #
    prog.add(x.shape[0])
  duration = time.time() - start_time
  return (nb_iter + 1,
          duration,
          [r if isinstance(r, np.ndarray) else r / (nb_iter + 1)
           for r in results])

# ====== get all optimizer ====== #
_optimizer_list = {}
for name in dir(K.optimizers):
  obj = getattr(K.optimizers, name)
  if isinstance(obj, type) and issubclass(obj, K.optimizers.Optimizer):
    _optimizer_list[name.lower()] = obj
# ===========================================================================
# Logistic regression
# ===========================================================================
class LogisticRegression(BaseEstimator, Evaluable):
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
  max_epoch : int
      maximum number of epoch for fitting
  max_iter : int
      maximum number of iteration for fitting
  optimizer : str
      name of optimizer (lower case must match the name in
      `odin.backend.optimizers`)
  learning_rate : float
      learning rate for the optimizer
  class_weight : {float, None, list of scalar}
      weight for each classes during training
  verbose : {bool, int}
      `0` or `False`, totally turn off all logging during fitting
      `1` or `True`, turn on the summary for fitting
       >= `2`, turn on progress of each iteration and summary
  path : str
      checkpoint path for backing the model after every
      improvement
  name : str
      specific name for tensorflow Op
  """

  def __init__(self, nb_classes, l1=0., l2=0.,
               fit_intercept=True, confusion_matrix=True,
               tol=1e-4, patience=3, rollback=True,
               batch_size=1024, max_epoch=100, max_iter=None,
               optimizer='adadelta', learning_rate=1.0, class_weight=None,
               dtype='float32', seed=5218,
               verbose=False, path=None, name=None):
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
    self._class_weight = class_weight
    # ====== flags ====== #
    self.l1 = float(l1)
    self.l2 = float(l2)
    self.fit_intercept = bool(fit_intercept)
    self.confusion_matrix = bool(confusion_matrix)
    # ====== internal states ====== #
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
    if not is_string(optimizer):
      raise ValueError("`optimizer` must be one of the following")
    optimizer = optimizer.lower()
    if optimizer not in _optimizer_list:
      raise ValueError("`optimizer` must be one of the following: %s" %
        str(list(_optimizer_list.keys())))
    self._optimizer = _optimizer_list[optimizer.lower()](lr=float(learning_rate))
    self._optimizer_name = optimizer
    self._optimizer_lr = learning_rate
    # ====== stop training ====== #
    self.tol = float(tol)
    self.patience = int(patience)
    self.rollback = bool(rollback)
    # ====== others ====== #
    self._train_history = []
    self._valid_history = []
    self._rand_state = np.random.RandomState(seed=int(seed))
    self.verbose = int(verbose)

  # ==================== pickling ==================== #
  def __getstate__(self):
    return (self._labels, self._nb_classes, self._feat_dim, self._dtype,
      self._class_weight, self.l1, self.l2, self.fit_intercept,
      self.confusion_matrix, self._is_fitted, self._name, self._path,
      self.batch_size, self.max_epoch, self.max_iter,
      self.tol, self.patience, self.rollback,
      self._train_history, self._valid_history, self._rand_state, self.verbose,
      self._optimizer_name, self._optimizer_lr, self.parameters)

  def __setstate__(self, states):
    (self._labels, self._nb_classes, self._feat_dim, self._dtype,
     self._class_weight, self.l1, self.l2, self.fit_intercept,
     self.confusion_matrix, self._is_fitted, self._name, self._path,
     self.batch_size, self.max_epoch, self.max_iter,
     self.tol, self.patience, self.rollback,
     self._train_history, self._valid_history, self._rand_state, self.verbose,
     self._optimizer_name, self._optimizer_lr, parameters) = states
    self._optimizer = _optimizer_list[self._optimizer_name](
        lr=float(self._optimizer_lr))
    self._initialize(X=np.empty(shape=(1, self.feat_dim),
                                dtype=self.dtype))
    self.set_parameters(*parameters)

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
  def _initialize(self, X):
    # ====== check inputs dimensions ====== #
    if not hasattr(X, 'shape'):
      raise ValueError("`X` must have `shape` attribute.")
    feat_dim = np.prod(X.shape[1:])
    if self._feat_dim is None:
      self._feat_dim = feat_dim
    # validate input dimension
    if feat_dim != self._feat_dim:
      raise RuntimeError("Feature dimension mismatch %d and %d" %
                         (feat_dim, self.feat_dim))
    # check if tensorflow op initalized
    if hasattr(self, '_f_train'):
      return
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
      with K.variable_dtype(dtype=self.dtype):
        self._model = N.Dense(num_units=self.nb_classes,
                          W_init=init_ops.glorot_uniform_initializer(seed=self._rand_state.randint()),
                          b_init=b_init,
                          activation=K.linear)
        y_logits = self._model(self._X)
      y_prob = fn_activation(y_logits)
      # applying class weights
      class_weights = tf.constant(value=self._class_weight,
                                  dtype=self.dtype,
                                  name="class_weights")
      weights = tf.gather(class_weights,
                          tf.cast(self._y, 'int32') if self.nb_classes == 2 else
                          tf.argmax(self._y, axis=-1))
      # optimizer
      params = [v for v in self._model.variables
                if has_roles(v, Weight) or has_roles(v, Bias)]
      losses = fn_loss(self._y, y_logits, weights=weights)
      l1_norm = tf.norm(self._model.get('W'), ord=1) if self.l1 > 0. else 0
      l2_norm = tf.norm(self._model.get('W'), ord=2) if self.l2 > 0. else 0
      losses = losses + self.l1 * l1_norm + self.l2 * l2_norm
      acc = fn_acc(self._y, y_prob)
      updates = self._optimizer.get_updates(losses, params)
      # create function
      if self.confusion_matrix:
        cm = K.metrics.confusion_matrix(y_true=self._y, y_pred=y_prob,
                                        labels=self.nb_classes)
      metrics = [losses, acc, cm] if self.confusion_matrix else [losses, acc]
      self._f_train = K.function(inputs=(self._X, self._y),
                                 outputs=metrics,
                                 updates=updates,
                                 training=True)
      self._f_score = K.function(inputs=(self._X, self._y),
                                 outputs=metrics,
                                 training=False)
      self._f_pred_prob = K.function(inputs=self._X,
                                     outputs=y_prob,
                                     training=False)
      self._f_pred_logit = K.function(inputs=self._X,
                                      outputs=y_logits,
                                      training=False)
    return self

  def fit(self, X, y=None, cv=None):
    self._initialize(X)
    if not hasattr(X, 'shape') or not hasattr(X, '__iter__') or \
    not hasattr(X, '__len__'):
      raise ValueError("`X` must has 'shape', '__len__' and '__iter__' attributes")
    nb_train_samples = len(X)
    # convert to odin.fuel.Data if possible
    if isinstance(X, (np.ndarray, list, tuple)):
      X = F.as_data(X)
    if isinstance(y, (np.ndarray, list, tuple)):
      y = F.as_data(y)
    start_tr = 0
    end_tr = nb_train_samples
    # ====== check if cross validating ====== #
    create_it_cv = None
    if is_number(cv):
      cv = int(float(cv) * nb_train_samples) if cv < 1. else int(cv)
      end_tr = nb_train_samples - cv
      start_cv = end_tr
      end_cv = nb_train_samples
      nb_cv_samples = end_cv - start_cv
      create_it_cv = _create_it_func(X=X, y=y, batch_size=self.batch_size,
                                     start=start_cv, end=end_cv)
    elif isinstance(cv, (tuple, list)):
      X_cv, y_cv = cv
      nb_cv_samples = X_cv.shape[0]
      create_it_cv = _create_it_func(X=X_cv, y=y_cv, batch_size=self.batch_size,
                                     start=0, end=X_cv.shape[0])
    elif hasattr(cv, 'set_batch'):
      nb_cv_samples = cv.shape[0]
      create_it_cv = _create_it_func(X=cv, y=None, batch_size=self.batch_size,
                                     start=0, end=cv.shape[0])
    elif cv is not None:
      raise ValueError('`cv` can be float (0-1), tuple or list of X and y, '
                       'any object that have "shape" and "__iter__" attributes, '
                       'or None')
    # ====== preprocessing ====== #
    create_it = _create_it_func(X=X, y=y, batch_size=self.batch_size,
                                start=start_tr, end=end_tr)
    # ====== prepare ====== #
    curr_niter = sum(epoch[0] for epoch in self._train_history)
    curr_nepoch = len(self._train_history)
    curr_patience = int(self.patience)
    last_losses = None
    last_checkpoint = None
    best_epoch = None
    is_converged = False
    # ====== fitting ====== #
    while not is_converged:
      curr_nepoch += 1
      seed = self._rand_state.randint(0, 10e8)
      # ====== training ====== #
      nb_iter, duration, results = _fitting_helper(create_it(seed),
                                                   fn=self._f_train,
                                                   nb_samples=nb_train_samples,
                                                   nb_classes=self.nb_classes,
                                                   title='Epoch %d' % curr_nepoch)
      curr_niter += nb_iter
      self._train_history.append(
          (nb_train_samples, nb_iter, duration, results))
      # ====== cross validation ====== #
      if create_it_cv is not None:
        nb_iter, duration_valid, results = _fitting_helper(create_it_cv(seed),
                                                     fn=self._f_score,
                                                     nb_samples=nb_cv_samples,
                                                     nb_classes=self.nb_classes,
                                                     title="Validating")
        self._valid_history.append(
            (nb_train_samples, nb_iter, duration_valid, results))
        duration += duration_valid
      # ====== print log ====== #
      if self.verbose >= 2:
        print(ctext('#epoch:', 'cyan') + str(curr_nepoch),
              ctext('#iter:', 'cyan') + str(curr_niter),
              ctext("Loss:", 'yellow') + '%.5f' % results[0],
              ctext("Acc:", 'yellow') + '%.3f' % results[1],
              ctext("%.2f(s)" % duration, 'magenta'))
        if self.confusion_matrix and (curr_nepoch - 1) % 8 == 0:
          print(V.print_confusion(results[-1], labels=self.labels))
      # ====== early stopping ====== #
      losses = results[0]
      if last_checkpoint is None: # first check point
        last_checkpoint = self.parameters
      if last_losses is not None:
        # degraded, smaller is better
        if last_losses - losses <= self.tol:
          curr_patience -= 1
          if self.rollback:
            if self.verbose >= 2:
              wprint('[LogisticRegression] Rollback to the best checkpoint '
                     'at epoch:%s patience:%s' %
                     (ctext(best_epoch, 'cyan'),
                      ctext(curr_patience, 'cyan')))
            self.set_parameters(*last_checkpoint)
        # save best checkpoint
        else:
          last_checkpoint = self.parameters
          best_epoch = curr_nepoch
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
    # ====== print summary plot ====== #
    if self.verbose >= 1:
      train_losses = [epoch[-1][0] for epoch in self._train_history]
      print(V.print_bar(train_losses, height=12,
                        bincount=min(20, len(train_losses)),
                        title='Training Losses'))
      if create_it_cv is not None:
        valid_losses = [epoch[-1][0] for epoch in self._valid_history]
        print(V.print_bar(valid_losses, height=12,
                          bincount=min(20, len(train_losses)),
                          title='Validation Losses'))
      if self.confusion_matrix:
        print(ctext("======== Training Confusion Matrix ========", 'cyan'))
        print(V.print_confusion(arr=self._train_history[-1][-1][-1],
                                labels=self.labels))
        if create_it_cv is not None:
          print(ctext("======== Validation Confusion Matrix ========", 'cyan'))
          print(V.print_confusion(arr=self._valid_history[-1][-1][-1],
                                  labels=self.labels))
    # ====== reset to best points ====== #
    self.set_parameters(*last_checkpoint)
    self._is_fitted = True
    if self._path is not None:
      with open(self._path, 'wb') as f:
        pickle.dump(self, f)

  def predict(self, X):
    """Predict class labels for samples in X.

    Parameters
    ----------
    X : {array-like}, shape = [nb_samples, feat_dim]
        Samples.

    Returns
    -------
    C : array, shape = [nb_samples]
        Predicted class label per sample.
    """
    return np.argmax(self.predict_proba(X), axis=-1)

  def _predict(self, X, f_pred):
    if not self.is_fitted:
      raise RuntimeError("LogisticRegression hasn't been initialized or "
                         "fitted.")
    if hasattr(X, 'set_batch'):
      it = iter(X.set_batch(batch_size=self.batch_size, seed=None))
    elif hasattr(X, '__getitem__'):
      it = (X[start:end]
            for start, end in batching(batch_size=self.batch_size,
                                       n=X.shape[0]))
    else:
      raise ValueError("`X` must has attributes 'set_batch' or '__getitem__'")
    # ====== make prediction ====== #
    y = []
    prog = Progbar(target=X.shape[0], print_report=True,
                   print_summary=False, name="Predicting")
    for x in it:
      x = _preprocess_xy(x, y=None, nb_classes=self.nb_classes)
      y.append(f_pred(x))
      prog.add(x.shape[0])
    return np.concatenate(y, axis=0)

  def predict_logits(self, X):
    """Logits values estimates.
    The returned estimates for all classes are ordered by the
    label of classes.
    """
    return self._predict(X, f_pred=self._f_pred_logit)

  def predict_proba(self, X):
    """Probability estimates.
    The returned estimates for all classes are ordered by the
    label of classes.

    Parameters
    ----------
    X : array-like, shape = [nb_samples, feat_dim]

    Returns
    -------
    y : array-like, shape = [nb_samples, nb_classes]
        Returns the probability of the sample for each class in the model,
        where classes are ordered as they are in ``self.classes_``.
    """
    return self._predict(X, f_pred=self._f_pred_prob)

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
