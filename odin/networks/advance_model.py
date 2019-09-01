from __future__ import absolute_import, division, print_function

import copy
import inspect
import os
import shutil
from tempfile import mkdtemp
from types import ModuleType
from typing import Callable

import dill
import tensorflow as tf
from six import string_types
from tensorflow.python import saved_model
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import Layer
from tensorflow.python.util import nest


# ===========================================================================
# Helpers
# ===========================================================================
def _is_list_of_layers(obj):
  pass


# ===========================================================================
# Main
# ===========================================================================
class AdvanceModel(Model):
  """ The advance model improving the serialization and deserialization of
  complex Model

  Parameters
  ----------
  parameters : {`dict` or `None`}
    recording the arguments given to `__init__`, recommend passing
    the `locals()` dictionary
  """

  def __init__(self, parameters=None, name=None):
    super(AdvanceModel, self).__init__(name=name)
    self.supports_masking = True
    self._build_input_shape = None

    if parameters is None:
      parameters = {}
    else:
      parameters = dict(parameters)
    kwargs = parameters.pop('kwargs', {})
    parameters.update(kwargs)
    parameters.pop('self', None)
    parameters.pop('__class__', None)
    # tricky recursive reference of overriding classes
    parameters.pop('parameters', None)
    self._parameters = parameters
    self._optimizer_weights = None
    self._optimizer = None
    self._history = {}

  @property
  def train_history(self):
    history = {}
    for key, val in self._history.items():
      if 'val_' != key[:4]:
        history[key] = val
    return history

  @property
  def valid_history(self):
    history = {}
    for key, val in self._history.items():
      if 'val_' == key[:4] and key[4:] in self._history:
        history[key] = val
    return history

  @property
  def optimizer(self):
    if self._optimizer_weights is not None and \
      self._optimizer is not None and \
        len(self._optimizer.weights) == len(self._optimizer_weights):
      # TODO: this is bad solution to resume the training
      self._optimizer.set_weights(self._optimizer_weights)
      self._optimizer_weights = None
    return self._optimizer

  @optimizer.setter
  def optimizer(self, optz):
    self._optimizer = optz

  def __getstate__(self):
    configs = dill.dumps(self.get_config())
    weights = dill.dumps(self.get_weights())
    optimizer = self.optimizer
    if optimizer is not None:
      optimizer = [
          optimizer.__class__,
          optimizer.get_config(),
          optimizer.get_weights()
          if self._optimizer_weights is None else self._optimizer_weights,
      ]
      optimizer = dill.dumps(optimizer)
    return configs, weights, optimizer

  def __setstate__(self, states):
    configs, weights, optimizer = states
    configs = dill.loads(configs)
    weights = dill.loads(weights)
    clone = self.__class__.from_config(configs)
    self.__dict__.update(clone.__dict__)
    self.set_weights(weights)
    if optimizer is not None:
      cls, configs, weights = dill.loads(optimizer)
      optimizer = cls.from_config(configs)
      self.compile(optimizer)
      self._optimizer_weights = weights

  @property
  def custom_objects(self):
    """ This property could be overrided to provide custom Layer class
    Expect a dictionary mapping from class name to the class itself for
    deserialization, or, list of modules
    """
    return {}

  @property
  def parameters(self):
    return dict(self._parameters)

  @base_layer_utils.default
  def build(self, input_shape=None):
    # if default is enable, it mean the build method is not overrided
    # and won't be called during __call__
    if self._is_graph_network:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
    else:
      if input_shape is None:
        if self._build_input_shape is None:
          raise ValueError('You must provide an `input_shape` argument.')
        else:
          input_shape = self._build_input_shape
      # do not convert input_shape to tuple, multiple inputs
      # will create a ListWrapper of multiple TensorShape.
      self._build_input_shape = input_shape
      super(AdvanceModel, self).build(input_shape)
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    # This to make sure input_shapes is recorded after every call
    if self._build_input_shape is None:
      input_list = nest.flatten(inputs)
      input_shapes = None
      if all(hasattr(x, 'shape') for x in input_list):
        input_shapes = nest.map_structure(lambda x: x.shape, inputs)
      self._build_input_shape = input_shapes
    return super(AdvanceModel, self).__call__(inputs, *args, **kwargs)

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
    """Trains the model for a fixed number of epochs (iterations on a dataset).

    Arguments:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset or a dataset iterator. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample weights)`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, dataset
          iterator, generator, or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, dataset, dataset iterators,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            Note that the progress bar is not particularly useful when
            logged to a file, so verbose=2 is recommended when not running
            interactively (eg, in a production environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, dataset iterator, generator or
           `keras.utils.Sequence` instance.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - tuple `(x_val, y_val)` of Numpy arrays or tensors
              - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
              - dataset or a dataset iterator
            For the first two cases, `batch_size` must be provided.
            For the last case, `validation_steps` must be provided.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch').
            'batch' is a special option for dealing with the
            limitations of HDF5 data; it shuffles in batch-sized chunks.
            Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`. This argument is not
            supported when `x` is a dataset, dataset iterator, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset or a dataset iterator, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
        validation_steps: Only relevant if `validation_data` is provided and
            is a dataset or dataset iterator. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If validation_data is a `tf.data` dataset
            or a dataset iterator, and 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted.
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.Container` instance (e.g. list, tuple, etc.). If an
            integer, specifies how many training epochs to run before a new
            validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1. If 0, will execute the generator on the main
            thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.
        **kwargs: Used for backwards compatibility.

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """
    history = super(AdvanceModel,
                    self).fit(x=x,
                              y=y,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              class_weight=class_weight,
                              sample_weight=sample_weight,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_freq=validation_freq,
                              max_queue_size=max_queue_size,
                              workers=workers,
                              use_multiprocessing=use_multiprocessing,
                              **kwargs)
    for key, val in history.history.items():
      if key not in self._history:
        self._history[key] = []
      self._history[key] += list(val)
    return history

  def get_config(self):
    subclass_types = []
    for t in type.mro(type(self)):
      if t == AdvanceModel:
        break
      subclass_types.append(t)
    default_types = [t for t in type.mro(type(self)) if t not in subclass_types]

    default_keys = []
    default_source = ''
    for t in default_types:
      default_keys += dir(t)
      # some class is builtin, so impossible to get the source code
      try:
        default_source += inspect.getsource(t)
      except:
        pass

    source = ''
    for t in subclass_types:
      source += inspect.getsource(t)

    attributes = {}
    layer_attributes = {}  # mapping attribute_name -> Layer
    for key, val in self.__dict__.items():
      # TODO: not a good solution here, history make recursive reference
      # but couldn't found in Model soruce code where it is setted
      if key in ('_parameters', 'history') or key in self.parameters:
        continue
      if isinstance(val, Layer):
        layer_attributes[id(val)] = key
      elif isinstance(val, tf.Variable):
        pass
      elif key not in default_keys and \
        'self.' + key not in default_source and \
          'self.' + key in source:
        try:
          attr = getattr(self, key)
          if not inspect.ismethod(attr) and \
            not isinstance(attr, property) and \
              not isinstance(attr, classmethod):
            attributes[key] = val
        except AttributeError:
          pass

    layer_configs = []
    for layer in self.layers:
      layer_configs.append({
          'class_name': layer.__class__.__name__,
          'config': layer.get_config(),
          'attribute': layer_attributes.get(id(layer), None)
      })

    config = {
        'name': self.name,
        'layers': copy.deepcopy(layer_configs),
        'parameters': self.parameters,
        'build_input_shape': self._build_input_shape,
        'attributes': attributes,
        'history': self._history,
    }
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'name' in config:
      name = config['name']
      build_input_shape = config['build_input_shape']
      layer_configs = config['layers']
      parameters = config['parameters']
      attributes = config['attributes']
      history = config['history']
    else:
      name = None
      build_input_shape = None
      layer_configs = config
      parameters = {}
      attributes = {}
      history = {}
    # create new instance
    if 'name' in inspect.getfullargspec(cls.__init__).args:
      parameters['name'] = name
    model = cls(**parameters)
    model._history = history
    # set all the attributes
    for key, val in attributes.items():
      setattr(model, key, val)
    # preprocessing the custom_objects
    if custom_objects is None:
      custom_objects = {}
    if hasattr(model, 'custom_objects'):
      model_objects = model.custom_objects
      if isinstance(model_objects, (tuple, list)):
        for obj in model_objects:
          if isinstance(obj, type):
            custom_objects[obj.__name__] = obj
          elif isinstance(obj, ModuleType):
            for i, j in inspect.getmembers(obj):
              if isinstance(j, type) and issubclass(j, Layer):
                custom_objects[i] = j
          elif inspect.isfunction(obj) or inspect.ismethod(obj):
            custom_objects[obj.__name__] = obj
          else:
            raise ValueError(
                "Cannot process value with type %s for custom_objects" +
                " (module, type or callable are supported)" % str(type(obj)))
      elif isinstance(model_objects, dict):
        custom_objects.update(model_objects)
      else:
        raise ValueError(
            "Class %s should return custom_objects of type dictionary or list,"
            " but the returned value is %s" %
            (str(cls), str(type(model_objects))))
    # deserialize all layers (NOTE: this no created layer is assigned
    # to our model, just to check all serialization go OK)
    layers = []
    for layer_config in layer_configs:
      attr = layer_config.pop('attribute')
      try:
        layer = layer_module.deserialize(layer_config,
                                         custom_objects=custom_objects)
      except ValueError:
        layer = None
      layers.append((attr, layer))
    # build if necessary
    if not model.inputs and build_input_shape is not None:
      model.build(build_input_shape)
    # check if all layers is deserialized, if any Layer is missing
    # that mean the Layer is externally added later after build,
    # then we add it back again to the Model (this logic might be fault)
    if len(model.layers) != len(layers):
      raise RuntimeError("No support for this case")
    return model
