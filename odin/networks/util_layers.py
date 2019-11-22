from __future__ import absolute_import, division, print_function

import types
from collections import Iterable

from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import (Activation, BatchNormalization,
                                            Dense, Lambda, Layer)
from tensorflow.python.ops import variable_scope

__all__ = ['Identity', 'Parallel', 'BatchRenormalization']


def _Lambda_call(self, *args, **kwargs):
  arguments = self.arguments
  arguments.update(kwargs)
  if not self._fn_expects_training_arg:
    arguments.pop('training', None)
  if not self._fn_expects_mask_arg:
    arguments.pop('mask', None)
  with variable_scope.variable_creator_scope(self._variable_creator):
    return self.function(*args, **arguments)


# simple hack make Lambda for flexible
Lambda.call = _Lambda_call


class ModuleList(Sequential):
  r""" Holds submodules in a list.
  :class:`~odin.networks.ModuleList` can be indexed like a regular Python list,
  but modules it contains are properly registered, and will be visible by all
  :class:`~keras.layers.Layer` methods.

  Arguments:
    modules (iterable, optional): an iterable of `Layer` to add
  """

  def __init__(self, modules=None, name=None):
    super().__init__(layers=modules, name=name)

  def modules(self):
    for l in self.layers:
      yield l

  def named_modules(self):
    for l in self.layers:
      yield l.name, l

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return self.__class__(self.layers.values()[idx])
    else:
      return self.layers[idx]

  def __setitem__(self, idx, module):
    idx = int(idx)
    return setattr(self, str(idx), module)

  def __delitem__(self, idx):
    if not self.layers:
      raise TypeError('There are no layers in the model.')
    ids = list(range(len(self._layers)))[idx]
    layers = self._layers[idx]
    if not isinstance(layers, (tuple, list)):
      layers = [layers]
      ids = [ids]

    self._layers = [l for i, l in enumerate(self._layers) if i not in ids]
    for layer in layers:
      self._layer_call_argspecs.pop(layer)
    # removed all layer
    if not self.layers:
      self.outputs = None
      self.inputs = None
      self.built = False
    # modifying the outputs and re-build in case of static graph
    elif self._is_graph_network:
      self.layers[-1]._outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
      self.built = True

  def __len__(self):
    return len(self.layers)

  def __iter__(self):
    return iter(self.layers)

  def __iadd__(self, modules):
    return self.extend(modules)

  def insert(self, index, module):
    raise NotImplementedError()

  def append(self, module):
    r"""Appends a given layer to the end of the list.

    Arguments:
        module (keras.Layer): module to append
    """
    self.add(module)
    return self

  def extend(self, modules):
    r"""Appends layers from a Python iterable to the end of the list.

    Arguments:
        modules (iterable): iterable of modules to append
    """
    if not isinstance(modules, Iterable):
      raise TypeError("ModuleList.extend should be called with an "
                      "iterable, but got " + type(modules).__name__)
    for module in modules:
      self.add(module)
    return self


class BatchRenormalization(BatchNormalization):
  r""" Shortcut for batch renormalization

  References
  ----------
  [1] S. Ioffe, “Batch Renormalization: Towards Reducing Minibatch Dependence in
  Batch-Normalized Models,” arXiv:1702.03275 [cs], Feb. 2017.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               virtual_batch_size=None,
               adjustment=None,
               name=None,
               **kwargs):
    super().__init__(name=name,
                     axis=axis,
                     momentum=momentum,
                     epsilon=epsilon,
                     center=center,
                     scale=scale,
                     beta_initializer=beta_initializer,
                     gamma_initializer=gamma_initializer,
                     moving_mean_initializer=moving_mean_initializer,
                     moving_variance_initializer=moving_variance_initializer,
                     beta_regularizer=beta_regularizer,
                     gamma_regularizer=gamma_regularizer,
                     beta_constraint=beta_constraint,
                     gamma_constraint=gamma_constraint,
                     renorm=True,
                     renorm_clipping=renorm_clipping,
                     renorm_momentum=renorm_momentum,
                     fused=fused,
                     trainable=trainable,
                     virtual_batch_size=virtual_batch_size,
                     adjustment=adjustment,
                     **kwargs)


class Identity(Layer):

  def __init__(self, name=None):
    super(Identity, self).__init__(name=name)
    self.supports_masking = True

  def call(self, inputs, training=None):
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape


class Parallel(Sequential):
  """ Similar design to keras `Sequential` but simultanously applying
  all the layer on the input and return all the results.

  This layer is important for implementing multitask learning.
  """

  def call(self, inputs, training=None, mask=None, **kwargs):  # pylint: disable=redefined-outer-name
    if self._is_graph_network:
      if not self.built:
        self._init_graph_network(self.inputs, self.outputs, name=self.name)
      return super(Parallel, self).call(inputs, training=training, mask=mask)

    outputs = []
    for layer in self.layers:
      # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
      # are the outputs of `layer` applied to `inputs`. At the end of each
      # iteration `inputs` is set to `outputs` to prepare for the next layer.
      kw = {}
      argspec = self._layer_call_argspecs[layer].args
      if 'mask' in argspec:
        kw['mask'] = mask
      if 'training' in argspec:
        kw['training'] = training
      # support custom keyword argument also
      for k, v in kwargs.items():
        if k in argspec:
          kw[k] = v

      o = layer(inputs, **kw)
      outputs.append(o)

    return tuple(outputs)

  def compute_output_shape(self, input_shape):
    shape = []
    for layer in self.layers:
      shape.append(layer.compute_output_shape(input_shape))
    return tuple(shape)

  def compute_mask(self, inputs, mask):
    outputs = self.call(inputs, mask=mask)
    return [o._keras_mask for o in outputs]
