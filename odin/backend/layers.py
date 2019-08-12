from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Activation, Dense, Layer
from tensorflow.python.keras.utils import tf_utils


class Identity(Layer):

  def __init__(self, name=None):
    super(Identity, self).__init__(name=name)
    self.supports_masking = True

  def call(self, inputs, training=None):
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape


class DeterministicDense(Sequential):

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               name=None):
    from odin.bay.distribution_layers import DeterministicLayer
    layers = [
        Dense(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        ),
        DeterministicLayer(vectorized=True)
    ]
    super(DeterministicDense, self).__init__(layers=layers, name=name)


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

  @property
  def _object_identifier(self):
    return '_tf_keras_parallel'
