import numpy as np
import six
import tensorflow as tf
from tensorflow_probability.python.bijectors.masked_autoregressive import (
    AutoregressiveNetwork, _create_degrees, _create_input_order,
    _make_dense_autoregressive_masks, _make_masked_constraint,
    _make_masked_initializer)
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = ['AutoregressiveDense']


class AutoregressiveDense(AutoregressiveNetwork):
  """ Masked autoregressive network - a generalized version of MADE.

  MADE is autoencoder which require equality in the number of dimensions
  between input and output.

  MAN enables these numbers to be different.
  """

  def build(self, input_shape):
    """See tfkl.Layer.build."""
    assert self._event_shape is not None, \
      'Unlike MADE, MAN require specified event_shape at __init__'
    # `event_shape` wasn't specied at __init__, so infer from `input_shape`.
    self._input_size = input_shape[-1]
    # Construct the masks.
    self._input_order = _create_input_order(
        self._input_size,
        self._input_order_param,
    )
    units = [] if self._hidden_units is None else list(self._hidden_units)
    units.append(self._event_size)
    masks = _make_dense_autoregressive_masks(
        params=self._params,
        event_size=self._input_size,
        hidden_units=units,
        input_order=self._input_order,
        hidden_degrees=self._hidden_degrees,
    )
    masks = masks[:-1]
    masks[-1] = np.reshape(
        np.tile(masks[-1][..., tf.newaxis], [1, 1, self._params]),
        [masks[-1].shape[0], self._event_size * self._params])
    self._masks = masks
    # create placeholder for ouput
    inputs = tf.keras.Input((self._input_size,), dtype=self.dtype)
    outputs = [inputs]
    if self._conditional:
      conditional_input = tf.keras.Input((self._conditional_size,),
                                         dtype=self.dtype)
      inputs = [inputs, conditional_input]
    # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
    #  [..., self._event_size] -> [..., self._hidden_units[0]].
    #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
    #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
    layer_output_sizes = list(
        self._hidden_units) + [self._event_size * self._params]
    for k in range(len(self._masks)):
      autoregressive_output = tf.keras.layers.Dense(
          layer_output_sizes[k],
          activation=None,
          use_bias=self._use_bias,
          kernel_initializer=_make_masked_initializer(self._masks[k],
                                                      self._kernel_initializer),
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          kernel_constraint=_make_masked_constraint(self._masks[k],
                                                    self._kernel_constraint),
          bias_constraint=self._bias_constraint,
          dtype=self.dtype)(outputs[-1])
      if (self._conditional and
          ((self._conditional_layers == 'all_layers') or
           ((self._conditional_layers == 'first_layer') and (k == 0)))):
        conditional_output = tf.keras.layers.Dense(
            layer_output_sizes[k],
            activation=None,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=None,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=None,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=None,
            dtype=self.dtype)(conditional_input)
        outputs.append(
            tf.keras.layers.Add()([autoregressive_output, conditional_output]))
      else:
        outputs.append(autoregressive_output)
      # last hidden layer, activation
      if k + 1 < len(self._masks):
        outputs.append(
            tf.keras.layers.Activation(self._activation)(outputs[-1]))
    self._network = tf.keras.models.Model(inputs=inputs, outputs=outputs[-1])
    # Allow network to be called with inputs of shapes that don't match
    # the specs of the network's input layers.
    self._network.input_spec = None
    # Record that the layer has been built.
    super(AutoregressiveNetwork, self).build(input_shape)

  def call(self, x, conditional_input=None):
    """Transforms the inputs and returns the outputs.

    Suppose `x` has shape `batch_shape + event_shape` and `conditional_input`
    has shape `conditional_batch_shape + conditional_event_shape`. Then, the
    output shape is:
    `broadcast(batch_shape, conditional_batch_shape) + event_shape + [params]`.

    Also see `tfkl.Layer.call` for some generic discussion about Layer calling.

    Args:
      x: A `Tensor`. Primary input to the layer.
      conditional_input: A `Tensor. Conditional input to the layer. This is
        required iff the layer is conditional.

    Returns:
      y: A `Tensor`. The output of the layer. Note that the leading dimensions
         follow broadcasting rules described above.
    """
    with tf.name_scope(self.name or 'MaskedAutoregressiveNetwork_call'):
      x = tf.convert_to_tensor(x, dtype=self.dtype, name='x')
      input_shape = ps.shape(x)
      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
      if self._conditional:
        if conditional_input is None:
          raise ValueError('`conditional_input` must be passed as a named '
                           'argument')
        conditional_input = tf.convert_to_tensor(conditional_input,
                                                 dtype=self.dtype,
                                                 name='conditional_input')
        conditional_batch_shape = ps.shape(conditional_input)[:-1]
        if tensorshape_util.rank(conditional_input.shape) == 1:
          conditional_input = conditional_input[tf.newaxis, ...]
        x = [x, conditional_input]
        output_shape = ps.concat([
            ps.broadcast_shape(conditional_batch_shape, input_shape[:-1]),
            (self._event_size,)
        ],
                                 axis=0)
      else:
        output_shape = ps.concat([input_shape[:-1], (self._event_size,)],
                                 axis=0)
      return tf.reshape(self._network(x),
                        tf.concat([output_shape, [self._params]], axis=0))
