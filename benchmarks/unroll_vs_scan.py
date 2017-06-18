# ===========================================================================
# Unroll:
#   Time: 0.360456 (sec)
#   Time: 0.179551 (sec)
#   1.42749e+12
# Scan:
#   Time: 0.001625 (sec)
#   Time: 0.083745 (sec)
#   1.42749e+12
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'
import numpy as np

from odin import backend as K, utils
import tensorflow as tf


def Scan1(fn,
         sequences=None,
         outputs_info=None,
         n_steps=None,
         backwards=False,
         name=None):
    """
    Note
    ----
    backwards mode only invert sequences then iterate over them
    """
    # ====== check sequences ====== #
    if sequences is None:
        sequences = []
    elif not isinstance(sequences, (tuple, list)):
        sequences = [sequences]
    sequences = [tf.unstack(seq, num=get_shape(seq)[0], axis=0)
                 for seq in sequences]
    if backwards:
        for i in sequences:
            i.reverse()
    # ====== check output info ====== #
    if outputs_info is None:
        outputs_info = []
    elif not isinstance(outputs_info, (tuple, list)):
        outputs_info = [outputs_info]
    else:
        outputs_info = list(outputs_info)
    nb_outputs = len(outputs_info)
    # ====== start iteration ====== #
    successive_outputs = [list() for i in range(nb_outputs)]
    outputs = outputs_info
    for step, inputs in enumerate(zip(*sequences)):
        inputs = list(inputs) + list(outputs)
        outputs = fn(*inputs)
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        for i, o in enumerate(outputs):
            successive_outputs[i].append(o)
        if n_steps is not None and step + 1 >= n_steps:
            break
    outputs = [tf.pack(output) for output in successive_outputs]
    if nb_outputs == 1:
        outputs = outputs[0]
    return outputs


def Scan2(fn,
         sequences=None,
         outputs_info=None,
         n_steps=None,
         backwards=False,
         name=None):
    """
    Note
    ----
    backwards mode only invert sequences then iterate over them
    """
    # ====== check sequences ====== #
    if sequences is None:
        sequences = []
    elif not isinstance(sequences, (tuple, list)):
        sequences = [sequences]
    if backwards:
        sequences = [tf.reverse(seq, axis=(0,)) for seq in sequences]
    if n_steps:
        sequences = [seq[:n_steps] for seq in sequences]
    # ====== check output info ====== #
    if outputs_info is None:
        outputs_info = []
    elif not isinstance(outputs_info, (tuple, list)):
        outputs_info = [outputs_info]
    else:
        outputs_info = list(outputs_info)
    nb_outputs = len(outputs_info)

    # ====== modified step function ====== #
    def step_(outputs, inputs):
        inputs = inputs + outputs
        outputs = fn(*inputs)
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        else:
            outputs = list(outputs)
        return outputs
    outputs = tf.scan(step_, sequences,
                initializer=outputs_info,
                parallel_iterations=32, back_prop=True,
                swap_memory=False, infer_shape=True,
                name=name)
    # consistent return as theano
    if nb_outputs == 1:
        outputs = outputs[0]
    return outputs


# ====== simulate data ====== #
def doit(_, x, y, z):
    z += K.sum(x + y) + K.sum(K.pow(_, 2))
    return z

sequences = [
    K.placeholder(shape=(600, None)),
    K.variable(np.arange(0, 1200).reshape(-1, 2)),
    K.variable(np.arange(1200, 2400).reshape(-1, 2))
]

outputs_info = K.zeros(shape=(1200,))

X = np.random.rand(600, 3000)
# ====== tf.scan ====== #
y = Scan2(doit,
          sequences=sequences,
          outputs_info=outputs_info,
          n_steps=None,
          backwards=True,
          name=None)
print('Scan:')
with utils.UnitTimer():
    f2 = K.function(sequences[0], y)
with utils.UnitTimer(12):
    for i in range(12):
        _ = f2(X)
print(np.sum(_))
# ====== unroll ====== #
y = Scan1(doit,
         sequences=sequences,
         outputs_info=outputs_info,
         n_steps=None,
         backwards=True,
         name=None)
print('Unroll:')
with utils.UnitTimer():
    f1 = K.function(sequences[0], y)
with utils.UnitTimer(12):
    for i in range(12):
        _ = f1(X)
print(np.sum(_))
