from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'theano,gpu,float32'

from odin import backend

import numpy as np
import theano
from theano import tensor as T

from odin.utils import UnitTimer
import time

const1 = theano.shared(np.random.rand(25, 25))
const2 = theano.shared(np.random.rand(18, 18))

# ===========================================================================
# Strict scan
# ===========================================================================


def step_strict(s1, o1, c1, c2):
    return (T.dot(o1, s1) + T.dot(o1.T, s1) + T.dot(o1, s1.T) + T.dot(o1.T, s1.T) +
        T.sum(const1) + T.sum(const2**2) + T.sum(const1**3) +
        T.sum(const2.T) + T.sum(const2.T**2) + T.sum(const2.T**3))

outputs, update = theano.scan(step_strict,
    sequences=theano.shared(np.arange(12 * 12 * 12 * 8 * 8).reshape(12 * 12 * 12, 8, 8)),
    outputs_info=theano.shared(np.ones((8, 8))),
    non_sequences=[const1, const2],
    strict=True)

f_strict = theano.function(inputs=[], outputs=outputs, allow_input_downcast=True)


# ===========================================================================
# Non-strict scan
# ===========================================================================
def step_non(s1, o1):
    return (T.dot(o1, s1) + T.dot(o1.T, s1) + T.dot(o1, s1.T) + T.dot(o1.T, s1.T) +
        T.sum(const1) + T.sum(const2**2) + T.sum(const1**3) +
        T.sum(const2.T) + T.sum(const2.T**2) + T.sum(const2.T**3))

outputs, update = theano.scan(step_non,
    sequences=theano.shared(np.arange(12 * 12 * 12 * 8 * 8).reshape(12 * 12 * 12, 8, 8)),
    outputs_info=theano.shared(np.ones((8, 8))),
    strict=False)
f_non = theano.function(inputs=[], outputs=outputs, allow_input_downcast=True)

time.sleep(0.5)

for i in range(3):
    print('Non-strict scan:')
    with UnitTimer(8):
        for i in range(8):
            f_non()

    print('Strict scan:')
    with UnitTimer(8):
        for i in range(8):
            f_strict()

# Non - strict scan:
# Time: 0.064988 (sec)
# Strict scan:
# Time: 0.058314 (sec)
# Non - strict scan:
# Time: 0.059891 (sec)
# Strict scan:
# Time: 0.067796 (sec)
# Non - strict scan:
# Time: 0.059809 (sec)
# Strict scan:
# Time: 0.065363 (sec)
