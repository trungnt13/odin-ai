from __future__ import print_function, division, absolute_import

import cPickle

from odin.basic import add_role

# ==================== import utilities modules ==================== #
from .basic_ops import *
from .advance_ops import *
# from . import init
# from . import optimizers


def pickling_variable(v, target=None):
    """ This function only apply for trainable parameters
    """
    if isinstance(v, str):
        value, dtype, name, roles = cPickle.loads(v)
        v = variable(value, dtype=dtype, name=name, target=target)
        for i in roles:
            add_role(v, i)
        return v
    elif is_trainable_variable(v):
        obj = [get_value(v, borrow=False), v.dtype, v.name,
               getattr(v.tag, 'roles', [])]
        return cPickle.dumps(obj, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        raise Exception('Variable must be in string form or trainable variable'
                        ' (i.e. SharedVariable in theano)')
