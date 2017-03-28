from __future__ import print_function, division, absolute_import

import os
from six.moves import cPickle

from odin.utils import is_string, is_path
from odin.basic import (add_role, add_updates, add_auxiliary_variable,
                        add_shape, get_shape)

# ==================== import utilities modules ==================== #
from .basic_ops import *
from .advance_ops import *
from . import init
from . import optimizers


def pickling_variable(v):
    """ This function only apply for trainable parameters
    Warning
    -------
    This pickling method won't save "auxiliary_variables" and "updates"
    tag of variables
    """
    # load variable
    if is_string(v):
        # check if is a path
        try:
            if is_path(v) and os.path.exists(v):
                v = open(v, 'r')
        except Exception, e:
            print('[Error]pickling_variable:', e)
        # otherwise load directly
        name, value, dtype, roles = cPickle.loads(v)
        v = variable(value, dtype=dtype, name=name)
        for i in roles:
            add_role(v, i)
        return v
    elif is_trainable_variable(v):
        name = v.name if ':' not in v.name else v.name.split(':')[0]
        value = get_value(v)
        dtype = v.dtype.as_numpy_dtype if hasattr(v.dtype, 'as_numpy_dtype') else v.dtype
        # ====== shape and roles ====== #
        roles = getattr(v.tag, 'roles', [])
        return cPickle.dumps([name, value, dtype, roles],
                             protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        raise Exception('Variable must be in string form or trainable variable'
                        ' (i.e. SharedVariable in theano)')
