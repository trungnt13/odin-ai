from __future__ import print_function, division, absolute_import

import cPickle
from abc import ABCMeta, abstractmethod
from six import add_metaclass

from odin.config import auto_config, RNG_GENERATOR
from odin.roles import add_role
from odin.annotations import add_annotation

config = auto_config()
FLOATX = config.floatX

if config['backend'] == 'theano':
    from .theano import *
elif config['backend'] == 'tensorflow':
    from .tensorflow import *

from . import init


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
        # cannot pickle annotations, because annotation is
        # function object with parameters per se
        # getattr(v.tag, 'annotations', [])
        return cPickle.dumps(obj, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        raise Exception('Variable must be in string form or trainable variable'
                        ' (i.e. SharedVariable in theano)')
