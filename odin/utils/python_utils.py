from __future__ import print_function, division, absolute_import

import inspect
from six import string_types, add_metaclass
from collections import defaultdict

import numpy as np

_non_override_list = ('__setattr__', '__getattr__', '__getattribute__',
                      '__subclasshook__', '__getnewargs__', '__reduce__',
                      '__reduce_ex__', '__new__', '__init__')

_support_primitive = (tuple, list, str, int, float, dict)


def _create_theproxy_(attrname):
    def __theproxy__(self, *args, **kwargs):
        t = self.ref_value
        if t is None:
            t = self.default_value
        return getattr(t, attrname)(*args, **kwargs)
    return __theproxy__


class _MetaPrimitive(type):

    """ This meta class is helpful for creating reference class
    while keeping its primitive type. """

    def __new__(meta, name, bases, attrdict):
        # ====== check bases classes ====== #
        # check if is primitive ref
        is_primitive_ref = any(i is PrimitiveRef for i in bases)
        if not is_primitive_ref:
            raise ValueError("the bases class of %s must contains PrimitiveRef."
                             % name)
        # check if contain primitive support type
        primitive_type = [i for i in bases if i in _support_primitive]
        if len(primitive_type) == 0:
            raise ValueError("Only support primitive types for the first base, "
                             "but given: %s" % str(bases))
        primitive_type = primitive_type[0]
        # ====== get the primitive type ====== #
        dummy_instance = primitive_type()
        for s in dir(dummy_instance):
            dummy_attr = getattr(dummy_instance, s)
            if type(dummy_attr).__name__ in ('builtin_function_or_method',
                                             'method-wrapper') and \
            s not in attrdict and s not in _non_override_list:
                attrdict[s] = _create_theproxy_(s)
        return super(_MetaPrimitive, meta).__new__(meta, name, bases, attrdict)

    def __init__(cls, name, bases, attrdict):
        super(_MetaPrimitive, cls).__init__(name, bases, attrdict)


class PrimitiveRef(object):
    """ PrimitiveRef """

    def __new__(cls, obj, default=None):
        main_type = [i for i in inspect.getmro(cls)
                     if i in _support_primitive][0]
        if default is None:
            default = main_type()
        if not isinstance(default, main_type):
            raise ValueError("default value must be instance of type: '%s', but "
                             "was given type: '%s'" %
                             (main_type.__name__, type(default).__name__))
        # ====== create new instance ====== #
        instance = super(PrimitiveRef, cls).__new__(cls)
        instance._default = default
        return instance

    def __init__(self, obj, default=None):
        super(PrimitiveRef, self).__init__()
        self._obj = obj

    # ==================== others ==================== #
    @property
    def default_value(self):
        return self._default

    @property
    def ref_value(self):
        return self._obj


@add_metaclass(_MetaPrimitive)
class DtypeRef(PrimitiveRef, str):

    @property
    def ref_value(self):
        v = self._obj
        if hasattr(v, 'dtype'):
            return str(v.dtype)
        elif isinstance(v, (string_types, np.ndarray, np.dtype)):
            return str(v)
        return None


@add_metaclass(_MetaPrimitive)
class ShapeRef(PrimitiveRef, tuple):

    @property
    def ref_value(self):
        v = self._obj
        if hasattr(v, 'shape'):
            return tuple(v.shape)
        elif isinstance(v, (tuple, list, np.ndarray)):
            return tuple(v)
        return None


# ===========================================================================
# Async file IO
# ===========================================================================
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, method):
        method.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(method)
