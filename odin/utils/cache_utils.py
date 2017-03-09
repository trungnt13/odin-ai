from __future__ import print_function, division, absolute_import

import os
import inspect
from joblib import Memory
from functools import wraps
from six.moves import builtins
from decorator import FunctionMaker

from collections import defaultdict, OrderedDict

# to set the cache dir, set the environment CACHE_DIR
__cache_dir = os.environ.get("CACHE_DIR", os.path.join(os.path.expanduser('~'), '.odin_cache'))
# check cache_dir
if not os.path.exists(__cache_dir):
    os.mkdir(__cache_dir)
elif os.path.isfile(__cache_dir):
    raise ValueError("Invalid cache directory at path:" + __cache_dir)
__memory = Memory(cachedir=__cache_dir,
                 mmap_mode="c", compress=False, verbose=0)


def cache_disk(function):
    """ Lazy evaluation of function, by storing the results to the disk,
    and not rerunning the function twice for the same arguments.

    It works by explicitly saving the output to a file and it is
    designed to work with non-hashable and potentially large input
    and output data types such as numpy arrays

    Example
    -------
    >>> import numpy as np
    >>> @cache_memory
    >>> def doit(x):
    ...     y = np.random.rand(1208, 1208)
    ...     print("Function called:", np.sum(y))
    ...     return y
    >>> for _ in range(12):
    ...     print(np.sum(doit(8)))
    """
    def decorator_apply(dec, func):
        """Decorate a function by preserving the signature even if dec
        is not a signature-preserving decorator.
        This recipe is derived from
        http://micheles.googlecode.com/hg/decorator/documentation.html#id14
        """
        return FunctionMaker.create(
            func, 'return decorated(%(signature)s)',
            dict(decorated=dec(func)), __wrapped__=func)
    return decorator_apply(__memory.cache, function)


# ===========================================================================
# Cache
# ===========================================================================
__CACHE = defaultdict(lambda: ([], [])) #KEY_ARGS, RET_VALUE


def cache_memory(func, *attrs):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    All the input and output are cached in the memory (i.e. RAM), and it
    requires hash-able inputs to compare the footprint of function.

    Parameters
    ----------
    args : str or list(str)
        list of object attributes in comparation for selecting cache value

    Example
    -------
    >>> class ClassName(object):
    >>>     def __init__(self, arg):
    >>>         super(ClassName, self).__init__()
    >>>         self.arg = arg
    >>>     @cache_memory('arg')
    >>>     def abcd(self, a):
    >>>         return np.random.rand(*a)
    >>>     def e(self):
    >>>         pass
    >>> x = c.abcd((10000, 10000))
    >>> x = c.abcd((10000, 10000)) # return cached value
    >>> c.arg = 'test'
    >>> x = c.abcd((10000, 10000)) # return new value
    '''
    if not inspect.ismethod(func) and not inspect.isfunction(func):
        attrs = (func,) + attrs
        func = None

    if builtins.any(not isinstance(i, str) for i in attrs):
        raise ValueError('Tracking attribute must be string represented name of'
                         ' attribute, but given attributes have types: {}'
                         ''.format(tuple(map(type, attrs))))

    def wrap_function(func):
        # ====== fetch arguments order ====== #
        _ = inspect.getargspec(func)
        args_name = _.args
        # reversed 2 time so everything in the right order
        if _.defaults is not None:
            args_defaults = OrderedDict(reversed([(i, j)
                for i, j in zip(reversed(_.args), reversed(_.defaults))]))
        else:
            args_defaults = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            input_args = list(args)
            excluded = {i: j for i, j in zip(args_name, input_args)}
            # check default kwargs
            for i, j in args_defaults.iteritems():
                if i in excluded: # already input as positional argument
                    continue
                if i in kwargs: # specified value
                    input_args.append(kwargs[i])
                else: # default value
                    input_args.append(j)
            # ====== create cache_key ====== #
            object_vars = {k: getattr(args[0], k) for k in attrs
                           if hasattr(args[0], k)}
            cache_key = (input_args, object_vars)
            # ====== check cache ====== #
            key_list = __CACHE[id(func)][0]
            value_list = __CACHE[id(func)][1]
            # get old cached value
            if cache_key in key_list:
                idx = key_list.index(cache_key)
                return value_list[idx]
            # call the function to get new value
            else:
                value = func(*args, **kwargs)
                key_list.append(cache_key)
                value_list.append(value)
                return value
        return wrapper

    # return wrapped function
    if func is None:
        return wrap_function
    return wrap_function(func)
