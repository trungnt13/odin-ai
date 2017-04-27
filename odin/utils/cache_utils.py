from __future__ import print_function, division, absolute_import

import os
import shutil
import inspect
from joblib import Memory
from functools import wraps
from six import string_types
from six.moves import builtins
from decorator import FunctionMaker, decorator

from collections import defaultdict, OrderedDict

# to set the cache dir, set the environment CACHE_DIR
__cache_dir = os.environ.get("CACHE_DIR", os.path.join(os.path.expanduser('~'), '.odin_cache'))
# check cache_dir
if not os.path.exists(__cache_dir):
    os.mkdir(__cache_dir)
elif os.path.isfile(__cache_dir):
    raise ValueError("Invalid cache directory at path:" + __cache_dir)
# Don't use memmap anymore, carefull when cache big numpy ndarray results
__memory = Memory(cachedir=__cache_dir,
                  mmap_mode=None, compress=False, verbose=0)


def get_cache_path():
    return __cache_dir


def clear_disk_cache():
    if os.path.exists(__cache_dir):
        shutil.rmtree(__cache_dir)


def cache_disk(function):
    """ Lazy evaluation of function, by storing the results to the disk,
    and not rerunning the function twice for the same arguments.

    It works by explicitly saving the output to a file and it is
    designed to work with non-hashable and potentially large input
    and output data types such as numpy arrays

    Note
    ----
    Any change in the function code will result re-cache

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


def clear_mem_cache():
    __CACHE.clear()


def cache_memory(func, *attrs):
    '''Decorator. Caches the returned value and called arguments of
    a function.

    All the input and output are cached in the memory (i.e. RAM), and it
    requires hash-able inputs to compare the footprint of function.

    Parameters
    ----------
    attrs : str or list(str)
        list of object attributes in comparation for selecting cache value

    Note
    ----
    enable strict mode by specify "__strict__" in the `attrs`, this mode
    turns off caching by default but activated when "__cache__" appeared in
    the argument

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
    strict_mode = False
    if not inspect.ismethod(func) and not inspect.isfunction(func):
        attrs = (func,) + attrs
        func = None
    # check if strict mode is enable
    if '__strict__' in attrs:
        strict_mode = True
        attrs = [a for a in attrs if a != '__strict__']
    # check if all attrs is string
    if builtins.any(not isinstance(i, string_types) for i in attrs):
        raise ValueError('Tracking attribute must be string represented name of'
                         ' attribute, but given attributes have types: {}'
                         ''.format(tuple(map(type, attrs))))

    def wrap_function(func):
        # ====== fetch arguments in order ====== #
        _ = inspect.getargspec(func)
        args_name = _.args
        # reversed 2 time so everything in the right order
        if _.defaults is not None:
            args_defaults = OrderedDict(reversed(
                [(i, j) for i, j in zip(reversed(_.args), reversed(_.defaults))]
            ))
        else:
            args_defaults = OrderedDict()

        # ====== wraps the function ====== #
        @wraps(func)
        def wrapper(*args, **kwargs):
            # ====== check if strict_mode and caching enable ====== #
            if strict_mode:
                if any(isinstance(a, string_types) and a == '__cache__'
                       for a in args):
                    args = tuple([a for a in args
                                  if not isinstance(a, string_types) or a != '__cache__'])
                elif '__cache__' in kwargs:
                    kwargs.pop('__cache__')
                else:
                    # no cache just call the function
                    return func(*args, **kwargs)
            # ====== additional arguments ====== #
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


def cache_memory_strict():
    pass
