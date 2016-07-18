# -*- coding: utf-8 -*-
# ===========================================================================
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import sys
import __builtin__

from collections import OrderedDict, defaultdict
from collections import MutableMapping
from functools import wraps, partial
import inspect
from six.moves import zip, zip_longest
import types
import cPickle

import numpy as np

from odin import SIG_TERMINATE_ITERATOR
from odin.utils import is_path


__all__ = [
    'cache',
    'typecheck',
    'autoattr',
    'autoinit',
    'abstractstatic',
    'functionable',
    'singleton',
    'terminatable_iterator'
]

# ===========================================================================
# Cache
# ===========================================================================
_CACHE = defaultdict(lambda: ([], [])) #KEY_ARGS, RET_VALUE


def cache(func, *attrs):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

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
    >>>     @cache('arg')
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

    if __builtin__.any(not isinstance(i, str) for i in attrs):
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
            key_list = _CACHE[id(func)][0]
            value_list = _CACHE[id(func)][1]
            if cache_key in key_list:
                idx = key_list.index(cache_key)
                return value_list[idx]
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


# ===========================================================================
# Type enforcement
# ===========================================================================
def _info(fname, expected, actual, flag):
    '''Convenience function outputs nicely formatted error/warning msg.'''
    def to_str(t):
        s = []
        for i in t:
            if not isinstance(i, (tuple, list)):
                s.append(str(i).split("'")[1])
            else:
                s.append('(' + ', '.join([str(j).split("'")[1] for j in i]) + ')')
        return ', '.join(s)

    expected, actual = to_str(expected), to_str(actual)
    ftype = 'method'
    msg = "'{}' {} ".format(fname, ftype) \
        + ("inputs", "outputs")[flag] + " ({}), but ".format(expected) \
        + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg


def _compares_types(argtype, force_types):
    # True if types is satisfied the force_types
    for i, j in zip(argtype, force_types):
        if isinstance(j, (tuple, list)):
            if i not in j:
                return False
        elif i != j:
            return False
    return True


def typecheck(inputs=None, outputs=None, debug=2):
    '''Function/Method decorator. Checks decorated function's arguments are
    of the expected types.

    Parameters
    ----------
    inputs : types
        The expected types of the inputs to the decorated function.
        Must specify type for each parameter.
    outputs : types
        The expected type of the decorated function's return value.
        Must specify type for each parameter.
    debug : int, str
        Optional specification of 'debug' level:
        0:'ignore', 1:'warn', 2:'raise'

    Examples
    --------
    >>> # Function typecheck
    >>> @typecheck(inputs=(int, str, float), outputs=(str))
    >>> def function(a, b, c):
    ...     return b
    >>> function(1, '1', 1.) # no error
    >>> function(1, '1', 1) # error, final argument must be float
    ...
    >>> # method typecheck
    >>> class ClassName(object):
    ...     @typecheck(inputs=(str, int), outputs=int)
    ...     def method(self, a, b):
    ...         return b
    >>> x = ClassName()
    >>> x.method('1', 1) # no error
    >>> x.method(1, '1') # error

    '''
    if inspect.ismethod(inputs) or inspect.isfunction(inputs):
        raise ValueError('You must specify either [inputs] types or [outputs]'
                         ' types arguments.')
    # ====== parse debug ====== #
    if isinstance(debug, str):
        debug_str = debug.lower()
        if 'raise' in debug_str:
            debug = 2
        elif 'warn' in debug_str:
            debug = 1
        else:
            debug = 0
    elif debug not in (0, 1, 2):
        debug = 2
    # ====== check types ====== #
    if inputs is not None and not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    if outputs is not None and not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)

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
            ### main logic
            if debug is 0: # ignore
                return func(*args, **kwargs)
            ### Check inputs
            if inputs is not None:
                # main logic
                length = int(min(len(input_args), len(inputs)))
                argtypes = tuple(map(type, input_args))
                # TODO: smarter way to check argtypes for methods
                if not _compares_types(argtypes[:length], inputs[:length]) and\
                    not _compares_types(argtypes[1:length + 1], inputs[:length]): # wrong types
                    msg = _info(func.__name__, inputs, argtypes, 0)
                    if debug is 1:
                        print('TypeWarning:', msg)
                    elif debug is 2:
                        raise TypeError(msg)
            ### get results
            results = func(*args, **kwargs)
            ### Check outputs
            if outputs is not None:
                res_types = ((type(results),)
                             if not isinstance(results, (tuple, list))
                             else tuple(map(type, results)))
                length = min(len(res_types), len(outputs))
                if len(outputs) > len(res_types) or \
                    not _compares_types(res_types[:length], outputs[:length]):
                    msg = _info(func.__name__, outputs, res_types, 1)
                    if debug is 1:
                        print('TypeWarning: ', msg)
                    elif debug is 2:
                        raise TypeError(msg)
            ### finally everything ok
            return results
        return wrapper
    return wrap_function


# ===========================================================================
# Auto set attributes
# ===========================================================================
def autoattr(*args, **kwargs):
    '''
    Example
    -------
    >>> class ClassName(object):
    ..... def __init__(self):
    ......... super(ClassName, self).__init__()
    ......... self.arg1 = 1
    ......... self.arg2 = False
    ...... @autoattr('arg1', arg1=lambda x: x + 1)
    ...... def test1(self):
    ......... print(self.arg1)
    ...... @autoattr('arg2')
    ...... def test2(self):
    ......... print(self.arg2)
    >>> c = ClassName()
    >>> c.test1() # arg1 = 2
    >>> c.test2() # arg2 = True

    '''
    if len(args) > 0 and (inspect.ismethod(args[0]) or inspect.isfunction(args[0])):
        raise ValueError('You must specify at least 1 *args or **kwargs, all '
                         'attributes in *args will be setted to True, likewise, '
                         'all attributes in **kwargs will be setted to given '
                         'value.')
    attrs = {i: True for i in args}
    attrs.update(kwargs)

    def wrap_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            if len(args) > 0:
                for i, j in attrs.iteritems():
                    if hasattr(args[0], i):
                        if hasattr(j, '__call__'):
                            setattr(args[0], str(i), j(getattr(args[0], i)))
                        else:
                            setattr(args[0], str(i), j)
            return results
        return wrapper

    return wrap_function


# ===========================================================================
# Auto store all arguments when init class
# ===========================================================================
def autoinit(func):
    """ For checking what arguments have been assigned to the object:
    `_arguments` (dictionary)
    """
    if not inspect.isfunction(func):
        raise ValueError("Only accept function as input argument "
                         "(autoinit without any parameters")
    attrs, varargs, varkw, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assigned_arguments = []
        # handle default values
        if defaults is not None:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                setattr(self, attr, val)
                assigned_arguments.append(attr)
        # handle positional arguments (excluded self)
        positional_attrs = attrs[1:]
        for attr, val in zip(positional_attrs, args):
            setattr(self, attr, val)
            assigned_arguments.append(attr)
        # handle varargs
        if varargs:
            remaining_args = args[len(positional_attrs):]
            setattr(self, varargs, remaining_args)
            assigned_arguments.append(varargs)
        # handle varkw
        if kwargs:
            for attr, val in kwargs.iteritems():
                try:
                    setattr(self, attr, val)
                    assigned_arguments.append(attr)
                except: # ignore already predifined attr
                    pass
        # call the init
        _ = func(self, *args, **kwargs)
        # just the right moments
        assigned_arguments = {i: getattr(self, i) for i in assigned_arguments}
        if hasattr(self, '_arguments'):
            self._arguments.update(assigned_arguments)
        else:
            setattr(self, '_arguments', assigned_arguments)
        return _
    return wrapper


# ===========================================================================
# Abstract static
# ===========================================================================
class abstractstatic(staticmethod):
    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

# ===========================================================================
# Python utilities
# ===========================================================================
_primitives = (bool, int, float, str,
               tuple, list, dict, type, types.ModuleType, types.FunctionType,
               types.NoneType, types.TypeType)


def _serialize_function_sandbox(function):
    '''environment, dictionary (e.g. globals(), locals())
    Returns
    -------
    dictionary : cPickle dumps-able dictionary to store as text
    '''
    import re
    import marshal
    from array import array

    sys_module = re.compile('__\w+__')

    environment = function.func_globals
    func_module = function.__module__

    sandbox = OrderedDict()

    def func_to_str(func):
        # conver to byte
        return cPickle.dumps(array("B", marshal.dumps(func.func_code)))

    # ====== serialize primitive type ====== #
    seen_function = False
    for name, val in environment.iteritems():
        # ignore system modules
        if sys_module.match(name) is not None:
            continue
        # primitive type
        if type(val) in _primitives:
            # function might nested, so cannot find it in globals()
            if val == function: seen_function = True
            # print(k, v.__module__ if hasattr(v, '__module__') else v.__name__, func_module)
            typ = type(val)
            if isinstance(val, types.ModuleType): # special case: import module
                val = val.__name__
                typ = 'module'
            elif val is None: # for some reason, pickle cannot serialize None type
                val = None
                typ = 'None'
            elif inspect.isfunction(val): # special case: function
                # imported function
                _ = '_main' if function == val else ''
                if val.__module__ != func_module:
                    typ = 'imported_function'
                    val = (val.__name__, val.__module__)
                # defined function in the same script file
                else:
                    typ = 'defined_function'
                    val = (val.func_name, func_to_str(val))
                typ += _
        # check if object is pickle-able
        else:
            try:
                val_new = cPickle.loads(cPickle.dumps(val))
                if val_new.__dict__ != val.__dict__:
                    raise Exception
                typ = 'object'
            except: # not pickle-albe, just ignore it
                typ = None
        # Finnally add to sandbox
        if typ is not None:
            sandbox[name] = (typ, val)
    # ====== not seen the main function ====== #
    if not seen_function: # mark the main function with "_main"
        sandbox['random_name_12082518'] = ('defined_function_main',
                                           (function.func_name,
                                            func_to_str(function)))
    # else: # looking for static method in each class
    #     for i in sandbox.values():
    #         if hasattr(i, self._function_name):
    #             f = getattr(i, self._function_name)
    #             if inspect.isfunction(f) and inspect.getsource(f) == self._source:
    #                 self._function = f
    return sandbox


def _deserialize_function_sandbox(sandbox):
    '''
    environment : dictionary
        create by `serialize_sandbox`
    '''
    import marshal
    import importlib

    def str_to_func(s, sandbox, name):
        func = marshal.loads(cPickle.loads(s).tostring())
        func = types.FunctionType(func, sandbox, name)
        return func

    environment = {}
    defined_function = []
    main_func = None
    # first pass we deserialize all type except function type
    for name, (typ, val) in sandbox.iteritems():
        if typ == 'None':
            val = None
        elif typ == 'module':
            val = importlib.import_module(val)
        elif typ in _primitives or typ == 'object':
            pass
        elif 'imported_function' in typ:
            val = getattr(importlib.import_module(val[1]), val[0])
            if '_main' in typ: main_func = val
        elif 'defined_function' in typ:
            val = str_to_func(val[1], globals(), name=val[0])
            if '_main' in typ: main_func = val
            defined_function.append(name)
        else:
            raise ValueError('Unsupport deserializing type: {}, '
                             'value: {}'.format(typ, val))
        environment[name] = val
    # ====== create all defined function ====== #
    # second pass, function all funciton and set it globales to new environment
    for name in defined_function:
        func = environment[name]
        func.func_globals.update(environment)
    return main_func, environment


class functionable(object):

    """ Class handles save and load a function with its arguments

    Parameters
    ----------
    arg: list
        arguments list for given function
    kwargs: dict
        keyword arguments for given function

    Note
    ----
    *All defaults arguments must be specified within the source code of function
    (i.e. mean(x) must be converted to mean(x, axis=None, keepdims=False))
    This class does not support nested functions
    All the complex objects must be created in the function
    """

    def __init__(self, func, *args, **kwargs):
        super(functionable, self).__init__()
        # default arguments lost during pickling so need to store them
        import inspect
        final_args = OrderedDict()
        spec = inspect.getargspec(func)
        for i, j in zip(spec.args, args): # positional arguments
            final_args[i] = j
        final_args.update(kwargs)
        if spec.defaults is not None:
            final_args.update(reversed([(i, j)
                            for i, j in zip(reversed(spec.args), reversed(spec.defaults))
                            if i not in final_args]))

        self._function = func
        # random id that identify this function
        self._function_name = func.func_name
        self._function_order = spec.args
        self._function_kwargs = final_args
        self._sandbox = _serialize_function_sandbox(func)
        try: # sometime cannot get the source
            self._source = inspect.getsource(self._function)
        except:
            self._source = None

    @property
    def function(self):
        return self._function

    @property
    def name(self):
        return self._function_name

    @property
    def args_order(self):
        return self._function_order

    @property
    def kwargs(self):
        return self._function_kwargs

    @property
    def source(self):
        return self._source

    def __call__(self, *args, **kwargs):
        kwargs_ = dict(self._function_kwargs)
        kwargs_.update(kwargs)
        for i, j in zip(self._function_order, args):
            kwargs_[i] = j
        return self._function(**kwargs_)

    def __str__(self):
        s = 'Name:   %s\n' % self._function_name
        s += 'kwargs: %s\n' % str(self._function_kwargs)
        s += 'Sandbox:%s\n' % str(len(self._sandbox))
        s += str(self._source)
        return s

    def __eq__(self, other):
        if self._function == other._function and \
           self._function_kwargs == other._function_kwargs:
            return True
        return False

    # ==================== update kwargs ==================== #
    def __setitem__(self, key, value):
        if not isinstance(key, (str, int, float, long)):
            raise ValueError('Only accept string for kwargs key or int for '
                             'index of args, but type(key)={}'.format(type(key)))
        if isinstance(key, str):
            if key in self._function_kwargs:
                self._function_kwargs[key] = value
        else:
            key = int(key)
            if key < len(self._function_kwargs):
                key = self._function_kwargs.keys()[key]
                self._function_kwargs[key] = value

    def __getitem__(self, key):
        if not isinstance(key, (str, int, float, long)):
            raise ValueError('Only accept string for kwargs key or int for '
                             'index of args, but type(key)={}'.format(type(key)))
        if isinstance(key, str):
            return self._function_kwargs[key]
        return self._function_kwargs(int(key))

    # ==================== Pickling methods ==================== #
    def __getstate__(self):
        # conver to byte
        return (self._function_kwargs,
                self._function_order,
                self._function_name,
                self._sandbox,
                self._source)

    def __setstate__(self, states):
        (self._function_kwargs,
        self._function_order,
        self._function_name,
        self._sandbox,
        self._source) = states

        # ====== deserialize the function ====== #
        self._function, sandbox = _deserialize_function_sandbox(self._sandbox)
        if self._function is None:
            raise AttributeError('Cannot find function with name={} in sandbox'
                                 ''.format(self._function_name))


# ===========================================================================
# Singleton metaclass
# ===========================================================================
def singleton(cls):
    ''' Singleton for class instance, all __init__ with same arguments return
    same instance

    Note
    ----
    call .dispose() to fully destroy a Singeleton
    '''
    if not isinstance(cls, type):
        raise Exception('singleton decorator only accept class without any '
                        'addition parameter to the decorator.')

    orig_vars = cls.__dict__.copy()
    slots = orig_vars.get('__slots__')
    if slots is not None:
        if isinstance(slots, str):
            slots = [slots]
        for slots_var in slots:
            orig_vars.pop(slots_var)
    orig_vars.pop('__dict__', None)
    orig_vars.pop('__weakref__', None)
    return Singleton(cls.__name__, cls.__bases__, orig_vars)


class Singleton(type):
    _instances = defaultdict(list) # (arguments, instance)

    @staticmethod
    def _dispose(instance):
        clz = instance.__class__
        Singleton._instances[clz] = [(args, ins)
                                     for args, ins in Singleton._instances[clz]
                                     if ins != instance]

    def __call__(cls, *args, **kwargs):
        spec = inspect.getargspec(cls.__init__)
        kwspec = {}
        if spec.defaults is not None:
            kwspec.update(zip(reversed(spec.args), reversed(spec.defaults)))
        kwspec.update(zip(spec.args[1:], args))
        kwspec.update(kwargs)
        # convert all path to abspath to make sure same path are the same
        for i, j in kwspec.iteritems():
            if is_path(j):
                kwspec[i] = os.path.abspath(j)
        # check duplicate instances
        instances = Singleton._instances[cls]
        for arguments, instance in instances:
            if arguments == kwspec:
                return instance
        # not found old instance
        instance = super(Singleton, cls).__call__(*args, **kwargs)
        instances.append((kwspec, instance))
        setattr(instance, 'dispose',
                types.MethodType(Singleton._dispose, instance))
        return instance

# Override the module's __call__ attribute
# sys.modules[__name__] = cache


# ===========================================================================
# Helper for iterator
# ===========================================================================
def terminatable_iterator(func=None, finish_callback=None):
    """ Make an iterator terminatable by using
    iterator.send(SIG_TERMINATE_ITERATOR)

    Parameters
    ----------
    finish_callback : callable
        a function take 1 argument, which indicates iterator was forced to
        be terminated or not.

    Example
    -------
    >>> from odin import SIG_TERMINATE_ITERATOR
    >>> @terminatable_iterator(finish_callback=lambda x: print('end'))
    >>> def itgen():
    ...     for i in range(10):
    ...         yield i
    >>> it = itgen()
    >>> for i in it:
    ...     print(i)
    ...     if i == 5:
    ...         it.send(SIG_TERMINATE_ITERATOR)
    >>> # 0 1 2 3 4 5 'end'

    Note
    ----
    Do NOT modify `func` argument yourself.
    """
    if finish_callback is not None and \
    (not hasattr(finish_callback, '__call__') or
     len(inspect.getargspec(finish_callback).args) == 0):
        raise ValueError('finish_callback must be callable and accept at least '
                         '1 argument, which indicate iterator was forced to '
                         'be terminated or not.')

    def wrap_func(func):
        @wraps(func)
        def it_func(*args, **kwargs):
            it = func(*args, **kwargs)
            forced_to_terminate = False
            for i in it:
                if (yield i) == SIG_TERMINATE_ITERATOR:
                    forced_to_terminate = True
                    break
            # ====== ending the iterator ====== #
            if finish_callback is not None:
                finish_callback(forced_to_terminate)
            if forced_to_terminate:
                yield

        return it_func

    if func is None:
        return wrap_func
    return wrap_func(func)
