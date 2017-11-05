from __future__ import division, absolute_import, print_function

import os
import re
import shutil
import inspect
import numbers
import warnings
from itertools import chain
from functools import wraps
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from six.moves import zip, range, cPickle
from six import add_metaclass, types, string_types

import numpy as np

from odin import backend as K
from odin.utils.decorators import functionable
from odin.utils import (as_tuple, as_list, uuid, cache_memory, is_number,
                        is_string, is_path, is_primitives, ctext,
                        ShapeRef, DtypeRef, flatten_list, get_all_files)
from odin.backend.role import (add_role, has_roles, Parameter, Weight, Bias)

import tensorflow as tf


# ===========================================================================
# Global NNOp manager
# ===========================================================================
def get_all_nnops(model_scope=None, op_type=None):
    """ Return a dictionary of (name, nnops) for all created NNOp """
    allops = NNOp._ALL_NNOPS.values()
    if model_scope is not None:
        if not is_string(model_scope): model_scope = model_scope.name
        allops = [o for o in allops if o.name[:len(model_scope)] == model_scope]
    if op_type is not None:
        op_type = [i for i in as_tuple(op_type)
                   if is_string(op_type) or issubclass(op_type, NNOp)]
        allops = [o for o in allops
                  if any(o.__class__.__name__ == t if is_string(t)
                         else isinstance(o, t)
                         for t in op_type)]
    return allops


def _assign_new_nnop(nnops):
    if not isinstance(nnops, NNOp):
        raise ValueError("The new assigned NNOp must be instance of odin.nnet.NNOp "
                         ", but the given object has type: %s" % str(type(nnops)))
    name = nnops.name
    if name in NNOp._ALL_NNOPS:
        raise RuntimeError("Another NNOp of type: '%s', and name: '%s' has "
                           "already existed." % (type(NNOp._ALL_NNOPS[name]), name))
    NNOp._ALL_NNOPS[name] = nnops


# ===========================================================================
# Context manager
# ===========================================================================
__ARGS_SCOPE_STACK = [defaultdict(dict)]
_NAME_SCOPE = ''
_NNOP_ID = defaultdict(int)


@contextmanager
def nnop_scope(scope=None, id_start=None, ops=[], **kwargs):
    """Stores the default arguments for the given set of applied_nnops.

    For usage, please see examples at top of the file.

    Parameters
    ----------
    applied_nnops: List or tuple string, type, or NNOp
        a dictionary containing the current scope. When list_ops_or_scope is a
        dict, kwargs must be empty. When list_ops_or_scope is a list or tuple,
        then every op in it need to be decorated with @add_arg_scope to work.
    **kwargs: keyword=value that will define the defaults for each op in
        list_ops. All the ops need to accept the given set of arguments.

    Return
    ------
    the current_scope, which is a dictionary of {op: {arg: value}}

    Raises
    ------
    TypeError: if list_ops is not a list or a tuple.
    ValueError: if any op in list_ops has not be decorated with @add_arg_scope.

    Note
    ----
    if the name scope is given, an increasement ID is generated for
    duplicated NNOp instead of UUID.
    """
    # ====== prepare Name Scope ====== #
    global _NAME_SCOPE, _NNOP_ID
    # name scope
    if is_string(scope):
        if scope not in _NAME_SCOPE:
            _NAME_SCOPE = str(scope) if len(_NAME_SCOPE) == 0 else \
                _NAME_SCOPE + '/' + str(scope)
    # name scope
    if is_number(id_start):
        _NNOP_ID[_NAME_SCOPE] = int(id_start)
    elif isinstance(id_start, list) and is_number(id_start[0]):
        _NNOP_ID[_NAME_SCOPE] = int(id_start[0])
    # ====== update Arguments Scope ====== #
    ops = as_tuple(ops)
    # copy prevous scopes
    args_scope = defaultdict(dict)
    for i, j in __ARGS_SCOPE_STACK[-1].iteritems():
        args_scope[i] = j.copy()
    # update new scopes
    for o in ops:
        args_scope[o].update(kwargs)
    __ARGS_SCOPE_STACK.append(args_scope)
    # ====== return the scope ====== #
    with tf.variable_scope(_NAME_SCOPE):
        yield scope
    # ====== reset everything ====== #
    __ARGS_SCOPE_STACK.pop()
    _NAME_SCOPE = '/'.join(_NAME_SCOPE.split('/')[:-1])
    if isinstance(id_start, list):
        id_start[0] = _NNOP_ID[0]


def get_name_scope():
    return _NAME_SCOPE


def get_args_scope():
    return __ARGS_SCOPE_STACK[-1].copy()


# ===========================================================================
# Helper
# ===========================================================================
def _check_shape(s):
    if callable(s): return functionable(s)
    if is_number(s) or s is None:
        s = (s,)
    elif isinstance(s, np.ndarray):
        s = s.tolist()
    return tuple([int(i) if is_number(i) else None for i in s])


def _check_dtype(dtype):
    if callable(dtype): return functionable(dtype)
    # ====== check dtype ====== #
    if dtype is None:
        dtype = K.floatX
    elif isinstance(dtype, np.dtype) or is_string(dtype):
        dtype = str(dtype)
    elif isinstance(dtype, VariableDescriptor):
        dtype = DtypeRef(dtype)
    elif isinstance(dtype, tf.DType):
        dtype = dtype.base_dtype.name
    return dtype


def _shape_compare(shape1, shape2):
    """Return True if shape1 == shape2"""
    if len(shape1) != len(shape2):
        return False
    for s1, s2 in zip(shape1, shape2):
        if s1 != s2:
            return False
    return True


# ===========================================================================
# Input descriptor
# ===========================================================================
class VariableDescriptor(object):
    """ VariableDescriptor
    Store all the necessary information to create placeholder as input
    to any ComputationalGraph.

    Parameters
    ----------
    shape: tuple, list, TensorVariable, callable
        if TensorVariable is given, shape and dtype will be taken from
        given variable. if a callable object is given, the object must
        return shape information when called without any argument.
    dtype: str, numpy.dtype, callable
        dtype of input variable
    name: str, None, callable
        specific name for the variable

    Note
    ----
    This object is pickle-able and comparable
    """

    def __init__(self, shape, dtype=None, name=None):
        super(VariableDescriptor, self).__init__()
        # ====== placeholder ====== #
        self.__placeholder = None
        self._name = name if name is None else str(name)
        # Given a TensorVariabe, we don't want to pickle TensorVariable,
        # so copy all necessary information
        if K.is_tensor(shape):
            if dtype is None:
                self._dtype = _check_dtype(shape.dtype)
            self._shape = shape.get_shape().as_list()
            # store the placeholder so don't have to create it again
            self.__placeholder = shape
        # input the VariableDescriptor directly
        elif isinstance(shape, VariableDescriptor):
            self._shape = ShapeRef(shape)
            self._dtype = DtypeRef(shape) if dtype is None else _check_dtype(dtype)
        # input regular information flow
        else:
            self._shape = _check_shape(shape)
            self._dtype = _check_dtype(dtype)
        # ====== create reference ====== #
        # trick to store self in x, hence, no closure
        self._shape_ref = ShapeRef(self)
        self._dtype_ref = DtypeRef(self)

    # ==================== pickle ==================== #
    def __getstate__(self):
        return (self._shape, self._shape_ref,
                self._dtype, self._dtype_ref, self._name)

    def __setstate__(self, states):
        (self._shape, self._shape_ref,
         self._dtype, self._dtype_ref, self._name) = states
        self.__placeholder = None

    # ==================== properties ==================== #
    def set_placeholder(self, plh):
        if not K.is_placeholder(plh):
            raise ValueError("a placholder must be specified.")
        if plh.get_shape().as_list() == self.shape and \
        _check_dtype(plh.dtype) == self.dtype:
            self.__placeholder = plh
        else:
            raise ValueError("This VariableDescriptor require input with shape=%s,"
                             "and dtype=%s, but given a placholder with shape=%s, "
                             "dtype=%s." % (str(self.shape), self.dtype,
                            str(plh.get_shape().as_list()), _check_dtype(plh.dtype)))
        return self

    @property
    def placeholder(self):
        if self.__placeholder is None:
            self.__placeholder = K.placeholder(
                shape=self.shape, dtype=self.dtype, name=self.name)
        return self.__placeholder

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape() if callable(self._shape) else self._shape

    @property
    def shape_ref(self):
        """ ref is callable reference to the shape information of
        this descriptor, it will return the actual shape if you
        call it. """
        return self._shape_ref

    @property
    def dtype(self):
        return self._dtype() if callable(self._dtype) else self._dtype

    @property
    def dtype_ref(self):
        """ ref is callable reference to the dtype information of
        this descriptor, it will return the actual dtype if you
        call it. """
        return self._dtype_ref

    # ==================== override ==================== #
    def __str__(self):
        return "<VarDesc - name:%s shape:%s dtype:%s init:%s>" % \
        (str(self.name), str(self.shape), str(self.dtype),
         False if self.__placeholder is None else True)

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        # ====== compare to a TensorVariable ====== #
        if K.is_tensor(other):
            other = VariableDescriptor(
                shape=other.get_shape().as_list(),
                dtype=_check_dtype(other.dtype))
        # ====== compare to a VariableDescriptor ====== #
        if isinstance(other, VariableDescriptor):
            if _shape_compare(self.shape, other.shape) \
            and self.dtype == other.dtype:
                return 0
        # ====== compare to a shape tuple (ignore the dtype) ====== #
        elif isinstance(other, (tuple, list)):
            return 0 if _shape_compare(self.shape, other) else 1
        return 1


# ===========================================================================
# Main Ops
# ===========================================================================
class _NNOp_Meta(ABCMeta):
    """ arguments scope for the NNOp

    Note
    ----
    you can only modify the arguments and kwarguments using __call__
    from MetaClass, not __new__ of instance class.
    """
    def __call__(clazz, *args, **kwargs):
        NO_ARGUMENT = '[__no_argument__]'
        # getting the default arguments to check user intentionally override
        # default argument.
        spec = inspect.getargspec(clazz.__init__)
        # ignore the self argument
        default_args = OrderedDict([(i, NO_ARGUMENT) for i in spec.args[1:]])
        if spec.defaults is not None:
            for name, value in zip(spec.args[::-1], spec.defaults[::-1]):
                default_args[name] = value
        # ====== upate the current argument scope ====== #
        # get current scope
        key_name = [clazz, str(clazz), clazz.__name__]
        current_scope = get_args_scope()
        for n in key_name:
            if n in current_scope:
                default_args.update(current_scope[n])
        # ====== update user specified args and kwargs ====== #
        # update the new arguments into default arguments
        new_args = OrderedDict([(name, args[i]) if i < len(args) else
                                (name, default)
                                for i, (name, default) in enumerate(default_args.iteritems())])
        new_args.update(kwargs)
        return super(_NNOp_Meta, clazz).__call__(*[], **new_args)


@add_metaclass(_NNOp_Meta)
class NNOp(object):
    """ Basics of all Neural Network operators

    Properties
    ----------
    name: str
        identity of the operator, this name is the scope for its operator
        and should be unique.
    T: NNOp
        transpose operator of this one (NOTE: some ops does not support
        transpose and raise NotImplementedError)
    parameters: list of variables
        list of all parameters associated with this operator scope

    Abstract
    --------
    _apply(self, x, **kwargs): resulted variables
        apply take a list of variables and custom parameters to compute
        output variables
    _initialize(self, **kwargs):
        create parameters

    Override
    --------
    _transpose(self): NNOp
        return another NNOp which is transposed version of this ops

    Note
    ----
    All NNOp are pickle-able!
    You must use: protocol=cPickle.HIGHEST_PROTOCOL when dump NNOp.
    if NNOp is applied to a list of inputs, it will process each input seperated.
    """
    _ALL_NNOPS = {}

    def __new__(clazz, *args, **kwargs):
        # cPickle call __new__
        if len(args) == 1 and len(kwargs) == 0 and \
        (is_string(args[0]) and '[__name__]' in args[0]):
            name = args[0].replace('[__name__]', '')
            if name in NNOp._ALL_NNOPS:
                instance = NNOp._ALL_NNOPS[name]
                if not isinstance(instance, clazz):
                    raise RuntimeError("Found duplicated NNOp with type: '%s', "
                        "which is different from pickled type: '%s'" %
                        (type(instance), clazz))
                return instance
        # New for first time create instance
        # ====== update Op name if it is None ====== #
        name = kwargs.get('name', None)
        # automatic generate name
        if name is None:
            name = clazz.__name__
            if len(_NAME_SCOPE) > 0:
                name = _NAME_SCOPE + '/' + name + "_" + str(_NNOP_ID[_NAME_SCOPE])
                _NNOP_ID[_NAME_SCOPE] += 1
            else:
                name = name + "_" + str(uuid())
        # regulation for the NNOp name
        elif is_string(name):
            if '/' in name or ':' in name:
                raise ValueError("NNOp cannot contain '\\' or ':', given name is: %s" % name)
            # add name scope
            if _NAME_SCOPE is not None:
                name = _NAME_SCOPE + '/' + name
                _NNOP_ID[0] += 1
        else:
            raise ValueError("name for NNOp must be string, but given name "
                             "has type: %s" % (name))
        # ====== check duplicated Op name ====== #
        if name in NNOp._ALL_NNOPS:
            old_clazz = NNOp._ALL_NNOPS[name].__class__
            if clazz != old_clazz:
                raise RuntimeError("Found predefined NNOp with type: %s, but "
                    "the new NNOp has type: %s" % (old_clazz, clazz))
            return NNOp._ALL_NNOPS[name]
        # ====== allocate new Op ====== #
        new_op = super(NNOp, clazz).__new__(clazz)
        new_op._name = name
        return new_op

    def __init__(self, **kwargs):
        # ====== create default NNOp name ====== #
        if not hasattr(self, '_name'):
            raise ValueError("NNOp must be given a name when initialized")
        self._save_states = {'_name': self._name}
        # list of VariableDescriptor
        self._input_desc = []
        self._transpose_ops = None
        self._is_initialized = False
        # mapping: variable_name -> (tensorflow_name, 'tensor' or 'variable')
        self._variable_info = OrderedDict()
        # special flags to detect if cPickle called with protocol >= 2
        self._new_args_called = False

    def _check_input_desc(self, inputs):
        inputs = [VariableDescriptor(shape=i) for i in as_tuple(inputs)]
        # first time initialized the input description
        if len(self._input_desc) == 0:
            self._input_desc = inputs
            for i, j in enumerate(self._input_desc):
                j._name = '%s_inp%.2d' % (self.name, i)
        else:
            inputs = [j if i is None else i
                      for i, j in zip(inputs, self._input_desc)]
            inputs = inputs + self._input_desc[len(inputs):]
        return [i.placeholder for i in inputs]

    # ==================== pickling method ==================== #
    def __getstate__(self):
        if not self._new_args_called:
            raise RuntimeError(
                "You must use argument `protocol=cPickle.HIGHEST_PROTOCOL` "
                "when using `pickle` or `cPickle` to be able pickling NNOp.")
        self._new_args_called = False
        return self._save_states

    def __setstate__(self, states):
        self._new_args_called = False
        self._save_states = states
        for key, val in self._save_states.iteritems():
            setattr(self, key, val)
        # # ====== check exist NNOp ====== #
        name = self.name
        if name in NNOp._ALL_NNOPS:
            if type(NNOp._ALL_NNOPS[name]) != type(self):
                raise RuntimeError("Found duplicated NNOp with name: '%s' and type: '%s'"
                    % (name, str(type(NNOp._ALL_NNOPS[name]))))
            # cannot find '[__name__]' due to not specify protocol
            # hence the __getnewargs__ is not called
            elif NNOp._ALL_NNOPS[name] != self:
                raise RuntimeError("Mismatch NNOp, two NNOps with the same name "
                                   "are initizlied:\n%s\nis different from:\n%s" %
                                   (str(NNOp._ALL_NNOPS[name]), str(self)))

    def __getnewargs__(self):
        self._new_args_called = True
        return ('[__name__]' + self.name,)

    # ==================== properties ==================== #
    @cache_memory
    def get_variable(self, name, shape=None, initializer=None, roles=[]):
        """
        Parameters
        ----------
        name: str
            name for the variable
        shape: tuple, list
            expected shape for given variable
        initializer: variable, numpy.ndarray, function
            specification for initializing the weights, if a function
            is given, the arguments must contain `shape`
        roles: `odin.backend.role.Role`
            categories of this variable
        """
        if shape is not None:
            shape = tuple(shape)  # convert to tuple if needed
            if any(d <= 0 or d is None for d in shape):
                raise ValueError((
                    "Cannot create param with a non-positive shape dimension. "
                    "Tried to create param with shape=%r, name=%r") %
                    (shape, name))
        if initializer is None and shape is None:
            if name not in self._variable_info:
                raise ValueError("Cannot find variable with name: %s for NNOps "
                    "with name: %s" % (name, self.name))
            var_name, t = self._variable_info[name]
            # get variable
            if t == 'variable':
                var = K.get_all_variables(full_name=var_name)
                if len(var) == 0:
                    raise RuntimeError("Cannot find variable with name: %s" % var_name)
                var = var[0]
            # get tensor
            elif t == 'tensor':
                name, footprint = var_name
                op = K.get_operations(footprint=footprint)
                if len(op) == 0:
                    raise RuntimeError("Cannot find any Op with given footprint: %s" % footprint)
                var = op[0]._outputs[int(name.split(':')[-1])]
            # only care about the first variable
            return add_role(var, roles)
        #####################################
        # 0. initializing function.
        if callable(initializer):
            var = initializer(shape)
        # is a scalar
        elif is_number(initializer):
            var = np.full(shape=shape, fill_value=initializer, dtype='float32')
        # else actual tensor
        else:
            var = initializer
        #####################################
        # 1. Numpy ndarray.
        if isinstance(var, np.ndarray):
            var = K.variable(var, shape=shape, name=name)
            self._variable_info[name] = (var.name, 'variable')
        #####################################
        # 2. Shared variable, just check the shape.
        elif K.is_variable(var):
            _shape = var.get_shape().as_list()
            if shape is not None and tuple(shape) != tuple(_shape):
                raise Exception('Require variable with shape=%s, but was given different '
                                'shape=%s, name:%s.' %
                                (str(shape), str(_shape), str(name)))
            self._variable_info[name] = (var.name, 'variable')
        #####################################
        # 3. expression, we can only check number of dimension.
        elif K.is_tensor(var):
            # We cannot check the shape here, Theano expressions (even shared
            # variables) do not have a fixed compile-time shape. We can check the
            # dimensionality though.
            # Note that we cannot assign a name here. We could assign to the
            # `name` attribute of the variable, but the user may have already
            # named the variable and we don't want to override this.
            if shape is not None and var.get_shape().ndims != len(shape):
                raise Exception("parameter with name=%s has %d dimensions, should be "
                                "%d" % (name, var.get_shape().ndims, len(shape)))
            self._variable_info[name] = ((var.name, K.get_operation_footprint(var.op)),
                                         'tensor')
        #####################################
        # 4. Exception.
        else:
            raise RuntimeError("cannot initialize parameters: 'spec' is not "
                               "a numpy array, a Theano expression, or a "
                               "callable")
        # ====== assign annotations ====== #
        return add_role(var, roles)

    @property
    def name(self):
        return self._name

    @property
    def T(self):
        """ Return new ops which is transpose of this ops """
        if self._transpose_ops is None:
            self._transpose_ops = self._transpose()
            if not isinstance(self._transpose_ops, NNOp):
                raise ValueError("The _transposed method must return NNOp."
                                 "but the returned object has type=%s" %
                                 str(type(self._transpose_ops)))
        return self._transpose_ops

    @property
    def variables(self):
        """ Get all variables related to this Op"""
        # created variable from `get_variable`
        allname = [name for _, (name, t) in self._variable_info.iteritems()
                   if t == 'variable']
        allvars = [v for v in K.get_all_variables() if v.name in allname]
        # related variables to all `Tensor`
        tensors = [self.get_variable(name)
                   for name, (info, t) in self._variable_info.iteritems()
                   if t == 'tensor']
        tensors = K.ComputationGraph(tensors).variables
        # all variables within the scope
        scope_vars = K.get_all_variables(scope=self.name)
        return sorted(set(allvars + tensors + scope_vars),
                      key=lambda x: x.name)

    @property
    def parameters(self):
        """ return all TensorVariables which have the PARAMETER role"""
        return [i for i in self.variables if has_roles(i, Parameter)]

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def placeholders(self):
        """ Create list of placeholder to represent inputs of this NNOp
        """
        x = [i.placeholder for i in self._input_desc]
        return x[0] if len(x) == 1 else x

    @property
    def nb_input(self):
        return len(self._input_desc)

    @property
    def input_shape(self):
        x = [i.shape for i in self._input_desc]
        return x[0] if len(x) == 1 else x

    @property
    def input_shape_ref(self):
        x = [i.shape_ref for i in self._input_desc]
        return x[0] if len(x) == 1 else x

    def __setattr__(self, name, value):
        # this record all assigned attribute to pickle them later
        # check hasattr to prevent recursive loop at the beginning before
        # __init__ is called
        if hasattr(self, '_save_states') and name != '_save_states':
            # otherwise, only save primitive types
            if isinstance(value, _PRIMITIVE_TYPES):
                self._save_states[name] = value
        return super(NNOp, self).__setattr__(name, value)

    def __getattr__(self, name):
        # merge the attributes of ops wit its configuration
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self._variable_info:
            return self.get_variable(name)
        raise AttributeError("NNOp cannot find attribute with name: %s" % name)

    # ==================== abstract method ==================== #
    def _initialize(self, **kwargs):
        """ This function is only called once, for the first time you
        apply this Ops
        """
        return None

    @abstractmethod
    def _apply(self, X, **kwargs):
        raise NotImplementedError

    def _transpose(self):
        raise NotImplementedError

    # ==================== interaction method ==================== #
    def apply(self, X, **kwargs):
        # self.name can contain ModelDescriptor varable scope, hence,
        # remove the scope here
        name = self.name
        if '/' in name:
            name = name.split('/')[-1]
        with tf.variable_scope(name, reuse=self.is_initialized):
            # ====== initialize first ====== #
            # only select necessary arguments
            argspec = inspect.getargspec(self._initialize)
            keywords = {}
            # kwargs must be specified in args, or the _initialize
            # must accept **kwaobject, class_or_type_or_tuplergs
            for i, j in kwargs.iteritems():
                if argspec.keywords is not None or i in argspec.args:
                    keywords[i] = j
            # initialize the operator (call the initilazation process)
            X = self._check_input_desc(X)
            if not self._is_initialized:
                self._initialize(**keywords)
                self._is_initialized = True
                # only assign new NNOp if it is initialized
                _assign_new_nnop(self)
            # ====== calculate and return outputs ====== #
            rets = self._apply(X[0] if len(X) == 1 else X, **kwargs)
            return rets

    def __call__(self, X, **kwargs):
        return self.apply(X, **kwargs)

    def __str__(self):
        # ====== get all attrs ====== #
        all_attrs = dir(self)
        print_attrs = {}
        for name in all_attrs:
            if '_' != name[0] and (len(name) >= 2 and '__' != name[:2]) and\
            'name' != name and 'is_initialized' != name:
                attr = getattr(self, name)
                if is_primitives(attr) or inspect.isfunction(attr):
                    print_attrs[name] = attr
        print_attrs = sorted(print_attrs.iteritems(), key=lambda x: x[0])
        # ====== format the output ====== #
        ops_format = '<%s, name: %s, init: %s>\n'
        ops_format = ops_format % (ctext(self.__class__.__name__, 'cyan'),
                                   ctext(self.name, 'MAGENTA'),
                                   self._is_initialized)
        for i, j in print_attrs:
            ops_format += "\t%s: %s\n" % (ctext(i, 'yellow'), str(j))
        return ops_format[:-1]

    # ==================== Slicing ==================== #
    def __getitem__(self, key):
        return NNSliceOp(self, key)


_PRIMITIVE_TYPES = (tuple, list, dict, string_types, type(True),
                    types.FunctionType, numbers.Number, type(None),
                    K.rand.constant, NNOp, VariableDescriptor, type)


# ===========================================================================
# Helper
# ===========================================================================
class NNSliceOp(NNOp):

    def __init__(self, ops, slice):
        if not isinstance(ops, NNOp):
            raise ValueError('ops must be instance of NNOp, but was given argument '
                             'has %s' % str(type(ops)))
        super(NNSliceOp, self).__init__()
        self._ops = ops
        if not isinstance(slice, (tuple, list)):
            slice = [slice]
        self.slice = slice

    @property
    def variables(self):
        return self._ops.variables

    def _apply(self, X, **kwargs):
        y = self._ops.apply(X, **kwargs)
        return_list = True if isinstance(y, (tuple, list)) else False
        # apply slice and calculate the shape
        output = [i[self.slice] for i in as_tuple(y)]
        # return output
        if return_list:
            return output
        return output[0]

    def __str__(self):
        ops_format = '<ops: %s, name: %s, init: %s, slice: %s>'
        return ops_format % (self._ops.__class__.__name__, self._ops.name,
                             self._ops.is_initialized, str(self.slice))


class NNTransposeOps(NNOp):
    """ TransposeOps
    Create a transposed view of the origin NNOp
    """

    def __init__(self, ops, **kwargs):
        name = ops.name.split("/")[-1]
        super(NNTransposeOps, self).__init__(name=name + '_transpose')
        if not isinstance(ops, NNOp):
            raise ValueError("NNTransposeOps can only be applied for instance of "
                             "odin.nnet.NNOp, but was given type=%s" % str(type(ops)))
        self._transpose_ops = ops

    def _transpose(self):
        # return original Ops to prevent infinite useless loop of transpose
        return self._transpose_ops

    def _initialize(self, **kwargs):
        if not self._transpose_ops.is_initialized:
            raise RuntimeError("The original NNOp with name:%s have not been "
                               "initialized, you must call the original NNOp "
                               "first." % self._ops)

    def __str__(self):
        ops_format = '<original_ops: %s, name: %s, init: %s>'
        return ops_format % (self._transpose_ops.__class__.__name__,
                             self.name, self._transpose_ops.is_initialized and
                             self.is_initialized)


# ===========================================================================
# Simple ops
# ===========================================================================
class Dense(NNOp):

    def __init__(self, num_units,
                 W_init=K.rand.glorot_uniform,
                 b_init=K.rand.constant(0),
                 activation=K.linear,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.activation = (K.linear if activation is None else activation)
        self.W_init = W_init
        self.b_init = b_init
        self.num_units = num_units

    # ==================== abstract methods ==================== #
    def _transpose(self):
        # create the new dense
        return TransposeDense(self)

    def _initialize(self):
        input_shape = self.input_shape
        shape = (input_shape[-1], self.num_units)
        self.get_variable(name='W', shape=shape, initializer=self.W_init,
                          roles=Weight)
        if self.b_init is not None:
            self.get_variable(initializer=self.b_init,
                shape=(self.num_units,), name='b', roles=Bias)

    def _apply(self, X):
        # calculate projection
        activation = K.dot(X, self.W)
        # add the bias
        if self.b_init is not None:
            activation = activation + self.b
        # Nonlinearity might change the shape of activation
        return self.activation(activation)


class TransposeDense(NNTransposeOps):

    def _initialize(self):
        super(TransposeDense, self)._initialize()
        self.num_units = self.T.input_shape[-1]
        if self.T.b_init is not None:
            self.get_variable(initializer=self.T.b_init,
                shape=(self.num_units,), name='b', roles=Bias)

    def _apply(self, X):
        # calculate projection
        activation = K.dot(X, tf.transpose(self.T.W))
        if self.T.b_init is not None:
            activation = activation + self.b
        # Nonlinearity might change the shape of activation
        return self.T.activation(activation)


class ParametricRectifier(NNOp):
    """ This class is adpated from Lasagne:
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE
    A layer that applies parametric rectify activation to its input
    following [1]_ (http://arxiv.org/abs/1502.01852)
    Equation for the parametric rectifier linear unit:
    :math:`\\varphi(x) = \\max(x,0) + \\alpha \\min(x,0)`
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    alpha : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the alpha values. The
        shape must match the incoming shape, skipping those axes the alpha
        values are shared over (see the example below).
        See :func:`lasagne.utils.create_params` for more information.
    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the parameters of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the parameter over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, which corresponds to a single scalar parameter.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
     References
    ----------
    .. [1] K He, X Zhang et al. (2015):
       Delving Deep into Rectifiers: Surpassing Human-Level Performance on
       ImageNet Classification,
       http://link.springer.com/chapter/10.1007/3-540-49430-8_2
    Notes
    -----
    The alpha parameter dimensionality is the input dimensionality minus the
    number of axes it is shared over, which matches the same convention as
    the :class:`BiasLayer`.
    >>> layer = ParametricRectifierLayer((20, 3, 28, 28), shared_axes=(0, 3))
    >>> layer.alpha.get_value().shape
    (3, 28)
    """

    def __init__(self, alpha_init=K.rand.constant(0.25),
                 shared_axes='auto', **kwargs):
        super(ParametricRectifier, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        self.shared_axes = shared_axes

    # ==================== abstract methods ==================== #
    def _initialize(self):
        if self.shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif self.shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(self.shared_axes, int):
            self.shared_axes = (self.shared_axes,)

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ParametricRectifierLayer needs input sizes for "
                             "all axes that alpha's are not shared over.")
        self.alpha = self.get_variable(initializer=self.alpha_init,
            shape=shape, name="alpha", roles=Parameter)

    def _apply(self, x):
        axes = iter(range(K.ndim(self.alpha)))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes)
                   for input_axis in range(K.ndim(x))]
        alpha = K.dimshuffle(self.alpha, pattern)
        return K.relu(x, alpha)
