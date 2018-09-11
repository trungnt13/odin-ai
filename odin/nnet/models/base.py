from __future__ import absolute_import

import os
import base64
from collections import Mapping
from six import add_metaclass
from abc import ABCMeta, abstractmethod

from odin.fuel import Dataset
from odin.nnet.base import NNOp, VariableDesc
from odin.utils import get_datasetpath, get_file, is_string, as_tuple, is_number

from zipfile import ZipFile, ZIP_DEFLATED


def _validate_shape_dtype(x):
  if not isinstance(x, tuple):
    return False
  if not len(x) == 2:
    return False
  shape, dtype = x
  # check shape
  if not isinstance(shape, tuple) and \
  all(is_number(i) or isinstance(i, type(None)) for i in x):
    return False
  # check dtype
  if not is_string(dtype):
    return False
  return True

@add_metaclass(ABCMeta)
class Model(NNOp):
  """ Model """
  ORIGIN = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLW0wZGVscy8=\n'
  BASE_DIR = get_datasetpath(root='~')

  def __init__(self, **kwargs):
    super(Model, self).__init__(**kwargs)
    input_info = self.get_input_info()
    if not isinstance(input_info, Mapping) or \
    len(input_info) == 0 or \
    not all(is_string(k) and _validate_shape_dtype(v)
            for k, v in input_info.items()):
      raise ValueError("`get_input_info` must return a (length > 0) Mapping "
          "of: 'input-name' -> (shape-tuple, dtype-string), but the "
          "returned value is: %s" % str(input_info))
    # ====== init kwargs_desc ====== #
    for name, (shape, dtype) in input_info.items():
      self._kwargs_desc[name] = VariableDesc(
          shape=shape, name=name, dtype=dtype)

  @abstractmethod
  def get_input_info(self):
    pass

  def get_loaded_param(self, name):
    ds = self.__class__.load_parameters()
    if is_string(name):
      return_1_param = True
    else:
      return_1_param = False
    name = as_tuple(name, t=str)
    if any(n not in ds for n in name):
      raise RuntimeError("Cannot find parameter with name:'%s' from loaded "
          "dataset at path: '%s'" % (name, ds.path))
    params = [ds[n][:] for n in name]
    return params[0] if return_1_param else tuple(params)

  @classmethod
  def load_parameters(clazz):
    # ====== all path ====== #
    name = clazz.__name__ + '.zip'
    path = os.path.join(base64.decodebytes(Model.ORIGIN).decode(), name)
    param_path = get_datasetpath(name=clazz.__name__, override=False)
    zip_path = os.path.join(Model.BASE_DIR, name)
    # ====== get params files ====== #
    if not os.path.exists(param_path) or \
    len(os.listdir(param_path)) == 0:
      get_file(name, origin=path, outdir=Model.BASE_DIR)
      zf = ZipFile(zip_path, mode='r', compression=ZIP_DEFLATED)
      zf.extractall(path=Model.BASE_DIR)
      zf.close()
      # check if proper unzipped
      if not os.path.exists(param_path) or \
      len(os.listdir(param_path)) == 0:
        raise RuntimeError("Zip file at path:%s is not proper unzipped, "
            "cannot find downloaded parameters at path: %s" %
            (zip_path, param_path))
      else:
        os.remove(zip_path)
    # ====== create and return the params dataset ====== #
    ds = Dataset(param_path, read_only=True)
    return ds
