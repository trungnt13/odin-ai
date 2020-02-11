from __future__ import absolute_import, division, print_function

import inspect
import os
import shutil
import sys

from six import string_types


def get_script_path(module=None, return_dir=False):
  r""" Return the path of the running script or the given module

  Example:

    >>> get_script_path(__name__)
    # return the path to current module

    >>> get_script_path()
    # return the path to runnings script, e.g. "python train.py" -> train.py
  """
  if module is None:
    path = os.path.dirname(sys.argv[0])
    path = os.path.join('.', path)
    path = os.path.abspath(path)
  elif isinstance(module, string_types):
    module = sys.modules[module]
    path = os.path.abspath(module.__file__)
  else:
    module = inspect.getmodule(module)
    path = os.path.abspath(module.__file__)
  if return_dir:
    path = os.path.dirname(path)
  return path


def get_script_name():
  """Return the name of the running scipt file without extension"""
  name = os.path.basename(sys.argv[0])
  name = os.path.splitext(name)[0]
  return name


def get_folder_size(path):
  raise NotImplementedError


def clean_folder(path, filter=None, verbose=False):
  if os.path.exists(path) and os.path.isdir(path):
    for name in os.listdir(path):
      f = os.path.join(path, name)
      # filtering
      if filter is not None and callable(filter):
        if not filter(f):
          continue
      if verbose:
        print("Remove:", f)
      # remove
      if os.path.isfile(f):
        os.remove(f)
      elif os.path.isdir(f):
        shutil.rmtree(f)
