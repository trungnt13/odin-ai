from __future__ import absolute_import, division, print_function

import os
import shutil
import sys


def get_script_path():
  """Return the path of the script that calling this methods"""
  path = os.path.dirname(sys.argv[0])
  path = os.path.join('.', path)
  return os.path.abspath(path)


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
