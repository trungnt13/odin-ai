from __future__ import print_function, absolute_import, division

from .base import *
from .conv import *
from .noise import *
from .shape import *
from .sampling import *
from .normalization import *
from .embedding import *
from .helper import *
from .rnn import *

from . import models
from odin.utils import uuid, bin2folder, folder2bin


# ===========================================================================
# Helper method for serialize NNOp
# ===========================================================================
def serialize(nnops, path=None, save_variables=True, variables=[],
              output_mode='folder', override=False):
  """ Serialize NNOp or list of NNOp and all necessary variables
  to a folder.

  Parameters
  ----------
  nnops: NNOp, Object, or list; tuple of NNOp and Object
  path: str
      path to a folder
  save_variables: bool
      if True, save all variables related to all given NNOps
  variables: list of tensorflow Variables
      additional list of variables to be saved with this model
  output_mode: {'file', 'folder', 'bin'}
      if 'folder' (by default), original way tensorflow serialize
      all variables.
      if 'bin' (or 'binary'), conver all files in the folder to binary
      and save to a dictionary with its relative path.
      if 'file', use pickle to save all binary data to a file
  override: bool
      if True, remove existed folder to override everythin.

  Return
  ------
  path: str
      path to the folder that store NNOps and variables
  """
  # ====== check output_mode ====== #
  output_mode = str(output_mode).lower()
  if output_mode not in ('folder', 'file', 'bin'):
    raise ValueError('`output_mode` can be: folder, file, or bin.')
  if output_mode in ('folder', 'file') and path is None:
    raise ValueError('`path` cannot be None in "folder" or "file" '
                     'output mode.')
  path_folder = '/tmp/tmp_%s' % uuid(length=12) \
      if path is None or output_mode == 'file' else path
  # ====== checking path ====== #
  if os.path.exists(path_folder):
    if os.path.isfile(path_folder):
      raise ValueError("path: '%s' is NOT a folder." % path_folder)
    elif override:
      shutil.rmtree(path_folder)
      os.mkdir(path_folder)
  else:
    os.mkdir(path_folder)
  nnops_path = os.path.join(path_folder, 'nnops.ai')
  vars_path = os.path.join(path_folder, 'variables')
  # ====== getting save data ====== #
  vars = []
  if save_variables:
    for op in as_tuple(nnops):
      if hasattr(op, 'variables'):
        for v in as_tuple(op.variables):
          if K.is_variable(v):
            vars.append(v)
  vars = list(set(vars + as_list(variables)))
  # save NNOps
  with open(nnops_path, 'wb') as f:
    cPickle.dump(nnops, f, protocol=cPickle.HIGHEST_PROTOCOL)
  # save Variables
  if len(vars) > 0:
    K.save_variables(vars, vars_path)
  # ====== convert folder to file or binary ====== #
  if output_mode != 'folder':
    data = folder2bin(path_folder)
    if output_mode == 'bin':
      return data
    # check if override
    if os.path.exists(path):
      if override:
        if os.path.isfile(path):
          os.remove(path)
        else:
          shutil.rmtree(path)
      else:
        raise RuntimeError("File at path: %s exists, cannot override."
                           % path)
    # write file
    with open(path, 'wb') as f:
      cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    shutil.rmtree(path_folder)
  return path


def deserialize(path, force_restore_vars=False):
  """
  Parameters
  ----------
  force_restore_vars : bool
      if `False`, this is special tricks, the unpickled NNOp stay useless
      until its variables are restored,
      but if we restore the variables right away, it create a
      session and prevent any possibility of running
      tensorflow with multiprocessing
      => store the `_restore_vars_path` in NNOp for later,
      and restore the variable when the NNOp is actually in used.

  Note
  ----
  if `force_restore_vars = False`, this create 1 flaw,
  if the nested NNOp is called before the unpickled NNOp
  restore its variables, the nested Ops cannot acquire its
  variables.

  """
  data = None
  path_folder = '/tmp/tmp_%s' % uuid(12)
  delete_folder = True
  # ====== check path ====== #
  if is_string(path):
    # path to a file
    if os.path.isfile(path):
      with open(path, 'rb') as f:
        data = cPickle.load(f)
    # path to a folder
    elif os.path.isdir(path):
      path_folder = path
      delete_folder = False
    else: # pickle string
      data = cPickle.loads(path)
  # given data
  elif isinstance(path, dict):
    data = path
  # ====== check data ====== #
  if data is not None:
    bin2folder(data, path=path_folder)
  path = path_folder
  # ====== read normally from folder ====== #
  nnops_path = os.path.join(path, 'nnops.ai')
  vars_path = os.path.join(path, 'variables')
  # ====== load the NNOps ====== #
  if not os.path.exists(nnops_path):
    raise ValueError("Cannot file path to serialized NNOps at: %s" % nnops_path)
  with open(nnops_path, 'rb') as f:
    nnops = cPickle.load(f)
  # ====== load the Variables ====== #
  if os.path.exists(vars_path + '.index'):
    if force_restore_vars:
      K.restore_variables(vars_path)
      # delete cached folder
      if delete_folder:
        shutil.rmtree(path)
    else:
      nnops._restore_vars_path = vars_path
      nnops._delete_vars_folder = delete_folder
  return nnops
