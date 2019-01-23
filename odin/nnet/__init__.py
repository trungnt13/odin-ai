from __future__ import print_function, absolute_import, division

from odin.nnet.base import *
from odin.nnet.math_utils import *
from odin.nnet.conv import *
from odin.nnet.noise import *
from odin.nnet.shape import *
from odin.nnet.sampling import *
from odin.nnet.normalization import *
from odin.nnet.embedding import *
from odin.nnet.helper import *
from odin.nnet.rnn import *
from odin.nnet.time_delayed import *

from odin.nnet import models

from odin.utils import uuid, bin2folder, folder2bin

# ===========================================================================
# Helper method for serialize NNOp
# ===========================================================================
def serialize(nnops, path=None, save_variables=True, variables=[],
              binary_output=False, override=False):
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
  binary_output: bool (default: False)
      if `False` (by default), original way tensorflow serialize
      all variables, save all variables and nnop info to separated
      files within a folder `path`
      if `True`, convert all files in the folder to binary
      and save to a dictionary with its relative path, if
      `path` is not None, use pickle to save all binary data
      to a file
  override: bool
      if True, remove existed folder to override everything.

  Return
  ------
  path: str
      path to the folder that store NNOps and variables
  """
  # ====== check output_mode ====== #
  if path is None and not binary_output:
    raise ValueError('`path` cannot be None if `binary_output=False`')
  is_path_given = False if path is None else True
  if path is None:
    path = '/tmp/tmp' # default path
  path_folder = path + uuid(length=25) if binary_output else path
  # ====== getting save data and variables ====== #
  vars = []
  if save_variables:
    for op in as_tuple(nnops):
      if hasattr(op, 'variables'):
        for v in as_tuple(op.variables):
          if K.is_variable(v):
            vars.append(v)
  vars = list(set(vars + as_list(variables)))
  # ====== checking path ====== #
  # It is important to remove the `path_folder` AFTER getting all
  # the variables, since this can remove the path to restored
  # variables required in `op.variables`
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
  # save NNOps
  with open(nnops_path, 'wb') as f:
    cPickle.dump(nnops, f, protocol=cPickle.HIGHEST_PROTOCOL)
  # save Variables
  if len(vars) > 0:
    K.save_variables(vars, vars_path)
  # ====== convert folder to file or binary ====== #
  if binary_output:
    data = folder2bin(path_folder)
    # only return binary data
    if not is_path_given:
      shutil.rmtree(path_folder)
      return data
    # given path, save binary to path
    # check if override
    if os.path.exists(path):
      if override:
        if os.path.isfile(path):
          os.remove(path)
        else:
          shutil.rmtree(path)
      else:
        raise RuntimeError("File at path: %s exists, cannot override." % path)
    # write file
    with open(path, 'wb') as f:
      cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    shutil.rmtree(path_folder)
  return path


def deserialize(path_or_data, force_restore_vars=True):
  """
  Parameters
  ----------
  path_or_data : {string, dict}
      if a path is given (i.e. string types), load dumped model from
      given folder
      if a dictionary is given, load binary data directly

  force_restore_vars : bool (default=True)
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
  delete_after = True
  # ====== check path ====== #
  if is_string(path_or_data):
    # path to a file
    if os.path.isfile(path_or_data):
      with open(path_or_data, 'rb') as f:
        data = cPickle.load(f)
    # path to a folder
    elif os.path.isdir(path_or_data):
      path_folder = path_or_data
      delete_after = False
    else: # pickle string
      data = cPickle.loads(path_or_data)
  # given data
  elif isinstance(path_or_data, dict):
    data = path_or_data
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
      if delete_after:
        shutil.rmtree(path)
    else:
      nnops._set_restore_info(vars_path, delete_after)
  return nnops
