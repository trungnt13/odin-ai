import functools
import inspect
import os
import re
import sys
import tempfile
import traceback
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, List, Optional, Union
import logging

import numpy as np
from odin.utils import as_tuple, clean_folder, clear_folder, get_all_folder
from odin.utils.crypto import md5_checksum
from six import string_types

try:
  from hydra import __version__
  assert int(__version__.split('.')[0]) >= 1, \
    "Require hydra-core version >= 1.0.0"
  from hydra._internal.config_loader_impl import ConfigLoaderImpl
  from hydra._internal.config_search_path_impl import ConfigSearchPathImpl
  from hydra._internal.utils import _run_hydra, get_args_parser
  from hydra.experimental import compose, initialize
  from hydra.types import TaskFunction
  from hydra.utils import HydraConfig, to_absolute_path
  from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError as e:
  print(e)

__all__ = [
    'pretty_print', 'flatten_config', 'hash_config', 'save_to_yaml', \
    'get_hydra_config', 'get_overrides', 'get_output_dir', 'get_sweep_dir', \
    'run_hydra'
]

# ===========================================================================
# Helpers
# ===========================================================================
YAML_REGEX = re.compile(r"\w+: \w+")
OVERRIDE_PATTERN = re.compile(r"\A[\+\~]?[\w\.\\\@]+=[\w\(\)\[\]\{\}\,\.\']+")
JOBS_PATTERN = re.compile(r"\A-{1,2}j=?(\d+)\Z")
LIST_PATTERN = re.compile(r"\A-{1,2}l(ist)?\Z")
SUMMARY_PATTERN = re.compile(r"\A-{1,2}summary\Z")
REMOVE_EXIST_PATTERN = re.compile(r"\A-{1,2}override\Z")
RESET_PATTERN = re.compile(r"\A-+r(eset)?\Z")
TIME_FMT = r'%d%b%y_%H%M%S'
HYDRA_TIME_FMT = r"${now:%d%b%y_%H%M%S}"
logger = logging.getLogger(__name__)


def _insert_argv(key, value, is_value_string=True):
  if not any(key in i for i in sys.argv):
    if is_value_string:
      sys.argv.insert(1, f"{key}='{value}'")
    else:
      sys.argv.insert(1, f"{key}={value}")


def _save_config_to_tempdir(cfg: Union[str, dict, list, tuple]):
  config_dir = tempfile.mkdtemp(prefix="hydra_", suffix="_odin")
  path = os.path.join(config_dir, 'base.yaml')
  # convert everything to string
  if isinstance(cfg, (list, tuple)):
    cfg = dict(cfg)
  if isinstance(cfg, dict):
    cfg = DictConfig({str(i): j for i, j in cfg.items()})
    cfg = OmegaConf.to_yaml(cfg)
  elif isinstance(cfg, DictConfig):
    cfg = OmegaConf.to_yaml(cfg)
  elif isinstance(cfg, string_types):
    ...
  else:
    raise ValueError(f"No support for serialize config of type: {type(cfg)}")
  # save to file
  with open(path, 'w') as f:
    f.write(cfg)
  return config_dir, 'base'


def _abspath(path):
  if '$' in path:
    if not re.search(r"\$\{\w+\}", path):
      raise ValueError("Wrong specification for env variable %s, use ${.}" %
                       path)
    path = os.path.expandvars(path)
  if '$' in path:
    raise ValueError("Invalid path '%s', empty env variable" % path)
  path = os.path.abspath(os.path.expanduser(path))
  return path


def pretty_print(cfg: dict, ncol=3, return_str=False) -> str:
  ncol = int(ncol)
  text = ''
  if isinstance(cfg, DictConfig):
    pretty = OmegaConf.to_yaml(cfg).split('\n')
  else:
    pretty = ['%s: %s' % (str(i), str(j)) for i, j in pretty.items()]
  max_len = max(len(i) for i in pretty)
  fmt = f'%-{max_len}s'
  for i, s in enumerate(pretty):
    s = fmt % s.strip()
    if i % ncol in list(range(1, ncol - 1)):
      text += " | %s" % s
    elif i % ncol == (ncol - 1):
      text += " | %s\n" % s
    else:
      text += ' ' + s
  if return_str:
    return text
  print(text)
  return text


def flatten_config(cfg: dict, base='', max_depth=-1) -> dict:
  r""" Flatten a dictionary and its sub-dictionary into a single dictionary
  by concatenating the key with '.' character """
  c = {}
  # use stack for DFS
  stack = [(0, base, k, v) for k, v in cfg.items()]
  while len(stack) > 0:
    # depth, base, key, value
    d, b, k, v = stack.pop()
    # name
    if '.' in k:
      raise KeyError(f"Invalid key {k} contain '.' character")
    if len(b) > 0:
      k = b + '.' + k
    # pre-processing
    if isinstance(v, ListConfig):
      v = list(v)
    elif isinstance(v, DictConfig):
      v = dict(v)
    # going deeper
    if isinstance(v, dict) and len(v) > 0 and (max_depth < 0 or
                                               d < max_depth - 1):
      for i, j in v.items():
        stack.append((d + 1, k, i, j))
    else:
      assert k not in c, (f"Duplicated key={k}, the config is {cfg}")
      c[k] = v
  return c


def hash_config(cfg: DictConfig,
                exclude_keys: Optional[List[str]] = None,
                length=6) -> str:
  """Create an unique hash code from `DictConfig`

  Arguments:
    cfg : {dict}
      the configuration to generate an unique identity
    exclude_keys : list of string,
      given keys will be ignored from theconfiguration
    length: {int}
      maximum length of the hash code.

  Return:
    a hash string
  """
  assert isinstance(cfg, (DictConfig, dict))
  # process exclude_keys
  if exclude_keys is not None and len(exclude_keys) > 0:
    exclude_keys = as_tuple(exclude_keys, t=string_types)
    cfg = deepcopy(cfg)
    for key in exclude_keys:
      if '.' in key:
        key = key.split('.')
        attr = cfg
        for i in key[:-1]:
          attr = getattr(attr, i)
        del attr[key[-1]]
      else:
        del cfg[key]
  cfg = flatten_config(cfg, base='', max_depth=-1)
  return md5_checksum(cfg)[:int(length)]


def save_to_yaml(config: DictConfig, path: Optional[str] = None) -> str:
  """Save the given configuration to given path in YAML format

  Parameters
  ----------
  config : DictConfig
      the configuration
  path : Optional[str], optional
      the path to output YAML file. If None, use the `get_output_dir()/config.yaml`
      , by default None

  Returns
  -------
  str
      path to the output YAML file
  """
  if path is None:
    output_dir = get_output_dir()
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    path = os.path.join(output_dir, 'config.yaml')
  else:
    path = _abspath(path)
  with open(path, 'w') as f:
    f.write(OmegaConf.to_yaml(config, sort_keys=True))
  return path


def get_hydra_config() -> DictConfig:
  r""" Return the configuration """
  return HydraConfig.get()


def get_overrides() -> str:
  """Return the configuration overrides"""
  return HydraConfig.get().job.override_dirname


def get_output_dir(subfolder: Optional[str] = None) -> str:
  """Return an unique output dir based on the configuration overrides"""
  path = HydraConfig.get().run.dir
  if subfolder is not None:
    name = os.path.basename(path)
    path = os.path.dirname(path)
    path = os.path.join(path, subfolder, name)
    if not os.path.exists(path):
      os.makedirs(path)
  return path


def get_sweep_dir() -> str:
  return HydraConfig.get().sweep.dir


# ===========================================================================
# Main Function
# ===========================================================================
def run_hydra(output_dir: str = '/tmp/outputs',
              exclude_keys: List[str] = []) -> Callable[[TaskFunction], Any]:
  """ A modified main function of Hydra-core for flexibility
  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

  Useful commands:
    - `hydra/launcher=joblib` enable joblib launcher
    - `hydra.launcher.n_jobs=-1` set maximum number of processes
    - `--list` or `--summary` : list all exist experiments
    - `-j2` : run multi-processing (with 2 processes)
    - `--override` : override existed model of the given experiment
    - `--reset` : remove all files and folder in the output_dir

  Examples
  --------
  ```
  @experimenter.run_hydra()
  def run(cfg: DictConfig):
    print(cfg, type(cfg))

  run("/tmp/conf/base.yaml")
  ```

  Note
  ------
  to add `db` sub-config file to `database` object, from command line
      `python tmp.py db=mysql`, with the `#@package database`
      on top of the `mysql.yaml`, otherwise, from `base.yaml`
  ```
  defaults:
    - db: mysql
  ```

  Adding db to specific attribute of `database` object, from command line
      `python tmp.py db@database.src=mysql`, otherwise, from `base.yaml`
  ```
  defaults:
    - db@database.src: mysql
  ```
  """
  output_dir = _abspath(output_dir)
  ### check if reset all the experiments
  for i, a in enumerate(list(sys.argv)):
    if RESET_PATTERN.match(a):
      print('*Reset all experiments:')
      clean_folder(output_dir, verbose=True)
      sys.argv.pop(i)
      break
  ### create the log dir
  log_dir = os.path.join(output_dir, 'logs')
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  def main_decorator(task_function: TaskFunction) -> Callable[[], None]:

    @functools.wraps(task_function)
    def decorated_main(
        config: Union[str, dict, list, tuple, DictConfig]) -> Any:
      ### string
      if isinstance(config, string_types):
        # path to a config file
        if os.path.isfile(config):
          config_name = os.path.basename(config).replace(".yaml", "")
          config_path = os.path.dirname(config)
        # path to a directory
        elif os.path.isdir(config):
          config_path = config
          if not os.path.exists(os.path.join(config_path, 'base.yaml')):
            config_name = "base"  # default name
          else:
            config_name = sorted([
                i for i in os.listdir(config_path) if '.yaml' in i
            ])[0].replace(".yaml", "")
        # YAML string
        else:
          config_path, config_name = _save_config_to_tempdir(config)
      ### dictionary, tuple, list, DictConfig
      else:
        config_path, config_name = _save_config_to_tempdir(config)
      ### list all experiments command
      for a in sys.argv:
        if LIST_PATTERN.match(a) or SUMMARY_PATTERN.match(a):
          print("Output dir:", output_dir)
          all_logs = defaultdict(list)
          for i in os.listdir(log_dir):
            name, time_str = i.replace('.log', '').split(':')
            all_logs[name].append((time_str, os.path.join(log_dir, i)))
          for fname in sorted(os.listdir(output_dir)):
            path = os.path.join(output_dir, fname)
            # basics meta
            print(
                f" {fname}", f"({len(os.listdir(path))} files)"
                if os.path.isdir(path) else "")
            # show the log files info
            if fname in all_logs:
              for time_str, log_file in all_logs[fname]:
                with open(log_file, 'r') as f:
                  log_data = f.read()
                  lines = log_data.split('\n')
                  n = len(lines)
                  print(
                      f'  log {datetime.strptime(time_str, TIME_FMT)} ({n} lines)'
                  )
                  for e in [l for l in lines if '[ERROR]' in l]:
                    print(f'   {e.split("[ERROR]")[1]}')
          exit()
      ### check if overrides provided
      is_overrided = False
      for a in sys.argv:
        match = OVERRIDE_PATTERN.match(a)
        if match and not any(k in match.string for k in exclude_keys):
          is_overrided = True
      ### formatting output dirs
      if is_overrided:
        override_id = r"${hydra.job.override_dirname}"
      else:
        override_id = r"default"
      ### check if enable remove exists experiment
      remove_exists = False
      for i, a in enumerate(list(sys.argv)):
        match = REMOVE_EXIST_PATTERN.match(a)
        if match:
          remove_exists = True
          sys.argv.pop(i)
          break
      ### parallel jobs provided
      jobs = 1
      for i, a in enumerate(list(sys.argv)):
        match = JOBS_PATTERN.match(a)
        if match:
          jobs = int(match.groups()[-1])
          sys.argv.pop(i)
          break
      if jobs > 1:
        _insert_argv(key="hydra/launcher",
                     value="joblib",
                     is_value_string=False)
        _insert_argv(key="hydra.launcher.n_jobs",
                     value=f"{jobs}",
                     is_value_string=False)
      ### running dirs
      _insert_argv(key="hydra.run.dir",
                   value=f"{output_dir}/{override_id}",
                   is_value_string=True)
      _insert_argv(key="hydra.sweep.dir",
                   value=f"{output_dir}/multirun/{HYDRA_TIME_FMT}",
                   is_value_string=True)
      _insert_argv(key="hydra.job_logging.handlers.file.filename",
                   value=f"{log_dir}/{override_id}:{HYDRA_TIME_FMT}.log",
                   is_value_string=True)
      _insert_argv(key="hydra.job.config.override_dirname.exclude_keys",
                   value=f"[{','.join([str(i) for i in exclude_keys])}]",
                   is_value_string=False)
      # no return value from run_hydra() as it may sometime actually run the task_function
      # multiple times (--multirun)
      args = get_args_parser()
      config_path = _abspath(config_path)
      ## prepare arguments for task_function
      spec = inspect.getfullargspec(task_function)
      ## run hydra
      @functools.wraps(task_function)
      def _task_function(_cfg):
        if remove_exists:
          output_dir = get_output_dir()
          dir_base = os.path.dirname(output_dir)
          dir_name = os.path.basename(output_dir)
          for folder in get_all_folder(dir_base):
            name = os.path.basename(folder)
            if dir_name == name:
              clear_folder(folder, verbose=True)
        # catch exception, continue running in case
        try:
          task_function(_cfg)
        except Exception as e:
          _, value, tb = sys.exc_info()
          for line in traceback.TracebackException(
              type(value), value, tb, limit=None).format(chain=None):
            logger.error(line)
          if jobs == 1:
            raise e

      _run_hydra(
          args_parser=args,
          task_function=_task_function,
          config_path=config_path,
          config_name=config_name,
          strict=None,
      )

    return decorated_main

  return main_decorator


# ===========================================================================
# Search functions
# ===========================================================================
