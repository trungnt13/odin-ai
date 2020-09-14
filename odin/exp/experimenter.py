import functools
import inspect
import os
import re
import sys
import tempfile
from copy import deepcopy
from typing import Any, Callable, List, Optional, Union

from six import string_types

from odin.utils import as_tuple
from odin.utils.crypto import md5_checksum

try:
  from hydra import __version__
  assert int(__version__.split('.')[0]) >= 1, \
    "Require hydra-core version >= 1.0.0"
  from hydra._internal.config_loader_impl import ConfigLoaderImpl
  from hydra._internal.config_search_path_impl import ConfigSearchPathImpl
  from hydra._internal.utils import _run_hydra, get_args_parser
  from hydra.types import TaskFunction
  from hydra.utils import HydraConfig, to_absolute_path
  from omegaconf import DictConfig, OmegaConf, ListConfig
  from hydra.experimental import initialize, compose
except ImportError as e:
  print(e)

__all__ = [
    'pretty_print', 'flatten_config', 'hash_config', \
    'get_hydra_config', 'get_overrides', 'get_output_dir', 'get_sweep_dir', \
    'run_hydra'
]

# ===========================================================================
# Helpers
# ===========================================================================
YAML_REGEX = re.compile(r"\w+: \w+")
OVERRIDE_PATTERN = re.compile(r"\A[\+\~]?[\w\.\\\@]+=[\w\(\)\[\]\{\}\,\.\']+")


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
  r""" Create an unique hash code from `DictConfig`

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


def get_hydra_config() -> DictConfig:
  return HydraConfig.get()


def get_overrides() -> str:
  return HydraConfig.get().job.override_dirname


def get_output_dir() -> str:
  r""" This is the same as `os.getcwd()` """
  if len(get_overrides()) == 0:
    return os.path.join(HydraConfig.get().run.dir, 'defaults')
  return HydraConfig.get().run.dir


def get_sweep_dir() -> str:
  return HydraConfig.get().sweep.dir


# ===========================================================================
# Main Function
# ===========================================================================
def run_hydra(output_dir: str = './outputs',
              exclude_keys: List[str] = []) -> Callable[[TaskFunction], Any]:
  r""" A modified main function of Hydra-core for flexibility
  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

  Useful commands:
    - `hydra/launcher=joblib` enable joblib launcher
    - `hydra.launcher.n_jobs=-1` set maximum number of processes


  Example:
  ```
  @experimenter.run_hydra()
  def run(cfg: DictConfig):
    print(cfg, type(cfg))

  run("/tmp/conf/base.yaml")
  ```

  Note:
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
  log_dir = os.path.join(output_dir, 'logs')
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  def main_decorator(task_function: TaskFunction) -> Callable[[], None]:

    @functools.wraps(task_function)
    def decorated_main(
        config: Union[str, dict, list, tuple, DictConfig]) -> Any:
      ## string
      if isinstance(config, string_types):
        if os.path.isfile(config):
          config_name = os.path.basename(config).replace(".yaml", "")
          config_path = os.path.dirname(config)
        elif os.path.isdir(config):
          config_path = config
          if not os.path.exists(os.path.join(config_path, 'base.yaml')):
            config_name = "base"  # default name
          else:
            config_name = sorted([
                i for i in os.listdir(config_path) if '.yaml' in i
            ])[0].replace(".yaml", "")
        elif len(YAML_REGEX.findall(config)) > 1:
          config_path, config_name = _save_config_to_tempdir(config)
        else:
          raise ValueError(
              f"No support for string config with format: {config}")
      ## dictionary, tuple, list, DictConfig
      else:
        config_path, config_name = _save_config_to_tempdir(config)
      ### check if output dir is provided
      is_overrided = False
      for a in sys.argv:
        print(a, OVERRIDE_PATTERN.match(a))
        if OVERRIDE_PATTERN.match(a):
          is_overrided = True
      time_fmt = r"${now:%j_%H%M%S}"
      if is_overrided:
        override_id = r"${hydra.job.override_dirname}"
      else:
        override_id = r"default"
      _insert_argv(key="hydra.run.dir",
                   value=f"{output_dir}/{override_id}",
                   is_value_string=True)
      _insert_argv(key="hydra.sweep.dir",
                   value=f"{output_dir}/multirun/{time_fmt}",
                   is_value_string=True)
      _insert_argv(key="hydra.job_logging.handlers.file.filename",
                   value=f"{log_dir}/{override_id}:{time_fmt}.log",
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
      _run_hydra(
          args_parser=args,
          task_function=task_function,
          config_path=config_path,
          config_name=config_name,
          strict=None,
      )

    return decorated_main

  return main_decorator


# ===========================================================================
# Search functions
# ===========================================================================
