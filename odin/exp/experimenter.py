import inspect
import itertools
import logging
import os
import re
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from numbers import Number

from hydra._internal.config_loader import ConfigLoader
from hydra._internal.core_plugins import BasicLauncher
from hydra._internal.hydra import Hydra, HydraConfig
from hydra._internal.pathlib import Path
from hydra._internal.utils import (create_config_search_path, get_args_parser,
                                   run_hydra)
from hydra.plugins.common.utils import (configure_log, filter_overrides,
                                        run_job, setup_globals,
                                        split_config_path)
from omegaconf import DictConfig, open_dict
from six import string_types

from odin.utils import as_tuple
from odin.utils.crypto import md5_checksum
from odin.utils.mpi import MPI

# ===========================================================================
# Helpers
# ===========================================================================
log = logging.getLogger(__name__)


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


def _overrides(overrides):
  if isinstance(overrides, dict):
    overrides = [
        "%s=%s" % (str(key), \
          ','.join([str(v) for v in value])
                   if isinstance(value, (tuple, list)) else str(value))
        for key, value in overrides.items()
    ]
  return overrides


# ===========================================================================
# Hydra Launcher
# ===========================================================================
class ParallelLauncher(BasicLauncher):
  r"""Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved"""

  def __init__(self, ncpu=2):
    super().__init__()
    self.ncpu = int(ncpu)

  def launch(self, job_overrides):
    setup_globals()
    configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
    sweep_dir = self.config.hydra.sweep.dir
    Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)
    log.info("Launching {} jobs locally".format(len(job_overrides)))

    def run_task(job):
      idx, overrides = job
      log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))
      sweep_config = self.config_loader.load_sweep_config(
          self.config, list(overrides))
      with open_dict(sweep_config):
        sweep_config.hydra.job.id = idx
        sweep_config.hydra.job.num = idx
      HydraConfig().set_config(sweep_config)
      ret = run_job(
          config=sweep_config,
          task_function=self.task_function,
          job_dir_key="hydra.sweep.dir",
          job_subdir_key="hydra.sweep.subdir",
      )
      configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
      return (idx, ret)

    jobs = list(enumerate(job_overrides))
    runs = sorted(
        [ret for ret in MPI(jobs=jobs, func=run_task, ncpu=self.ncpu, batch=1)])
    runs = [i[1] for i in runs]
    return runs


def _to_sqltype(obj):
  if obj is None:
    return "NULL"
  if isinstance(obj, string_types):
    return "TEXT"
  if isinstance(obj, Number):
    obj = float(obj)
    if obj.is_integer():
      return "INTEGER"
    return "REAL"
  return "BLOB"


# ===========================================================================
# SQLlite
# ===========================================================================
class ExperimentManager():

  def __init__(self, exp):
    assert isinstance(exp, Experimenter), \
      "exp must be instance of Experimenter, but given: %s" % str(type(exp))
    self.path = _abspath(os.path.join(exp._save_path, 'exp.log'))

  def update_status(self):
    pass

  def close(self):
    pass


class SQLExperimentManager(ExperimentManager):

  @property
  def connection(self) -> sqlite3.Connection:
    if not hasattr(self, '_connection'):
      self._connection = sqlite3.connect(self.path,
                                         timeout=self.timeout,
                                         check_same_thread=True)
    return self._connection

  @contextmanager
  def cursor(self) -> sqlite3.Cursor:
    if not hasattr(self, '_cursor'):
      self._cursor = self.connection.cursor()
    yield self._cursor
    self._connection.commit()

  def close(self):
    if hasattr(self, '_connection'):
      if hasattr(self, '_cursor'):
        self._cursor.close()
        del self._cursor
      self._connection.close()
      del self._connection


# ===========================================================================
# Main class
# ===========================================================================
_INSTANCES = {}


class Experimenter():

  def __new__(cls, save_path, *args, **kwargs):
    # ====== from pickling ====== #
    if save_path is None:
      return super(Experimenter, cls).__new__(cls)
    # ====== found exist singleton ====== #
    save_path = _abspath(save_path)
    if save_path in _INSTANCES:
      print("SAME INSTANCE")
      return _INSTANCES[save_path]
    # ====== create new instance ====== #
    new_instance = super(Experimenter, cls).__new__(cls)
    _INSTANCES[save_path] = new_instance
    return new_instance

  def __init__(self, save_path, config_path="", ncpu=1):
    # already init, return by singleton
    if hasattr(self, '_configs'):
      return
    self.ncpu = int(ncpu)
    # ====== check save path ====== #
    self._save_path = _abspath(save_path)
    if os.path.isfile(self._save_path):
      raise ValueError("save_path='%s' must be a folder" % self._save_path)
    if not os.path.exists(self._save_path):
      os.mkdir(self._save_path)
    self._manager = ExperimentManager(self)
    # ====== load configs ====== #
    self.config_path = _abspath(config_path)
    assert os.path.isfile(self.config_path), \
      "Config file does not exist: %s" % self.config_path
    search_path = create_config_search_path(os.path.dirname(self.config_path))
    self.config_loader = ConfigLoader(config_search_path=search_path,
                                      default_strict=False)
    self._configs = self.load_configuration()

  ####################### Static helpers
  @staticmethod
  def hash_config(cfg: DictConfig) -> str:
    assert isinstance(cfg, (DictConfig, dict))
    return md5_checksum(cfg)[:8]

  @staticmethod
  def match_arguments(func: callable, cfg: DictConfig, ignores=[],
                      **defaults) -> dict:
    kwargs = {}
    spec = inspect.getfullargspec(func)
    args = spec.args
    if 'self' == args[0]:
      args = args[1:]
    for name in args:
      if name in cfg:
        kwargs[name] = cfg[name]
      elif name in defaults:
        kwargs[name] = defaults[name]
    for name in as_tuple(ignores, t=str):
      if name in kwargs:
        del kwargs[name]
    return kwargs

  ####################### Property
  @property
  def manager(self) -> ExperimentManager:
    return self._manager

  @property
  def db_path(self):
    return os.path.join(self._save_path, 'exp.db')

  @property
  def configs(self) -> DictConfig:
    return self._configs.copy()

  ####################### Helpers
  def get_output_path(self, overrides=[]):
    overrides = _overrides(overrides)
    keys = self.hash(overrides)
    if isinstance(keys, string_types):
      keys = [keys]
    paths = []
    for k in keys:
      p = os.path.join(self._save_path, 'exp_%s' % k)
      if not os.path.exists(p):
        os.mkdir(p)
      paths.append(p)
    return paths[0] if len(paths) == 1 else tuple(paths)

  def parse_overrides(self, overrides=[]):
    overrides = _overrides(overrides)
    lists = []
    for s in overrides:
      key, value = s.split("=")
      assert key in self._configs, \
        "Cannot find key='%s' in default configs, possible key are: %s" % \
          (key, ', '.join(self._configs.keys()))
      lists.append(["{}={}".format(key, val) for val in value.split(",")])
    lists = list(itertools.product(*lists))
    return lists

  def load_configuration(self, overrides=[]):
    overrides = _overrides(overrides)
    # create the config
    returns = []
    for overrides in self.parse_overrides(overrides):
      cfg = self.config_loader.load_configuration(\
        config_file=os.path.basename(self.config_path),
        overrides=list(overrides),
        strict=False)
      del cfg['hydra']
      returns.append(cfg)
    return returns[0] if len(returns) == 1 else returns

  def hash(self, overrides=[]) -> str:
    r""" An unique hash string of length 8 based on the parsed configurations.
    """
    overrides = _overrides(overrides)
    cfg = self.load_configuration(overrides)
    if isinstance(cfg, DictConfig):
      return Experimenter.hash_config(cfg)
    return [Experimenter.hash_config(cfg) for c in cfg]

  def clear_all_experiments(self, verbose=True):
    input("<Enter> to continue remove all experiments ...")
    for path in os.listdir(self._save_path):
      path = os.path.join(self._save_path, path)
      if os.path.isdir(path):
        shutil.rmtree(path)
      elif os.path.isfile(path):
        os.remove(path)
      if verbose:
        print(" Remove:", path)
    return self

  ####################### Base methods
  def on_load_data(self, cfg: DictConfig):
    pass

  def on_load_model(self, cfg: DictConfig):
    pass

  def on_train(self, cfg: DictConfig):
    pass

  def on_clean(self, cfg: DictConfig):
    pass

  ####################### Basic logics
  def _run(self, cfg: DictConfig):
    self.on_load_data(cfg)
    self.on_load_model(cfg)
    self.on_train(cfg)
    self.on_clean(cfg)

  def run(self, overrides=[]):
    overrides = _overrides(overrides)
    ncpu = self.ncpu

    def _multirun(self, config_file, task_function, overrides):
      # Initial config is loaded without strict (individual job configs may have strict).
      from hydra._internal.plugins import Plugins
      cfg = self.compose_config(config_file=config_file,
                                overrides=overrides,
                                strict=False,
                                with_log_configuration=True)
      HydraConfig().set_config(cfg)
      sweeper = Plugins.instantiate_sweeper(config=cfg,
                                            config_loader=self.config_loader,
                                            task_function=task_function)
      # override launcher for using multiprocessing
      if ncpu > 1:
        sweeper.launcher = ParallelLauncher(ncpu=ncpu)
        sweeper.launcher.setup(config=cfg,
                               config_loader=self.config_loader,
                               task_function=task_function)
      return sweeper.sweep(arguments=cfg.hydra.overrides.task)

    old_multirun = Hydra.multirun
    Hydra.multirun = _multirun

    try:
      if len(overrides) > 0:
        sys.argv += overrides
      args_parser = get_args_parser()
      run_hydra(
          args_parser=args_parser,
          task_function=self._run,
          config_path=self.config_path,
          strict=False,
      )
    except KeyboardInterrupt:
      sys.exit(-1)
    except SystemExit:
      pass
    Hydra.multirun = old_multirun
