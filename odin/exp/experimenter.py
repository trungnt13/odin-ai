from __future__ import absolute_import, division, print_function

import inspect
import itertools
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from numbers import Number

from pandas import DataFrame
from six import string_types

from odin.utils import (as_tuple, clean_folder, get_all_files,
                        get_formatted_datetime)
from odin.utils.crypto import md5_checksum, md5_folder
from odin.utils.mpi import MPI

try:
  from hydra._internal.config_loader import ConfigLoader
  from hydra._internal.core_plugins import BasicLauncher
  from hydra._internal.hydra import Hydra, HydraConfig
  from hydra._internal.pathlib import Path
  from hydra._internal.utils import (create_config_search_path, get_args_parser,
                                     run_hydra)
  from hydra.plugins.common.utils import (configure_log, filter_overrides,
                                          run_job, setup_globals,
                                          split_config_path)
  from omegaconf import DictConfig, OmegaConf, open_dict
except ImportError as e:
  raise ImportError(
      "Experimenter requires hydra-core library, 'pip install hydra-core'")

# ===========================================================================
# Helpers
# ===========================================================================
LOGGER = logging.getLogger("Experimenter")
_APP_HELP = """
${hydra.help.header}

-ncpu (number of process for multirun)

--reset (remove all exists experiments' results)

== Configuration groups ==
Compose your configuration from those groups (group=option)

$APP_CONFIG_GROUPS

== Arguments help ==

%s


== Config ==
Override anything in the config (foo.bar=value)

$CONFIG

"""

YAML_REGEX = re.compile(r"\w+: \w+")


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


def _overrides(overrides) -> list:
  if isinstance(overrides, dict):
    overrides = [
        "%s=%s" % (str(key), \
          ','.join([str(v) for v in value])
                   if isinstance(value, (tuple, list)) else str(value))
        for key, value in overrides.items()
    ]
  elif isinstance(overrides, string_types):
    overrides = [overrides]
  elif overrides is None:
    overrides = []
  return list(overrides)


def _all_keys(d, base):
  keys = []
  for k, v in d.items():
    if len(base) > 0:
      k = base + "." + k
    keys.append(k)
    if hasattr(v, 'items'):
      keys += _all_keys(v, k)
  return keys


def _prepare_conditions(conditions={}):
  conditions = [cond.split('=') for cond in _overrides(conditions)]
  conditions = {
      key: set([i.strip() for i in values.split(',')]) \
        for key, values in conditions
  }
  return conditions


# ===========================================================================
# Hydra Launcher
# ===========================================================================
class ParallelLauncher(BasicLauncher):
  r"""Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

  This launcher won't run parallel task if `ncpu<=1`
  """

  def __init__(self, ncpu=1):
    super().__init__()
    self.ncpu = ncpu

  def launch(self, job_overrides):
    setup_globals()
    configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
    sweep_dir = self.config.hydra.sweep.dir
    Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)
    LOGGER.info("Launching {} jobs locally".format(len(job_overrides)))

    def run_task(job):
      idx, overrides = job
      LOGGER.info("\t#{} : {}".format(idx,
                                      " ".join(filter_overrides(overrides))))
      sweep_config = self.config_loader.load_sweep_config(
          self.config, list(overrides))
      with open_dict(sweep_config):
        # id is concatenated overrides here
        sweep_config.hydra.job.id = '_'.join(sorted(overrides))
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

    if self.ncpu > 1:
      jobs = list(enumerate(job_overrides))
      runs = sorted([
          ret
          for ret in MPI(jobs=jobs, func=run_task, ncpu=int(self.ncpu), batch=1)
      ])
      runs = [i[1] for i in runs]
    else:
      runs = [run_task(job)[1] for job in enumerate(job_overrides)]
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
# Main class
# ===========================================================================
_INSTANCES = {}


class Experimenter():
  r""" Experiment management using hydra

  Arguments:
    save_path : path to a folder for saving the experiments
    config_path : String. Two option for providing the configuration file
      - path to a yaml file : base configuraition
      - the yaml content itself, stored in string
    ncpu : number of process when multirun (-m) option is enable.
    exclude_keys : list of String. Keys will be excluded when hashing
      the configuration to create experiments' ID.
    consistent_model : a Boolean. If True, check if MD5 of saved model
      is the same as MD5 of loaded model.

  Methods:
    on_load_data(cfg: DictConfig)
      called at the beginning, everytime, for loading data
    on_create_model(cfg: DictConfig)
      called only when first train a model with given configuration
    on_load_model(cfg: DictConfig, path: str)
      called when pretrained model detected
    on_train(cfg: DictConfig, model_path: str)
      call when training start

  """

  def __init__(self,
               save_path,
               config_path="",
               ncpu=1,
               exclude_keys=[],
               consistent_model=True):
    # already init, return by singleton
    if hasattr(self, '_configs'):
      return
    self.ncpu = int(ncpu)
    ### check save path
    self._save_path = _abspath(save_path)
    if os.path.isfile(self._save_path):
      raise ValueError("save_path='%s' must be a folder" % self._save_path)
    if not os.path.exists(self._save_path):
      os.mkdir(self._save_path)
    ### load configs
    # the config given by string
    if not os.path.exists(config_path) and \
      len(YAML_REGEX.findall(config_path)) > 1:
      config_dir = os.path.join(self.save_path, "configs")
      if not os.path.exists(config_dir):
        os.makedirs(config_dir)
      path = os.path.join(config_dir, 'base.yaml')
      with open(path, 'w') as f:
        f.write(config_path.strip())
      config_path = path
    self.config_path = _abspath(config_path)
    assert os.path.isfile(self.config_path), \
      "Config file does not exist: %s" % self.config_path
    search_path = create_config_search_path(os.path.dirname(self.config_path))
    self.config_loader = ConfigLoader(config_search_path=search_path,
                                      default_strict=False)
    self._configs = self.load_configuration()
    ### others
    self._all_keys = set(_all_keys(self._configs, base=""))
    self._exclude_keys = as_tuple(exclude_keys, t=string_types)
    self.consistent_model = bool(consistent_model)
    self._train_mode = True

  def train(self):
    r""" Prepare this experimenter for training models """
    self._train_mode = True
    return self

  def eval(self):
    r""" Prevent further changes to models, and prepare for evaluation """
    self._train_mode = False
    return self

  ####################### Static helpers
  @staticmethod
  def remove_keys(cfg: DictConfig, keys=[], copy=True) -> DictConfig:
    r""" Remove list of keys from given configs """
    assert isinstance(cfg, (DictConfig, dict))
    exclude_keys = as_tuple(keys, t=string_types)
    if len(exclude_keys) > 0:
      if copy:
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
    return cfg

  @staticmethod
  def hash_config(cfg: DictConfig, exclude_keys=[]) -> str:
    r"""
    cfg : dictionary of configuration to generate an unique identity
    exclude_keys : list of string, given keys will be ignored from the
      configuraiton
    """
    assert isinstance(cfg, (DictConfig, dict))
    cfg = Experimenter.remove_keys(cfg, copy=True, keys=exclude_keys)
    return md5_checksum(cfg)[:8]

  @staticmethod
  def match_arguments(func: callable,
                      cfg: DictConfig,
                      exclude_args=[],
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
    for name in as_tuple(exclude_args, t=str):
      if name in kwargs:
        del kwargs[name]
    return kwargs

  ####################### Property
  @property
  def save_path(self):
    return self._save_path

  @property
  def exclude_keys(self):
    return self._exclude_keys

  @property
  def db_path(self):
    return os.path.join(self._save_path, 'exp.db')

  @property
  def configs(self) -> DictConfig:
    return deepcopy(self._configs)

  ####################### Helpers
  def write_history(self, *msg):
    with open(os.path.join(self._save_path, 'history.txt'), 'a+') as f:
      f.write("[%s]" % get_formatted_datetime(only_number=False))
      for i, m in enumerate(msg):
        sep = " " if i == 0 else "\t"
        f.write("%s%s\n" % (sep, str(m)))
    return self

  def get_output_path(self, cfg: DictConfig = None):
    if cfg is None:
      cfg = self.configs
    key = Experimenter.hash_config(cfg, self.exclude_keys)
    path = os.path.join(self._save_path, 'exp_%s' % key)
    if not os.path.exists(path):
      os.mkdir(path)
    return path

  def get_model_path(self, cfg: DictConfig = None):
    output_path = self.get_output_path(cfg)
    return os.path.join(output_path, 'model')

  def get_hydra_path(self):
    path = os.path.join(self._save_path, 'hydra')
    if not os.path.exists(path):
      os.mkdir(path)
    return path

  def get_config_path(self, cfg: DictConfig = None, datetime=False):
    output_path = self.get_output_path(cfg)
    if datetime:
      return os.path.join(
          output_path,
          'configs_%s.yaml' % get_formatted_datetime(only_number=False))
    return os.path.join(output_path, 'configs.yaml')

  def parse_overrides(self, overrides=[], **configs):
    overrides = _overrides(overrides) + _overrides(configs)
    lists = []
    for s in overrides:
      key, value = s.split("=")
      assert key in self._all_keys, \
        "Cannot find key='%s' in default configs, possible key are: %s" % \
          (key, ', '.join(self._all_keys))
      lists.append(["{}={}".format(key, val) for val in value.split(",")])
    lists = list(itertools.product(*lists))
    return lists

  def load_configuration(self, overrides=[], **configs):
    overrides = _overrides(overrides) + _overrides(configs)
    # create the config
    returns = []
    for overrides in self.parse_overrides(overrides):
      cfg = self.config_loader.load_configuration(\
        config_file=os.path.basename(self.config_path),
        overrides=list(overrides),
        strict=True)
      del cfg['hydra']
      returns.append(cfg)
    return returns[0] if len(returns) == 1 else returns

  def hash(self, overrides=[], **configs) -> str:
    r""" An unique hash string of length 8 based on the parsed configurations.
    """
    overrides = _overrides(overrides) + _overrides(configs)
    cfg = self.load_configuration(overrides)
    if isinstance(cfg, DictConfig):
      return Experimenter.hash_config(cfg, self.exclude_keys)
    return [Experimenter.hash_config(cfg, self.exclude_keys) for c in cfg]

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
  @property
  def args_help(self) -> dict:
    r""" Return a mapping argument name to list of allowed values """
    return {}

  def on_load_data(self, cfg: DictConfig):
    r""" Cleaning """
    pass

  def on_create_model(self, cfg: DictConfig):
    r""" Cleaning """
    pass

  def on_load_model(self, cfg: DictConfig, model_path: str):
    r""" Cleaning """
    pass

  def on_train(self, cfg: DictConfig, model_path: str):
    r""" Cleaning """
    pass

  ####################### Basic logics
  def _run(self, cfg: DictConfig):
    # the cfg is dispatched by hydra.run_job, we couldn't change anything here
    logger = LOGGER
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=DeprecationWarning)
      # prepare the paths
      model_path = self.get_model_path(cfg)
      md5_path = model_path + '.md5'
      # check the configs
      config_path = self.get_config_path(cfg, datetime=True)
      with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
      logger.info("Save config: %s" % config_path)
      # load data
      self.on_load_data(cfg)
      logger.info("Loaded data")
      # create or load model
      if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        # check if the loading model is consistent with the saved model
        if os.path.exists(md5_path) and self.consistent_model:
          md5_loaded = md5_folder(model_path)
          with open(md5_path, 'r') as f:
            md5_saved = f.read().strip()
          assert md5_loaded == md5_saved, \
            "MD5 of saved model mismatch, probably files are corrupted"
        model = self.on_load_model(cfg, model_path)
        if model is None:
          raise RuntimeError(
              "The implementation of on_load_model must return the loaded model."
          )
        logger.info("Loaded model: %s" % model_path)
      else:
        self.on_create_model(cfg)
        logger.info("Create model: %s" % model_path)
      # training
      self.on_train(cfg, model_path)
      logger.info("Finish training")
      # saving the model hash
      if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        with open(md5_path, 'w') as f:
          f.write(md5_folder(model_path))
        logger.info("Save model:%s" % model_path)

  def run(self, overrides=[], ncpu=None, **configs):
    r"""

    Arguments:
      strict: A Boolean, strict configurations prevent the access to
        unknown key, otherwise, the config will return `None`.

    Example:
      exp = SisuaExperimenter(ncpu=1)
      exp.run(
          overrides={
              'model': ['sisua', 'dca', 'vae'],
              'dataset.name': ['cortex', 'pbmc8kly'],
              'train.verbose': 0,
              'train.epochs': 2,
              'train': ['adam'],
          })
    """
    overrides = _overrides(overrides) + _overrides(configs)
    strict = False
    command = ' '.join(sys.argv)
    # parse ncpu
    if ncpu is None:
      ncpu = self.ncpu
    ncpu = int(ncpu)
    for idx, arg in enumerate(list(sys.argv)):
      if 'ncpu' in arg:
        if '=' in arg:
          ncpu = int(arg.split('=')[-1])
          sys.argv.pop(idx)
        else:
          ncpu = int(sys.argv[idx + 1])
          sys.argv.pop(idx)
          sys.argv.pop(idx)
        break
    # check reset
    for idx, arg in enumerate(list(sys.argv)):
      if arg in ('--reset', '--clear', '--clean'):
        configs_filter = lambda f: 'configs' != f.split('/')[-1]
        if len(get_all_files(self._save_path, filter_func=configs_filter)) > 0:
          old_exps = '\n'.join([
              " - %s" % i
              for i in os.listdir(self._save_path)
              if configs_filter(i)
          ])
          inp = input("<Enter> to clear all exists experiments:"
                      "\n%s\n'n' to cancel, otherwise continue:" % old_exps)
          if inp.strip().lower() != 'n':
            clean_folder(self._save_path, filter=configs_filter, verbose=True)
        sys.argv.pop(idx)
    # check multirun
    is_multirun = any(',' in ovr for ovr in overrides) or \
      any(',' in arg and '=' in arg for arg in sys.argv)
    # write history
    self.write_history(command, "overrides: %s" % str(overrides),
                       "strict: %s" % str(strict), "ncpu: %d" % ncpu,
                       "multirun: %s" % str(is_multirun))
    # generate app help
    hlp = '\n\n'.join([
        "%s - %s" % (str(key), ', '.join(sorted(as_tuple(val, t=str))))
        for key, val in dict(self.args_help).items()
    ])

    def _run(self, config_file, task_function, overrides):
      if is_multirun:
        raise RuntimeError(
            "Performing single run with multiple overrides in hydra "
            "(use '-m' for multirun): %s" % str(overrides))
      cfg = self.compose_config(config_file=config_file,
                                overrides=overrides,
                                strict=strict,
                                with_log_configuration=True)
      HydraConfig().set_config(cfg)
      return run_job(
          config=cfg,
          task_function=task_function,
          job_dir_key="hydra.run.dir",
          job_subdir_key=None,
      )

    def _multirun(self, config_file, task_function, overrides):
      # Initial config is loaded without strict (individual job configs may have strict).
      from hydra._internal.plugins import Plugins
      cfg = self.compose_config(config_file=config_file,
                                overrides=overrides,
                                strict=strict,
                                with_log_configuration=True)
      HydraConfig().set_config(cfg)
      sweeper = Plugins.instantiate_sweeper(config=cfg,
                                            config_loader=self.config_loader,
                                            task_function=task_function)
      # override launcher for using multiprocessing
      sweeper.launcher = ParallelLauncher(ncpu=ncpu)
      sweeper.launcher.setup(config=cfg,
                             config_loader=self.config_loader,
                             task_function=task_function)
      return sweeper.sweep(arguments=cfg.hydra.overrides.task)

    old_multirun = (Hydra.run, Hydra.multirun)
    Hydra.run = _run
    Hydra.multirun = _multirun

    try:
      # append the new override
      if len(overrides) > 0:
        sys.argv += overrides
      # help for arguments
      if '--help' in sys.argv:
        # sys.argv.append("hydra.help.header='**** %s ****'" %
        #                 self.__class__.__name__)
        # sys.argv.append("hydra.help.template=%s" % (_APP_HELP % hlp))
        # TODO : fix bug here
        pass
      # append the hydra log path
      job_fmt = "/${now:%d%b%y_%H%M%S}"
      sys.argv.insert(1, "hydra.run.dir=%s" % self.get_hydra_path() + job_fmt)
      sys.argv.insert(1, "hydra.sweep.dir=%s" % self.get_hydra_path() + job_fmt)
      sys.argv.insert(1, "hydra.sweep.subdir=${hydra.job.id}")
      # sys.argv.append(r"hydra.job_logging.formatters.simple.format=" +
      #                 r"[\%(asctime)s][\%(name)s][\%(levelname)s] - \%(message)s")
      args_parser = get_args_parser()
      run_hydra(
          args_parser=args_parser,
          task_function=self._run,
          config_path=self.config_path,
          strict=strict,
      )
    except KeyboardInterrupt:
      sys.exit(-1)
    except SystemExit:
      pass
    Hydra.run = old_multirun[0]
    Hydra.multirun = old_multirun[1]
    # update the summary
    self.summary()
    return self

  ####################### For evaluation
  def fetch_exp_cfg(self, conditions={}, require_model=True) -> dict:
    r"""

    Arguments:
      require_model : a Boolean. If True, only return exp with saved model

    Return:
      A dictionary mapping from path to experiments and list of configs
    """
    conditions = _prepare_conditions(conditions)

    def get_attr(c, name):
      if '.' in name:
        for key in name.split('.'):
          c = c.get(key)
        return c
      return c[name]

    # prepare the path
    path = self._save_path
    exp_path = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if 'exp_' == name[:4]
    ]
    # filter path with require_model
    if require_model:
      exp_path = list(
          filter(lambda x: os.path.isdir(os.path.join(x, 'model')), exp_path))
    ret = {}
    for path in exp_path:
      cfg = sorted([
          os.path.join(path, i) for i in os.listdir(path) if 'configs_' == i[:8]
      ],
                   key=lambda x: get_formatted_datetime(
                       only_number=False,
                       convert_text=x.split('_')[-1].split('.')[0]).timestamp())
      if len(cfg) > 0:
        if len(conditions) > 0:
          last_cfg = cfg[-1]  # lastest config
          with open(last_cfg, 'r') as f:
            last_cfg = OmegaConf.load(f)
          # filter the conditions
          if all(
              get_attr(last_cfg, key) in val
              for key, val in conditions.items()):
            ret[path] = cfg
          del last_cfg
        else:
          ret[path] = cfg
    return ret

  def sample_model(self, conditions={}, seed=1):
    exp_cfg = self.fetch_exp_cfg(conditions)
    if len(exp_cfg) == 0:
      raise RuntimeError("Cannot find model with configuration: %s" %
                         str(conditions))
    random.seed(seed)
    exp, cfg = random.choice(list(exp_cfg.items()))
    model = self.on_load_model(cfg, os.path.join(exp, 'model'))
    cfg = OmegaConf.load(cfg[-1])
    return model, cfg

  def search(self, conditions={}, load_model=True, return_config=True):
    r"""
    Arguments:
      conditions : a Dictionary
      return_config : a Boolean

    Returns:
      list of (model, config) tuple or list of loaded model
        (if `return_config=False`)
    """
    # ====== filtering the model ====== #
    found_models = []
    found_cfg = []
    for path, cfg in self.fetch_exp_cfg(conditions).items():
      cfg = cfg[-1]  # lastest config
      with open(cfg, 'r') as f:
        cfg = OmegaConf.load(f)
      found_models.append(os.path.join(path, 'model'))
      found_cfg.append(cfg)
    # ====== load the found models ====== #
    if len(found_models) == 0:
      raise RuntimeError("Cannot find model satisfying conditions: %s" %
                         str(conditions))
    if load_model:
      found_models = [
          self.on_load_model(cfg, path)
          for cfg, path in zip(found_cfg, found_models)
      ]
    if return_config:
      found_models = list(zip(found_models, found_cfg))
    return found_models

  def summary(self, save_files=True) -> DataFrame:
    r""" Save a table of experiment ID and all their attributes to an
    excel file and a html file. """
    exp_cfg = self.fetch_exp_cfg(require_model=False)
    records = []
    index = []
    for path, cfg in exp_cfg.items():
      index.append(os.path.basename(path))
      model_exist = os.path.isdir(os.path.join(path, 'model'))
      report = {"#run": len(cfg), "model": str(model_exist)[0]}
      cfg = [("",
              Experimenter.remove_keys(OmegaConf.load(cfg[-1]),
                                       copy=False,
                                       keys=self.exclude_keys))]
      # use stack instead of recursive
      while len(cfg) > 0:
        root, items = cfg.pop()
        for key, val in items.items():
          name = root + key
          if hasattr(val, 'items'):
            cfg.append((name + '.', val))
          else:
            report[name] = val
      records.append(report)
    # ====== save dataframe to html ====== #
    df = DataFrame(records, index=index)
    if save_files:
      with open(os.path.join(self._save_path, 'summary.html'), 'w') as f:
        f.write(df.to_html())
      with open(os.path.join(self._save_path, 'summary.txt'), 'w') as f:
        f.write(df.to_string(index=False))
      try:
        df.to_excel(os.path.join(self._save_path, 'summary.xlsx'))
      except ModuleNotFoundError as e:
        print("Cannot save summary to excel file: %s" % str(e))
    return df
