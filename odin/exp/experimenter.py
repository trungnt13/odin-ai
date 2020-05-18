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
import traceback
import types
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO
from numbers import Number

import numpy as np
import tensorflow as tf
from pandas import DataFrame
from six import string_types

from odin.exp.scores import ScoreBoard
from odin.utils import (as_tuple, clean_folder, get_all_files,
                        get_formatted_datetime, struct)
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
  from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
except ImportError as e:
  BasicLauncher = object
  DictConfig = dict
  OmegaConf = dict
  warnings.warn(
      "Experimenter requires hydra-core library, 'pip install hydra-core'")

# ===========================================================================
# Helpers
# ===========================================================================
LOGGER = logging.getLogger("Experimenter")
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


def pretty_config(cfg: dict, ncol=4) -> str:
  ncol = int(ncol)
  text = ''
  if hasattr(cfg, 'pretty'):
    pretty = cfg.pretty().split('\n')
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
  return text


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


class ModelList(list):

  def to_dataframe(self) -> DataFrame:
    return self.to_df()

  def to_df(self) -> DataFrame:
    cols = list(self[0].keys())
    df = DataFrame([[i[k] for k in cols] for i in self], columns=cols)
    return df

  def __getitem__(self, key):
    if isinstance(key, string_types):
      for i in self:
        if i.hash == key:
          return i
      raise KeyError("Cannot find model with hash key: %s" % key)
    return super().__getitem__(key)


# ===========================================================================
# Main class
# ===========================================================================
_INSTANCES = {}


class Experimenter():
  r""" Experiment management using hydra

  For `--eval`, `--plot`, `--compare`,  the Experimenter will find all models
  then filter out the model by provided. For `training`, the exact set of models
  will provided by the arguments. For example:

    - During training `ds=mnist` run only the default model on MNIST dataset.
    - During eval `ds=mnist` run all model trained on MNIST dataset

  Common workflow with `Experimenter`:
  ```
  input exp : Experimenter
  for prog in n_processes:
    on prog:
      on_train() # default, no flag
  for prog in n_processes:
    on prog:
      on_eval() # --eval
      on_plot() # --plot
  on_compare() # --compare
  ```

  Arguments:
    save_path : path to a folder for saving the experiments
    config_path : String. Two option for providing the configuration file
      - path to a yaml file : base configuration
      - the yaml content itself, stored in string
    ncpu : number of process when multirun (-m) option is enable.
    exclude_keys : list of String. Keys will be excluded when hashing
      the configuration to create experiments' ID.
    hash_length : an Integer (default: `5`). The length of hash key that is
      unique for each experiment configuration, the longer the less chance
      for hash collision.

  Methods:
    on_load_data(cfg)
      called at the beginning, everytime, for loading data
    on_create_model(cfg, model_dir, md5)
      called only when first train a model with given configuration
    on_train(cfg, output_dir, model_dir)
      call when training start
    on_eval(cfg, output_dir)
      call with `--eval` option for evaluation
    on_plot(cfg, output_dir)
      call with `--plot` option for visualization the results
    on_compare(models, save_path)
      call with `--compare` to generate analysis of comparing multiple models
      `--load` option can be added to force loading the trained model using
      `on_create_model`

  Database:
    List of default tables and columns:
    - 'run' (store the called running script and its arguments)
        path, date, overrides, strict, ncpu, multirun
    - 'config' (store the configuration of each run)
        hash, ds, model

  Example
  ```
  CONFIG = r"data: 1\nmodel: 2"
  exp = Experimenter(save_path="/tmp/exptmp", config_path=CONFIG)
  exp.run()
  # python main.py vae=betavae,factorvae ds=mnist,shapes3d -m -ncpu 2
  # python main.py vae=betavae,factorvae ds=mnist,shapes3d -m -ncpu 2 --eval
  # python main.py vae=betavae,factorvae ds=mnist,shapes3d -m -ncpu 2 --plot
  # python main.py ds=mnist --compare --load
  ```
  """

  def __init__(self,
               save_path,
               config_path,
               ncpu=1,
               exclude_keys=[],
               hash_length=5):
    # already init, return by singleton
    if hasattr(self, '_configs'):
      return
    self.ncpu = int(ncpu)
    self.hash_length = int(hash_length)
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
    ### running configuration
    self._db = None
    self._running_configs = None
    self._running_hash = None
    self._running_mode = 'train'
    self._override_mode = False

  @property
  def is_training(self):
    return self._running_mode == 'train'

  def train(self):
    self._running_mode = 'train'
    return self

  def eval(self):
    self._running_mode = 'eval'
    return self

  def plot(self):
    self._running_mode = 'plot'
    return self

  def hash_config(self, cfg: DictConfig, exclude_keys=[]) -> str:
    r"""
    cfg : dictionary of configuration to generate an unique identity
    exclude_keys : list of string, given keys will be ignored from the
      configuration
    """
    assert isinstance(cfg, (DictConfig, dict))
    cfg = Experimenter.remove_keys(cfg, copy=True, keys=exclude_keys)
    cfg = flatten_config(cfg, base='', max_depth=-1)
    return md5_checksum(cfg)[:self.hash_length]

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
  def db(self) -> ScoreBoard:
    r""" Path to the database recording all experiments configuration and
    results """
    if self._db is None:
      self._db = ScoreBoard(os.path.join(self._save_path, 'exp.db'))
    return self._db

  @property
  def configs(self) -> DictConfig:
    return deepcopy(self._configs)

  ####################### Helpers
  def _write_history(self, path, *msg):
    date = get_formatted_datetime(only_number=False)
    self.db.write(table='run',
                  path=path,
                  date=date,
                  **dict([m.split(':') for m in msg]))
    with open(os.path.join(self._save_path, 'history.txt'), 'a+') as f:
      f.write("[%s]" % date)
      f.write("path: %s\n" % path)
      for i, m in enumerate(msg):
        sep = " " if i == 0 else "\t"
        f.write("%s%s\n" % (sep, str(m)))
    return self

  def get_output_dir(self, cfg: DictConfig = None):
    if cfg is None:
      cfg = self.configs
    key = self.hash_config(cfg, self.exclude_keys)
    path = os.path.join(self._save_path, 'exp_%s' % key)
    if not os.path.exists(path):
      os.mkdir(path)
    return path

  def get_model_dir(self, cfg: DictConfig = None):
    path = self.get_output_dir(cfg)
    path = os.path.join(path, 'model')
    if not os.path.exists(path):
      os.mkdir(path)
    return path

  def get_hydra_path(self):
    path = os.path.join(self._save_path, 'hydra')
    if not os.path.exists(path):
      os.mkdir(path)
    return path

  def get_config_path(self, cfg: DictConfig = None, datetime=False):
    output_path = self.get_output_dir(cfg)
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
      return self.hash_config(cfg, self.exclude_keys)
    return [self.hash_config(cfg, self.exclude_keys) for c in cfg]

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

  ####################### Database access
  def get_scores(self, table, hashes=None, *columns) -> dict:
    r""" Get saved scores for given model based on their hash value

    Arguments:
      table : a String, name of SQL table
      hashes : {`None`, a String, list of String}.
        List of hash code, i.e. the identity of model, generated based on
        its running configuration. If `None`, select all

    Return:
      a Dictionary, mapping from `hash key` to `columns`
    Example
    ```
    self.get_scores('score', [i.hash for i in configs], ['mllk'])
    ```
    """
    table = str(table).lower().strip()
    # prepare columns
    columns = [str(i) for i in tf.nest.flatten(columns)]
    if len(columns) > 0:
      columns = ','.join(['hash'] + columns)
    else:
      columns = '*'
    # prepare where
    hashes = [str(i) for i in tf.nest.flatten(hashes)]
    if hashes is None or len(hashes) == 0:
      where = ""
    else:
      where = "WHERE hash in (%s)" % ','.join("'%s'" % i for i in hashes)
    # run the SQL query
    query = f"SELECT {columns} FROM {table} {where};"
    rows = self.select(query)
    return {r[0]: r[1] if len(r) == 2 else r[1:] for r in rows}

  def save_scores(self, table, override=False, **scores):
    r""" Save scores to the SQLite database, the hash key (primary key) is
    determined by the running configuration. """
    if self._running_hash:
      hash_key = self._running_hash
    else:
      cfg = self._running_configs
      if cfg is None:
        cfg = self._configs
      hash_key = self.hash_config(cfg, self.exclude_keys)
    self.db.write(table=table,
                  unique='hash',
                  override=override,
                  hash=hash_key,
                  **scores)
    return self

  def get_all_configs(self) -> DataFrame:
    col_names = self.db.get_column_names('config')
    df = DataFrame(self.db.get_table('config'), columns=col_names)
    return df

  def select(self, query):
    r""" Run a select query on the database """
    return self.db.select(query)

  def get_models(self, conditions="", load_models=False):
    r""" Select all model in the database `exp.db` given the conditions

    Return:
      list of Dictionary
    """
    regex = re.compile(r'\w+[=].+')
    if isinstance(conditions, string_types):
      conditions = conditions.split(' ')
    arguments = [i for i in conditions if '-' != i[0]]
    conditions = []
    expect_condition = True
    ## convert condition to where statement
    for i, ov in enumerate(arguments):
      if regex.findall(ov):
        if not expect_condition:
          conditions.append("AND")
        expect_condition = False  # now we need connector
        key, vals = ov.split('=')
        sql_vals = []
        for v in vals.split(','):
          if v.isdigit():
            v = float(v)
            if v.is_integer():
              v = int(v)
          else:
            v = "'%s'" % v
          sql_vals.append(v)
        if len(sql_vals) > 1:
          conditions.append(f"config.'{key}' in ({','.join(sql_vals)})")
        else:
          conditions.append(f"config.'{key}'={sql_vals[0]}")
      else:  # expect and, or ... connector here
        expect_condition = True
        conditions.append(ov)
    conditions = ' '.join(conditions) if len(conditions) > 0 else ""
    ## create the query
    where = f"WHERE {conditions}" if len(conditions) > 0 else ""
    query = f"SELECT * FROM config {where};"
    configs = self.select(query)
    colname = [row[1] for row in self.select("PRAGMA table_info(config);")]
    ## select only available model
    _path = lambda row: os.path.join(self.save_path, 'exp_%s' % row[0], 'model')
    configs = [
        # all values must be primitive
        [i.tolist() if isinstance(i, np.ndarray) else i
         for i in row]
        for row in configs
        if os.path.exists(_path(row)) and os.listdir(_path(row))
    ]
    # create the model
    models = ModelList()
    attrs = set([i[0] for i in inspect.getmembers(self)])
    # iterate and get all the relevant config
    for cfg in configs:
      cfg = DictConfig(dict(zip(colname, cfg)))
      if load_models:
        print("Loading model:")
        print(pretty_config(cfg))
        # load data
        for k in self.exclude_keys:  # set the default excluded key
          if k not in cfg:
            cfg[k] = self.configs[k]
        self.on_load_data(cfg)
        # load model
        self.on_create_model(cfg, self.get_model_dir(cfg), md5=None)
        # just get the newly added attributes
        cfg = struct(**cfg)
        for k, v in [i for i in inspect.getmembers(self) if i[0] not in attrs]:
          cfg[k] = v
      # add the model
      models.append(cfg)
    return models

  ####################### Base methods
  def on_load_data(self, cfg: DictConfig):
    print("LOAD DATA!")

  def on_create_model(self, cfg: DictConfig, model_dir: str, md5: str = None):
    print("CREATE MODEL!", cfg, model_dir, md5)

  def on_train(self, cfg: DictConfig, output_dir: str, model_dir: str):
    print("TRAINING:", cfg, output_dir, model_dir)

  def on_eval(self, cfg: DictConfig, output_dir: str):
    print("EVALUATING:", cfg, output_dir)

  def on_plot(self, cfg: DictConfig, output_dir: str):
    print("PLOTTING:", cfg, output_dir)

  def on_compare(self, models: ModelList, save_path: str):
    print("COMPARING:", save_path)

  ####################### Compare multiple model
  def _run_comparison(self, argv, load_models):
    models = self.get_models(conditions=argv[1:], load_models=load_models)
    ## finally call the compare function
    self.on_compare(models, self.save_path)

  ####################### Basic logics
  def _call_and_catch(self, method, **kwargs):
    r""" Return True to stop the run, False for continue running """
    method_name = method.__func__.__name__
    try:
      method(**kwargs)
      return False
    except:
      text = StringIO()
      traceback.print_exception(*sys.exc_info(),
                                limit=None,
                                file=text,
                                chain=True)
      text.seek(0)
      text = text.read().strip()
      LOGGER.error("\n" + text)
      self.db.write(table='error',
                    hash=self._running_hash,
                    method=method_name,
                    traceback=text,
                    config=self._running_configs.pretty(),
                    datetime=get_formatted_datetime(only_number=False))
      return True

  def _run(self, cfg: DictConfig):
    cfg = deepcopy(cfg)
    # store config in database
    hash_key = self.hash_config(cfg, self.exclude_keys)
    self.db.write(table='config',
                  unique='hash',
                  hash=hash_key,
                  **flatten_config(cfg))
    self._running_configs = cfg
    self._running_hash = hash_key
    # the cfg is dispatched by hydra.run_job, we couldn't change anything here
    logger = LOGGER
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=DeprecationWarning)
      ## prepare the paths
      output_dir = self.get_output_dir(cfg)
      if self._override_mode:
        logger.info("Override experiment at path: %s" % output_dir)
        shutil.rmtree(output_dir)
      model_dir = self.get_model_dir(cfg)
      md5_path = model_dir + '.md5'
      ## check the configs
      config_path = self.get_config_path(cfg, datetime=True)
      with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
      logger.info("Save config: %s" % config_path)
      ## load data
      if self._call_and_catch(self.on_load_data, cfg=cfg):
        return
      logger.info("Loaded data")
      ## create or load model
      md5_saved = None
      if os.path.exists(md5_path):
        with open(md5_path, 'r') as f:
          md5_saved = f.read().strip()
      if self._call_and_catch(self.on_create_model,
                              cfg=cfg,
                              model_dir=model_dir,
                              md5=md5_saved):
        return
      logger.info("Create model: %s (md5:%s)" % (model_dir, str(md5_saved)))
      ## training
      logger.info("Start experiment in mode '%s'" % self._running_mode)
      if self._running_mode == 'train':
        if self._call_and_catch(self.on_train,
                                cfg=cfg,
                                output_dir=output_dir,
                                model_dir=model_dir):
          return
        logger.info("Finish training")
      elif self._running_mode == 'eval':
        if self._call_and_catch(self.on_eval, cfg=cfg, output_dir=output_dir):
          return
        logger.info("Finish evaluating")
      elif self._running_mode == 'plot':
        if self._call_and_catch(self.on_plot, cfg=cfg, output_dir=output_dir):
          return
        logger.info("Finish plotting")
      ## saving the model hash
      if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
        md5 = md5_folder(model_dir)
        with open(md5_path, 'w') as f:
          f.write(md5)
        logger.info("The model stored at path:%s  (MD5: %s)" % (model_dir, md5))

  ####################### main
  def run(self, overrides=[], ncpu=None, **configs):
    r""" Extra options for controlling the experiment:

      `--eval` : run in evaluation mode
      `--plot` : run in plotting mode
      `--reset` or `--clear` : remove all exist experiments
      `--override` : override existed experiment

    Arguments:
      strict: A Boolean, strict configurations prevent the access to
        unknown key, otherwise, the config will return `None`.

    Example:
    ```
    exp = SisuaExperimenter(ncpu=1)
    exp.run(
        overrides={
            'model': ['sisua', 'dca', 'vae'],
            'dataset.name': ['cortex', 'pbmc8kly'],
            'train.verbose': 0,
            'train.epochs': 2,
            'train': ['adam'],
        })
    ```
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
    ## check functional fixed arguments
    run_comparison = False
    load_model_comparison = False
    remove_items = []
    for arg in list(sys.argv):
      if arg in ('--load',):
        load_model_comparison = True
        remove_items.append(arg)
      elif arg in ('--compare', '-compare'):
        run_comparison = True
        remove_items.append(arg)
      elif arg in ('--override', '-override'):
        self._override_mode = True
        remove_items.append(arg)
      elif arg in ('--train', '-train'):
        self.train()
        remove_items.append(arg)
      elif arg in ('--plot', '-plot'):
        self.plot()
        remove_items.append(arg)
      elif arg in ('--eval', '-eval'):
        self.eval()
        remove_items.append(arg)
      elif arg in ('--reset', '--clear', '--clean'):
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
        remove_items.append(arg)
    # remove all checked arguments
    for i in remove_items:
      sys.argv.remove(i)
    # check multirun
    is_multirun = any(',' in ovr for ovr in overrides) or \
      any(',' in arg and '=' in arg for arg in sys.argv)
    # write history
    self._write_history(command, "overrides:%s" % str(overrides),
                        "strict:%s" % str(strict), "ncpu:%d" % ncpu,
                        "multirun:%s" % str(is_multirun))
    ## preprocessing running mode in comparison mode
    regex = re.compile(r'\w+[=].+')
    # run in comparison mode
    if run_comparison:
      self._run_comparison(sys.argv, load_model_comparison)
      return self
    elif self._running_mode in ('eval', 'plot'):
      configs = self.get_models(sys.argv[1:], load_models=False)
      if len(configs) == 0:
        raise RuntimeError("Cannot find trained models with configuration: %s" %
                           str(sys.argv[1:]))
      # get all config of relevant models
      kw = defaultdict(list)
      for cfg in configs:
        for k, v in cfg.items():
          args = kw[k]
          if all(i != v for i in args):
            args.append(v)
      # convert the dictionary config to overrides, need to flatten the config
      # here to ensure the same key appeared in the database and in def_configs
      def_configs = flatten_config(self.configs)
      args = []
      for k, v in kw.items():
        if k in def_configs and (len(v) > 1 or def_configs[k] != v[0]):
          args.append('%s=%s' % (k, ','.join([str(i) for i in v])))
      # remove old overrides and assign the new ones
      sys.argv = [i for i in sys.argv if not regex.findall(i)]
      sys.argv = sys.argv[0:1] + args + sys.argv[1:]
    # show warning if multirun is necessary
    if any(regex.findall(i) for i in sys.argv) and \
      not ('-m' in sys.argv or '--multirun' in sys.argv):
      warnings.warn(
          "Multiple overrides are provided but multirun mode not enable, "
          "-m for enabling multirun, %s" % str(sys.argv))

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
      # append the hydra log path
      job_fmt = "/${now:%d%b%y_%H%M%S}"
      sys.argv.insert(1, "hydra.run.dir=%s" % self.get_hydra_path() + job_fmt)
      sys.argv.insert(1, "hydra.sweep.dir=%s" % self.get_hydra_path() + job_fmt)
      sys.argv.insert(1, "hydra.sweep.subdir=${hydra.job.id}")
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
    try:
      self.summary()
    except Exception as e:
      print("Error:", e)
    return self

  def __call__(self, **kwargs):
    return self.run(**kwargs)

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
      cfg = sorted(
          [
              os.path.join(path, i)
              for i in os.listdir(path)
              if 'configs_' == i[:8]
          ],
          key=lambda x: get_formatted_datetime(only_number=False,
                                               convert_text=x.split('_')[-1].
                                               split('.')[0]).timestamp(),
      )
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
          self.on_create_model(cfg, path, None)
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
      report = {
          "expid": os.path.basename(path),
          "#run": len(cfg),
          "model": str(model_exist)[0]
      }
      cfg = [("",
              Experimenter.remove_keys(OmegaConf.load(cfg[-1]),
                                       copy=True,
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
    return df
