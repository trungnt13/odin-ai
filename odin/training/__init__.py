from collections import Mapping

import inspect

import numpy as np
import tensorflow as tf

from .trainer import Task, Timer, MainLoop
from .callbacks import *
from .scenarios import train
