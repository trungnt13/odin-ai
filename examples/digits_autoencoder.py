# ===========================================================================
# Using TIDIGITS dataset to predict gender (Boy, Girl, Woman, Man)
# ===========================================================================
# Saved WAV file format:
#     0) [train|test]
#     1) [m|w|b|g] (alias for man, women, boy, girl)
#     2) [age]
#     3) [dialectID]
#     4) [speakerID]
#     5) [production]
#     6) [digit_sequence]
#     => "train_g_08_17_as_a_4291815"
# ===========================================================================
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'cpu,float32,seed=12082518'
import shutil
import pickle

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score

from odin import backend as K, nnet as N, fuel as F, visual as V
from odin.stats import train_valid_test_split, freqcount
from odin import ml
from odin import training
from odin import preprocessing as pp
from odin.visual import print_dist, print_confusion, print_hist
from odin.utils import (get_logpath, get_modelpath, get_datasetpath,
                        Progbar, unique_labels, chain, get_formatted_datetime,
                        as_tuple_of_shape, stdio, ctext, ArgController)
# ===========================================================================
# Input arguments
# ===========================================================================
args = ArgController(
).add('path', 'path to preprocessed TIDIGITS dataset'
).parse()
# ===========================================================================
# path
# ===========================================================================
# path to preprocessed dataset
ds = F.Dataset(args.path, read_only=True)
# ====== general path ====== #
EXP_DIR = '/home/trung/data/exp_digit'
if not os.path.exists(EXP_DIR):
  os.mkdir(EXP_DIR)
# ====== start logging ====== #
LOG_PATH = os.path.join(EXP_DIR,
                        'log_%s.txt' % get_formatted_datetime(only_number=True))
stdio(LOG_PATH)
print("Exp-dir:", ctext(EXP_DIR, 'cyan'))
print("Log path:", ctext(LOG_PATH, 'cyan'))
# ===========================================================================
# Prepare data
# ===========================================================================
train_list = [(name, (start, end))
              for name, (start, end) in ds['indices']
              if 'train_' in name]
test_list = [(name, (start, end))
             for name, (start, end) in ds['indices']
             if 'test_' in name]
print("#Train:", len(train_list))
print("#Test:", len(test_list))
# ===========================================================================
# Test the synthesis
# ===========================================================================
name, (s1, e1) = sorted(list(ds['indices'].items()),
                        key=lambda x: x[-1][0])[12]
s2, e2 = ds['indices_raw'][name]
y = ds['spec'][s1:e1]
raw = ds['raw'][s2:e2]
print(name)
print(y.shape)
raw1 = pp.signal.ispec(y[:],
                       frame_length=int(0.05 * 16000),
                       step_length=int(0.0125 * 16000),
                       db=False,
                       nb_iter=100,
                       normalize=True,
                       de_preemphasis=0.97)
print(raw1)
for i in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
  print(np.percentile(raw1, i))
pp.speech.save('/tmp/tmp1.wav', s=raw[:].astype('float32'), sr=16000)
pp.speech.save('/tmp/tmp2.wav', s=raw1[:].astype('float32'), sr=16000)
