# -*- coding: utf-8 -*-
# This script merge multiple dataset (e.g. mspec, mspec_musan, mspec_rirs)
# into single dataset for training
from __future__ import print_function, division, absolute_import
import os
import shutil

import numpy as np

from odin import fuel as F
from odin.utils import ctext, Progbar

# ===========================================================================
# Configurations
# ===========================================================================
# path to all input dataset
inpath = [
    '/home/trung/data/SRE_FEAT/mfcc',
    '/home/trung/data/SRE_FEAT/mfcc_musan',
    '/home/trung/data/SRE_FEAT/mfcc_rirs',
]
# path to single output dataset
outpath = '/home/trung/data/SRE_FEAT/mfcc_musan_rirs'
# name of feature to be merged
features = ['mspec', 'mfcc']
# name of other metadata to be merge
other_features = [
    'dsname',
    'duration',
    'spkid',
    'path'
]
# ===========================================================================
# Validating all configurations
# ===========================================================================
assert all(os.path.exists(i) for i in inpath), \
    "All inpath must exists"
assert all(os.path.exists(os.path.join(i, j))
           for i in inpath for j in features), \
    "All inpath/%s must exists" % features
assert all(os.path.exists(os.path.join(i, 'indices_%s' % j))
           for i in inpath for j in features), \
    "All inpath/indices_%s must exists" % features
assert all(os.path.exists(os.path.join(i, j))
           for i in inpath
           for j in other_features), \
    "All inpath/%s must exists" % str(other_features)
# ====== check outpath ====== #
if os.path.exists(outpath):
  shutil.rmtree(outpath)
os.mkdir(outpath)
# ===========================================================================
# Start merging
# ===========================================================================
for feat_name in features:
  # ====== check all indices not overlap ====== #
  all_indices = [F.MmapDict(path=os.path.join(i, 'indices_%s' % feat_name),
                        read_only=True)
                 for i in inpath]
  _ = []
  for ids in all_indices:
    _ += list(ids.keys())
  assert len(_) == len(set(_)), "Overlap indices name"
  # ====== initialize ====== #
  out_data = None
  out_indices = {}
  start = 0
  curr_nfile = 0
  prog = Progbar(target=sum(len(i) for i in all_indices),
                 print_summary=True,
                 name=outpath)

  for i, path in enumerate(inpath):
    in_data = F.MmapData(path=os.path.join(path, feat_name),
                         read_only=True)
    in_indices = all_indices[i]
    # initialize
    if out_data is None:
      out_data = F.MmapData(path=os.path.join(outpath, feat_name),
                            dtype=in_data.dtype,
                            shape=(0,) + in_data.shape[1:],
                            read_only=False)
    # copy data
    for name, (s, e) in list(in_indices.items()):
      X = in_data[s:e]
      out_data.append(X)
      out_indices[name] = (start, start + X.shape[0])
      start += X.shape[0]
      # update progress
      prog['name'] = name[:48]
      prog['inpath'] = in_data.path
      prog.add(1)
      curr_nfile += 1
      # periodically flush
      if curr_nfile % 25000 == 0:
        out_data.flush()
    # close everything
    in_data.close()
  out_data.flush()
  out_data.close()
  # ====== create indices ====== #
  _ = F.MmapDict(path=os.path.join(outpath, 'indices_%s' % feat_name),
                 read_only=False)
  for name, (start, end) in out_indices.items():
    _[name] = (start, end)
  _.flush(save_all=True)
  _.close()
  # ====== validate the data ====== #
  out_data = F.MmapData(path=os.path.join(outpath, feat_name),
                        read_only=True)
  out_indices = F.MmapDict(path=os.path.join(outpath, 'indices_%s' % feat_name),
                           read_only=True)
  for ids, path in zip(all_indices, inpath):
    in_data = F.MmapData(path=os.path.join(path, feat_name),
                         read_only=True)
    for name in np.random.choice(a=list(ids.keys()), size=88, replace=False):
      x1 = ids[name]
      x1 = in_data[x1[0]:x1[1]][:]

      x2 = out_indices[name]
      x2 = out_data[x2[0]:x2[1]][:]

      assert np.all(x1 == x2), "Failed copying the data"

    in_data.close()

  out_data.close()
  out_indices.close()
# ===========================================================================
# Copy meta data
# ===========================================================================
prog = Progbar(target=sum(len(i) for i in all_indices) * len(other_features),
               print_summary=True,
               name=outpath)
for feat_name in other_features:
  out_feat = F.MmapDict(path=os.path.join(outpath, feat_name),
                        read_only=False)
  for path in inpath:
    in_feat = F.MmapDict(path=os.path.join(path, feat_name),
                         read_only=True)
    for key, val in in_feat.items():
      out_feat[key] = val
      prog['feat'] = feat_name
      prog['name'] = key[:48]
      prog['inpath'] = in_feat.path
      prog['outpath'] = out_feat.path
      prog.add(1)
    out_feat.flush(save_all=True)
  # end one feature
  out_feat.flush(save_all=True)
  out_feat.close()
