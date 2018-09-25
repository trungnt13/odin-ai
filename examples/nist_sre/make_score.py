from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'cpu=4,float32'

import numpy as np
import tensorflow as tf

from odin import preprocessing as pp
from odin import fuel as F, nnet as N, backend as K
from odin.utils import (get_module_from_path, get_script_path, ctext,
                        Progbar)

from helpers import (SCORING_DATASETS, SCORE_SYSTEM_NAME, SCORE_SYSTEM_ID,
                     PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE, EXP_DIR,
                     get_model_path, NCPU, get_logpath, prepare_dnn_feeder_recipe)

SCORE_DIR = os.path.join(EXP_DIR, 'scores')
if not os.path.exists(SCORE_DIR):
  os.mkdir(SCORE_DIR)
# ===========================================================================
# Some helper
# ===========================================================================
def _check_running_feature_extraction(feat_dir, feat_name, n_files):
  # True mean need to run the feature extraction
  if not os.path.exists(feat_dir):
    return True
  if not os.path.exists(os.path.join(feat_dir,
                                     'indices_%s' % feat_name)):
    return True
  try:
    indices = F.MmapDict(path=os.path.join(feat_dir,
                                           'indices_%s' % feat_name),
                         read_only=True)
    n_indices = len(indices)
    indices.close()
  except Exception as e:
    return True
  if n_indices != n_files:
    return True
  return False
# ===========================================================================
# Searching for extractor
# ===========================================================================
extractor_name = FEATURE_RECIPE.split("_")[0]
extractor = get_module_from_path(identifier=extractor_name,
                                 path=get_script_path(),
                                 prefix='feature_recipes')[0]
extractor = extractor()
# ====== extract the feature if not exists ====== #
scoring_features = {}
for dsname, file_list in SCORING_DATASETS.items():
  feat_dir = os.path.join(PATH_ACOUSTIC_FEATURES,
                          '%s_%s' % (dsname, extractor_name))
  log_path = get_logpath(name='%s_%s.log' % (dsname, extractor_name),
                         increasing=True, odin_base=False, root=EXP_DIR)
  if _check_running_feature_extraction(feat_dir,
                                       feat_name=extractor_name,
                                       n_files=len(file_list)):
    with np.warnings.catch_warnings():
      np.warnings.filterwarnings('ignore')
      processor = pp.FeatureProcessor(jobs=file_list,
                                      path=feat_dir,
                                      extractor=extractor,
                                      ncpu=NCPU, override=True,
                                      identifier='name',
                                      log_path=log_path,
                                      stop_on_failure=False)
      processor.run()
  # store the extracted dataset
  scoring_features[dsname] = F.Dataset(path=feat_dir,
                                       read_only=True)
# ====== check the duration ====== #
for dsname, ds in scoring_features.items():
  for fname, dur in ds['duration'].items():
    dur = float(dur)
    if dur < 5:
      raise RuntimeError("Dataset: '%s' contains file: '%s', duration='%f' < 5(s)"
        % (dsname, fname, dur))
# ===========================================================================
# Searching for trained system
# ===========================================================================
model_dir, _, _ = get_model_path(system_name=SCORE_SYSTEM_NAME)
all_model = []
for path in os.listdir(model_dir):
  path = os.path.join(model_dir, path)
  if 'model.ai.' in path:
    all_model.append(path)
if len(all_model) == 0:
  final_model = os.path.join(model_dir, 'model.ai')
  assert os.path.exists(final_model), \
  "Cannot find pre-trained model at path: %s" % model_dir
else:
  all_model = sorted(all_model, key= lambda x: int(x[-1]))
  final_model = all_model[SCORE_SYSTEM_ID]
print("Found pre-trained at:", ctext(final_model, 'cyan'))
# ===========================================================================
# Extract the x-vector
# ===========================================================================
if 'xvec' == SCORE_SYSTEM_NAME:
  # ====== load the network ====== #
  x_vec = N.deserialize(path=final_model,
                        force_restore_vars=True)
  # ====== get output tensors ====== #
  y_logit = x_vec()
  y_proba = tf.nn.softmax(y_logit)
  X = K.ComputationGraph(y_proba).placeholders[0]
  z = K.ComputationGraph(y_proba).get(roles=N.Dense, scope='LatentOutput',
                                      beginning_scope=False)[0]
  f_z = K.function(inputs=X, outputs=z, training=False)
  print('Inputs:', ctext(X, 'cyan'))
  print('Latent:', ctext(z, 'cyan'))
  # ====== prepare the data ====== #
  recipe = prepare_dnn_feeder_recipe()
  for dsname, ds in scoring_features.items():
    feeder = F.Feeder(
        data_desc=F.IndexedData(data=ds[extractor_name],
                                indices=ds['indices_%s' % extractor_name]),
        batch_mode='file', ncpu=8)
    feeder.set_recipes(recipe)
    # ====== init ====== #
    output_name = []
    output_data = []
    prog = Progbar(target=len(feeder),
                   print_summary=True,
                   name='Making prediction on: %s' % dsname)
    # ====== make prediction ====== #
    for name, idx, X in feeder.set_batch(batch_size=100000,
                                         seed=None, shuffle_level=0):
      assert idx == 0, "File '%s' longer than maximum batch size" % name
      z = f_z(X)
      if z.shape[0] > 1:
        z = np.mean(z, axis=0, keepdims=True)
      output_name.append(name)
      output_data.append(z)
      # update the progress
      prog['ds'] = dsname
      prog['name'] = name[:48]
      prog['latent'] = z.shape
      prog.add(X.shape[0])
    # ====== post-processing ====== #
    output_name = np.array(output_name)
    output_data = np.array(output_data)
    print(output_name)
    print(output_data)
    print(output_name.shape, output_name.dtype)
    print(output_data.shape, output_data.dtype)
# ===========================================================================
# Extract the i-vector
# ===========================================================================
elif 'ivec' == SCORE_SYSTEM_NAME:
  raise NotImplementedError
# ===========================================================================
# Unknown system
# ===========================================================================
else:
  raise RuntimeError("No support for system: %s" % SCORE_SYSTEM_NAME)
