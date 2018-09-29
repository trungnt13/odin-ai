from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu'
import pickle
from collections import OrderedDict, defaultdict

import numpy as np
from scipy.io import savemat
from scipy import stats

import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from odin.ml import PLDA, Scorer
from odin import preprocessing as pp
from odin import fuel as F, nnet as N, backend as K
from odin.utils import (get_module_from_path, get_script_path, ctext,
                        Progbar)

from helpers import (SCORING_DATASETS, SCORE_SYSTEM_NAME, SCORE_SYSTEM_ID,
                     IS_LDA, PLDA_MAXIMUM_LIKELIHOOD, PLDA_SHOW_LLK,
                     PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE,
                     get_model_path, NCPU, get_logpath, prepare_dnn_feeder_recipe,
                     sre_file_list, Config, BACKEND_DATASET,
                     EXP_DIR, SCORE_DIR, BACKEND_DIR, RESULT_DIR)
# ===========================================================================
# Some helper
# ===========================================================================
def _check_running_feature_extraction(feat_dir, feat_name, n_files):
  # True mean need to run the feature extraction
  if not os.path.exists(feat_dir):
    return True
  indices_path = os.path.join(feat_dir, 'indices_%s' % feat_name)
  if not os.path.exists(indices_path):
    return True
  try:
    indices = F.MmapDict(path=indices_path, read_only=True)
    n_indices = len(indices)
    indices.close()
  except Exception as e:
    import traceback
    traceback.print_exc()
    print("Loading indices error: '%s'" % str(e), "at:", indices_path)
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
print(extractor)
# mapping from
# scoring_data_name -> [features 2-D array,
#                       indices {name: (start, end)},
#                       spkid_or_meta {name: spkid_or_meta},
#                       path {name: path}]
scoring_features = {}
training_ds = F.Dataset(path=os.path.join(PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE),
                        read_only=True)
all_training_dataset = set(training_ds['dsname'].values())
# ====== extract the feature if not exists ====== #
print("Acoustic feature extraction:")
for dsname, file_list in sorted(SCORING_DATASETS.items(),
                                key=lambda x: x[0]):
  # acoustic features already extracted in training dataset
  if dsname in all_training_dataset:
    X = training_ds[extractor_name]
    indices = {name: (start, end)
               for name, (start, end) in training_ds['indices_%s' % extractor_name].items()
               if training_ds['dsname'][name] == dsname}
    meta = {name: meta
            for name, meta in training_ds['spkid'].items()
            if name in indices}
    if 'path' in training_ds:
      path = {name: path
              for name, path in training_ds['path'].items()
              if name in indices}
    else:
      path = None
    print("  Name  :", ctext(dsname, 'cyan'))
    print("   #Files:", ctext(len(indices), 'cyan'))
    print("   Load dataset:", ctext(training_ds.path, 'cyan'))
    scoring_features[dsname] = [X, indices, meta, path]
    continue
  # extract acoustic feature from scratch
  feat_dir = os.path.join(PATH_ACOUSTIC_FEATURES,
                          '%s_%s' % (dsname, extractor_name))
  log_path = get_logpath(name='%s_%s.log' % (dsname, extractor_name),
                         increasing=True, odin_base=False, root=EXP_DIR)
  # check if need running the feature extraction
  if _check_running_feature_extraction(feat_dir,
                                       feat_name=extractor_name,
                                       n_files=len(file_list)):
    with np.warnings.catch_warnings():
      np.warnings.filterwarnings('ignore')
      processor = pp.FeatureProcessor(jobs=file_list,
                                      path=feat_dir,
                                      extractor=extractor,
                                      ncpu=NCPU,
                                      override=True,
                                      identifier='name',
                                      log_path=log_path,
                                      stop_on_failure=False)
      processor.run()
  # store the extracted dataset
  print("  Name  :", ctext(dsname, 'cyan'))
  print("   #Files:", ctext(len(file_list), 'cyan'))
  print("   Load dataset:", ctext(feat_dir, 'cyan'))
  ds = F.Dataset(path=feat_dir, read_only=True)
  scoring_features[dsname] = [
      ds[extractor_name],
      dict(ds['indices_%s' % extractor_name].items()),
      dict(ds['spkid'].items()),
      dict(ds['path'].items()),
  ]
# ====== check the duration ====== #
# for dsname, ds in scoring_features.items():
#   for fname, dur in ds['duration'].items():
#     dur = float(dur)
#     if dur < 5:
#       raise RuntimeError("Dataset: '%s' contains file: '%s', duration='%f' < 5(s)"
#         % (dsname, fname, dur))
# ===========================================================================
# Searching for trained system
# ===========================================================================
sys_dir, _, _ = get_model_path(system_name=SCORE_SYSTEM_NAME,
                               logging=False)
sys_name = os.path.basename(sys_dir)
all_sys = []
for path in os.listdir(sys_dir):
  path = os.path.join(sys_dir, path)
  if 'model.ai.' in path:
    all_sys.append(path)
# ====== get the right model based on given system index ====== #
if len(all_sys) == 0:
  final_sys = os.path.join(sys_dir, 'model.ai')
  sys_index = ''
  assert os.path.exists(final_sys), \
  "Cannot find pre-trained model at path: %s" % sys_dir
else:
  all_sys = sorted(all_sys,
                   key=lambda x: int(x.split('.')[-1]))
  final_sys = all_sys[SCORE_SYSTEM_ID]
  sys_index = '.' + final_sys.split('.')[-1]
# ====== print the log ====== #
print("Searching pre-trained model:")
print("  Found pre-trained at:", ctext(final_sys, 'cyan'))
print("  System name         :", ctext(sys_name, 'cyan'))
print("  System index        :", ctext(sys_index, 'cyan'))
# just check one more time
assert os.path.exists(final_sys), \
"Cannot find pre-trained model at: '%s'" % final_sys
# ===========================================================================
# All system must extract following information
# ===========================================================================
# mapping from
# dataset_name -> {'name': 1-D array [n_samples],
#                  'meta': 1-D array [n_samples], # (e.g. 'test', 'enroll', 'unlabeled')
#                  'path': 1-D array [n_samples], # (path to original audio)
#                  'data': 2-D array [n_samples, n_latent_dim]}
all_scores = {}

# mapping of data for training the backend
# dataset_name -> {'X': 2-D array [n_samples, n_latent_dim],
#                  'y': 1-D array [n_samples]}
all_backend = {}
# ===========================================================================
# Extract the x-vector for enroll and trials
# ===========================================================================
if 'xvec' == SCORE_SYSTEM_NAME:
  # ====== load the network ====== #
  x_vec = N.deserialize(path=final_sys,
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
  # ====== recipe for feeder ====== #
  recipe = prepare_dnn_feeder_recipe()
  # ==================== extract x-vector for enroll and trials ==================== #
  for dsname, (ds_feat, ds_indices, ds_meta, ds_path) in sorted(
      scoring_features.items(), key=lambda x: x[0]):
    n_files = len(ds_indices)
    # ====== check exist scores ====== #
    score_path = os.path.join(SCORE_DIR,
                              '%s%s.%s' % (sys_name, sys_index, dsname))
    if os.path.exists(score_path):
      with open(score_path, 'rb') as f:
        scores = pickle.load(f)
        if (len(scores['name']) == len(scores['meta']) ==
                len(scores['path']) == len(scores['data']) <= n_files):
          all_scores[dsname] = scores
          print(' - Loaded scores at:', ctext(score_path, 'cyan'))
          if len(scores['name']) != n_files:
            print(' - [WARNING] Extracted scores only for: %s/%s (files)' %
              (ctext(len(scores['name']), 'lightcyan'),
               ctext(n_files, 'cyan')))
          continue # skip the calculation
    # ====== create feeder ====== #
    feeder = F.Feeder(
        data_desc=F.IndexedData(data=ds_feat, indices=ds_indices),
        batch_mode='file', ncpu=8)
    feeder.set_recipes(recipe)
    # ====== init ====== #
    output_name = []
    output_meta = []
    output_path = []
    output_data = []
    # progress bar
    prog = Progbar(target=len(feeder), print_summary=True,
                   name=os.path.basename(score_path))
    prog.set_summarizer('#File', fn=lambda x: x[-1])
    prog.set_summarizer('#Batch', fn=lambda x: x[-1])
    # ====== make prediction ====== #
    curr_nfile = 0
    for batch_idx, (name, idx, X) in enumerate(feeder.set_batch(
        batch_size=100000, seed=None, shuffle_level=0)):
      assert idx == 0, "File '%s' longer than maximum batch size" % name
      curr_nfile += 1
      z = f_z(X)
      if z.shape[0] > 1:
        z = np.mean(z, axis=0, keepdims=True)
      output_name.append(name)
      output_meta.append(ds_meta[name])
      output_path.append(None if ds_path is None else ds_path[name])
      output_data.append(z)
      # update the progress
      prog['ds'] = dsname
      prog['name'] = name[:48]
      prog['latent'] = z.shape
      prog['#File'] = curr_nfile
      prog['#Batch'] = batch_idx + 1
      prog.add(X.shape[0])
    # ====== post-processing ====== #
    output_name = np.array(output_name)
    output_meta = np.array(output_meta)
    output_path = np.array(output_path)
    output_data = np.concatenate(output_data, axis=0)
    # ====== save the score ====== #
    with open(score_path, 'wb') as f:
      scores = {'name': output_name,
                'meta': output_meta,
                'path': output_path,
                'data': output_data.astype('float32')}
      pickle.dump(scores, f)
      all_scores[dsname] = scores
  # ==================== Extract the x-vector for training the backend ==================== #
  assert len(BACKEND_DATASET) > 0, \
  "Datasets for training the backend must be provided"
  print("Backend dataset:", ctext(BACKEND_DATASET, 'cyan'))
  feature_name = FEATURE_RECIPE.split('_')[0]
  ids_name = 'indices_%s' % feature_name
  indices = training_ds[ids_name]
  indices_dsname = {i: j for i, j in training_ds['dsname'].items()}
  indices_spkid = {i: j for i, j in training_ds['spkid'].items()}
  # ====== extract vector for each dataset ====== #
  for dsname in sorted(BACKEND_DATASET):
    path = os.path.join(BACKEND_DIR,
                        sys_name + sys_index + '.' + dsname)
    print("Processing ...", ctext(os.path.basename(path), 'yellow'))
    # ====== indices ====== #
    indices_ds = [(name, (start, end))
                  for name, (start, end) in indices.items()
                  if indices_dsname[name] == dsname]
    print("  Found: %s (files)" %
      ctext(len(indices_ds), 'cyan'))
    print("  Found: %s (speakers)" %
      ctext(len(set(indices_spkid[i[0]] for i in indices_ds)), 'cyan'))
    # skip if no files found
    if len(indices_ds) == 0:
      print("  Skip the calculation!")
      continue
    # ====== found exists vectors ====== #
    if os.path.exists(path):
      with open(path, 'rb') as fin:
        vectors = pickle.load(fin)
        if len(vectors['X']) == len(vectors['y']) and \
        len(vectors['X']) > 0:
          print("  Loaded vectors:",
                ctext(vectors['X'].shape, 'cyan'),
                ctext(vectors['y'].shape, 'cyan'))
          all_backend[dsname] = vectors
          continue
    # ====== create feeder ====== #
    feeder = F.Feeder(
        data_desc=F.IndexedData(data=training_ds[feature_name],
                                indices=indices_ds),
        batch_mode='file', ncpu=8)
    feeder.set_recipes(recipe)
    prog = Progbar(target=len(feeder), print_summary=True,
                   name="Extracting vector for: %s - %d (files)" %
                   (dsname, len(indices_ds)))
    # ====== extracting vectors ====== #
    Z_out = []
    y_out = []
    for name, idx, X in feeder.set_batch(
        batch_size=100000, seed=None, shuffle_level=0):
      assert idx == 0, "File '%s' longer than maximum batch size" % name
      # get the latent
      z = f_z(X)
      if z.shape[0] > 1:
        z = np.mean(z, axis=0, keepdims=True)
      Z_out.append(z)
      y_out.append(indices_spkid[name])
      # update the progress
      prog['name'] = name[:48]
      prog.add(X.shape[0])
    # ====== post processing ====== #
    Z_out = np.concatenate(Z_out).astype('float32')
    y_out = np.array(y_out)
    with open(path, 'wb') as fout:
      pickle.dump({'X': Z_out,
                   'y': y_out},
                  fout)
    print('  Extracted:', ctext(Z_out.shape, 'cyan'), y_out.shape)
    # ====== store the backend vectors ====== #
    all_backend[dsname] = {'X': Z_out,
                           'y': y_out}
# ===========================================================================
# Extract the i-vector
# ===========================================================================
elif 'ivec' == SCORE_SYSTEM_NAME:
  raise NotImplementedError
# ===========================================================================
# Extract the end-to-end system
# ===========================================================================
elif 'e2e' == SCORE_SYSTEM_NAME:
  raise NotImplementedError
# ===========================================================================
# Unknown system
# ===========================================================================
else:
  raise RuntimeError("No support for system: %s" % SCORE_SYSTEM_NAME)
# ===========================================================================
# Prepare data for training the backend
# ===========================================================================
assert len(all_backend) > 0
X_backend = []
y_backend = []
n_speakers = 0
for dsname, vectors in all_backend.items():
  X, y = vectors['X'], vectors['y']
  # add the data
  X_backend.append(X)
  # add the labels
  y_backend += y.tolist()
  # create label list
  n_speakers += len(np.unique(y))
# create mapping of spk to integer label
all_speakers = sorted(set(y_backend))
spk2label = {j: i
             for i, j in enumerate(all_speakers)}
# make sure no overlap speaker among dataset
assert len(all_speakers) == n_speakers
# create the training data
X_backend = np.concatenate(X_backend, axis=0)
y_backend = np.array([spk2label[i] for i in y_backend])
print("Training data for backend:")
print("  #Speakers:", ctext(n_speakers, 'cyan'))
print("  X        :", ctext(X_backend.shape, 'cyan'))
print("  y        :", ctext(y_backend.shape, 'cyan'))
# ====== fast checking the array ====== #
print("Check backend data statistics:")
print("  Mean  :", ctext(np.mean(X_backend), 'cyan'))
print("  Std   :", ctext(np.std(X_backend), 'cyan'))
print("  Max   :", ctext(np.max(X_backend), 'cyan'))
print("  Min   :", ctext(np.min(X_backend), 'cyan'))
print("  NaN   :", ctext(np.any(np.isnan(X_backend)), 'cyan'))
n = int(np.prod(X_backend.shape))
n_non_zeros = np.count_nonzero(X_backend)
print("  #Zeros: %s/%s or %.1f%%" %
  (ctext(n - n_non_zeros, 'lightcyan'),
   ctext(n, 'cyan'),
   (n - n_non_zeros) / n * 100))
# ====== optional save data to matlab for testing ====== #
with open('/tmp/xvecs.mat', 'wb') as ftmp:
  savemat(ftmp, {'X': np.array(X_backend.astype('float32'), order='F'),
                 'y': np.array(y_backend.astype('int32'), order='F')})
# ===========================================================================
# Now scoring
# ===========================================================================
for dsname, scores in sorted(all_scores.items(),
                             key=lambda x: x[0]):
  print("Scoring:", ctext(dsname, 'yellow'))
  # load the scores
  (seg_name, seg_meta,
   seg_path, seg_data) = (scores['name'], scores['meta'],
                          scores['path'], scores['data'])
  name_2_data = {i: j
                 for i, j in zip(seg_name, seg_data)}
  name_2_ext = {i: '' if j is None else os.path.splitext(j)[-1]
                for i, j in zip(seg_name, seg_path)}
  # get the enroll and trials list
  enroll_name = '%s_enroll' % dsname
  trials_name = '%s_trials' % dsname
  if enroll_name in sre_file_list and trials_name in sre_file_list:
    # ====== checking the trials ====== #
    trials = np.array([(i, j)
                       for i, j in sre_file_list[trials_name][:, :2]
                       if j in name_2_data])
    print("  Missing trials: %s/%s" %
      (ctext(len(sre_file_list[trials_name]) - len(trials), 'lightcyan'),
       ctext(len(sre_file_list[trials_name]), 'cyan')))
    # ====== checking the enrollments ====== #
    enroll = np.array([(i, j)
                       for i, j in sre_file_list[enroll_name][:, :2]
                       if j in name_2_data])
    print("  Missing enroll: %s/%s" %
      (ctext(len(sre_file_list[enroll_name]) - len(enroll), 'lightcyan'),
       ctext(len(sre_file_list[enroll_name]), 'cyan')))
    # ====== skip the scoring if necessary ====== #
    if len(trials) == 0 or len(enroll) == 0:
      print("  Skip scoring for:", ctext(dsname, 'yellow'))
      continue
    # ====== create the enrollments data ====== #
    models = OrderedDict()
    # for now we don't care about channel (or size) information
    for model_id, segment_id in enroll[:, :2]:
      if model_id not in models:
        models[model_id] = []
      models[model_id].append(name_2_data[segment_id])
    # calculate the x-vector for each model
    models = OrderedDict([
        (model_id, np.mean(seg_list, axis=0, keepdims=True))
        for model_id, seg_list in models.items()
    ])
    model_2_index = {j: i for i, j in enumerate(models.keys())}
    X_models = np.concatenate(list(models.values()), axis=0)
    print("  Enroll:", ctext(X_models.shape, 'cyan'))
    # ====== create the trials list ====== #
    X_trials = np.concatenate([name_2_data[i][None, :] for i in trials[:, 1]],
                              axis=0)
    print("  Trials:", ctext(X_trials.shape, 'cyan'))
    # ====== training the plda ====== #
    if IS_LDA:
      print("  Fitting LDA ...")
      lda = LinearDiscriminantAnalysis(n_components=200)
      X_backend = lda.fit_transform(X=X_backend, y=y_backend)
    plda = PLDA(n_phi=150,
                centering=True, wccn=True, unit_length=True,
                n_iter=20, random_state=Config.SUPER_SEED,
                verbose=2 if PLDA_SHOW_LLK else 1)
    if PLDA_MAXIMUM_LIKELIHOOD:
      print("  Fitting PLDA maximum likelihood ...")
      plda.fit_maximum_likelihood(X=X_backend, y=y_backend)
    plda.fit(X=X_backend, y=y_backend)
    y_scores = plda.predict_log_proba(X=X_trials, X_model=X_models)
    print("  Scores:", ctext(y_scores.shape, 'cyan'))
    # ====== write the scores to file ====== #
    score_path = os.path.join(RESULT_DIR,
                              '%s%s.%s.csv' % (sys_name, sys_index, dsname))
    with open(score_path, 'w') as fout:
      fout.write('\t'.join(['modelid', 'segmentid', 'side', 'LLR']) + '\n')
      for i, (model_id, seg_id) in enumerate(trials):
        score = '%f' % y_scores[i][model_2_index[model_id]]
        fout.write('\t'.join([model_id, seg_id + name_2_ext[seg_id], 'a', score]) + '\n')
    print("  Saved trials:", ctext(score_path, 'cyan'))
  else:
    raise RuntimeError(
        "Cannot find '%s_trials.csv' and '%s_enroll.csv' for dataset: %s" %
        (dsname, dsname, dsname))
