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
                        Progbar, stdio, get_logpath, get_formatted_datetime)
from odin.stats import describe

from helpers import (SCORING_DATASETS, BACKEND_DATASETS,
                     SCORE_SYSTEM_NAME, SCORE_SYSTEM_ID,
                     N_PLDA, N_LDA, PLDA_MAXIMUM_LIKELIHOOD, PLDA_SHOW_LLK,
                     PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE, FEATURE_NAME,
                     get_model_path, NCPU, get_logpath, prepare_dnn_feeder_recipe,
                     sre_file_list, Config,
                     EXP_DIR, VECTORS_DIR, RESULT_DIR,
                     filter_utterances)
# ====== scoring log ====== #
stdio(get_logpath(name='make_score.log', increasing=True,
                  odin_base=False, root=EXP_DIR))
print('=' * 48)
print(get_formatted_datetime(only_number=False))
print("System name    :", SCORE_SYSTEM_NAME)
print("System id      :", SCORE_SYSTEM_ID)
print("Feature recipe :", FEATURE_RECIPE)
print("Feature name   :", FEATURE_NAME)
print("Backend dataset:", ','.join(BACKEND_DATASETS.keys()))
print("Scoring dataset:", ','.join(SCORING_DATASETS.keys()))
print('=' * 48)
# ===========================================================================
# Some helper
# ===========================================================================
def _check_running_feature_extraction(feat_dir, n_files):
  # True mean need to run the feature extraction
  if not os.path.exists(feat_dir):
    return True
  indices_path = os.path.join(feat_dir, 'indices_%s' % FEATURE_NAME)
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
# ====== generate path ====== #
def get_vectors_outpath(dsname):
  return os.path.join(VECTORS_DIR, '%s%s.%s' % (sys_name, sys_index, dsname))
# ===========================================================================
# Searching for extractor
# ===========================================================================
EXTRACTOR_NAME = FEATURE_RECIPE.split("_")[0]
extractor = get_module_from_path(identifier=EXTRACTOR_NAME,
                                 path=get_script_path(),
                                 prefix='feature_recipes')
assert len(extractor) > 0, \
    "Cannot find extractor with name: %s" % EXTRACTOR_NAME
extractor = extractor[0]()
# ====== initializing ====== #
# mapping from
# scoring_data_name -> [features 2-D array,
#                       indices {name: (start, end)},
#                       spkid_or_meta {name: spkid_or_meta},
#                       path {name: path}]
acoustic_features = {}
training_ds = F.Dataset(path=os.path.join(PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE),
                        read_only=True)
all_training_dataset = set(training_ds['dsname'].values())
print("All training dataset:", ctext(all_training_dataset, 'cyan'))
# ====== extract the feature if not exists ====== #
for dsname, file_list in sorted(list(SCORING_DATASETS.items()) + list(BACKEND_DATASETS.items()),
                                key=lambda x: x[0]):
  # acoustic features already extracted in training dataset
  if dsname in all_training_dataset:
    assert FEATURE_NAME in training_ds, \
        "Cannot find feature with name: %s, from: %s" % (FEATURE_NAME, training_ds.path)
    X = training_ds[FEATURE_NAME]
    indices = {name: (start, end)
               for name, (start, end) in training_ds['indices_%s' % FEATURE_NAME].items()
               if training_ds['dsname'][name] == dsname}
    # we use everything for PLDA
    indices = filter_utterances(X, indices, training_ds['spkid'],
                remove_min_length=False,
                remove_min_uttspk=True if 'voxceleb' in dsname else False,
                n_speakers=800 if 'voxceleb' in dsname else None,
                ncpu=4, title=dsname)
    meta = {name: meta
            for name, meta in training_ds['spkid'].items()
            if name in indices}
    path = {name: path
            for name, path in training_ds['path'].items()
            if name in indices}
    acoustic_features[dsname] = [X, indices, meta, path]
    continue
  # extract acoustic feature from scratch
  feat_dir = os.path.join(PATH_ACOUSTIC_FEATURES,
                          '%s_%s' % (dsname, EXTRACTOR_NAME))
  log_path = get_logpath(name='%s_%s.log' % (dsname, EXTRACTOR_NAME),
                         increasing=True, odin_base=False, root=EXP_DIR)
  # check if need running the feature extraction
  if _check_running_feature_extraction(feat_dir, n_files=len(file_list)):
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
  ds = F.Dataset(path=feat_dir, read_only=True)
  assert FEATURE_NAME in ds, \
      "Cannot find feature with name: %s, from: %s" % (FEATURE_NAME, ds.path)
  acoustic_features[dsname] = [
      ds[FEATURE_NAME],
      dict(ds['indices_%s' % FEATURE_NAME].items()),
      dict(ds['spkid'].items()),
      dict(ds['path'].items()),
  ]
# ====== print log ====== #
print("Acoustic features:")
for dsname, (X, indices, y, path) in sorted(acoustic_features.items(),
                                            key=lambda x: x[0]):
  all_utt_length = dict([(name, end - start)
                         for name, (start, end) in indices.items()])
  print("  %s" % ctext(dsname, 'yellow'))
  print("   #Files         :", ctext(len(indices), 'cyan'))
  print("   #Noise         : %s/%s" % (
      ctext(len([i for i in indices if '/' in i]), 'lightcyan'),
      ctext(len(indices), 'cyan')))
  print("   Loaded features:", ctext(X.path, 'cyan'))
  print("   Utt length     :", describe(list(all_utt_length.values()), shorten=True))
  print("   Min length(+8) :")
  min_length = min(all_utt_length.values())
  for name, length in all_utt_length.items():
    if length <= min_length + 8:
      print('    %s | %s' % (name.split('/')[0], path[name]))
# ===========================================================================
# All system must extract following information
# ===========================================================================
# mapping from
# dataset_name -> 'name': 1-D array [n_samples],
#                 # (path to original audio)
#                 'path': 1-D array [n_samples],
#                 # Extracted latent vectors
#                 'X': 2-D array [n_samples, n_latent_dim]}
#                 # speaker label or meta-data (e.g. 'test', 'enroll', 'unlabeled')
#                 'y': 1-D array [n_samples],
all_vectors = {}
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
  # ==================== extract x-vector from acoustic features ==================== #
  for dsname, (ds_feat, ds_indices, ds_meta, ds_path) in sorted(
      acoustic_features.items(), key=lambda x: x[0]):
    n_files = len(ds_indices)
    # ====== check exist scores ====== #
    vector_outpath = get_vectors_outpath(dsname)
    if os.path.exists(vector_outpath):
      with open(vector_outpath, 'rb') as f:
        vectors = pickle.load(f)
        if (len(vectors['name']) == len(vectors['y']) ==
                len(vectors['path']) == len(vectors['X']) <= n_files):
          all_vectors[dsname] = vectors
          print(' - Loaded vectors at:', ctext(vector_outpath, 'yellow'))
          if len(vectors['name']) != n_files:
            print('    [WARNING] Extracted scores only for: %s/%s (files)' %
              (ctext(len(vectors['name']), 'lightcyan'),
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
                   name='Extract vectors: %s' % dsname)
    # ====== make prediction ====== #
    for batch_idx, (name, idx, X) in enumerate(feeder.set_batch(
        batch_size=100000, seed=None, shuffle_level=0)):
      assert idx == 0, "File '%s' longer than maximum batch size" % name
      z = f_z(X)
      if z.shape[0] > 1:
        z = np.mean(z, axis=0, keepdims=True)
      output_name.append(name)
      output_meta.append(ds_meta[name])
      output_path.append(ds_path[name])
      output_data.append(z)
      # update the progress
      prog['ds'] = dsname
      prog['name'] = name[:48]
      prog['latent'] = z.shape
      prog['outpath'] = vector_outpath
      prog.add(X.shape[0])
    # ====== post-processing ====== #
    output_name = np.array(output_name)
    output_meta = np.array(output_meta)
    output_path = np.array(output_path)
    output_data = np.concatenate(output_data, axis=0)
    # ====== save the score ====== #
    with open(vector_outpath, 'wb') as f:
      scores = {'name': output_name,
                'path': output_path,
                'X': output_data.astype('float32'),
                'y': output_meta}
      pickle.dump(scores, f)
      all_vectors[dsname] = scores
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
all_backend_data = {name: all_vectors[name]
                    for name in BACKEND_DATASETS.keys()}
X_backend = []
y_backend = []
n_speakers = 0
for dsname, vectors in all_backend_data.items():
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
# ******************** optional save data to matlab for testing ******************** #
with open('/tmp/backend.mat', 'wb') as ftmp:
  savemat(ftmp, {'X': np.array(X_backend.astype('float32'), order='F'),
                 'y': np.array(y_backend.astype('int32'), order='F')})
for dsname in SCORING_DATASETS.keys():
  vectors = all_vectors[dsname]
  with open(os.path.join('/tmp', '%s.mat' % dsname), 'wb') as ftmp:
    y = []
    for i in range(len(vectors['X'])):
      name = vectors['name'][i]
      path = vectors['path'][i]
      if path is not None:
        name += os.path.splitext(path)[-1]
      y.append(name)
    savemat(ftmp, {'X': np.array(vectors['X'].astype('float32'), order='F'),
                   'y': np.array(y)})
# ===========================================================================
# Training the PLDA
# ===========================================================================
# ====== training the LDA ====== #
if N_LDA > 0:
  print("  Fitting LDA ...")
  lda = LinearDiscriminantAnalysis(n_components=N_LDA)
  X_backend = lda.fit_transform(X=X_backend, y=y_backend)
  lda_transform = lda.transform
else:
  lda_transform = lambda x: x
# ====== training the PLDA ====== #
plda = PLDA(n_phi=N_PLDA,
            centering=True, wccn=True, unit_length=True,
            n_iter=20, random_state=Config.SUPER_SEED,
            verbose=2 if PLDA_SHOW_LLK else 1)
if PLDA_MAXIMUM_LIKELIHOOD:
  print("  Fitting PLDA maximum likelihood ...")
  plda.fit_maximum_likelihood(X=lda_transform(X_backend), y=y_backend)
plda.fit(X=lda_transform(X_backend), y=y_backend)
# ===========================================================================
# Now scoring
# ===========================================================================
for dsname, scores in sorted(all_vectors.items(),
                             key=lambda x: x[0]):
  # ====== skip non scoring dataset ====== #
  if dsname not in SCORING_DATASETS:
    continue
  # ====== proceed ====== #
  print("Scoring:", ctext(dsname, 'yellow'))
  # load the scores
  (seg_name, seg_meta,
   seg_path, seg_data) = (scores['name'], scores['y'],
                          scores['path'], scores['X'])
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
    # ====== extract scores ====== #
    y_scores = plda.predict_log_proba(X=lda_transform(X_trials),
                                      X_model=X_models)
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
